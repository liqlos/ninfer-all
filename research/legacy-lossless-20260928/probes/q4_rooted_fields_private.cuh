#pragma once

#include "ops/linear/gguf/q4_fields.cuh"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::gguf::detail::q4_rooted_fields {

using namespace vec;
using q4_fields::Metadata;
using q4_fields::decode_half;

template <int T, unsigned DWidth, unsigned DBase, unsigned DMinWidth, unsigned DMinBase>
__launch_bounds__(128) __global__ void kernel(VecArgs p, Metadata metadata,
                                               const std::uint8_t* d,
                                               const std::uint8_t* dmin,
                                               std::uint64_t row_begin) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    constexpr int k = 5120;
    constexpr int slices = k / 32;
    constexpr int blocks_per_row = k / 256;
    for (int row0 = (blockIdx.x * 4 + warp) * 2; row0 < p.rows;
         row0 += gridDim.x * 8) {
        float acc[2][T];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
#pragma unroll
            for (int j = 0; j < T; ++j) { acc[r][j] = 0.0f; }
        }
#pragma unroll
        for (int it = 0; it < k / 1024; ++it) {
            int a[T][8];
            float activation_d[T], activation_sum[T];
#pragma unroll
            for (int j = 0; j < T; ++j) {
                const float2 ds = __half22float2(p.ds[std::int64_t(j) * slices + lane + 32 * it]);
                activation_d[j] = ds.x;
                activation_sum[j] = ds.y;
                const int4* q = reinterpret_cast<const int4*>(p.qs + std::int64_t(j) * k + 32 * (lane + 32 * it));
                const int4 v0 = q[0], v1 = q[1];
                a[j][0] = v0.x; a[j][1] = v0.y; a[j][2] = v0.z; a[j][3] = v0.w;
                a[j][4] = v1.x; a[j][5] = v1.y; a[j][6] = v1.z; a[j][7] = v1.w;
            }
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                const int row = row0 + r;
                if (row < p.rows) {
                    const std::uint32_t global_block = (std::uint32_t(row_begin) + std::uint32_t(row)) *
                                                       blocks_per_row + std::uint32_t(4 * it + lane / 8);
                    int scale, minimum;
                    vec::scale_min_k4(metadata.scales + std::uint64_t(global_block) * 12, lane % 8,
                                      scale, minimum);
                    const std::uint16_t d_bits = decode_half<DWidth, DBase>(d, global_block);
                    const std::uint16_t dmin_bits = decode_half<DMinWidth, DMinBase>(dmin, global_block);
                    const float2 dm = make_float2(__half2float(__ushort_as_half(d_bits)),
                                                  __half2float(__ushort_as_half(dmin_bits)));
                    const uint4* q = reinterpret_cast<const uint4*>(p.weight + std::uint64_t(global_block) * 128 +
                                                                     32 * ((lane % 8) / 2));
                    const uint4 q0 = q[0], q1 = q[1];
                    const int shift = 4 * ((lane % 8) & 1);
                    Slice sl;
                    sl.f[0] = dm.x * float(scale);
                    sl.f[1] = dm.y * float(minimum);
                    const std::uint32_t words[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
#pragma unroll
                    for (int i = 0; i < 8; ++i) { sl.w[i] = int((words[i] >> shift) & 0x0F0F0F0Fu); }
#pragma unroll
                    for (int j = 0; j < T; ++j) {
                        acc[r][j] += Decoder<GGML_TYPE_Q4_K>::dot(sl, a[j], activation_d[j], activation_sum[j]);
                    }
                }
            }
        }
#pragma unroll
        for (int r = 0; r < 2; ++r) {
#pragma unroll
            for (int j = 0; j < T; ++j) {
#pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1) {
                    acc[r][j] += __shfl_xor_sync(0xffffffffu, acc[r][j], offset);
                }
                if (lane == 0 && row0 + r < p.rows) { vec_store(p.out, row0 + r, j, acc[r][j]); }
            }
        }
    }
}

template <int T>
static void launch(VecArgs args, Metadata metadata, const std::uint8_t* d,
                   const std::uint8_t* dmin, std::uint64_t row_begin,
                   cudaStream_t stream) {
    if (args.k != 5120 || args.rows <= 0 || args.rows > 16384 || row_begin > 16384 ||
        row_begin + std::uint64_t(args.rows) > 16384 || !args.weight || !args.qs ||
        !args.ds || !metadata.scales || !d || !dmin) {
        throw std::invalid_argument("invalid rooted Q4 field vector launch");
    }
    if (row_begin > std::numeric_limits<std::uint64_t>::max() - std::uint64_t(args.rows)) {
        throw std::invalid_argument("rooted Q4 field row range overflows");
    }
    const auto fields = metadata.fields;
    if (fields.d_width != 12 || fields.d_exponent_base != 0 || fields.d_exponent_bits != 2 ||
        fields.dmin_width != 13 || fields.dmin_exponent_base != 2 || fields.dmin_exponent_bits != 3) {
        throw std::invalid_argument("unsupported rooted Q4 field encoding");
    }
    const std::uint64_t blocks_per_row = std::uint64_t(args.k / 256);
    const std::uint64_t row_code_bytes = blocks_per_row * 128;
    const std::uint64_t row_scale_bytes = blocks_per_row * 12;
    args.weight -= row_begin * row_code_bytes;
    metadata.scales -= row_begin * row_scale_bytes;
    int active = 0;
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,
        kernel<T, 12, 0, 13, 2>, 128, 0), "rooted Q4 occupancy");
    const int groups = (args.rows + 7) / 8;
    const int grid = std::min(groups, std::max(active, 1) * device_facts().sm_count);
    q4_rooted_fields::kernel<T, 12, 0, 13, 2><<<grid, 128, 0, stream>>>(args, metadata, d, dmin, row_begin);
    check(cudaGetLastError(), "rooted Q4 launch");
}

}
