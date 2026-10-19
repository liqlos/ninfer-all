#pragma once

#include "q4_fields.cuh"

#include <algorithm>
#include <stdexcept>

namespace ninfer::ops::gguf::detail::q4_fused_v9r2_fields {

using namespace vec;
using Metadata = q4_fields::Metadata;

struct ScaleArgs {
    VecArgs vec;
    const std::uint8_t* dm;
    const std::uint8_t* qs;
    const std::uint8_t* raw_metadata;
    const std::uint8_t* d_code;
    const std::uint8_t* dmin_code;
    std::uint32_t d_exp_min;
    std::uint32_t dmin_exp_min;
    std::uint32_t d_exp_bits;
    std::uint32_t dmin_exp_bits;
    std::uint32_t sign_bits;
    bool coded;
};

template <int T>
__launch_bounds__(128) __global__ void kernel(ScaleArgs p) {
    static_assert(T >= 1 && T <= 8);
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    constexpr int k = 5120;
    constexpr int blocks_per_row = k / 256;
    for (int row0 = blockIdx.x * 4 + warp; row0 < p.vec.rows;
         row0 += gridDim.x * 4) {
        float acc[2][T];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
#pragma unroll
            for (int j = 0; j < T; ++j) { acc[r][j] = 0.0f; }
        }
#pragma unroll
        for (int it = 0; it < k / 1024; ++it) {
            int a[T][8];
            float d[T], sum[T];
#pragma unroll
            for (int j = 0; j < T; ++j) {
                const float2 ds = __half22float2(p.vec.ds[std::int64_t(j) * (k / 32) + lane + 32 * it]);
                d[j] = ds.x;
                sum[j] = ds.y;
                const int4* q = reinterpret_cast<const int4*>(p.vec.qs + std::int64_t(j) * k + 32 * (lane + 32 * it));
                const int4 v0 = q[0], v1 = q[1];
                a[j][0] = v0.x; a[j][1] = v0.y; a[j][2] = v0.z; a[j][3] = v0.w;
                a[j][4] = v1.x; a[j][5] = v1.y; a[j][6] = v1.z; a[j][7] = v1.w;
            }
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                const int row = row0 + r * p.vec.rows;
                const std::uint32_t block = std::uint32_t(row) * std::uint32_t(blocks_per_row) + std::uint32_t(4 * it + lane / 8);
                int sc, mn;
                scale_min_k4(p.raw_metadata + block * 12, lane % 8, sc, mn);
                const std::uint16_t dword = q4_fields::decode_half<13, 0, false>(p.d_code, block);
                const std::uint16_t dminword = q4_fields::decode_half<13, 2, false>(p.dmin_code, block);
                const float2 dm = make_float2(__half2float(__ushort_as_half(dword)), __half2float(__ushort_as_half(dminword)));
                Slice sl;
                sl.f[0] = dm.x * float(sc);
                sl.f[1] = dm.y * float(mn);
                const uint4* q = reinterpret_cast<const uint4*>(p.qs + block * 128 + 32 * ((lane % 8) / 2));
                const uint4 q0 = q[0], q1 = q[1];
                const int shift = 4 * ((lane % 8) & 1);
                const std::uint32_t words[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
#pragma unroll
                for (int i = 0; i < 8; ++i) { sl.w[i] = int((words[i] >> shift) & 0x0F0F0F0Fu); }
#pragma unroll
                for (int j = 0; j < T; ++j) { acc[r][j] += Decoder<GGML_TYPE_Q4_K>::dot(sl, a[j], d[j], sum[j]); }
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
            }
        }
        if (lane == 0) {
#pragma unroll
            for (int j = 0; j < T; ++j) { vec_store(p.vec.out, row0, j, vec_silu(acc[0][j]) * acc[1][j]); }
        }
    }
}

template <int T>
static void launch_fused(VecArgs args, Metadata metadata, const std::uint8_t* d,
                         const std::uint8_t* dmin, cudaStream_t stream) {
    static_assert(T >= 1 && T <= 8);
    const auto fields = metadata.fields;
    if (args.rows != 17408 || args.k != 5120 || args.out.column_stride < args.rows ||
        !args.weight || !args.qs || !args.ds || !metadata.scales || !d || !dmin ||
        (args.out.bf16 == nullptr) == (args.out.f32 == nullptr) ||
        fields.d_width != 13 || fields.d_exponent_base != 0 || fields.d_exponent_bits != 3 ||
        fields.dmin_width != 13 || fields.dmin_exponent_base != 2 || fields.dmin_exponent_bits != 3) {
        throw std::invalid_argument("invalid fused Q4 r2 field vector launch");
    }
    ScaleArgs kernel_args{args, nullptr, args.weight, metadata.scales, d, dmin, 0, 2, 3, 3, 0, true};
    int active = 0;
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, kernel<T>, 128, 0),
          "fused Q4 r2 field occupancy");
    const int grid = std::min((args.rows + 3) / 4, std::max(active, 1) * device_facts().sm_count);
    kernel<T><<<grid, 128, 0, stream>>>(kernel_args);
    check(cudaGetLastError(), "fused Q4 r2 field launch");
}

}
