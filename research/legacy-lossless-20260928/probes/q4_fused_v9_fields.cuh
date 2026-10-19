#pragma once

#include "core/weight.h"
#include "ops/linear/gguf/ggml_bridge_vec.cuh"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::gguf::detail::q4_fields {

using namespace vec;

struct Metadata {
    const std::uint8_t* scales;
    Q4FieldEncoding fields;
};

template <unsigned Width, unsigned ExponentBase, bool WordRead>
__device__ __forceinline__ std::uint16_t decode_half(const std::uint8_t* stream,
                                                      std::uint32_t block) {
    static_assert(Width == 12 || Width == 13);
    const std::uint32_t bit = block * Width;
    if constexpr (WordRead && Width == 13) {
        const auto* words = reinterpret_cast<const std::uint32_t*>(stream);
        const std::uint32_t at = bit >> 5;
        const std::uint32_t code = __funnelshift_r(words[at], words[at + 1], bit & 31u) & 8191u;
        return std::uint16_t((code & 0x03ffu) | ((ExponentBase + (code >> 10)) << 10));
    }
    const std::uint32_t at = bit >> 3;
    const unsigned shift = bit & 7u;
    std::uint32_t value = std::uint32_t(stream[at]) | (std::uint32_t(stream[at + 1]) << 8);
    if constexpr (Width == 13) {
        if (shift > 3) { value |= std::uint32_t(stream[at + 2]) << 16; }
    }
    const std::uint32_t code = (value >> shift) & ((1u << Width) - 1u);
    return std::uint16_t((code & 0x03ffu) | ((ExponentBase + (code >> 10)) << 10));
}

template <int T, int K, unsigned DWidth, unsigned DBase, unsigned DMinWidth, unsigned DMinBase, bool Fused = false>
__launch_bounds__(128) __global__ void kernel(VecArgs p, Metadata metadata,
                                               const std::uint8_t* d,
                                               const std::uint8_t* dmin,
                                               std::uint64_t row_begin) {
    static_assert(K % 1024 == 0);
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    constexpr int k = K;
    constexpr int slices = k / 32;
    constexpr int blocks_per_row = k / 256;
    constexpr int step = Fused ? 1 : 2;
    for (int row0 = (blockIdx.x * 4 + warp) * step; row0 < p.rows;
         row0 += gridDim.x * 4 * step) {
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
                const int row = Fused ? row0 + r * p.rows : row0 + r;
                if (Fused || row < p.rows) {
                    const std::uint32_t local_block = std::uint32_t(row) * blocks_per_row +
                                                      std::uint32_t(4 * it + lane / 8);
                    const std::uint32_t global_block = (std::uint32_t(Fused ? 0 : row_begin) + std::uint32_t(row)) *
                                                       blocks_per_row + std::uint32_t(4 * it + lane / 8);
                    int scale, minimum;
                    vec::scale_min_k4(metadata.scales + local_block * 12, lane % 8, scale, minimum);
                    const std::uint16_t d_bits = decode_half<DWidth, DBase, T == 4 && K == 17408>(d, global_block);
                    const std::uint16_t dmin_bits = decode_half<DMinWidth, DMinBase, T == 4 && K == 17408>(dmin, global_block);
                    const float2 dm = make_float2(__half2float(__ushort_as_half(d_bits)),
                                                  __half2float(__ushort_as_half(dmin_bits)));
                    const uint4* q = reinterpret_cast<const uint4*>(p.weight + local_block * 128 +
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
                if constexpr (!Fused) {
                    if (lane == 0 && row0 + r < p.rows) { vec_store(p.out, row0 + r, j, acc[r][j]); }
                }
            }
        }
        if constexpr (Fused) {
            if (lane == 0) {
#pragma unroll
                for (int j = 0; j < T; ++j) { vec_store(p.out, row0, j, vec_silu(acc[0][j]) * acc[1][j]); }
            }
        }
    }
}

template <int T, int K, unsigned DWidth>
static void launch_fixed(VecArgs args, Metadata metadata, const std::uint8_t* d,
                         const std::uint8_t* dmin, std::uint64_t row_begin,
                         cudaStream_t stream) {
    int active = 0;
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, kernel<T, K, DWidth, 0, 13, 2>, 128, 0),
          "Q4 fixed-field occupancy");
    const int groups = (args.rows + 7) / 8;
    const int grid = std::min(groups, std::max(active, 1) * device_facts().sm_count);
    kernel<T, K, DWidth, 0, 13, 2><<<grid, 128, 0, stream>>>(args, metadata, d, dmin, row_begin);
    check(cudaGetLastError(), "Q4 fixed-field launch");
}

template <int T>
static void launch(VecArgs args, Metadata metadata, const std::uint8_t* d,
                   const std::uint8_t* dmin, std::uint64_t row_begin,
                   cudaStream_t stream) {
    const std::uint64_t root_rows = args.k == 5120 ? (metadata.fields.d_width == 13 ? 34816 : 16384) : args.k == 17408 ? 5120 : 0;
    if (!root_rows || args.rows <= 0 || std::uint64_t(args.rows) > root_rows || row_begin > root_rows ||
        std::uint64_t(args.rows) > root_rows - row_begin || !args.weight || !args.qs ||
        !args.ds || !metadata.scales || !d || !dmin) {
        throw std::invalid_argument("invalid Q4 field vector launch");
    }
    const Q4FieldEncoding fields = metadata.fields;
    if constexpr (T == 4) {
        if ((reinterpret_cast<std::uintptr_t>(d) & 3u) ||
            (reinterpret_cast<std::uintptr_t>(dmin) & 3u)) {
            throw std::invalid_argument("Q4 field word plane is not aligned");
        }
    }
    if (row_begin > std::numeric_limits<std::uint64_t>::max() - std::uint64_t(args.rows)) {
        throw std::invalid_argument("Q4 field row range overflows");
    }
    const std::uint64_t max_blocks = std::numeric_limits<std::uint32_t>::max() / 13u;
    const std::uint64_t blocks_per_row = std::uint64_t(args.k / 256);
    if (row_begin + std::uint64_t(args.rows) > max_blocks / blocks_per_row) {
        throw std::invalid_argument("Q4 field bit index exceeds 32-bit decoder range");
    }
    if (args.k == 5120 && fields.d_width == 12 && fields.d_exponent_base == 0 && fields.d_exponent_bits == 2 &&
        fields.dmin_width == 13 && fields.dmin_exponent_base == 2 && fields.dmin_exponent_bits == 3) {
        launch_fixed<T, 5120, 12>(args, metadata, d, dmin, row_begin, stream);
        return;
    }
    if (args.k == 5120 && fields.d_width == 13 && fields.d_exponent_base == 0 && fields.d_exponent_bits == 3 &&
        fields.dmin_width == 13 && fields.dmin_exponent_base == 2 && fields.dmin_exponent_bits == 3) {
        launch_fixed<T, 5120, 13>(args, metadata, d, dmin, row_begin, stream);
        return;
    }
    if (args.k == 17408 && fields.d_width == 13 && fields.d_exponent_base == 0 && fields.d_exponent_bits == 3 &&
        fields.dmin_width == 13 && fields.dmin_exponent_base == 2 && fields.dmin_exponent_bits == 3) {
        launch_fixed<T, 17408, 13>(args, metadata, d, dmin, row_begin, stream);
        return;
    }
    throw std::invalid_argument("unsupported Q4 field encoding specialization");
}

template <int T>
static void launch_fused(VecArgs args, Metadata metadata, const std::uint8_t* d,
                         const std::uint8_t* dmin, cudaStream_t stream) {
    static_assert(T == 4);
    const auto fields = metadata.fields;
    if (args.rows != 17408 || args.k != 5120 || args.out.column_stride < args.rows ||
        !args.weight || !args.qs || !args.ds || !metadata.scales || !d || !dmin ||
        (args.out.bf16 == nullptr) == (args.out.f32 == nullptr) ||
        fields.d_width != 13 || fields.d_exponent_base != 0 || fields.d_exponent_bits != 3 ||
        fields.dmin_width != 13 || fields.dmin_exponent_base != 2 || fields.dmin_exponent_bits != 3) {
        throw std::invalid_argument("invalid fused Q4 field vector launch");
    }
    int active = 0;
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, kernel<T, 5120, 13, 0, 13, 2, true>, 128, 0),
          "fused Q4 field occupancy");
    const int grid = std::min((args.rows + 3) / 4, std::max(active, 1) * device_facts().sm_count);
    kernel<T, 5120, 13, 0, 13, 2, true><<<grid, 128, 0, stream>>>(args, metadata, d, dmin, 0);
    check(cudaGetLastError(), "fused Q4 field launch");
}

}
