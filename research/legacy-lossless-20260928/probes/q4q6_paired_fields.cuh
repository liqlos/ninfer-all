#pragma once

#include "ops/linear/gguf/q4_fields.cuh"
#include "q6_regbalance_fields.cuh"

namespace ninfer::ops::gguf::detail::paired_split {

using namespace vec;

struct Args {
    VecArgs vector;
    VecStore first;
    VecStore second;
    int split_rows;
};

inline void validate(const Args& p) {
    const auto valid = [](const VecStore& output, int rows) {
        const bool f32 = output.f32 != nullptr && output.bf16 == nullptr;
        const bool bf16 = output.bf16 != nullptr && output.f32 == nullptr;
        return (f32 || bf16) && output.column_stride == rows &&
               output.epilogue == Epilogue::Store && output.gate == nullptr && output.gate_column_stride == 0;
    };
    if (p.vector.rows != 16384 || p.vector.k != 5120 || p.split_rows != 10240 ||
        !p.vector.weight || !p.vector.qs || !p.vector.ds ||
        !valid(p.first, p.split_rows) || !valid(p.second, p.vector.rows - p.split_rows)) {
        throw std::invalid_argument("invalid paired split-store vectors");
    }
}

__device__ __forceinline__ void store(const Args& p, int row, int column, float value) {
    if (row < p.split_rows) { vec_store(p.first, row, column, value); }
    else { vec_store(p.second, row - p.split_rows, column, value); }
}

template <int T, unsigned DWidth, unsigned DBase, unsigned DMinWidth, unsigned DMinBase>
__launch_bounds__(128) __global__ void q4_kernel(Args pair,
    q4_fields::Metadata metadata, const std::uint8_t* d, const std::uint8_t* dmin) {
    constexpr int K = 5120;
    constexpr int slices = K / 32;
    constexpr int blocks_per_row = K / 256;
    const VecArgs p = pair.vector;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    for (int row0 = (blockIdx.x * 4 + warp) * 2; row0 < p.rows; row0 += gridDim.x * 8) {
        float acc[2][T];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
#pragma unroll
            for (int j = 0; j < T; ++j) { acc[r][j] = 0.0f; }
        }
#pragma unroll
        for (int it = 0; it < K / 1024; ++it) {
            int a[T][8];
            float activation_d[T], activation_sum[T];
#pragma unroll
            for (int j = 0; j < T; ++j) {
                const float2 ds = __half22float2(p.ds[std::int64_t(j) * slices + lane + 32 * it]);
                activation_d[j] = ds.x;
                activation_sum[j] = ds.y;
                const int4* q = reinterpret_cast<const int4*>(p.qs + std::int64_t(j) * K + 32 * (lane + 32 * it));
                const int4 v0 = q[0], v1 = q[1];
                a[j][0] = v0.x; a[j][1] = v0.y; a[j][2] = v0.z; a[j][3] = v0.w;
                a[j][4] = v1.x; a[j][5] = v1.y; a[j][6] = v1.z; a[j][7] = v1.w;
            }
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                const int row = row0 + r;
                if (row < p.rows) {
                    const std::uint32_t block = std::uint32_t(row) * blocks_per_row + std::uint32_t(4 * it + lane / 8);
                    int scale, minimum;
                    vec::scale_min_k4(metadata.scales + block * 12, lane % 8, scale, minimum);
                    const std::uint16_t d_bits = q4_fields::decode_half<DWidth, DBase, false>(d, block);
                    const std::uint16_t dmin_bits = q4_fields::decode_half<DMinWidth, DMinBase, false>(dmin, block);
                    const float2 dm = make_float2(__half2float(__ushort_as_half(d_bits)),
                                                  __half2float(__ushort_as_half(dmin_bits)));
                    const uint4* q = reinterpret_cast<const uint4*>(p.weight + std::int64_t(block) * 128 +
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
                if (lane == 0 && row0 + r < p.rows) { store(pair, row0 + r, j, acc[r][j]); }
            }
        }
    }
}

template <int T>
static void launch_q4(Args args, q4_fields::Metadata metadata, const std::uint8_t* d,
                      const std::uint8_t* dmin, std::uint64_t row_begin, cudaStream_t stream) {
    if constexpr (T != 4) { throw std::invalid_argument("paired Q4 probe requires T4"); }
    validate(args);
    if (args.vector.rows != 16384 || args.split_rows != 10240 || args.vector.k != 5120 ||
        metadata.fields.d_width != 12 || metadata.fields.d_exponent_base != 0 || metadata.fields.d_exponent_bits != 2 ||
        metadata.fields.dmin_width != 13 || metadata.fields.dmin_exponent_base != 2 || metadata.fields.dmin_exponent_bits != 3 ||
        row_begin != 0 || (reinterpret_cast<std::uintptr_t>(d) & 3u) ||
        (reinterpret_cast<std::uintptr_t>(dmin) & 3u)) {
        throw std::invalid_argument("unsupported paired Q4 parent descriptor");
    }
    int active = 0;
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, q4_kernel<T, 12, 0, 13, 2>, 128, 0), "paired Q4 occupancy");
    const int grid = std::min((args.vector.rows + 7) / 8, std::max(active, 1) * device_facts().sm_count);
    q4_kernel<T, 12, 0, 13, 2><<<grid, 128, 0, stream>>>(args, metadata, d, dmin);
    check(cudaGetLastError(), "paired Q4 launch");
}

template <int T>
__launch_bounds__(128, 5) __global__ void q6_kernel(Args pair, const std::uint8_t* d_bits,
    std::uint8_t d_width, std::uint8_t d_exp_base, std::uint8_t d_exp_bits,
    std::uint8_t d_sign_variable, std::uint8_t d_sign_value) {
    asm volatile (".pragma \"enable_smem_spilling\";");
    const VecArgs p = pair.vector;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    constexpr int K = 5120;
    constexpr int slices = K / 32;
    constexpr int blocks_per_row = K / 256;
    constexpr int tile_bytes = 832;
    constexpr int row_bytes = (blocks_per_row / 4) * tile_bytes;
    for (int row0 = (blockIdx.x * 4 + warp) * 2; row0 < p.rows; row0 += gridDim.x * 8) {
        float acc[2][T];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
#pragma unroll
            for (int j = 0; j < T; ++j) { acc[r][j] = 0.0f; }
        }
        for (int it = 0; lane + 32 * it < slices; ++it) {
            int a[T][8];
            float d[T], sum[T];
#pragma unroll
            for (int j = 0; j < T; ++j) {
                const float2 ds = __half22float2(p.ds[std::int64_t(j) * slices + lane + 32 * it]);
                d[j] = ds.x;
                sum[j] = ds.y;
                const int4* q = reinterpret_cast<const int4*>(p.qs + std::int64_t(j) * K + 32 * (lane + 32 * it));
                const int4 v0 = q[0], v1 = q[1];
                a[j][0] = v0.x; a[j][1] = v0.y; a[j][2] = v0.z; a[j][3] = v0.w;
                a[j][4] = v1.x; a[j][5] = v1.y; a[j][6] = v1.z; a[j][7] = v1.w;
            }
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                const int row = row0 + r;
                if (row < p.rows) {
                    const int slot = lane / 8;
                    const int tile_index = it;
                    const std::uint8_t* tile = p.weight + std::int64_t(row) * row_bytes + std::int64_t(tile_index) * tile_bytes;
                    Slice sl;
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const int h = (lane % 8) / 4;
                        const int q = lane % 4;
                        const int group = h * 2 + (q & 1);
                        const std::size_t lo_offset = std::size_t(slot * 128 + group * 32 + i * 4);
                        const std::size_t hi_offset = std::size_t(512 + slot * 64 + h * 32 + i * 4);
                        const std::uint32_t lo = q6_regbalance::load_word(tile + lo_offset);
                        const std::uint32_t hi = q6_regbalance::load_word(tile + hi_offset);
                        const std::uint32_t code = ((lo >> (4 * (q >> 1))) & 0x0F0F0F0Fu) |
                            (((hi >> (2 * q)) << 4) & 0x30303030u);
                        sl.w[i] = int((code + 0x60606060u) ^ 0x80808080u);
                    }
                    const int h = (lane % 8) / 4;
                    const int q = lane % 4;
                    const std::size_t scale_offset = std::size_t(768 + slot * 16 + h * 8 + 2 * q);
                    const std::uint8_t* scale_ptr = tile + scale_offset;
                    const std::int8_t s0 = *reinterpret_cast<const std::int8_t*>(scale_ptr);
                    const std::int8_t s1 = *reinterpret_cast<const std::int8_t*>(scale_ptr + 1);
                    const std::uint32_t block_index = std::uint32_t(row) * blocks_per_row + std::uint32_t(tile_index * 4 + slot);
                    const std::uint32_t code = q6_regbalance::read_bits_device12(d_bits, block_index);
                    const std::uint16_t half_bits = std::uint16_t((code & 0x07ffu) | ((code >> 11) << 15));
                    const float block_d = __half2float(__ushort_as_half(half_bits));
                    sl.f[0] = block_d * float(s0);
                    sl.f[1] = block_d * float(s1);
#pragma unroll
                    for (int j = 0; j < T; ++j) { acc[r][j] += Decoder<GGML_TYPE_Q6_K>::dot(sl, a[j], d[j], sum[j]); }
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
                if (lane == 0 && row0 + r < p.rows) { store(pair, row0 + r, j, acc[r][j]); }
            }
        }
    }
}

template <int T>
static void launch_q6(Args args, const std::uint8_t* d_bits,
                      std::uint8_t d_width, std::uint8_t d_exp_base, std::uint8_t d_exp_bits,
                      std::uint8_t d_sign_variable, std::uint8_t d_sign_value, std::uint64_t row_begin, cudaStream_t stream) {
    if constexpr (T != 4) { throw std::invalid_argument("paired Q6 probe requires T4"); }
    validate(args);
    if (args.vector.rows != 16384 || args.split_rows != 10240 || args.vector.k != 5120 ||
        d_width != 12 || d_exp_base != 0 || d_exp_bits != 1 || !d_sign_variable || row_begin != 0 ||
        (reinterpret_cast<std::uintptr_t>(d_bits) & 3u)) {
        throw std::invalid_argument("unsupported paired Q6 parent descriptor");
    }
    int active = 0;
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, q6_kernel<T>, 128, 0), "paired Q6 occupancy");
    const int grid = std::min((args.vector.rows + 7) / 8, std::max(active, 1) * device_facts().sm_count);
    q6_kernel<T><<<grid, 128, 0, stream>>>(args, d_bits, d_width, d_exp_base, d_exp_bits, d_sign_variable, d_sign_value);
    check(cudaGetLastError(), "paired Q6 launch");
}

}