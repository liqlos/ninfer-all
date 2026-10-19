#pragma once

#include "ops/linear/gguf/ggml_bridge_vec.cuh"

namespace ninfer::ops::gguf::detail::q6_shared_activation_fields {

using namespace vec;

__device__ __forceinline__ uint32_t load_word(const uint8_t* p) {
    return *reinterpret_cast<const uint32_t*>(p);
}

__device__ __forceinline__ uint32_t read_bits_device(const uint8_t* data, uint32_t at, uint32_t width) {
    const uint32_t byte = at >> 3;
    const uint32_t shift = at & 7;
    uint32_t value = uint32_t(data[byte]) | (uint32_t(data[byte + 1]) << 8);
    if (shift + width > 16) { value |= uint32_t(data[byte + 2]) << 16; }
    return (value >> shift) & ((1u << width) - 1u);
}

__device__ __forceinline__ uint32_t read_bits_device12(const uint8_t* data, uint32_t block) {
    const uint32_t byte = (block * 3u) >> 1;
    const uint32_t shift = (block & 1u) << 2;
    const uint32_t pair = uint32_t(data[byte]) | (uint32_t(data[byte + 1]) << 8);
    return (pair >> shift) & 0x0fffu;
}

__device__ __forceinline__ int decode_word(uint32_t lo, uint32_t hi, int q) {
    const uint32_t code = ((lo >> (4 * (q >> 1))) & 0x0F0F0F0Fu) |
                          (((hi >> (2 * q)) << 4) & 0x30303030u);
    return int(code ^ 0xE0E0E0E0u ^ ((code & 0x20202020u) * 6u));
}

template <int T, bool Transposed, bool CompressedD = false, bool FastWidth12 = false, int Threads = 128, bool VectorLoads = false, bool SharedActivation = false>
__launch_bounds__(Threads) __global__ void kernel(VecArgs p, const uint8_t* d_bits,
                                                        uint8_t d_width, uint8_t d_exp_base,
                                                        uint8_t d_exp_bits, uint8_t d_sign_variable,
                                                        uint8_t d_sign_value, uint32_t row_begin) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int k = p.k;
    const int slices = k / 32;
    const int blocks_per_row = k / 256;
    const int tile_bytes = CompressedD ? 832 : 840;
    const int row_bytes = (blocks_per_row / 4) * tile_bytes;
    __shared__ int4 shared_q_words[T][320];
    __shared__ half2 shared_ds[T][160];
    if constexpr (SharedActivation) {
        static_assert(T == 1);
        for (int i = threadIdx.x; i < 320; i += Threads) {
            shared_q_words[0][i] = reinterpret_cast<const int4*>(p.qs)[i];
        }
        for (int i = threadIdx.x; i < 160; i += Threads) { shared_ds[0][i] = p.ds[i]; }
        __syncthreads();
    }
    for (int row0 = (blockIdx.x * (Threads / 32) + warp) * 2; row0 < p.rows;
         row0 += gridDim.x * (Threads / 16)) {
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
                const int activation_slice = int64_t(j) * slices + lane + 32 * it;
                const float2 ds = __half22float2(SharedActivation ? shared_ds[j][lane + 32 * it] : p.ds[activation_slice]);
                d[j] = ds.x;
                sum[j] = ds.y;
                if constexpr (SharedActivation) {
                    const int4 v0 = shared_q_words[j][2 * (lane + 32 * it)];
                    const int4 v1 = shared_q_words[j][2 * (lane + 32 * it) + 1];
                    a[j][0] = v0.x; a[j][1] = v0.y; a[j][2] = v0.z; a[j][3] = v0.w;
                    a[j][4] = v1.x; a[j][5] = v1.y; a[j][6] = v1.z; a[j][7] = v1.w;
                } else {
                    const int4* q = reinterpret_cast<const int4*>(p.qs + int64_t(j) * k + 32 * (lane + 32 * it));
                    const int4 v0 = q[0], v1 = q[1];
                    a[j][0] = v0.x; a[j][1] = v0.y; a[j][2] = v0.z; a[j][3] = v0.w;
                    a[j][4] = v1.x; a[j][5] = v1.y; a[j][6] = v1.z; a[j][7] = v1.w;
                }
            }
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                const int row = row0 + r;
                if (row < p.rows) {
                    const int slot = lane / 8;
                    const int tile_index = it;
                    const uint8_t* tile = p.weight + int64_t(row) * row_bytes + int64_t(tile_index) * tile_bytes;
                    Slice sl;
                    if constexpr (VectorLoads) {
                        static_assert(CompressedD && !Transposed);
                        const int h = (lane % 8) / 4;
                        const int q = lane % 4;
                        const int group = h * 2 + (q & 1);
                        const size_t lo_offset = size_t(slot * 128 + group * 32);
                        const size_t hi_offset = size_t(512 + slot * 64 + h * 32);
                        {
                            const uint4 lo = *reinterpret_cast<const uint4*>(tile + lo_offset);
                            const uint4 hi = *reinterpret_cast<const uint4*>(tile + hi_offset);
                            sl.w[0] = decode_word(lo.x, hi.x, q);
                            sl.w[1] = decode_word(lo.y, hi.y, q);
                            sl.w[2] = decode_word(lo.z, hi.z, q);
                            sl.w[3] = decode_word(lo.w, hi.w, q);
                        }
                        {
                            const uint4 lo = *reinterpret_cast<const uint4*>(tile + lo_offset + 16);
                            const uint4 hi = *reinterpret_cast<const uint4*>(tile + hi_offset + 16);
                            sl.w[4] = decode_word(lo.x, hi.x, q);
                            sl.w[5] = decode_word(lo.y, hi.y, q);
                            sl.w[6] = decode_word(lo.z, hi.z, q);
                            sl.w[7] = decode_word(lo.w, hi.w, q);
                        }
                    } else {
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const int h = (lane % 8) / 4;
                        const int q = lane % 4;
                        const int group = h * 2 + (q & 1);
                        const size_t lo_offset = Transposed ? size_t((i * 4 + group) * 16 + slot * 4) :
                                                              size_t(slot * 128 + group * 32 + i * 4);
                        const size_t hi_offset = Transposed ? size_t(512 + (i * 2 + h) * 16 + slot * 4) :
                                                              size_t(512 + slot * 64 + h * 32 + i * 4);
                        const uint32_t lo = load_word(tile + lo_offset);
                        const uint32_t hi = load_word(tile + hi_offset);
                        const uint32_t code = ((lo >> (4 * (q >> 1))) & 0x0F0F0F0Fu) |
                                              (((hi >> (2 * q)) << 4) & 0x30303030u);
                        if constexpr (FastWidth12) { sl.w[i] = int((code + 0x60606060u) ^ 0x80808080u); }
                        else { sl.w[i] = int(code ^ 0xE0E0E0E0u ^ ((code & 0x20202020u) * 6u)); }
                    }
                    }
                    const int h = (lane % 8) / 4;
                    const int q = lane % 4;
                    const size_t scale_offset = Transposed ? size_t(768 + (h * 8 + 2 * q) * 4 + slot) :
                                                              size_t(768 + slot * 16 + h * 8 + 2 * q);
                    const uint8_t* scale_ptr = tile + scale_offset;
                    const int8_t s0 = *reinterpret_cast<const int8_t*>(scale_ptr);
                    const int8_t s1 = *reinterpret_cast<const int8_t*>(scale_ptr + (Transposed ? 4 : 1));
                    float block_d;
                    if constexpr (CompressedD) {
                        const uint32_t block_index = (row_begin + uint32_t(row)) * uint32_t(blocks_per_row) + uint32_t(tile_index * 4 + slot);
                        if constexpr (FastWidth12) {
                            const uint32_t code = read_bits_device12(d_bits, block_index);
                            const uint16_t half_bits = uint16_t((code & 0x07ffu) | ((code >> 11) << 15));
                            block_d = __half2float(__ushort_as_half(half_bits));
                        } else {
                            const uint32_t code = read_bits_device(d_bits, block_index * uint32_t(d_width), d_width);
                            const uint32_t exp_mask = (1u << d_exp_bits) - 1u;
                            const uint32_t exponent = uint32_t(d_exp_base) + ((code >> 10) & exp_mask);
                            const uint32_t sign = d_sign_variable ? ((code >> (10 + d_exp_bits)) & 1u) : d_sign_value;
                            const uint16_t half_bits = uint16_t((code & 0x03ffu) | (exponent << 10) | (sign << 15));
                            block_d = __half2float(__ushort_as_half(half_bits));
                        }
                    } else {
                        block_d = half_at(tile + 832 + slot * 2);
                    }
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
                if (lane == 0 && row0 + r < p.rows) { vec_store(p.out, row0 + r, j, acc[r][j]); }
            }
        }
    }
}

template <int T, bool Transposed, bool CompressedD = false>
static void launch(VecArgs args, const uint8_t* d_bits,
                              uint8_t d_width, uint8_t d_exp_base,
                              uint8_t d_exp_bits, uint8_t d_sign_variable, uint8_t d_sign_value, uint32_t row_begin,
                              cudaStream_t stream) {
    bool fast_width12 = false;
    if constexpr (T == 4 && !Transposed && CompressedD) {
        fast_width12 = args.k == 5120 && args.out.column_stride > 0 &&
                       (args.out.column_stride < 100000 ||
                        (args.rows == 248320 && args.out.column_stride == 248320)) &&
                       d_width == 12 && d_exp_base == 0 && d_exp_bits == 1 && d_sign_variable != 0;
    }
    if constexpr (T == 4 && !Transposed && CompressedD) {
        if (args.k == 17408 && args.out.column_stride == 5120 && args.rows == 5120 &&
            d_width == 13 && d_exp_base == 0 && d_exp_bits == 2 && d_sign_variable != 0) {
            int active = 0;
            check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,
                kernel<T, Transposed, CompressedD, false, 128, true>, 128, 0), "Q6 down occupancy");
            const int groups = (args.rows + 7) / 8;
            const int grid = std::min(groups, std::max(active, 1) * device_facts().sm_count);
            kernel<T, Transposed, CompressedD, false, 128, true><<<grid, 128, 0, stream>>>(
                args, d_bits, d_width, d_exp_base, d_exp_bits, d_sign_variable, d_sign_value, row_begin);
            check(cudaGetLastError(), "Q6 down launch");
            return;
        }
    }
    int active = 0;
    if (fast_width12) {
        check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, kernel<T, Transposed, CompressedD, true>, 128, 0), "Q6 width-12 occupancy");
    } else {
        check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, kernel<T, Transposed, CompressedD>, 128, 0), "rearranged occupancy");
    }
    const int groups = (args.rows + 7) / 8;
    const int grid = std::min(groups, std::max(active, 1) * device_facts().sm_count);
    if (fast_width12) {
        kernel<T, Transposed, CompressedD, true><<<grid, 128, 0, stream>>>(
            args, d_bits, d_width, d_exp_base, d_exp_bits, d_sign_variable, d_sign_value, row_begin);
    } else {
        kernel<T, Transposed, CompressedD><<<grid, 128, 0, stream>>>(
            args, d_bits, d_width, d_exp_base, d_exp_bits, d_sign_variable, d_sign_value, row_begin);
    }
    check(cudaGetLastError(), "rearranged launch");
}


}
