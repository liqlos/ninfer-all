#pragma once

#include "operand.cuh"

template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void q6_word_group_load_tiles(
    Q6WGView operand, int* __restrict__ x_tile, int kbx0, int i_max, int stride) {
    static_assert(type == GGML_TYPE_Q6_K && J == 128 && !fallback);
    constexpr int nwarps = ggml_cuda_mmq_get_nthreads(type, J, fallback) / 32;
    constexpr int I = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);
    static_assert(nwarps == 8 && I == 128 && sram_stride == 76);
    static_assert(MMQ_TILE_NE_K == 32 && QI6_K == 32);
    const std::uint64_t origin = std::uint64_t(operand.row_begin) * operand.parent_B + kbx0;
    const int group = threadIdx.x & 15;
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += 2 * nwarps) {
        const int i = i0 + 2 * threadIdx.y + threadIdx.x / 16;
        const auto b = q6wg::block_address(operand, origin + std::uint64_t(i) * stride);
        const uint4 w = q6wg::unsigned_words(b, group);
        const int4 value = make_int4(
            __vsubss4(w.x, 0x20202020), __vsubss4(w.y, 0x20202020),
            __vsubss4(w.z, 0x20202020), __vsubss4(w.w, 0x20202020));
        *reinterpret_cast<int4*>(x_tile + i * sram_stride + group * 4) = value;
    }
    float* x_df = reinterpret_cast<float*>(x_tile + 64);
    int* x_sc = x_tile + 65;
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * 32) {
        const int i = (i0 + threadIdx.y * 32 + threadIdx.x) % I;
        const auto b = q6wg::block_address(operand, origin + std::uint64_t(i) * stride);
        x_df[i * sram_stride] = q6wg::block_d(b);
    }
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * 8) {
        const int i = (i0 + threadIdx.y * 8 + threadIdx.x / 4) % I;
        const auto b = q6wg::block_address(operand, origin + std::uint64_t(i) * stride);
        x_sc[i * sram_stride + threadIdx.x % 4] =
            *reinterpret_cast<const int*>(b.slab + 3072 + b.slot * 16 + (threadIdx.x % 4) * 4);
    }
    (void)i_max;
}
