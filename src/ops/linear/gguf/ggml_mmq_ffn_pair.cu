#include "ggml_bridge_ffn_pair.h"
#include "ggml_bridge_internal.cuh"

#include <mutex>

namespace ninfer::ops::gguf {

#if defined(NINFER_GGUF_FFN_PAIR_MMQ_FUSION)
namespace {

constexpr int kRows = 17408;
constexpr int kK = 5120;
constexpr int kColumns = 1280;
#if defined(NINFER_GGUF_FFN_PAIR_MMQ_J128)
constexpr int kJ = 128;
#else
constexpr int kJ = 64;
#endif
constexpr auto kType = GGML_TYPE_Q4_K;
__global__ __launch_bounds__(ggml_cuda_mmq_get_nthreads(kType, kJ, false), 1) void ffn_pair_kernel(
    const char* __restrict__ gate, const char* __restrict__ up,
    const int* __restrict__ activation, __nv_bfloat16* __restrict__ out,
    int x_tile_ints) {
    constexpr int kThreads = ggml_cuda_mmq_get_nthreads(kType, kJ, false);
    constexpr int kI = ggml_cuda_mmq_get_I(kType, kJ, false);
    constexpr int kYInts = GGML_PAD(kJ * MMQ_TILE_Y_K, kThreads);
    static_assert(kRows % kI == 0 && kColumns % kJ == 0);
    constexpr auto load = ggml_cuda_mmq_get_load_tiles<kType, kJ, false>();
    constexpr auto dot = ggml_cuda_mmq_get_vec_dot<kType, kJ, false>();
    constexpr int qk = ggml_cuda_type_traits<kType>::qk;
    constexpr int blocks_per_iter = ggml_cuda_mmq_get_K_vram(kType, kJ, false) / qk;
    constexpr int activation_ints = sizeof(block_q8_1_mmq) / sizeof(int);
    extern __shared__ int shared[];
    int* y = shared;
    int* ga = y + kYInts;
    int* ua = ga + x_tile_ints;
    float gs[kJ * kI / kThreads] = {0.0f};
    float us[kJ * kI / kThreads] = {0.0f};
    const int row = blockIdx.x * kI;
    const int column = blockIdx.y * kJ;
    const int weight_offset = row * (kK / qk);
    for (int kb = 0; kb < kK / qk; kb += blocks_per_iter) {
        load(gate, ga, weight_offset + kb, kI - 1, kK / qk);
        load(up, ua, weight_offset + kb, kI - 1, kK / qk);
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int* src = activation + kColumns *
                ((kb * qk / QK8_1_MMQ) * activation_ints + half * activation_ints) +
                column * activation_ints;
            for (int i = threadIdx.y * 32 + threadIdx.x; i < kJ * MMQ_TILE_Y_K;
                 i += kThreads) {
                y[i] = src[i];
            }
            __syncthreads();
            dot(ga, y, gs, half * MMQ_TILE_NE_K);
            dot(ua, y, us, half * MMQ_TILE_NE_K);
            __syncthreads();
        }
    }
    using Tile = tile<16, 8, int>;
    constexpr int ntx = ggml_cuda_mmq_get_rows_per_warp(kType, kJ, false) / Tile::I;
    const int i0 = (threadIdx.y / ntx) * (ntx * Tile::I);
#pragma unroll
    for (int j0 = 0; j0 < kJ; j0 += ntx * Tile::J) {
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
#pragma unroll
            for (int l = 0; l < Tile::ne; ++l) {
                const int j = j0 + (threadIdx.y % ntx) * Tile::J + Tile::get_j(l);
                const int i = i0 + n * Tile::I + Tile::get_i(l);
                const int at = (j0 / Tile::J + n) * Tile::ne + l;
                const float value = us[at] * detail::vec_silu(gs[at]);
                out[std::int64_t(column + j) * kRows + row + i] = __float2bfloat16(value);
            }
        }
    }
}

bool overlaps(const void* a, std::size_t na, const void* b, std::size_t nb) {
    const auto x = reinterpret_cast<std::uintptr_t>(a);
    const auto y = reinterpret_cast<std::uintptr_t>(b);
    return x <= y ? y - x < na : x - y < nb;
}

}
#endif

bool ffn_q4_k_mmq_pair(const void* gate, const void* up, std::int64_t row_bytes,
                        const void* activation, int columns, __nv_bfloat16* out,
                        cudaStream_t stream) {
#if defined(NINFER_GGUF_FFN_PAIR_MMQ_FUSION)
    constexpr std::size_t weight_bytes = std::size_t(kRows) * (kK / 256) * sizeof(block_q4_K);
    constexpr std::size_t output_bytes = std::size_t(kRows) * kColumns * sizeof(__nv_bfloat16);
    constexpr std::size_t activation_bytes = std::size_t(kK / QK8_1_MMQ) * kColumns * sizeof(block_q8_1_mmq);
    if (columns != kColumns || row_bytes != (kK / 256) * sizeof(block_q4_K) ||
        !gate || !up || !activation || !out ||
        ((reinterpret_cast<std::uintptr_t>(gate) | reinterpret_cast<std::uintptr_t>(up) |
          reinterpret_cast<std::uintptr_t>(activation) | reinterpret_cast<std::uintptr_t>(out)) & 15) ||
        overlaps(out, output_bytes, gate, weight_bytes) ||
        overlaps(out, output_bytes, up, weight_bytes) ||
        overlaps(out, output_bytes, activation, activation_bytes)) {
        return false;
    }
    const auto& device = detail::device_facts();
    if (device.cc != 860) { return false; }
    const auto config = ggml_cuda_mmq_get_config(kType, kJ, false, device.cc);
    const std::size_t x_bytes = ggml_cuda_mmq_get_nbytes_shared_x(config, device.cc);
    const std::size_t shared_bytes = GGML_PAD(kJ * MMQ_TILE_Y_K, config.nthreads) * sizeof(int) + 2 * x_bytes;
    if (shared_bytes > device.shared_per_block_optin) { return false; }
    int ordinal = 0;
    detail::check(cudaGetDevice(&ordinal), "FFN pair device");
    if (ordinal < 0 || ordinal >= 16) { return false; }
    static std::once_flag raised[16];
    std::call_once(raised[ordinal], [&] {
        detail::check(cudaFuncSetAttribute(ffn_pair_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared_bytes)),
            "FFN pair shared memory");
    });
    ffn_pair_kernel<<<dim3(kRows / config.I, kColumns / kJ), dim3(32, config.nthreads / 32), shared_bytes, stream>>>(
        static_cast<const char*>(gate), static_cast<const char*>(up),
        static_cast<const int*>(activation), out, static_cast<int>(x_bytes / sizeof(int)));
    detail::check(cudaGetLastError(), "FFN pair launch");
    return true;
#else
    GGML_UNUSED(gate);
    GGML_UNUSED(up);
    GGML_UNUSED(row_bytes);
    GGML_UNUSED(activation);
    GGML_UNUSED(columns);
    GGML_UNUSED(out);
    GGML_UNUSED(stream);
    return false;
#endif
}

}
