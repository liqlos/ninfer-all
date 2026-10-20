#include "probe_api.h"

#ifndef NINFER_GGUF_FFN_FOLDED16_MMQ
#define NINFER_GGUF_FFN_FOLDED16_MMQ 0
#endif

#if NINFER_GGUF_FFN_FOLDED16_MMQ != 0 && NINFER_GGUF_FFN_FOLDED16_MMQ != 1
#error NINFER_GGUF_FFN_FOLDED16_MMQ_must_be_zero_or_one
#endif

#if NINFER_GGUF_FFN_FOLDED16_MMQ

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 860
#error NINFER_GGUF_FFN_FOLDED16_MMQ_requires_SM86
#endif

#if defined(NINFER_GGUF_Q4_MMQ_ASYNC_X) || defined(NINFER_GGUF_Q4_MMQ_ASYNC_Y) || \
    defined(NINFER_GGUF_Q6_MMQ_ASYNC_Y) || defined(NINFER_GGUF_Q4_MMQ_CHILD_CTA) || \
    defined(NINFER_GGUF_Q4_MMQ_K_STREAM) || defined(NINFER_GGUF_Q4_MMQ_PACKED_DM) || \
    defined(NINFER_GGUF_Q4_MMQ_BIASED_DOT) || defined(NINFER_GGUF_Q4_MMQ_OUTPUT_CS) || \
    defined(NINFER_GGUF_Q4_MMQ_COMPACT_A) || defined(NINFER_GGUF_FFN_PAIR_MMQ_FUSION) || \
    defined(NINFER_GGUF_Q6_MMQ_A_X4)
#error NINFER_GGUF_FFN_FOLDED16_MMQ_requires_isolated_native_MMQ
#endif

#include "ggml-cuda/mmq.cuh"
#include "ops/linear/gguf/ggml_bridge_internal.cuh"

#include <cstddef>
#include <limits>
#include <mutex>

#if !defined(NINFER_Q4_FOLDED16_GENERATED_MMQ) || \
    !defined(NINFER_Q4_FOLDED16_GENERATED_LOAD) || \
    !defined(NINFER_Q4_FOLDED16_GENERATED_DOT)
#error NINFER_GGUF_FFN_FOLDED16_MMQ_requires_all_pinned_generated_headers
#endif

namespace {

constexpr int kRows = 17408;
constexpr int kK = 5120;
constexpr int kColumns = 1280;
constexpr int kI = 128;
constexpr int kJ = 128;
constexpr int kOutputRowsPerTile = 64;
constexpr int kThreads = 256;
constexpr int kWarps = 8;
constexpr int kRowBytes = 2880;
constexpr int kSharedBytes = 57856;
constexpr std::size_t kWeightBytes = std::size_t(kRows) * kRowBytes;
constexpr std::size_t kActivationBytes = std::size_t(kK / 128) * kColumns * 144;
constexpr std::size_t kOutputBytes = std::size_t(kRows) * kColumns * sizeof(__nv_bfloat16);
constexpr std::size_t kPlaneBytes = std::size_t(kRows) * kColumns * sizeof(float);

static_assert(sizeof(block_q4_K) == 144);
static_assert(sizeof(block_q8_1_mmq) == 144);
static_assert(kK / QK_K * sizeof(block_q4_K) == kRowBytes);
static_assert((kJ + kJ * 36 + kI * 76) * sizeof(int) == kSharedBytes);
static_assert(kRows % kOutputRowsPerTile == 0 && kColumns % kJ == 0);

template <bool Diagnostic>
__global__ __launch_bounds__(kThreads, 1) void q4_folded_ffn_kernel(
        const char* __restrict__ gate, const char* __restrict__ up,
        const int* __restrict__ activation, __nv_bfloat16* __restrict__ output,
        float* __restrict__ diagnostic_gate, float* __restrict__ diagnostic_up,
        const int blocks_per_k, const int stride_row, const int ncols_y) {
    constexpr int warp_size = 32;
    constexpr int qk = ggml_cuda_type_traits<GGML_TYPE_Q4_K>::qk;
    constexpr int ne_block = QK8_1_MMQ;
    constexpr int sz = sizeof(block_q8_1_mmq) / sizeof(int);
    constexpr ggml_cuda_mmq_vec_dot_t vec_dot =
        ggml_cuda_mmq_get_vec_dot<GGML_TYPE_Q4_K, kJ, false>();
#if defined(__CUDA_ARCH__)
    static_assert(ggml_cuda_get_physical_warp_size() == warp_size);
    static_assert(ggml_cuda_mmq_get_nthreads(GGML_TYPE_Q4_K, kJ, false) == kThreads);
    static_assert(ggml_cuda_mmq_get_I(GGML_TYPE_Q4_K, kJ, false) == kI);
    static_assert(ggml_cuda_mmq_get_sram_stride(GGML_TYPE_Q4_K, kJ, false) == 76);
    static_assert(ggml_cuda_mmq_get_rows_per_warp(GGML_TYPE_Q4_K, kJ, false) == 32);
    static_assert(ggml_cuda_mmq_get_K_vram(GGML_TYPE_Q4_K, kJ, false) == 256);
#endif
    extern __shared__ int folded_shared[];
    int* tile_y = folded_shared + kJ;
    int* tile_x = tile_y + kJ * MMQ_TILE_Y_K;
    const int row0 = (blockIdx.x / (kColumns / kJ)) * kOutputRowsPerTile;
    const int column0 = (blockIdx.x % (kColumns / kJ)) * kJ;
    const int* y = activation + column0 * sz;
    float sum[kI * kJ / kThreads] = {0.0f};

#pragma unroll 1
    for (int kb0 = 0; kb0 < blocks_per_k; ++kb0) {
        ggml_cuda_mmq_load_tiles_q4_K_folded16<GGML_TYPE_Q4_K, kJ, false>(
            gate, tile_x, kb0, kI - 1, stride_row, up, row0);
        {
            const int* by0 = y + ncols_y * (kb0 * qk / ne_block) * sz;
#pragma unroll
            for (int l0 = 0; l0 < kJ * MMQ_TILE_Y_K; l0 += kWarps * warp_size) {
                int l = l0 + threadIdx.y * warp_size + threadIdx.x;
                tile_y[l] = by0[l];
            }
        }
        __syncthreads();
        vec_dot(tile_x, tile_y, sum, 0);
        __syncthreads();
        {
            const int* by0 = y + ncols_y * ((kb0 * qk / ne_block) * sz + sz);
#pragma unroll
            for (int l0 = 0; l0 < kJ * MMQ_TILE_Y_K; l0 += kWarps * warp_size) {
                int l = l0 + threadIdx.y * warp_size + threadIdx.x;
                tile_y[l] = by0[l];
            }
        }
        __syncthreads();
        vec_dot(tile_x, tile_y, sum, MMQ_TILE_NE_K);
        __syncthreads();
    }

    using tile_C = ggml_cuda_mma::tile<16, 8, int>;
#pragma unroll
    for (int j0 = 0; j0 < kJ; j0 += 16) {
#pragma unroll
        for (int l = 0; l < tile_C::ne; ++l) {
            const int row = row0 + (threadIdx.y / 2) * 16 + tile_C::get_i(l);
            const int column = column0 + j0 + (threadIdx.y % 2) * 8 + tile_C::get_j(l);
            const int at = column * kRows + row;
            const float gate_value = sum[(j0 / 8) * tile_C::ne + l];
            const float up_value = sum[(j0 / 8 + 1) * tile_C::ne + l];
            if constexpr (Diagnostic) {
                diagnostic_gate[at] = gate_value;
                diagnostic_up[at] = up_value;
            }
            const float value = up_value * ninfer::ops::gguf::detail::vec_silu(gate_value);
            output[at] = __float2bfloat16(value);
        }
    }
}

struct AddressRange {
    std::uintptr_t begin;
    std::uintptr_t end;
};

bool address_range(const void* pointer, std::size_t bytes, AddressRange& range) {
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (begin == 0 || (begin & 15U) != 0 ||
        bytes > std::numeric_limits<std::uintptr_t>::max() - begin) {
        return false;
    }
    range = {begin, begin + bytes};
    return true;
}

bool overlaps(const AddressRange& a, const AddressRange& b) {
    return a.begin < b.end && b.begin < a.end;
}

bool supported_device() {
    using ninfer::ops::gguf::detail::check;
    struct DeviceState {
        std::once_flag once;
        bool supported = false;
    };
    static DeviceState devices[16];
    int device = -1;
    check(cudaGetDevice(&device), "folded16 current device");
    if (device < 0 || device >= 16) {
        return false;
    }
    auto& state = devices[device];
    std::call_once(state.once, [&state, device] {
        cudaDeviceProp properties{};
        check(cudaGetDeviceProperties(&properties, device), "folded16 device properties");
        if (properties.major != 8 || properties.minor != 6 ||
            properties.sharedMemPerBlockOptin < kSharedBytes) {
            return;
        }
        check(cudaFuncSetAttribute(q4_folded_ffn_kernel<false>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, kSharedBytes),
              "folded16 timed shared memory");
        check(cudaFuncSetAttribute(q4_folded_ffn_kernel<true>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, kSharedBytes),
              "folded16 diagnostic shared memory");
        state.supported = true;
    });
    return state.supported;
}

}

#endif

bool q4_folded_ffn_run(const void* gate, const void* up, std::int64_t row_bytes,
                       const void* activation, int columns, __nv_bfloat16* output,
                       float* diagnostic_gate, float* diagnostic_up, cudaStream_t stream) {
#if NINFER_GGUF_FFN_FOLDED16_MMQ
    if (row_bytes != kRowBytes || columns != kColumns ||
        (diagnostic_gate == nullptr) != (diagnostic_up == nullptr)) {
        return false;
    }
    AddressRange inputs[3]{};
    AddressRange outputs[3]{};
    if (!address_range(gate, kWeightBytes, inputs[0]) ||
        !address_range(up, kWeightBytes, inputs[1]) ||
        !address_range(activation, kActivationBytes, inputs[2]) ||
        !address_range(output, kOutputBytes, outputs[0])) {
        return false;
    }
    const int count = diagnostic_gate == nullptr ? 1 : 3;
    if (count == 3 &&
        (!address_range(diagnostic_gate, kPlaneBytes, outputs[1]) ||
         !address_range(diagnostic_up, kPlaneBytes, outputs[2]))) {
        return false;
    }
    for (int i = 0; i < count; ++i) {
        for (const auto& input : inputs) {
            if (overlaps(outputs[i], input)) {
                return false;
            }
        }
        for (int j = 0; j < i; ++j) {
            if (overlaps(outputs[i], outputs[j])) {
                return false;
            }
        }
    }
    if (!supported_device()) {
        return false;
    }
    const dim3 grid((kRows / kOutputRowsPerTile) * (kColumns / kJ));
    const dim3 block(32, kWarps);
    if (count == 1) {
        q4_folded_ffn_kernel<false><<<grid, block, kSharedBytes, stream>>>(
            static_cast<const char*>(gate), static_cast<const char*>(up),
            static_cast<const int*>(activation), output, nullptr, nullptr,
            kK / QK_K, row_bytes / sizeof(block_q4_K), columns);
    } else {
        q4_folded_ffn_kernel<true><<<grid, block, kSharedBytes, stream>>>(
            static_cast<const char*>(gate), static_cast<const char*>(up),
            static_cast<const int*>(activation), output, diagnostic_gate, diagnostic_up,
            kK / QK_K, row_bytes / sizeof(block_q4_K), columns);
    }
    ninfer::ops::gguf::detail::check(cudaGetLastError(), "folded16 kernel launch");
    return true;
#else
    static_cast<void>(gate);
    static_cast<void>(up);
    static_cast<void>(row_bytes);
    static_cast<void>(activation);
    static_cast<void>(columns);
    static_cast<void>(output);
    static_cast<void>(diagnostic_gate);
    static_cast<void>(diagnostic_up);
    static_cast<void>(stream);
    return false;
#endif
}
