#include "pack_api.h"

#ifndef NINFER_GGUF_FFN_FOLDED16_PACK
#define NINFER_GGUF_FFN_FOLDED16_PACK 0
#endif

#if NINFER_GGUF_FFN_FOLDED16_PACK != 0 && NINFER_GGUF_FFN_FOLDED16_PACK != 1
#error NINFER_GGUF_FFN_FOLDED16_PACK_must_be_zero_or_one
#endif

#if NINFER_GGUF_FFN_FOLDED16_PACK

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 860
#error NINFER_GGUF_FFN_FOLDED16_PACK_requires_SM86
#endif

#if defined(NINFER_GGUF_Q4_MMQ_ASYNC_X) || defined(NINFER_GGUF_Q4_MMQ_ASYNC_Y) || \
    defined(NINFER_GGUF_Q6_MMQ_ASYNC_Y) || defined(NINFER_GGUF_Q4_MMQ_CHILD_CTA) || \
    defined(NINFER_GGUF_Q4_MMQ_K_STREAM) || defined(NINFER_GGUF_Q4_MMQ_PACKED_DM) || \
    defined(NINFER_GGUF_Q4_MMQ_BIASED_DOT) || defined(NINFER_GGUF_Q4_MMQ_OUTPUT_CS) || \
    defined(NINFER_GGUF_Q4_MMQ_COMPACT_A) || defined(NINFER_GGUF_FFN_PAIR_MMQ_FUSION) || \
    defined(NINFER_GGUF_Q6_MMQ_A_X4) || defined(NINFER_GGUF_FFN_FOLDED16_MMQ)
#error NINFER_GGUF_FFN_FOLDED16_PACK_requires_isolated_native_MMQ
#endif

#include "ggml-cuda/mmq.cuh"
#include "ops/linear/gguf/ggml_bridge_internal.cuh"
#include "q4_folded_ffn_pack_kernel.cuh"

#include <cstddef>
#include <limits>
#include <mutex>

#if !defined(NINFER_Q4_FOLDED16_PACK_GENERATED_MMQ) || \
    !defined(NINFER_Q4_FOLDED16_PACK_GENERATED_LOAD) || \
    !defined(NINFER_Q4_FOLDED16_PACK_GENERATED_DOT) || \
    !defined(NINFER_Q4_FOLDED16_PACK_GENERATED_KERNEL)
#error NINFER_GGUF_FFN_FOLDED16_PACK_requires_all_pinned_generated_headers
#endif

namespace {

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
    check(cudaGetDevice(&device), "folded16 pack current device");
    if (device < 0 || device >= 16) {
        return false;
    }
    auto& state = devices[device];
    std::call_once(state.once, [&state, device] {
        cudaDeviceProp properties{};
        check(cudaGetDeviceProperties(&properties, device), "folded16 pack device properties");
        if (properties.major != 8 || properties.minor != 6 ||
            properties.sharedMemPerBlockOptin < kSharedBytes) {
            return;
        }
        check(cudaFuncSetAttribute(q4_folded_ffn_pack_kernel<MMQ_Q8_1_DS_LAYOUT_D4, false>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, kSharedBytes),
              "folded16 pack D4 timed shared memory");
        check(cudaFuncSetAttribute(q4_folded_ffn_pack_kernel<MMQ_Q8_1_DS_LAYOUT_DS4, false>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, kSharedBytes),
              "folded16 pack DS4 timed shared memory");
        check(cudaFuncSetAttribute(q4_folded_ffn_pack_kernel<MMQ_Q8_1_DS_LAYOUT_D4, true>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, kSharedBytes),
              "folded16 pack D4 diagnostic shared memory");
        check(cudaFuncSetAttribute(q4_folded_ffn_pack_kernel<MMQ_Q8_1_DS_LAYOUT_DS4, true>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, kSharedBytes),
              "folded16 pack DS4 diagnostic shared memory");
        state.supported = true;
    });
    return state.supported;
}

template <mmq_q8_1_ds_layout Layout, bool Diagnostic>
void launch(const void* gate, const void* up, std::int64_t row_bytes,
            const void* activation, int columns, void* packed_output,
            float* diagnostic_gate, float* diagnostic_up, __nv_bfloat16* diagnostic_bf16,
            cudaStream_t stream) {
    const dim3 grid((kRows / kOutputRowsPerTile) * (kColumns / kJ));
    const dim3 block(32, kWarps);
    q4_folded_ffn_pack_kernel<Layout, Diagnostic><<<grid, block, kSharedBytes, stream>>>(
        static_cast<const char*>(gate), static_cast<const char*>(up),
        static_cast<const int*>(activation), static_cast<block_q8_1_mmq*>(packed_output),
        diagnostic_gate, diagnostic_up, diagnostic_bf16,
        kK / QK_K, row_bytes / sizeof(block_q4_K), columns);
}

}

#endif

bool q4_folded_ffn_pack_run(
        const void* gate, const void* up, std::int64_t row_bytes,
        const void* activation, int columns, void* packed_output, int layout,
        float* diagnostic_gate, float* diagnostic_up, __nv_bfloat16* diagnostic_bf16,
        cudaStream_t stream) {
#if NINFER_GGUF_FFN_FOLDED16_PACK
    const bool diagnostic = diagnostic_gate != nullptr;
    if (row_bytes != kRowBytes || columns != kColumns ||
        (layout != MMQ_Q8_1_DS_LAYOUT_D4 && layout != MMQ_Q8_1_DS_LAYOUT_DS4) ||
        diagnostic != (diagnostic_up != nullptr) ||
        diagnostic != (diagnostic_bf16 != nullptr)) {
        return false;
    }
    AddressRange inputs[3]{};
    AddressRange outputs[4]{};
    if (!address_range(gate, kWeightBytes, inputs[0]) ||
        !address_range(up, kWeightBytes, inputs[1]) ||
        !address_range(activation, kActivationBytes, inputs[2]) ||
        !address_range(packed_output, kPackedBytes, outputs[0])) {
        return false;
    }
    const int count = diagnostic ? 4 : 1;
    if (diagnostic &&
        (!address_range(diagnostic_gate, kPlaneBytes, outputs[1]) ||
         !address_range(diagnostic_up, kPlaneBytes, outputs[2]) ||
         !address_range(diagnostic_bf16, kOutputBytes, outputs[3]))) {
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
    if (layout == MMQ_Q8_1_DS_LAYOUT_D4) {
        if (diagnostic) {
            launch<MMQ_Q8_1_DS_LAYOUT_D4, true>(
                gate, up, row_bytes, activation, columns, packed_output,
                diagnostic_gate, diagnostic_up, diagnostic_bf16, stream);
        } else {
            launch<MMQ_Q8_1_DS_LAYOUT_D4, false>(
                gate, up, row_bytes, activation, columns, packed_output,
                nullptr, nullptr, nullptr, stream);
        }
    } else if (diagnostic) {
        launch<MMQ_Q8_1_DS_LAYOUT_DS4, true>(
            gate, up, row_bytes, activation, columns, packed_output,
            diagnostic_gate, diagnostic_up, diagnostic_bf16, stream);
    } else {
        launch<MMQ_Q8_1_DS_LAYOUT_DS4, false>(
            gate, up, row_bytes, activation, columns, packed_output,
            nullptr, nullptr, nullptr, stream);
    }
    ninfer::ops::gguf::detail::check(cudaGetLastError(), "folded16 pack kernel launch");
    return true;
#else
    static_cast<void>(gate);
    static_cast<void>(up);
    static_cast<void>(row_bytes);
    static_cast<void>(activation);
    static_cast<void>(columns);
    static_cast<void>(packed_output);
    static_cast<void>(layout);
    static_cast<void>(diagnostic_gate);
    static_cast<void>(diagnostic_up);
    static_cast<void>(diagnostic_bf16);
    static_cast<void>(stream);
    return false;
#endif
}
