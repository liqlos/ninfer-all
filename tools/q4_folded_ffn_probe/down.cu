#include "whole_api.h"

#if defined(NINFER_GGUF_MMQ_STORE_FUSION) && NINFER_GGUF_MMQ_STORE_FUSION != 1
#error whole_ffn_down_requires_native_store_fusion
#endif
#ifndef NINFER_GGUF_MMQ_STORE_FUSION
#define NINFER_GGUF_MMQ_STORE_FUSION 1
#endif

#if defined(NINFER_GGUF_Q4_MMQ_ASYNC_X) || defined(NINFER_GGUF_Q4_MMQ_ASYNC_Y) || \
    defined(NINFER_GGUF_Q6_MMQ_ASYNC_Y) || defined(NINFER_GGUF_Q4_MMQ_CHILD_CTA) || \
    defined(NINFER_GGUF_Q4_MMQ_K_STREAM) || defined(NINFER_GGUF_Q4_MMQ_PACKED_DM) || \
    defined(NINFER_GGUF_Q4_MMQ_BIASED_DOT) || defined(NINFER_GGUF_Q4_MMQ_OUTPUT_CS) || \
    defined(NINFER_GGUF_Q4_MMQ_COMPACT_A) || defined(NINFER_GGUF_FFN_PAIR_MMQ_FUSION) || \
    defined(NINFER_GGUF_Q6_MMQ_A_X4) || defined(NINFER_GGUF_Q6_MMQ_DENSE_ENTRY) || \
    defined(NINFER_GGUF_Q6_MMQ_RAW32) || defined(NINFER_GGUF_SM86_MMQ_TILE_CAP) || \
    defined(NINFER_GGUF_MMQ_GATE_EPILOGUE_FUSION)
#error whole_ffn_down_requires_isolated_native_mmq
#endif

#define MatrixLaunch WholeFfnNativeDownLaunch
#define select_matrix_launch whole_ffn_native_down_select
#define matrix_fixup_bytes_impl whole_ffn_native_down_fixup_bytes
#define launch_matrix whole_ffn_native_down_launch
#define matrix_product_impl whole_ffn_native_down_product
#define matrix_bf16_store_plan_impl whole_ffn_native_down_store_plan
#define launch_matrix_bf16_store whole_ffn_native_down_launch_store
#define launch_matrix_bf16_gate_store whole_ffn_native_down_launch_gate
#define matrix_product_bf16_store_impl whole_ffn_native_down_store
#define matrix_product_bf16_gate_store_impl whole_ffn_native_down_gate
#define mul_mat_q_process_tile whole_ffn_native_down_process_tile
#define mul_mat_q whole_ffn_native_down_kernel
#define mul_mat_q_stream_k_fixup whole_ffn_native_down_fixup
#include "ops/linear/gguf/ggml_bridge_mmq.cuh"
#undef mul_mat_q_stream_k_fixup
#undef mul_mat_q
#undef mul_mat_q_process_tile
#undef matrix_product_bf16_gate_store_impl
#undef matrix_product_bf16_store_impl
#undef launch_matrix_bf16_gate_store
#undef launch_matrix_bf16_store
#undef matrix_bf16_store_plan_impl
#undef matrix_product_impl
#undef launch_matrix
#undef matrix_fixup_bytes_impl
#undef select_matrix_launch
#undef MatrixLaunch

namespace {

namespace detail = ninfer::ops::gguf::detail;
constexpr int kDownRows = 5120;
constexpr int kDownK = 17408;
constexpr int kTokens = 1280;

template <ggml_type Type>
bool plan_for(WholeFfnDownPlan* out) {
    if (out == nullptr || detail::device_facts().cc != 860) { return false; }
    const auto plan = detail::whole_ffn_native_down_select<Type>(kDownRows, kTokens);
    if (plan.J != 128 || plan.I != 128 || plan.threads != 256 ||
        plan.row_tiles != 40 || plan.column_tiles != 10 ||
        plan.blocks != 400 || !plan.stream_k || plan.fixup) {
        return false;
    }
    *out = {plan.J, plan.I, plan.threads, plan.blocks, plan.shared, plan.stream_k, plan.fixup};
    return true;
}

template <ggml_type Type>
bool run_for(const void* weight, const void* activation, __nv_bfloat16* output,
              cudaStream_t stream) {
    constexpr std::int64_t row_bytes =
        std::int64_t(kDownK / 256) * ggml_cuda_type_traits<Type>::bs;
    return detail::whole_ffn_native_down_store<Type>(
        weight, row_bytes, kDownRows, kDownK, activation, kTokens, output, kDownRows,
        stream, true);
}

template <ggml_type Type>
bool fp32_for(const void* weight, const void* activation, float* output, cudaStream_t stream) {
    if (weight == nullptr || activation == nullptr || output == nullptr) { return false; }
    constexpr std::int64_t row_bytes =
        std::int64_t(kDownK / 256) * ggml_cuda_type_traits<Type>::bs;
    detail::whole_ffn_native_down_product<Type>(
        weight, row_bytes, kDownRows, kDownK, activation, kTokens, output,
        kDownRows, nullptr, stream);
    return true;
}

}

bool whole_ffn_down_plan(int layout, WholeFfnDownPlan* plan) {
    if (layout == 0) { return plan_for<GGML_TYPE_Q6_K>(plan); }
    if (layout == 1) { return plan_for<GGML_TYPE_Q4_K>(plan); }
    return false;
}

bool whole_ffn_down_run(int layout, const void* weight, const void* activation,
                        __nv_bfloat16* output, cudaStream_t stream) {
    if (layout == 0) { return run_for<GGML_TYPE_Q6_K>(weight, activation, output, stream); }
    if (layout == 1) { return run_for<GGML_TYPE_Q4_K>(weight, activation, output, stream); }
    return false;
}

bool whole_ffn_down_fp32(int layout, const void* weight, const void* activation,
                         float* output, cudaStream_t stream) {
    if (layout == 0) { return fp32_for<GGML_TYPE_Q6_K>(weight, activation, output, stream); }
    if (layout == 1) { return fp32_for<GGML_TYPE_Q4_K>(weight, activation, output, stream); }
    return false;
}
