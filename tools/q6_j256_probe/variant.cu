#ifndef NINFER_Q6_J256_PROBE
#define NINFER_Q6_J256_PROBE 0
#endif

#if NINFER_Q6_J256_PROBE != 0 && NINFER_Q6_J256_PROBE != 1
#error Invalid_Q6_J256_probe_switch
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 860
#error Q6_J256_requires_SM86
#endif

#if defined(GGML_USE_HIP) || defined(NINFER_GGUF_Q6_MMQ_A_X4) || \
    defined(NINFER_GGUF_Q6_MMQ_ALIGNED_RAW32) || defined(NINFER_GGUF_Q6_MMQ_ASYNC_Y) || \
    defined(NINFER_GGUF_Q4_MMQ_ASYNC_Y) || defined(NINFER_GGUF_Q4_MMQ_ASYNC_X) || \
    defined(NINFER_GGUF_Q4_MMQ_CHILD_CTA) || defined(NINFER_GGUF_Q4_MMQ_K_STREAM) || \
    defined(NINFER_GGUF_Q4_MMQ_PACKED_DM) || defined(NINFER_GGUF_SM86_MMQ_TILE_CAP) || \
    defined(NINFER_GGUF_MMQ_GATE_EPILOGUE_FUSION) || defined(NINFER_GGUF_Q6_MMQ_DENSE_ENTRY)
#error Unrelated_MMQ_experiments_must_be_disabled
#endif

#define NINFER_GGUF_MMQ_STORE_FUSION 1

#if NINFER_Q6_J256_PROBE
#define PROBE_PLAN q6_j256_on_plan
#define PROBE_RUN q6_j256_on_run
#define mul_mat_q q6_j256_on_native_mul_mat_q
#define mul_mat_q_stream_k_fixup q6_j256_on_native_fixup
#define launch_matrix q6_j256_on_native_launch_matrix
#define select_matrix_launch q6_j256_on_native_select_matrix_launch
#define matrix_product_impl q6_j256_on_native_matrix_product_impl
#define matrix_fixup_bytes_impl q6_j256_on_native_matrix_fixup_bytes_impl
#define matrix_bf16_store_plan_impl q6_j256_on_native_matrix_bf16_store_plan_impl
#define matrix_product_bf16_store_impl q6_j256_on_native_matrix_product_bf16_store_impl
#else
#define PROBE_PLAN q6_j256_off_plan
#define PROBE_RUN q6_j256_off_run
#define mul_mat_q q6_j256_off_native_mul_mat_q
#define mul_mat_q_stream_k_fixup q6_j256_off_native_fixup
#define launch_matrix q6_j256_off_native_launch_matrix
#define select_matrix_launch q6_j256_off_native_select_matrix_launch
#define matrix_product_impl q6_j256_off_native_matrix_product_impl
#define matrix_fixup_bytes_impl q6_j256_off_native_matrix_fixup_bytes_impl
#define matrix_bf16_store_plan_impl q6_j256_off_native_matrix_bf16_store_plan_impl
#define matrix_product_bf16_store_impl q6_j256_off_native_matrix_product_bf16_store_impl
#endif

#include "variant_impl.cuh"
