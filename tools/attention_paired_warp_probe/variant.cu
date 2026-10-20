#ifndef NINFER_PROMPT_PAIRED_WARP
#define NINFER_PROMPT_PAIRED_WARP 0
#endif

#if NINFER_PROMPT_PAIRED_WARP != 0 && NINFER_PROMPT_PAIRED_WARP != 1
#error NINFER_PROMPT_PAIRED_WARP_must_be_zero_or_one
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 860
#error attention_paired_warp_probe_requires_SM86
#endif

#if NINFER_PROMPT_PAIRED_WARP
#include "attention_paired_generated.cuh"
#else
#define causal_attention_prompt_i8_fast_kernel causal_attention_prompt_i8_paired_off_kernel
#include "attention_native_generated.cuh"
namespace ninfer::ops {
template __global__ void causal_attention_prompt_i8_paired_off_kernel<
    CausalD256H24Kv4, PagedKVBatchMetadata<false>, 8, false, KvKeyCoding::Int8>(
    const __nv_bfloat16*, const std::int8_t*, const std::int8_t*,
    const __half*, const __half*, PagedKVBatchMetadata<false>,
    const std::int32_t*, float, __nv_bfloat16*, std::int32_t);
}
#undef causal_attention_prompt_i8_fast_kernel
#endif
