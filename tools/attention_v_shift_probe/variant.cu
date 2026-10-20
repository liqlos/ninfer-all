#include "probe_api.h"

#ifndef NINFER_PROMPT_V_SHIFT_CTA
#define NINFER_PROMPT_V_SHIFT_CTA 0
#endif

#if NINFER_PROMPT_V_SHIFT_CTA != 0 && NINFER_PROMPT_V_SHIFT_CTA != 1
#error NINFER_PROMPT_V_SHIFT_CTA_must_be_zero_or_one
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 860
#error attention_v_shift_probe_requires_SM86
#endif

#if NINFER_PROMPT_V_SHIFT_CTA
#define causal_attention_prompt_i8_fast_kernel causal_attention_prompt_i8_fast_v_shift_on_kernel
#define ATTENTION_V_SHIFT_LAUNCH attention_v_shift_on_wide_launch
#else
#define causal_attention_prompt_i8_fast_kernel causal_attention_prompt_i8_fast_v_shift_off_kernel
#define ATTENTION_V_SHIFT_LAUNCH attention_v_shift_off_wide_launch
#endif

#include "attention_v_shift_generated.cuh"

#ifndef NINFER_ATTENTION_V_SHIFT_GENERATED
#error attention_v_shift_probe_requires_pinned_generated_header
#endif

cudaError_t ATTENTION_V_SHIFT_LAUNCH(
    const ninfer::Tensor& q, const ninfer::Tensor& positions,
    const ninfer::Tensor& table_rows, float scale,
    const ninfer::PagedKVBatchLayerView& cache, ninfer::Tensor& out, cudaStream_t stream) {
    using Geometry = ninfer::ops::CausalD256H24Kv4;
    using Metadata = ninfer::ops::PagedKVBatchMetadata<false>;
    using Shape = ninfer::ops::CausalPromptI8FastShape<8, false>;
    static_assert(Geometry::QHeads == 24 && Geometry::KVHeads == 4);
    static_assert(ninfer::ops::kCausalPromptHeadDim == 256);
    static_assert(Shape::Threads == 256 && Shape::Br == 128 && Shape::SmemBytes == 100352);
    static_assert(Shape::SmemBytes + 2 * sizeof(int) == 100360);
    const std::int32_t tokens = q.ne[2];
    if (q.data == nullptr || out.data == nullptr || positions.data == nullptr ||
        table_rows.data == nullptr || cache.block_tables.data == nullptr ||
        cache.k_pages.data == nullptr || cache.v_pages.data == nullptr ||
        cache.k_scale_pages.data == nullptr || cache.v_scale_pages.data == nullptr ||
        (reinterpret_cast<std::uintptr_t>(cache.v_scale_pages.data) & 15U) != 0 ||
        q.dtype != ninfer::DType::BF16 || out.dtype != ninfer::DType::BF16 ||
        positions.dtype != ninfer::DType::I32 || table_rows.dtype != ninfer::DType::I32 ||
        cache.block_tables.dtype != ninfer::DType::I32 ||
        cache.v_scale_pages.dtype != ninfer::DType::FP16 ||
        cache.storage != ninfer::KvCacheStorage::Int8Group64 ||
        cache.head_dim != 256 || cache.num_kv_heads != 4 ||
        q.ne[0] != 256 || q.ne[1] != 24 || q.ne[3] != 1 ||
        tokens <= 0 || tokens > 65536 || cache.block_tables.ne[0] <= 0 ||
        positions.ne[0] != tokens || table_rows.ne[0] != 1 ||
        out.ne[0] != 256 || out.ne[1] != 24 || out.ne[2] != tokens || out.ne[3] != 1) {
        return cudaErrorInvalidValue;
    }
    const Metadata metadata{
        static_cast<const std::int32_t*>(cache.block_tables.data),
        nullptr,
        static_cast<const std::int32_t*>(table_rows.data),
        cache.block_tables.ne[0],
    };
    const cudaError_t attribute = cudaFuncSetAttribute(
        ninfer::ops::causal_attention_prompt_i8_fast_kernel<
            Geometry, Metadata, 8, false, ninfer::ops::KvKeyCoding::Int8>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, Shape::SmemBytes);
    if (attribute != cudaSuccess) { return attribute; }
    const dim3 grid(static_cast<unsigned>((tokens + Shape::Br - 1) / Shape::Br),
                    static_cast<unsigned>(Geometry::QHeads), 1u);
    ninfer::ops::causal_attention_prompt_i8_fast_kernel<
        Geometry, Metadata, 8, false, ninfer::ops::KvKeyCoding::Int8>
        <<<grid, Shape::Threads, Shape::SmemBytes, stream>>>(
            static_cast<const __nv_bfloat16*>(q.data),
            static_cast<const std::int8_t*>(cache.k_pages.data),
            static_cast<const std::int8_t*>(cache.v_pages.data),
            static_cast<const __half*>(cache.k_scale_pages.data),
            static_cast<const __half*>(cache.v_scale_pages.data), metadata,
            static_cast<const std::int32_t*>(positions.data), scale,
            static_cast<__nv_bfloat16*>(out.data), tokens);
    return cudaGetLastError();
}

#undef ATTENTION_V_SHIFT_LAUNCH
#undef causal_attention_prompt_i8_fast_kernel
