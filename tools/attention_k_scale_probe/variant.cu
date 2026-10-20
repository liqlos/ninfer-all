#include "probe_api.h"

#include <limits>

#ifndef NINFER_PROMPT_K_SCALE_FP32
#define NINFER_PROMPT_K_SCALE_FP32 0
#endif

#if NINFER_PROMPT_K_SCALE_FP32 != 0 && NINFER_PROMPT_K_SCALE_FP32 != 1
#error NINFER_PROMPT_K_SCALE_FP32_must_be_zero_or_one
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 860
#error attention_k_scale_probe_requires_SM86
#endif

#if NINFER_PROMPT_K_SCALE_FP32
#define causal_attention_prompt_i8_fast_kernel causal_attention_prompt_i8_fast_k_scale_on_kernel
#define ATTENTION_K_SCALE_LAUNCH attention_k_scale_on_wide_launch
#else
#define causal_attention_prompt_i8_fast_kernel causal_attention_prompt_i8_fast_k_scale_off_kernel
#define ATTENTION_K_SCALE_LAUNCH attention_k_scale_off_wide_launch
#endif

#include "attention_k_scale_generated.cuh"

#ifndef NINFER_ATTENTION_K_SCALE_GENERATED
#error attention_k_scale_probe_requires_pinned_generated_header
#endif

cudaError_t ATTENTION_K_SCALE_LAUNCH(
    const ninfer::Tensor& q, const ninfer::Tensor& positions,
    const ninfer::Tensor& table_rows, float scale,
    const ninfer::PagedKVBatchLayerView& cache, ninfer::Tensor& out, cudaStream_t stream) {
    using Geometry = ninfer::ops::CausalD256H24Kv4;
    using Metadata = ninfer::ops::PagedKVBatchMetadata<false>;
    using Shape = ninfer::ops::CausalPromptI8FastShape<8, false>;
    constexpr int SharedBytes = Shape::SmemBytes + NINFER_PROMPT_K_SCALE_FP32 * 1024;
    static_assert(Geometry::QHeads == 24 && Geometry::KVHeads == 4);
    static_assert(ninfer::ops::kCausalPromptHeadDim == 256);
    static_assert(Shape::Threads == 256 && Shape::Br == 128 && Shape::SmemBytes == 100352);
    static_assert(SharedBytes == (NINFER_PROMPT_K_SCALE_FP32 ? 101376 : 100352));
    const auto aligned = [](const void* pointer, std::uintptr_t bytes) {
        return (reinterpret_cast<std::uintptr_t>(pointer) & (bytes - 1)) == 0;
    };
    const auto contiguous = [](const ninfer::Tensor& tensor, std::int64_t bytes) {
        for (int dimension = 0; dimension < 4; ++dimension) {
            if (tensor.ne[dimension] <= 0 ||
                (tensor.ne[dimension] != 1 && tensor.nb[dimension] != bytes)) { return false; }
            if (dimension < 3 && tensor.ne[dimension] != 1) {
                if (bytes > std::numeric_limits<std::int64_t>::max() / tensor.ne[dimension]) {
                    return false;
                }
                bytes *= tensor.ne[dimension];
            }
        }
        return true;
    };
    const std::int32_t tokens = q.ne[2];
    if (q.data == nullptr || out.data == nullptr || positions.data == nullptr ||
        table_rows.data == nullptr || cache.block_tables.data == nullptr ||
        cache.k_pages.data == nullptr || cache.v_pages.data == nullptr ||
        cache.k_scale_pages.data == nullptr || cache.v_scale_pages.data == nullptr ||
        !aligned(q.data, 16) || !aligned(out.data, 16) ||
        !aligned(cache.k_pages.data, 16) || !aligned(cache.v_pages.data, 16) ||
        !aligned(cache.k_scale_pages.data, 8) || !aligned(cache.v_scale_pages.data, 8) ||
        !aligned(positions.data, 4) || !aligned(table_rows.data, 4) ||
        !aligned(cache.block_tables.data, 4) ||
        q.dtype != ninfer::DType::BF16 || out.dtype != ninfer::DType::BF16 ||
        positions.dtype != ninfer::DType::I32 || table_rows.dtype != ninfer::DType::I32 ||
        cache.block_tables.dtype != ninfer::DType::I32 ||
        cache.k_pages.dtype != ninfer::DType::I8 || cache.v_pages.dtype != ninfer::DType::I8 ||
        cache.k_scale_pages.dtype != ninfer::DType::FP16 ||
        cache.v_scale_pages.dtype != ninfer::DType::FP16 ||
        cache.storage != ninfer::KvCacheStorage::Int8Group64 ||
        cache.head_dim != 256 || cache.num_kv_heads != 4 ||
        q.ne[0] != 256 || q.ne[1] != 24 || q.ne[3] != 1 ||
        tokens <= 0 || tokens > 65536 || cache.block_tables.ne[0] <= 0 ||
        cache.block_tables.ne[2] != 1 || cache.block_tables.ne[3] != 1 ||
        positions.ne[0] != tokens || positions.ne[1] != 1 || positions.ne[2] != 1 || positions.ne[3] != 1 ||
        table_rows.ne[0] != 1 || table_rows.ne[1] != 1 || table_rows.ne[2] != 1 || table_rows.ne[3] != 1 ||
        cache.k_pages.ne[0] != 256 || cache.k_pages.ne[1] != 64 || cache.k_pages.ne[2] != 4 ||
        cache.v_pages.ne[0] != 256 || cache.v_pages.ne[1] != 64 || cache.v_pages.ne[2] != 4 ||
        cache.k_scale_pages.ne[0] != 4 || cache.k_scale_pages.ne[1] != 64 || cache.k_scale_pages.ne[2] != 4 ||
        cache.v_scale_pages.ne[0] != 4 || cache.v_scale_pages.ne[1] != 64 || cache.v_scale_pages.ne[2] != 4 ||
        cache.k_pages.ne[3] != cache.v_pages.ne[3] ||
        cache.k_pages.ne[3] != cache.k_scale_pages.ne[3] ||
        cache.k_pages.ne[3] != cache.v_scale_pages.ne[3] ||
        !contiguous(q, 2) || !contiguous(out, 2) || !contiguous(positions, 4) ||
        !contiguous(table_rows, 4) || !contiguous(cache.block_tables, 4) ||
        !contiguous(cache.k_pages, 1) || !contiguous(cache.v_pages, 1) ||
        !contiguous(cache.k_scale_pages, 2) || !contiguous(cache.v_scale_pages, 2) ||
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
        cudaFuncAttributeMaxDynamicSharedMemorySize, SharedBytes);
    if (attribute != cudaSuccess) { return attribute; }
    const dim3 grid(static_cast<unsigned>((tokens + Shape::Br - 1) / Shape::Br),
                    static_cast<unsigned>(Geometry::QHeads), 1u);
    ninfer::ops::causal_attention_prompt_i8_fast_kernel<
        Geometry, Metadata, 8, false, ninfer::ops::KvKeyCoding::Int8>
        <<<grid, Shape::Threads, SharedBytes, stream>>>(
            static_cast<const __nv_bfloat16*>(q.data),
            static_cast<const std::int8_t*>(cache.k_pages.data),
            static_cast<const std::int8_t*>(cache.v_pages.data),
            static_cast<const __half*>(cache.k_scale_pages.data),
            static_cast<const __half*>(cache.v_scale_pages.data), metadata,
            static_cast<const std::int32_t*>(positions.data), scale,
            static_cast<__nv_bfloat16*>(out.data), tokens);
    return cudaGetLastError();
}

#undef ATTENTION_K_SCALE_LAUNCH
#undef causal_attention_prompt_i8_fast_kernel
