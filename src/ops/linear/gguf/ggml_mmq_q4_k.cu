// The integer tensor-core product for one block type, in its own unit so the heavy template builds in
// parallel with the others.
#include "ggml_bridge_mmq.cuh"

NINFER_GGUF_MATRIX_INSTANCE(GGML_TYPE_Q4_K);
#if defined(NINFER_GGUF_MMQ_STORE_FUSION)
NINFER_GGUF_MMQ_BF16_STORE_INSTANCE(GGML_TYPE_Q4_K);
#endif
#if defined(NINFER_GGUF_MMQ_GATE_EPILOGUE_FUSION)
#if !defined(NINFER_GGUF_MMQ_STORE_FUSION)
template bool ninfer::ops::gguf::detail::matrix_bf16_store_plan_impl<GGML_TYPE_Q4_K>(
    int, int, int&, int&, bool&);
#endif
template bool ninfer::ops::gguf::detail::matrix_product_bf16_gate_store_impl<GGML_TYPE_Q4_K>(
    const void*, std::int64_t, int, int, const void*, int, __nv_bfloat16*, std::int64_t,
    const float*, std::int64_t, cudaStream_t);
#endif
