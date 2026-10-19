// The vector products for one block type, in their own unit so they build in parallel with the others.
#include "ggml_bridge_vec.cuh"

NINFER_GGUF_VECTOR_INSTANCE(GGML_TYPE_Q4_K);

#if defined(NINFER_GGUF_Q4_T4_IMMA_TEST) || defined(NINFER_GGUF_Q4_T4_FUSED_IMMA_TEST)
namespace ninfer::ops::gguf::detail {

void q4_t4_imma_baseline_for_test(const std::uint8_t* weight, std::int64_t row_bytes,
                                  int rows, int k, const std::uint8_t* activation,
                                  void* output, std::int64_t output_column_stride,
                                  cudaStream_t stream) {
    const VecArgs args{weight,
                       row_bytes,
                       rows,
                       k,
                       reinterpret_cast<const std::int8_t*>(activation),
                       reinterpret_cast<const half2*>(activation + vec_scales_offset(k, 4)),
                       VecStore{static_cast<__nv_bfloat16*>(output), nullptr,
                                output_column_stride, Epilogue::Store, nullptr, 0}};
    vec::launch_baseline<GGML_TYPE_Q4_K, 4, false, 5120>(args, stream);
}

#if defined(NINFER_GGUF_Q4_T4_FUSED_IMMA_TEST)
void q4_t4_fused_imma_baseline_for_test(const std::uint8_t* weight, std::int64_t row_bytes,
                                        int rows, int k, const std::uint8_t* activation,
                                        void* output, std::int64_t output_column_stride,
                                        cudaStream_t stream) {
    const VecStore store{static_cast<__nv_bfloat16*>(output), nullptr,
                         output_column_stride, Epilogue::Store, nullptr, 0};
    const VecArgs args{weight,
                       row_bytes,
                       rows,
                       k,
                       reinterpret_cast<const std::int8_t*>(activation),
                       reinterpret_cast<const half2*>(activation + vec_scales_offset(k, 4)),
                       store};
    vec::launch_baseline<GGML_TYPE_Q4_K, 4, true, 5120>(args, stream);
}
#endif

}
#endif
