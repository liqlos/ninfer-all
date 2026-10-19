// The vector products for one block type, in their own unit so they build in parallel with the others.
#include "ggml_bridge_vec.cuh"

NINFER_GGUF_VECTOR_INSTANCE(GGML_TYPE_Q6_K);

#if defined(NINFER_GGUF_Q6_T4_IMMA_TEST)
namespace ninfer::ops::gguf::detail {

void q6_t4_imma_baseline_for_test(const std::uint8_t* weight, std::int64_t row_bytes,
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
#if defined(NINFER_GGUF_Q6_STATIC_K)
    vec::launch_baseline<GGML_TYPE_Q6_K, 4, false, 5120>(args, stream);
#else
    vec::launch_baseline<GGML_TYPE_Q6_K, 4, false, 0>(args, stream);
#endif
}

}
#endif
