// The vector products for one block type, in their own unit so they build in parallel with the others.
#include "ggml_bridge_vec.cuh"

NINFER_GGUF_VECTOR_INSTANCE(GGML_TYPE_Q4_K);

namespace ninfer::ops::gguf::detail {

void q4_vector_baseline_for_test(const std::uint8_t* weight, std::int64_t row_bytes,
                                 int rows, int k, const std::uint8_t* activation, int columns,
                                 void* output, std::int64_t output_column_stride, bool fused,
                                 cudaStream_t stream) {
    if (k != 5120 || columns != 4 || rows <= 0) {
        throw std::invalid_argument("Q4 vector baseline test geometry is unsupported");
    }
    const VecArgs args{weight,
                       row_bytes,
                       rows,
                       k,
                       reinterpret_cast<const std::int8_t*>(activation),
                       reinterpret_cast<const half2*>(activation + vec_scales_offset(k, columns)),
                       VecStore{static_cast<__nv_bfloat16*>(output), nullptr,
                                output_column_stride, Epilogue::Store, nullptr, 0}};
    if (fused) {
        vec::launch_baseline<GGML_TYPE_Q4_K, 4, true, 5120>(args, stream);
    } else {
        vec::launch_baseline<GGML_TYPE_Q4_K, 4, false, 5120>(args, stream);
    }
}

}

#if defined(NINFER_GGUF_Q4_T4_IMMA_TEST)
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

}
#endif
