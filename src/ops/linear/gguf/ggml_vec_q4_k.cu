// The vector products for one block type, in their own unit so they build in parallel with the others.
#include "ggml_bridge_vec.cuh"

#include <stdexcept>

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

#if defined(NINFER_GGUF_VECTOR_PREFETCH2_TEST)
namespace ninfer::ops::gguf::detail {

template <int T>
void q4_vector_prefetch2_default_baseline(const VecArgs& args, cudaStream_t stream) {
    if (args.k == 5120) {
        vec::launch_baseline<GGML_TYPE_Q4_K, T, false, 5120>(args, stream);
    } else if (args.k == 17408) {
        vec::launch_baseline<GGML_TYPE_Q4_K, T, false, 17408>(args, stream);
    } else {
        throw std::invalid_argument("Q4 vector prefetch2 baseline supports K5120 or K17408");
    }
}

void q4_vector_prefetch2_baseline_for_test(const std::uint8_t* weight, std::int64_t row_bytes,
                                           int rows, int k, const std::uint8_t* activation,
                                           int columns, void* output,
                                           std::int64_t output_column_stride,
                                           cudaStream_t stream) {
    const VecArgs args{weight,
                       row_bytes,
                       rows,
                       k,
                       reinterpret_cast<const std::int8_t*>(activation),
                       reinterpret_cast<const half2*>(activation + vec_scales_offset(k, columns)),
                       VecStore{static_cast<__nv_bfloat16*>(output), nullptr,
                                output_column_stride, Epilogue::Store, nullptr, 0}};
    switch (columns) {
    case 1: return q4_vector_prefetch2_default_baseline<1>(args, stream);
    case 2: return q4_vector_prefetch2_default_baseline<2>(args, stream);
    case 3: return q4_vector_prefetch2_default_baseline<3>(args, stream);
    case 4: return q4_vector_prefetch2_default_baseline<4>(args, stream);
    default: throw std::invalid_argument("Q4 vector prefetch2 baseline supports T1..4");
    }
}

template <int T>
void q4_vector_prefetch2_fused_default_baseline(const VecArgs& args, cudaStream_t stream) {
    if (args.k == 5120) {
        vec::launch_baseline<GGML_TYPE_Q4_K, T, true, 5120>(args, stream);
    } else {
        throw std::invalid_argument("Q4 fused vector prefetch2 baseline supports K5120");
    }
}

void q4_vector_prefetch2_fused_baseline_for_test(
    const std::uint8_t* weight, std::int64_t row_bytes, int rows, int k,
    const std::uint8_t* activation, int columns, void* output,
    std::int64_t output_column_stride, cudaStream_t stream) {
    const VecArgs args{weight,
                       row_bytes,
                       rows,
                       k,
                       reinterpret_cast<const std::int8_t*>(activation),
                       reinterpret_cast<const half2*>(activation + vec_scales_offset(k, columns)),
                       VecStore{static_cast<__nv_bfloat16*>(output), nullptr,
                                output_column_stride, Epilogue::Store, nullptr, 0}};
    switch (columns) {
    case 1: return q4_vector_prefetch2_fused_default_baseline<1>(args, stream);
    case 2: return q4_vector_prefetch2_fused_default_baseline<2>(args, stream);
    case 3: return q4_vector_prefetch2_fused_default_baseline<3>(args, stream);
    case 4: return q4_vector_prefetch2_fused_default_baseline<4>(args, stream);
    default: throw std::invalid_argument("Q4 fused vector prefetch2 baseline supports T1..4");
    }
}

}
#endif

namespace ninfer::ops::gguf::detail {

void q4_vector_baseline_for_test(const std::uint8_t* weight, std::int64_t row_bytes,
                                 int rows, int k, const std::uint8_t* activation, int columns,
                                 void* output, std::int64_t output_column_stride, bool fused,
                                 cudaStream_t stream) {
    const VecArgs args{weight, row_bytes, rows, k,
                       reinterpret_cast<const std::int8_t*>(activation),
                       reinterpret_cast<const half2*>(activation + vec_scales_offset(k, columns)),
                       VecStore{static_cast<__nv_bfloat16*>(output), nullptr,
                                output_column_stride, Epilogue::Store, nullptr, 0}};
    switch (columns) {
    case 1: return vec::launch_baseline<GGML_TYPE_Q4_K, 1, false, 0>(args, stream);
    case 2: return vec::launch_baseline<GGML_TYPE_Q4_K, 2, false, 0>(args, stream);
    case 3: return vec::launch_baseline<GGML_TYPE_Q4_K, 3, false, 0>(args, stream);
    case 4:
        if (fused) {
            if (k != 5120) { throw std::invalid_argument("Q4 fused baseline requires K5120"); }
            return vec::launch_baseline<GGML_TYPE_Q4_K, 4, true, 5120>(args, stream);
        }
        return vec::launch_baseline<GGML_TYPE_Q4_K, 4, false, 0>(args, stream);
    default: throw std::invalid_argument("Q4 vector baseline supports T1..4");
    }
}

}
