// The vector products for one block type, in their own unit so they build in parallel with the others.
#include "ggml_bridge_vec.cuh"

#include <stdexcept>

NINFER_GGUF_VECTOR_INSTANCE(GGML_TYPE_Q6_K);

#if defined(NINFER_GGUF_VECTOR_PREFETCH2_TEST)
namespace ninfer::ops::gguf::detail {

template <int T>
void q6_vector_prefetch2_default_baseline(const VecArgs& args, cudaStream_t stream) {
#if defined(NINFER_GGUF_Q6_STATIC_K)
    if constexpr (T == 1 || T == 4) {
        if (args.k == 5120) {
            return vec::launch_baseline<GGML_TYPE_Q6_K, T, false, 5120, false>(args, stream);
        }
    }
#endif
    if (args.k != 5120 && args.k != 17408) {
        throw std::invalid_argument("Q6 vector prefetch2 baseline supports K5120 or K17408");
    }
    vec::launch_baseline<GGML_TYPE_Q6_K, T, false, 0, false>(args, stream);
}

void q6_vector_prefetch2_baseline_for_test(const std::uint8_t* weight, std::int64_t row_bytes,
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
    case 1: return q6_vector_prefetch2_default_baseline<1>(args, stream);
    case 2: return q6_vector_prefetch2_default_baseline<2>(args, stream);
    case 3: return q6_vector_prefetch2_default_baseline<3>(args, stream);
    case 4: return q6_vector_prefetch2_default_baseline<4>(args, stream);
    default: throw std::invalid_argument("Q6 vector prefetch2 baseline supports T1..4");
    }
}

}
#endif
