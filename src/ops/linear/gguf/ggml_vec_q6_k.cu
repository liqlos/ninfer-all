// The vector products for one block type, in their own unit so they build in parallel with the others.
#include "ggml_bridge_vec.cuh"

NINFER_GGUF_VECTOR_INSTANCE(GGML_TYPE_Q6_K);

namespace ninfer::ops::gguf::detail {

void q6_vector_baseline_for_test(const std::uint8_t* weight, std::int64_t row_bytes,
                                 int rows, int k, const std::uint8_t* activation, int columns,
                                 void* output, std::int64_t output_column_stride, bool fused,
                                 cudaStream_t stream) {
    if ((k != 5120 && k != 17408) || (columns != 1 && columns != 2 && columns != 4) ||
        rows <= 0 || fused) {
        throw std::invalid_argument("Q6 vector baseline test geometry is unsupported");
    }
    const VecArgs args{weight,
                       row_bytes,
                       rows,
                       k,
                       reinterpret_cast<const std::int8_t*>(activation),
                       reinterpret_cast<const half2*>(activation + vec_scales_offset(k, columns)),
                       VecStore{static_cast<__nv_bfloat16*>(output), nullptr,
                                output_column_stride, Epilogue::Store, nullptr, 0}};
    const auto run = [&]<int T>() {
        if (k == 5120) {
#if defined(NINFER_GGUF_Q6_STATIC_K)
            if constexpr (T == 1 || T == 4) {
                if (device_facts().cc == 860) {
                    return vec::launch_baseline<GGML_TYPE_Q6_K, T, false, 5120>(args, stream);
                }
            }
#endif
        }
        return vec::launch_baseline<GGML_TYPE_Q6_K, T, false, 0>(args, stream);
    };
    switch (columns) {
    case 1: return run.template operator()<1>();
    case 2: return run.template operator()<2>();
    case 4: return run.template operator()<4>();
    }
}

}
