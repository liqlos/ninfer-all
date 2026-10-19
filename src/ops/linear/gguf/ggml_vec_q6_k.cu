// The vector products for one block type, in their own unit so they build in parallel with the others.
#include "ggml_bridge_vec.cuh"

#include <stdexcept>

NINFER_GGUF_VECTOR_INSTANCE(GGML_TYPE_Q6_K);

#if defined(NINFER_GGUF_Q6_HEAD_ALIGNED_PLANES)
namespace ninfer::ops::gguf::detail {
namespace {

template <int T>
void launch_q6_head_aligned(const VecArgs& args, cudaStream_t stream) {
    if (args.k != 5120 || device_facts().cc != 860) {
        throw std::invalid_argument("Q6 head aligned vector route requires SM86 K5120");
    }
    VecArgs aligned = args;
    aligned.q6_head_aligned_planes = true;
    static const int resident = [] {
        int blocks = 0;
        check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                  &blocks, vec::kernel<GGML_TYPE_Q6_K, T, false, 5120, false>,
                  kVecWarps * 32, 0),
              "Q6 head aligned vector occupancy");
        return std::max(blocks, 1);
    }();
    constexpr int per_block = kVecWarps * 2;
    const int groups = (aligned.rows + per_block - 1) / per_block;
    const int blocks = std::min(groups, resident * device_facts().sm_count);
    vec::kernel<GGML_TYPE_Q6_K, T, false, 5120, false>
        <<<blocks, kVecWarps * 32, 0, stream>>>(aligned);
    check(cudaGetLastError(), "Q6 head aligned vector launch");
}

}

void q6_head_aligned_vec_launch(const VecArgs& args, int columns, cudaStream_t stream) {
    if (args.rows <= 0 || args.row_bytes != 4224 || (columns != 1 && columns != 4)) {
        throw std::invalid_argument("Q6 head aligned vector route has unsupported geometry");
    }
    switch (columns) {
    case 1: return launch_q6_head_aligned<1>(args, stream);
    case 4: return launch_q6_head_aligned<4>(args, stream);
    default: throw std::invalid_argument("Q6 head aligned vector route supports T1 or T4");
    }
}
}
#endif

#if defined(NINFER_GGUF_VECTOR_PREFETCH2_TEST)
namespace ninfer::ops::gguf::detail {

template <int T>
void q6_vector_prefetch2_default_baseline(const VecArgs& args, cudaStream_t stream) {
#if defined(NINFER_GGUF_Q6_STATIC_K)
    if constexpr (T == 1 || T == 4) {
        if (args.k == 5120) {
            return vec::launch_baseline<GGML_TYPE_Q6_K, T, false, 5120>(args, stream);
        }
    }
#endif
    if (args.k != 5120 && args.k != 17408) {
        throw std::invalid_argument("Q6 vector prefetch2 baseline supports K5120 or K17408");
    }
    vec::launch_baseline<GGML_TYPE_Q6_K, T, false, 0>(args, stream);
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
    if (k == 5120) {
        vec::launch_baseline<GGML_TYPE_Q6_K, 4, false, 5120>(args, stream);
        return;
    }
#endif
    vec::launch_baseline<GGML_TYPE_Q6_K, 4, false, 0>(args, stream);
}

}
#endif

namespace ninfer::ops::gguf::detail {

void q6_vector_baseline_for_test(const std::uint8_t* weight, std::int64_t row_bytes,
                                 int rows, int k, const std::uint8_t* activation, int columns,
                                 void* output, std::int64_t output_column_stride, cudaStream_t stream) {
    const VecArgs args{weight, row_bytes, rows, k,
                       reinterpret_cast<const std::int8_t*>(activation),
                       reinterpret_cast<const half2*>(activation + vec_scales_offset(k, columns)),
                       VecStore{static_cast<__nv_bfloat16*>(output), nullptr,
                                output_column_stride, Epilogue::Store, nullptr, 0}};
    switch (columns) {
    case 1:
#if defined(NINFER_GGUF_Q6_STATIC_K)
        if (k == 5120) { return vec::launch_baseline<GGML_TYPE_Q6_K, 1, false, 5120>(args, stream); }
#endif
        return vec::launch_baseline<GGML_TYPE_Q6_K, 1, false, 0>(args, stream);
    case 2: return vec::launch_baseline<GGML_TYPE_Q6_K, 2, false, 0>(args, stream);
    case 3: return vec::launch_baseline<GGML_TYPE_Q6_K, 3, false, 0>(args, stream);
    case 4:
#if defined(NINFER_GGUF_Q6_STATIC_K)
        if (k == 5120) { return vec::launch_baseline<GGML_TYPE_Q6_K, 4, false, 5120>(args, stream); }
#endif
        return vec::launch_baseline<GGML_TYPE_Q6_K, 4, false, 0>(args, stream);
    default: throw std::invalid_argument("Q6 vector baseline supports T1..4");
    }
}

}
