#include "q6_word_group_vector.cuh"
#include "host_checks.cuh"

namespace {

using namespace ninfer::ops::gguf::detail;

template <int T, bool Candidate>
void launch_vector(const VecArgs& args, const Q6WGView& operand, cudaStream_t stream) {
    int resident = 0;
    if constexpr (Candidate) {
#if NINFER_Q6_WORD_GROUP_PROBE
        check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &resident, vec::q6_word_group_vector_kernel<GGML_TYPE_Q6_K, T, false, 0, true, false, 2>,
            kVecWarps * 32, 0), "word-group vector occupancy");
#endif
    } else {
        check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &resident, vec::kernel<GGML_TYPE_Q6_K, T, false, 0, true, false, 2>,
            kVecWarps * 32, 0), "native vector occupancy");
    }
    if (resident <= 0) { throw std::runtime_error("zero selected vector occupancy"); }
    const int groups = (args.rows + kVecWarps * 2 - 1) / (kVecWarps * 2);
    const int blocks = std::min(groups, resident * device_facts().sm_count);
    if constexpr (Candidate) {
#if NINFER_Q6_WORD_GROUP_PROBE
        vec::q6_word_group_vector_kernel<GGML_TYPE_Q6_K, T, false, 0, true, false, 2>
            <<<blocks, kVecWarps * 32, 0, stream>>>(args, operand);
#endif
    } else {
        vec::kernel<GGML_TYPE_Q6_K, T, false, 0, true, false, 2>
            <<<blocks, kVecWarps * 32, 0, stream>>>(args);
    }
    check(cudaGetLastError(), "selected Q6 vector launch");
}

}

bool q6wg_candidate_enabled() {
    return NINFER_Q6_WORD_GROUP_PROBE == 1;
}

bool q6wg_validate_view(const Q6WGView& view) {
    return q6wg::validate_view(view);
}

std::size_t q6wg_vector_activation_bytes(int k, int columns) {
    if (k <= 0 || k % 256 || (columns != 1 && columns != 4)) { return 0; }
    return ninfer::ops::gguf::detail::vec_scales_offset(k, columns) +
           std::size_t(columns) * (k / 32) * sizeof(half2);
}

bool q6wg_vector_run(const Q6WGView& view, Q6WGReadBuffer activation, int columns,
                     const Q6WGOutput& output, cudaStream_t stream) {
    using namespace ninfer::ops::gguf;
    using namespace ninfer::ops::gguf::detail;
    if (!q6wg::validate_view(view) || view.rows > INT_MAX - kVecWarps * 2 ||
        (columns != 1 && columns != 4) || device_facts().cc != 860 ||
        (view.storage == Q6WGStorage::WordGroupSlab16 && !q6wg_candidate_enabled())) {
        return false;
    }
    const int k = static_cast<int>(view.parent_B * 256);
    const auto activation_bytes = q6wg_vector_activation_bytes(k, columns);
    std::size_t output_bytes = 0;
    if (activation.bytes < activation_bytes ||
        !q6wg::range_valid(activation.data, activation.bytes, 16) ||
        !q6wg::output_extent(output, view.rows, columns, output_bytes) ||
        !q6wg::output_disjoint(view, activation, output, output_bytes)) {
        return false;
    }
    const bool plain = output.mode == Q6WGOutputMode::PlainFP32;
    const VecStore out{
        plain ? nullptr : static_cast<__nv_bfloat16*>(output.buffer.data),
        plain ? static_cast<float*>(output.buffer.data) : nullptr,
        output.column_stride,
        output.mode == Q6WGOutputMode::AccumulateBF16 ? Epilogue::Accumulate : Epilogue::Store,
        nullptr, 0};
    const auto* values = static_cast<const std::int8_t*>(activation.data);
    const auto* scales = reinterpret_cast<const half2*>(
        static_cast<const std::uint8_t*>(activation.data) + vec_scales_offset(k, columns));
    const auto* weight = view.storage == Q6WGStorage::Native210
        ? view.base + std::uint64_t(view.row_begin) * view.parent_B * 210 : view.base;
    const VecArgs args{weight, std::int64_t(view.parent_B) * 210,
                       static_cast<int>(view.rows), k, values, scales, out};
    if (view.storage == Q6WGStorage::WordGroupSlab16) {
#if NINFER_Q6_WORD_GROUP_PROBE
        if (columns == 1) { launch_vector<1, true>(args, view, stream); }
        else { launch_vector<4, true>(args, view, stream); }
#else
        return false;
#endif
    } else {
        if (columns == 1) { launch_vector<1, false>(args, view, stream); }
        else { launch_vector<4, false>(args, view, stream); }
    }
    return true;
}
