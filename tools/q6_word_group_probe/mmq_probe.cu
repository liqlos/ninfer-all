#include "q6_word_group_mmq.cuh"
#include "host_checks.cuh"

namespace {

using namespace ninfer::ops::gguf::detail;

bool equal_plan(const Q6WGMatrixPlan& a, const Q6WGMatrixPlan& b) {
    return a.rows == b.rows && a.k == b.k && a.columns == b.columns &&
           a.J == b.J && a.I == b.I && a.threads == b.threads && a.shared == b.shared &&
           a.row_tiles == b.row_tiles && a.column_tiles == b.column_tiles &&
           a.stream_k == b.stream_k && a.blocks == b.blocks && a.fixup == b.fixup &&
           a.activation_bytes == b.activation_bytes && a.fixup_bytes == b.fixup_bytes;
}

template <bool Candidate, bool Store, bool Accumulate>
void launch_matrix_selected(const Q6WGMatrixPlan& plan, const Q6WGView& operand,
                            const int* activation, void* output, int stride,
                            float* fixup, cudaStream_t stream) {
    static std::once_flag raised[16];
    int device = 0;
    check(cudaGetDevice(&device), "word-group matrix device");
    if (device < 0 || device >= 16) { throw std::invalid_argument("unsupported device ordinal"); }
    std::call_once(raised[device], [&] {
        if constexpr (Candidate) {
#if NINFER_Q6_WORD_GROUP_PROBE
            check(cudaFuncSetAttribute(
                q6_word_group_mul_mat_q<GGML_TYPE_Q6_K, 128, false, GGML_PREC_Q8, Store, Accumulate>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(plan.shared)),
                "word-group matrix shared memory");
#endif
        } else {
            check(cudaFuncSetAttribute(
                mul_mat_q<GGML_TYPE_Q6_K, 128, false, GGML_PREC_Q8, Store, Accumulate>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(plan.shared)),
                "native matrix shared memory");
        }
    });
    const uint3 blocks_fd = init_fastdiv_values(plan.k / 256);
    const uint3 one_fd = init_fastdiv_values(1);
    const uint3 ntx_fd = init_fastdiv_values(plan.column_tiles);
    float* out = Store ? nullptr : static_cast<float*>(output);
    __nv_bfloat16* out_bf16 = Store ? static_cast<__nv_bfloat16*>(output) : nullptr;
    const dim3 grid(plan.blocks, 1, 1);
    const dim3 threads(32, 8, 1);
    if constexpr (Candidate) {
#if NINFER_Q6_WORD_GROUP_PROBE
        q6_word_group_mul_mat_q<GGML_TYPE_Q6_K, 128, false, GGML_PREC_Q8, Store, Accumulate>
            <<<grid, threads, plan.shared, stream>>>(
                operand, activation, nullptr, nullptr, out, out_bf16, fixup, nullptr,
                blocks_fd, plan.rows, plan.columns, operand.parent_B, plan.columns, stride,
                one_fd, one_fd, 0, 0, 0, one_fd, one_fd, 0, 0, 0, ntx_fd);
#endif
    } else {
        const auto* weight = reinterpret_cast<const char*>(
            operand.base + std::uint64_t(operand.row_begin) * operand.parent_B * 210);
        mul_mat_q<GGML_TYPE_Q6_K, 128, false, GGML_PREC_Q8, Store, Accumulate>
            <<<grid, threads, plan.shared, stream>>>(
                weight, activation, nullptr, nullptr, out, out_bf16, fixup, nullptr,
                blocks_fd, plan.rows, plan.columns, operand.parent_B, plan.columns, stride,
                one_fd, one_fd, 0, 0, 0, one_fd, one_fd, 0, 0, 0, ntx_fd);
    }
    check(cudaGetLastError(), "selected Q6 matrix launch");
    if constexpr (!Store) {
        if (plan.fixup) {
            mul_mat_q_stream_k_fixup<GGML_TYPE_Q6_K, 128, false>
                <<<dim3(plan.blocks, 4, 1), dim3(32, 4, 1), 0, stream>>>(
                    nullptr, nullptr, out, fixup, blocks_fd, plan.rows, plan.columns,
                    stride, one_fd, 0, one_fd, 0, ntx_fd);
            check(cudaGetLastError(), "selected Q6 native fixup launch");
        }
    }
}

template <bool Candidate>
void launch_mode(const Q6WGMatrixPlan& plan, const Q6WGView& view,
                 const int* activation, const Q6WGOutput& output,
                 float* fixup, cudaStream_t stream) {
    const int stride = static_cast<int>(output.column_stride);
    if (output.mode == Q6WGOutputMode::PlainFP32) {
        launch_matrix_selected<Candidate, false, false>(
            plan, view, activation, output.buffer.data, stride, fixup, stream);
    } else if (output.mode == Q6WGOutputMode::StoreBF16) {
        launch_matrix_selected<Candidate, true, false>(
            plan, view, activation, output.buffer.data, stride, nullptr, stream);
    } else {
        launch_matrix_selected<Candidate, true, true>(
            plan, view, activation, output.buffer.data, stride, nullptr, stream);
    }
}

}

int q6wg_matrix_activation_layout() {
    return MMQ_Q8_1_DS_LAYOUT_D4;
}

bool q6wg_matrix_plan(int rows, int k, int columns, Q6WGMatrixPlan* plan) {
    using namespace ninfer::ops::gguf::detail;
    if (!plan || rows <= 0 || rows % 128 || rows > INT_MAX - 128 ||
        columns <= 64 || columns > INT_MAX - 128 || k <= 0 || k % 256 ||
        std::int64_t(rows / 128) * ((columns + 127) / 128) > INT_MAX / 100 ||
        device_facts().cc != 860 || device_facts().sm_count <= 0) {
        return false;
    }
    const auto selected = select_matrix_launch<GGML_TYPE_Q6_K>(rows, columns);
    if (selected.J != 128 || selected.I != 128 || selected.threads != 256 ||
        selected.shared != 57856 || !selected.stream_k) {
        return false;
    }
    const std::size_t activation_bytes =
        std::size_t(k / 128) * columns * sizeof(block_q8_1_mmq) + kMatrixActivationSlack;
    const std::size_t fixup_bytes =
        selected.fixup ? std::size_t(selected.blocks) * 128 * 128 * sizeof(float) : 0;
    *plan = {rows, k, columns, selected.J, selected.I, selected.threads, selected.shared,
             selected.row_tiles, selected.column_tiles, selected.stream_k, selected.blocks,
             selected.fixup, activation_bytes, fixup_bytes};
    return true;
}

bool q6wg_matrix_run(const Q6WGMatrixPlan& plan, const Q6WGView& view,
                     Q6WGReadBuffer activation, int activation_layout,
                     const Q6WGOutput& output, Q6WGWriteBuffer fixup,
                     cudaStream_t stream) {
    if (!q6wg::validate_view(view) || view.rows != static_cast<unsigned>(plan.rows) ||
        view.parent_B * 256 != static_cast<unsigned>(plan.k) ||
        std::uint64_t(view.rows) * view.parent_B > INT_MAX ||
        activation_layout != MMQ_Q8_1_DS_LAYOUT_D4 ||
        (view.storage == Q6WGStorage::WordGroupSlab16 && !q6wg_candidate_enabled())) {
        return false;
    }
    Q6WGMatrixPlan expected{};
    if (!q6wg_matrix_plan(plan.rows, plan.k, plan.columns, &expected) || !equal_plan(plan, expected)) {
        return false;
    }
    std::size_t output_bytes = 0;
    if (activation.bytes < plan.activation_bytes ||
        !q6wg::range_valid(activation.data, activation.bytes, 4) ||
        !q6wg::output_extent(output, plan.rows, plan.columns, output_bytes) ||
        !q6wg::output_disjoint(view, activation, output, output_bytes) ||
        (output.mode != Q6WGOutputMode::PlainFP32 && plan.fixup)) {
        return false;
    }
    if (plan.fixup) {
        if (fixup.bytes < plan.fixup_bytes || !q6wg::range_valid(fixup.data, fixup.bytes, 4) ||
            q6wg::overlaps(fixup.data, fixup.bytes, view.base, view.bytes) ||
            q6wg::overlaps(fixup.data, fixup.bytes, activation.data, activation.bytes) ||
            q6wg::overlaps(fixup.data, fixup.bytes, output.buffer.data, output_bytes)) {
            return false;
        }
    } else if (fixup.data || fixup.bytes) {
        return false;
    }
    const auto* values = static_cast<const int*>(activation.data);
    if (view.storage == Q6WGStorage::WordGroupSlab16) {
#if NINFER_Q6_WORD_GROUP_PROBE
        launch_mode<true>(plan, view, values, output, static_cast<float*>(fixup.data), stream);
#else
        return false;
#endif
    } else {
        launch_mode<false>(plan, view, values, output, static_cast<float*>(fixup.data), stream);
    }
    return true;
}
