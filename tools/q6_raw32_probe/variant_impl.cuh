#include "ops/linear/gguf/ggml_bridge_mmq.cuh"
#include "probe_api.h"

#if defined(NINFER_GGUF_Q6_MMQ_A_X4)
#error Q6_A_X4_must_be_disabled
#endif

bool PROBE_PLAN(int rows, int columns, Q6Raw32Plan* plan) {
    using namespace ninfer::ops::gguf::detail;
    if (!plan || rows <= 0 || rows % 128 || columns <= 64 || device_facts().cc != 860) {
        return false;
    }
    const auto selected = select_matrix_launch<GGML_TYPE_Q6_K>(rows, columns);
    if (selected.J != 128 || selected.I != 128 || selected.threads != 256) { return false; }
    *plan = {selected.J, selected.I, selected.threads, selected.shared, selected.row_tiles,
             selected.column_tiles, selected.stream_k, selected.blocks, selected.fixup};
    return true;
}

bool PROBE_RUN(const Q6Raw32Plan* plan, const char* weight, int stride_row_blocks,
    int rows, int k, const int* activation, int activation_layout, int columns, float* out,
    __nv_bfloat16* out_bf16, bool accumulate, std::int64_t output_column_stride,
    float* fixup, cudaStream_t stream) {
    using namespace ninfer::ops::gguf::detail;
    if (!plan || !weight || !activation || k <= 0 || k % 512 || stride_row_blocks != k / 256 ||
        rows <= 0 || columns <= 64 || output_column_stride < rows ||
        output_column_stride > INT_MAX || (!!out == !!out_bf16) ||
        (accumulate && !out_bf16) || (out_bf16 && plan->fixup) ||
        (plan->fixup && !fixup) || activation_layout != MMQ_Q8_1_DS_LAYOUT_D4 ||
        (reinterpret_cast<std::uintptr_t>(weight) & 1u) ||
        (reinterpret_cast<std::uintptr_t>(activation) & 3u)) {
        return false;
    }
    Q6Raw32Plan expected{};
    if (!PROBE_PLAN(rows, columns, &expected) || plan->J != expected.J ||
        plan->I != expected.I || plan->threads != expected.threads ||
        plan->shared != expected.shared || plan->row_tiles != expected.row_tiles ||
        plan->column_tiles != expected.column_tiles || plan->stream_k != expected.stream_k ||
        plan->blocks != expected.blocks || plan->fixup != expected.fixup) {
        return false;
    }
    const auto output = reinterpret_cast<std::uintptr_t>(out ? static_cast<void*>(out)
                                                            : static_cast<void*>(out_bf16));
    const std::size_t output_bytes = std::size_t(output_column_stride) * columns *
                                    (out ? sizeof(float) : sizeof(__nv_bfloat16));
    if (output & (out ? 3u : 1u)) { return false; }
    const auto overlaps = [&](const void* pointer, std::size_t bytes) {
        const auto other = reinterpret_cast<std::uintptr_t>(pointer);
        return output <= other ? other - output < output_bytes : output - other < bytes;
    };
    if (overlaps(weight, std::size_t(rows) * stride_row_blocks * 210) ||
        overlaps(activation, std::size_t(k / 128) * columns * sizeof(block_q8_1_mmq) +
                             kMatrixActivationSlack) ||
        (fixup && overlaps(fixup, std::size_t(plan->blocks) * plan->J * plan->I * sizeof(float)))) {
        return false;
    }
    const MatrixLaunch selected{plan->J, plan->I, plan->threads, plan->shared,
        plan->row_tiles, plan->column_tiles, plan->stream_k, plan->blocks, plan->fixup};
    if (accumulate) {
        launch_matrix<GGML_TYPE_Q6_K, 128, true>(selected, weight, stride_row_blocks,
            rows, k, activation, columns, nullptr, out_bf16, output_column_stride, fixup, stream);
    } else {
        launch_matrix<GGML_TYPE_Q6_K, 128>(selected, weight, stride_row_blocks,
            rows, k, activation, columns, out, out_bf16, output_column_stride, fixup, stream);
    }
    return true;
}
