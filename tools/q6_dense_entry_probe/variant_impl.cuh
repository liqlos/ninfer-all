#ifndef NINFER_GGUF_Q6_MMQ_DENSE_ENTRY
#define NINFER_GGUF_Q6_MMQ_DENSE_ENTRY 0
#endif

#if defined(NINFER_GGUF_Q6_MMQ_A_X4) || defined(NINFER_GGUF_Q6_MMQ_ALIGNED_RAW32) || \
    defined(NINFER_GGUF_Q6_MMQ_ASYNC_Y) || defined(NINFER_GGUF_Q4_MMQ_ASYNC_Y) || \
    defined(NINFER_GGUF_Q4_MMQ_ASYNC_X) || defined(NINFER_GGUF_Q4_MMQ_CHILD_CTA) || \
    defined(NINFER_GGUF_Q4_MMQ_K_STREAM) || defined(NINFER_GGUF_Q4_MMQ_PACKED_DM) || \
    defined(NINFER_GGUF_SM86_MMQ_TILE_CAP) || defined(NINFER_GGUF_MMQ_GATE_EPILOGUE_FUSION)
#error Unrelated_MMQ_experiments_must_be_disabled
#endif

#include "ops/linear/gguf/ggml_bridge_mmq.cuh"
#include "probe_api.h"

bool PROBE_PLAN(int rows, int columns, Q6DenseEntryPlan* plan) {
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

bool PROBE_USES_DENSE(const Q6DenseEntryPlan* plan, int rows, int k, int columns) {
#if NINFER_GGUF_Q6_MMQ_DENSE_ENTRY
    using namespace ninfer::ops::gguf::detail;
    const bool shape = (rows == 5120 && k == 17408) ||
                       ((rows == 10240 || rows == 1024) && k == 5120);
    return plan && shape && device_facts().cc == 860 && columns == 1280 &&
           rows % 128 == 0 && columns % 128 == 0 && k > 0 && k % 512 == 0 &&
           plan->I == 128 && plan->J == 128 && plan->threads == 256 &&
           plan->stream_k && !plan->fixup && plan->row_tiles == rows / 128 &&
           plan->column_tiles == columns / 128 &&
           std::int64_t(plan->blocks) == std::int64_t(plan->row_tiles) * plan->column_tiles;
#else
    (void)plan;
    (void)rows;
    (void)k;
    (void)columns;
    return false;
#endif
}

#if NINFER_GGUF_Q6_MMQ_DENSE_ENTRY
template <bool bf16_store, bool accumulate>
static void launch_dense(const Q6DenseEntryPlan& plan, const char* weight, int stride_row_blocks,
    int k, const int* activation, int columns, float* out, __nv_bfloat16* out_bf16,
    int output_column_stride, cudaStream_t stream) {
    using namespace ninfer::ops::gguf::detail;
    static std::once_flag raised[16];
    int device = 0;
    check(cudaGetDevice(&device), "dense-entry device");
    std::call_once(raised[device & 15], [&] {
        check(cudaFuncSetAttribute(q6_dense_entry_kernel<bf16_store, accumulate>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   static_cast<int>(plan.shared)), "dense-entry shared memory");
    });
    q6_dense_entry_kernel<bf16_store, accumulate>
        <<<dim3(plan.blocks, 1, 1), dim3(32, 8, 1), plan.shared, stream>>>(
            weight, activation, out, out_bf16, stride_row_blocks, columns,
            output_column_stride, k / 256, init_fastdiv_values(plan.column_tiles));
    check(cudaGetLastError(), "dense-entry matrix launch");
}
#endif

bool PROBE_RUN(const Q6DenseEntryPlan* plan, const char* weight, int stride_row_blocks,
    int rows, int k, const int* activation, int activation_layout, int columns, float* out,
    __nv_bfloat16* out_bf16, bool accumulate, std::int64_t output_column_stride,
    float* fixup, cudaStream_t stream) {
    using namespace ninfer::ops::gguf::detail;
    if (!plan || !weight || !activation || k <= 0 || k % 512 || stride_row_blocks != k / 256 ||
        rows <= 0 || columns <= 64 || output_column_stride < rows ||
        output_column_stride > INT_MAX || output_column_stride > INT_MAX / columns ||
        (!!out == !!out_bf16) || (accumulate && !out_bf16) || (out_bf16 && plan->fixup) ||
        (plan->fixup && !fixup) || activation_layout != MMQ_Q8_1_DS_LAYOUT_D4 ||
        (reinterpret_cast<std::uintptr_t>(weight) & 1u) ||
        (reinterpret_cast<std::uintptr_t>(activation) & 3u)) {
        return false;
    }
    Q6DenseEntryPlan expected{};
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
#if NINFER_GGUF_Q6_MMQ_DENSE_ENTRY
    if (PROBE_USES_DENSE(plan, rows, k, columns) && fixup == nullptr) {
        if (accumulate) {
            launch_dense<true, true>(*plan, weight, stride_row_blocks, k, activation, columns,
                                     nullptr, out_bf16, static_cast<int>(output_column_stride), stream);
        } else if (out_bf16) {
            launch_dense<true, false>(*plan, weight, stride_row_blocks, k, activation, columns,
                                      nullptr, out_bf16, static_cast<int>(output_column_stride), stream);
        } else {
            launch_dense<false, false>(*plan, weight, stride_row_blocks, k, activation, columns,
                                       out, nullptr, static_cast<int>(output_column_stride), stream);
        }
        return true;
    }
#endif
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
