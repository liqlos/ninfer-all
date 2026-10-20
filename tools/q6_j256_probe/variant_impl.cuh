#include "ops/linear/gguf/ggml_bridge_mmq.cuh"
#include "probe_api.h"

static_assert(sizeof(block_q6_K) == 210 && sizeof(block_q8_1_mmq) == 144, "native block widths");

bool PROBE_PLAN(int rows, int k, int columns, Q6J256Plan* plan) {
    using namespace ninfer::ops::gguf::detail;
    if (!plan || rows != 5120 || k != 17408 || columns != 1280 ||
        device_facts().cc != 860 || device_facts().sm_count != 82) {
        return false;
    }
#if NINFER_Q6_J256_PROBE
    if (device_facts().shared_per_block_optin < 76800) { return false; }
    *plan = {256, 128, 512, 76800, 40, 5, false, 200, false};
#else
    const auto selected = select_matrix_launch<GGML_TYPE_Q6_K>(rows, columns);
    if (selected.J != 128 || selected.I != 128 || selected.threads != 256 ||
        selected.shared != 57856 || selected.row_tiles != 40 || selected.column_tiles != 10 ||
        !selected.stream_k || selected.blocks != 400 || selected.fixup) {
        return false;
    }
    *plan = {selected.J, selected.I, selected.threads, selected.shared, selected.row_tiles,
             selected.column_tiles, selected.stream_k, selected.blocks, selected.fixup};
#endif
    return true;
}

#if NINFER_Q6_J256_PROBE
template <bool bf16_store, bool bf16_accumulate>
static void launch_j256(const char* weight, const int* activation, float* out,
                        __nv_bfloat16* out_bf16, int stride, cudaStream_t stream) {
    using namespace ninfer::ops::gguf::detail;
    static std::once_flag raised[16];
    int device = 0;
    check(cudaGetDevice(&device), "Q6 J256 device");
    if (device < 0 || device >= 16) { throw std::invalid_argument("Q6 J256 device index"); }
    std::call_once(raised[device], [&] {
        check(cudaFuncSetAttribute(q6_j256_kernel<bf16_store, bf16_accumulate>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, 76800), "Q6 J256 shared memory");
    });
    q6_j256_kernel<bf16_store, bf16_accumulate>
        <<<dim3(200, 1, 1), dim3(32, 16, 1), 76800, stream>>>(weight, activation, out, out_bf16, stride);
    check(cudaGetLastError(), "Q6 J256 launch");
}
#endif

bool PROBE_RUN(const Q6J256Plan* plan, const char* weight, int stride_row_blocks,
    int rows, int k, const int* activation, int activation_layout, int columns,
    float* out, __nv_bfloat16* out_bf16, bool accumulate, std::int64_t output_column_stride,
    float* fixup, cudaStream_t stream) {
    using namespace ninfer::ops::gguf::detail;
    if (!plan || !weight || !activation || rows != 5120 || k != 17408 || columns != 1280 ||
        stride_row_blocks != 68 || fixup != nullptr || (!!out == !!out_bf16) ||
        (accumulate && !out_bf16) || output_column_stride < rows ||
        output_column_stride > INT_MAX / columns || activation_layout != MMQ_Q8_1_DS_LAYOUT_D4 ||
        (reinterpret_cast<std::uintptr_t>(weight) & 1u) ||
        (reinterpret_cast<std::uintptr_t>(activation) & 3u)) {
        return false;
    }
    Q6J256Plan expected{};
    if (!PROBE_PLAN(rows, k, columns, &expected) || plan->J != expected.J ||
        plan->I != expected.I || plan->threads != expected.threads || plan->shared != expected.shared ||
        plan->row_tiles != expected.row_tiles || plan->column_tiles != expected.column_tiles ||
        plan->stream_k != expected.stream_k || plan->blocks != expected.blocks || plan->fixup != expected.fixup) {
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
    if (overlaps(weight, std::size_t(rows) * stride_row_blocks * sizeof(block_q6_K)) ||
        overlaps(activation, std::size_t(k / 128) * columns * sizeof(block_q8_1_mmq) +
                             kMatrixActivationSlack)) {
        return false;
    }
#if NINFER_Q6_J256_PROBE
    if (accumulate) {
        launch_j256<true, true>(weight, activation, nullptr, out_bf16,
                               static_cast<int>(output_column_stride), stream);
    } else if (out_bf16) {
        launch_j256<true, false>(weight, activation, nullptr, out_bf16,
                                static_cast<int>(output_column_stride), stream);
    } else {
        launch_j256<false, false>(weight, activation, out, nullptr,
                                 static_cast<int>(output_column_stride), stream);
    }
#else
    const MatrixLaunch selected{plan->J, plan->I, plan->threads, plan->shared,
        plan->row_tiles, plan->column_tiles, plan->stream_k, plan->blocks, plan->fixup};
    if (accumulate) {
        launch_matrix<GGML_TYPE_Q6_K, 128, true>(selected, weight, stride_row_blocks,
            rows, k, activation, columns, nullptr, out_bf16, output_column_stride, nullptr, stream);
    } else {
        launch_matrix<GGML_TYPE_Q6_K, 128>(selected, weight, stride_row_blocks,
            rows, k, activation, columns, out, out_bf16, output_column_stride, nullptr, stream);
    }
#endif
    return true;
}
