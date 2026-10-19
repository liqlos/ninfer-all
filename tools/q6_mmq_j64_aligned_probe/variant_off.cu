#define NINFER_GGUF_MMQ_STORE_FUSION 1
#define mul_mat_q q6_mmq_j64_aligned_off_mul_mat_q
#define mul_mat_q_stream_k_fixup q6_mmq_j64_aligned_off_mul_mat_q_stream_k_fixup
#define launch_matrix q6_mmq_j64_aligned_off_launch_matrix
#define matrix_product_impl q6_mmq_j64_aligned_off_matrix_product_impl
#define matrix_fixup_bytes_impl q6_mmq_j64_aligned_off_matrix_fixup_bytes_impl
#define select_matrix_launch q6_mmq_j64_aligned_off_select_matrix_launch
#define matrix_bf16_store_plan_impl q6_mmq_j64_aligned_off_matrix_bf16_store_plan_impl
#define matrix_product_bf16_store_impl q6_mmq_j64_aligned_off_matrix_product_bf16_store_impl

#include "ops/linear/gguf/ggml_bridge_mmq.cuh"
#include "probe_api.h"

bool q6_mmq_j64_aligned_off_plan(int rows, int columns, Q6MmqProbePlan* plan) {
    const auto selected = ninfer::ops::gguf::detail::q6_mmq_j64_aligned_off_select_matrix_launch<GGML_TYPE_Q6_K>(rows, columns);
    plan->J = selected.J;
    plan->I = selected.I;
    plan->threads = selected.threads;
    plan->shared = selected.shared;
    plan->row_tiles = selected.row_tiles;
    plan->column_tiles = selected.column_tiles;
    plan->stream_k = selected.stream_k;
    plan->blocks = selected.blocks;
    plan->fixup = selected.fixup;
    return true;
}

void q6_mmq_j64_aligned_off_run(const Q6MmqProbePlan* plan, const char* weight,
                            int stride_row_blocks, int rows, int k, const int* activation,
                            int columns, float* out, __nv_bfloat16* out_bf16,
                            std::int64_t output_column_stride, float* fixup, cudaStream_t stream) {
    using namespace ninfer::ops::gguf::detail;
    const MatrixLaunch selected{plan->J, plan->I, plan->threads, plan->shared,
                                plan->row_tiles, plan->column_tiles, plan->stream_k,
                                plan->blocks, plan->fixup};
    if (plan->J != 128) { throw std::invalid_argument("Q6 MMQ probe requires native J128"); }
    q6_mmq_j64_aligned_off_launch_matrix<GGML_TYPE_Q6_K, 128>(
        selected, weight, stride_row_blocks, rows, k, activation, columns, out, out_bf16,
        output_column_stride, fixup, stream);
}
