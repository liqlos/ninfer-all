#define NINFER_GGUF_MMQ_STORE_FUSION 1
#define NINFER_GGUF_Q4_MMQ_X_BANK_SWIZZLE 1
#define mul_mat_q q4_mmq_x_bank_swizzle_on_mul_mat_q
#define mul_mat_q_stream_k_fixup q4_mmq_x_bank_swizzle_on_mul_mat_q_stream_k_fixup
#define launch_matrix q4_mmq_x_bank_swizzle_on_launch_matrix
#define matrix_product_impl q4_mmq_x_bank_swizzle_on_matrix_product_impl
#define matrix_fixup_bytes_impl q4_mmq_x_bank_swizzle_on_matrix_fixup_bytes_impl
#define select_matrix_launch q4_mmq_x_bank_swizzle_on_select_matrix_launch
#define matrix_bf16_store_plan_impl q4_mmq_x_bank_swizzle_on_matrix_bf16_store_plan_impl
#define matrix_product_bf16_store_impl q4_mmq_x_bank_swizzle_on_matrix_product_bf16_store_impl

#include "ops/linear/gguf/ggml_bridge_mmq.cuh"
#include "probe_api.h"

bool q4_mmq_x_bank_swizzle_on_plan(int rows, int columns, Q4MmqProbePlan* plan) {
    const auto selected = ninfer::ops::gguf::detail::q4_mmq_x_bank_swizzle_on_select_matrix_launch<GGML_TYPE_Q4_K>(rows, columns);
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

void q4_mmq_x_bank_swizzle_on_run(const Q4MmqProbePlan* plan, const char* weight,
                           int stride_row_blocks, int rows, int k, const int* activation,
                           int columns, float* out, __nv_bfloat16* out_bf16,
                           std::int64_t output_column_stride, float* fixup, cudaStream_t stream) {
    using namespace ninfer::ops::gguf::detail;
    const MatrixLaunch selected{plan->J, plan->I, plan->threads, plan->shared,
                                plan->row_tiles, plan->column_tiles, plan->stream_k,
                                plan->blocks, plan->fixup};
    if (plan->J != 128) { throw std::invalid_argument("Q4 MMQ probe requires native J128"); }
    q4_mmq_x_bank_swizzle_on_launch_matrix<GGML_TYPE_Q4_K, 128>(
        selected, weight, stride_row_blocks, rows, k, activation, columns, out, out_bf16,
        output_column_stride, fixup, stream);
}

__global__ void q4_mmq_x_bank_swizzle_layout_kernel(int* result) {
    __shared__ __align__(16) int values[128 * 76];
    for (int i = threadIdx.x; i < 128 * 64; i += 32) {
        const int row = i / 64, k = i % 64;
        values[row * 76 + (k ^ ((k & 32) >> 2))] = row * 10000 + k;
    }
    __syncthreads();
    for (int row0 = 0; row0 < 128; row0 += 16) for (int k0 = 0; k0 < 64; k0 += 8) {
        ggml_cuda_mma::tile<16, 8, int> actual;
        ggml_cuda_mma::load_ldmatrix(actual, values + row0 * 76 + (k0 ^ ((k0 & 32) >> 2)), 76);
        for (int r = 0; r < 4; ++r) {
            const int row = row0 + threadIdx.x / 4 + 8 * (r % 2);
            const int k = k0 + threadIdx.x % 4 + 4 * (r / 2);
            const int expected = row * 10000 + k;
            if (actual.x[r] != expected) atomicAdd(result, 1);
        }
    }
}
bool q4_mmq_x_bank_swizzle_layout_test() {
    int* device_result = nullptr;
    if (cudaMalloc(&device_result, sizeof(int)) != cudaSuccess) return false;
    if (cudaMemset(device_result, 0, sizeof(int)) != cudaSuccess) { cudaFree(device_result); return false; }
    q4_mmq_x_bank_swizzle_layout_kernel<<<1, 32>>>(device_result);
    int result = -1;
    const auto copied = cudaMemcpy(&result, device_result, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(device_result);
    return copied == cudaSuccess && result == 0;
}
