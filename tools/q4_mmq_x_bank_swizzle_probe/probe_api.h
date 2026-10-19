#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

struct Q4MmqProbePlan {
    int J;
    int I;
    int threads;
    std::size_t shared;
    int row_tiles;
    int column_tiles;
    bool stream_k;
    int blocks;
    bool fixup;
};

bool q4_mmq_x_bank_swizzle_off_plan(int rows, int columns, Q4MmqProbePlan* plan);
bool q4_mmq_x_bank_swizzle_on_plan(int rows, int columns, Q4MmqProbePlan* plan);
void q4_mmq_x_bank_swizzle_off_run(const Q4MmqProbePlan* plan, const char* weight,
                            int stride_row_blocks, int rows, int k, const int* activation,
                            int columns, float* out, __nv_bfloat16* out_bf16,
                            std::int64_t output_column_stride, float* fixup, cudaStream_t stream);
void q4_mmq_x_bank_swizzle_on_run(const Q4MmqProbePlan* plan, const char* weight,
                           int stride_row_blocks, int rows, int k, const int* activation,
                           int columns, float* out, __nv_bfloat16* out_bf16,
                           std::int64_t output_column_stride, float* fixup, cudaStream_t stream);
