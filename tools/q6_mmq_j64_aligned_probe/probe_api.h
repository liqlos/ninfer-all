#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

struct Q6MmqProbePlan {
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

bool q6_mmq_j64_aligned_off_plan(int rows, int columns, Q6MmqProbePlan* plan);
bool q6_mmq_j64_aligned_on_plan(int rows, int columns, Q6MmqProbePlan* plan);
void q6_mmq_j64_aligned_off_run(const Q6MmqProbePlan* plan, const char* weight,
                            int stride_row_blocks, int rows, int k, const int* activation,
                            int columns, float* out, __nv_bfloat16* out_bf16,
                            std::int64_t output_column_stride, float* fixup, cudaStream_t stream);
void q6_mmq_j64_aligned_on_run(const Q6MmqProbePlan* plan, const char* weight,
                           int stride_row_blocks, int rows, int k, const int* activation,
                           int columns, float* out, __nv_bfloat16* out_bf16,
                           std::int64_t output_column_stride, float* fixup, cudaStream_t stream);
