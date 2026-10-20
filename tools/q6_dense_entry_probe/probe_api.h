#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

struct Q6DenseEntryPlan {
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

using Q6DenseEntryPlanFunction = bool (*)(int, int, Q6DenseEntryPlan*);
using Q6DenseEntryRunFunction = bool (*)(const Q6DenseEntryPlan*, const char*, int, int, int,
    const int*, int, int, float*, __nv_bfloat16*, bool, std::int64_t, float*, cudaStream_t);

bool q6_dense_entry_off_plan(int rows, int columns, Q6DenseEntryPlan* plan);
bool q6_dense_entry_on_plan(int rows, int columns, Q6DenseEntryPlan* plan);
bool q6_dense_entry_off_uses_dense(const Q6DenseEntryPlan* plan, int rows, int k, int columns);
bool q6_dense_entry_on_uses_dense(const Q6DenseEntryPlan* plan, int rows, int k, int columns);
bool q6_dense_entry_off_run(const Q6DenseEntryPlan* plan, const char* weight, int stride_row_blocks,
    int rows, int k, const int* activation, int activation_layout, int columns, float* out,
    __nv_bfloat16* out_bf16, bool accumulate, std::int64_t output_column_stride,
    float* fixup, cudaStream_t stream);
bool q6_dense_entry_on_run(const Q6DenseEntryPlan* plan, const char* weight, int stride_row_blocks,
    int rows, int k, const int* activation, int activation_layout, int columns, float* out,
    __nv_bfloat16* out_bf16, bool accumulate, std::int64_t output_column_stride,
    float* fixup, cudaStream_t stream);
