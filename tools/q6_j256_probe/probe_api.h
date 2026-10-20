#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

struct Q6J256Plan {
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

bool q6_j256_off_plan(int rows, int k, int columns, Q6J256Plan* plan);
bool q6_j256_on_plan(int rows, int k, int columns, Q6J256Plan* plan);
bool q6_j256_off_run(const Q6J256Plan* plan, const char* weight, int stride_row_blocks,
    int rows, int k, const int* activation, int activation_layout, int columns,
    float* out, __nv_bfloat16* out_bf16, bool accumulate, std::int64_t output_column_stride,
    float* fixup, cudaStream_t stream);
bool q6_j256_on_run(const Q6J256Plan* plan, const char* weight, int stride_row_blocks,
    int rows, int k, const int* activation, int activation_layout, int columns,
    float* out, __nv_bfloat16* out_bf16, bool accumulate, std::int64_t output_column_stride,
    float* fixup, cudaStream_t stream);
