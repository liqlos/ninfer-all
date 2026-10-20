#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

struct Q6Raw32Plan {
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

using Q6Raw32PlanFunction = bool (*)(int, int, Q6Raw32Plan*);
using Q6Raw32RunFunction = bool (*)(const Q6Raw32Plan*, const char*, int, int, int,
    const int*, int, int, float*, __nv_bfloat16*, bool, std::int64_t, float*, cudaStream_t);

bool q6_raw32_off_plan(int rows, int columns, Q6Raw32Plan* plan);
bool q6_raw32_on_plan(int rows, int columns, Q6Raw32Plan* plan);
bool q6_raw32_off_run(const Q6Raw32Plan* plan, const char* weight, int stride_row_blocks,
    int rows, int k, const int* activation, int activation_layout, int columns, float* out,
    __nv_bfloat16* out_bf16, bool accumulate, std::int64_t output_column_stride,
    float* fixup, cudaStream_t stream);
bool q6_raw32_on_run(const Q6Raw32Plan* plan, const char* weight, int stride_row_blocks,
    int rows, int k, const int* activation, int activation_layout, int columns, float* out,
    __nv_bfloat16* out_bf16, bool accumulate, std::int64_t output_column_stride,
    float* fixup, cudaStream_t stream);
