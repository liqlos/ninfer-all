#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>

struct WholeFfnDownPlan {
    int J;
    int I;
    int threads;
    int blocks;
    std::size_t shared;
    bool stream_k;
    bool fixup;
};

bool whole_ffn_down_plan(int layout, WholeFfnDownPlan* plan);
bool whole_ffn_down_run(int layout, const void* weight, const void* activation,
                        __nv_bfloat16* output, cudaStream_t stream);
bool whole_ffn_down_fp32(int layout, const void* weight, const void* activation,
                         float* output, cudaStream_t stream);
