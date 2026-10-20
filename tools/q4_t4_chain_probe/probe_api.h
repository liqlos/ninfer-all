#pragma once

#include <cuda_runtime.h>
#include <cstdint>

#ifndef NINFER_Q4_T4_CHAIN_PROBE
#define NINFER_Q4_T4_CHAIN_PROBE 0
#endif

enum class Q4TCRoute : std::uint32_t {
    NativeW = 0,
    Chain = 1,
};

enum class Q4TCOutput : std::uint32_t {
    PlainFP32 = 0,
    StoreBF16 = 1,
    AccumulateBF16 = 2,
};

struct Q4TCArgs {
    const std::uint8_t* weight;
    std::int64_t row_bytes;
    int rows;
    int k;
    int columns;
    const std::int8_t* qs;
    const void* ds;
    void* output;
    std::int64_t output_column_stride;
    Q4TCOutput output_mode;
};

bool q4tc_candidate_enabled();
bool q4tc_candidate_shape(int rows, int k, int columns);
bool q4tc_run(const Q4TCArgs& args, Q4TCRoute route, cudaStream_t stream);
