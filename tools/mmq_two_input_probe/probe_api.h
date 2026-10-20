#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

enum class MmqTwoInputProfile : std::int32_t {
    Q4GateOrUp = 0,
    Q6Down = 1,
};

enum class MmqTwoInputEpilogue : std::int32_t {
    PlainFp32 = 0,
    StoreBf16 = 1,
    AccumulateBf16 = 2,
};

struct MmqTwoInputArguments {
    MmqTwoInputProfile profile;
    MmqTwoInputEpilogue epilogue;
    std::int32_t rows;
    std::int32_t k;
    std::int32_t columns;
    std::int32_t activation_layout;
    const void* weight;
    std::size_t weight_capacity_bytes;
    std::int64_t weight_row_stride_bytes;
    const void* packed_inputs;
    std::size_t packed_capacity_bytes;
    std::int64_t packed_sample_stride_bytes;
    void* output;
    std::size_t output_capacity_bytes;
    std::int64_t output_column_stride_elements;
    std::int64_t output_sample_stride_elements;
};

bool mmq_two_input_control(const MmqTwoInputArguments& arguments, cudaStream_t stream);
bool mmq_two_input_candidate(const MmqTwoInputArguments& arguments, cudaStream_t stream);
