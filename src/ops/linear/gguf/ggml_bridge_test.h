#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::gguf::detail {

void q4_vector_baseline_for_test(const std::uint8_t* weight, std::int64_t row_bytes,
                                 int rows, int k, const std::uint8_t* activation, int columns,
                                 void* output, std::int64_t output_column_stride, bool fused,
                                 cudaStream_t stream);

void q6_vector_baseline_for_test(const std::uint8_t* weight, std::int64_t row_bytes,
                                 int rows, int k, const std::uint8_t* activation, int columns,
                                 void* output, std::int64_t output_column_stride, bool fused,
                                 cudaStream_t stream);

void q4_t4_imma_baseline_for_test(const std::uint8_t* weight, std::int64_t row_bytes,
                                  int rows, int k, const std::uint8_t* activation,
                                  void* output, std::int64_t output_column_stride,
                                  cudaStream_t stream);

}
