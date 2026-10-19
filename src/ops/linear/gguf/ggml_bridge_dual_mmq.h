#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::gguf {

[[nodiscard]] std::size_t dual_q4_k_matrix_fixup_bytes(int rows, int columns);
[[nodiscard]] bool dual_q4_k_matrix_supported(int rows, int columns);

[[nodiscard]] bool dual_q4_k_matrix_product(const void* weight_a, std::int64_t row_bytes_a,
                              const void* weight_b, std::int64_t row_bytes_b,
                              int rows, int k, const void* activation, int columns,
                              float* out_a, std::int64_t out_column_stride_a,
                              float* out_b, std::int64_t out_column_stride_b,
                              void* fixup_a, void* fixup_b, cudaStream_t stream);

}
