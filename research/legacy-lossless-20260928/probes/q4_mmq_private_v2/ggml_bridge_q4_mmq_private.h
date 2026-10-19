#pragma once

#include "core/weight_view.h"
#include <cuda_runtime.h>
#include <cstdint>

namespace ninfer::ops::gguf {

void matrix_q4_fields_private(const Weight& weight, const void* activation, int columns,
                              float* out, std::int64_t out_column_stride, void* fixup,
                              cudaStream_t stream);

void matrix_q4_fields_fast_private(const Weight& weight, const void* activation, int columns,
                                   float* out, std::int64_t out_column_stride, void* fixup,
                                   cudaStream_t stream);

}
