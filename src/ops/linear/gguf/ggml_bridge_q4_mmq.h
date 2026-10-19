#pragma once

#include "ggml_bridge.h"

namespace ninfer::ops::gguf {

void matrix_q4_fields(const Weight& weight, const void* activation, int columns,
                      float* out, std::int64_t out_column_stride, void* fixup,
                      cudaStream_t stream);

}
