#pragma once

#include "ops/linear/gguf/ggml_bridge.h"

namespace ninfer::ops::gguf {

void vector_q4_fields_paired(const Weight& first_weight, const Weight& second_weight,
                             const void* activation, int columns, const VectorOutput& first_out,
                             const VectorOutput& second_out, cudaStream_t stream);

}
