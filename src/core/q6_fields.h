#pragma once

#include "core/weight.h"

#include <cuda_runtime.h>
#include <cstddef>

namespace ninfer {

inline constexpr std::size_t kQ6FieldsRawScratchBytes = 96ULL << 20;
void unpack_q6_fields(const Weight& weight, int first_row, int rows, void* raw,
                      cudaStream_t stream);

}
