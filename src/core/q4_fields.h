#pragma once

#include "core/weight.h"

#include <cuda_runtime.h>

namespace ninfer {

void unpack_q4_fields(const Weight& weight, int first_row, int rows, void* raw,
                      cudaStream_t stream);

}
