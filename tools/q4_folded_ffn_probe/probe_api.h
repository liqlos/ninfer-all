#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

bool q4_folded_ffn_run(const void* gate, const void* up, std::int64_t row_bytes,
                       const void* activation, int columns, __nv_bfloat16* output,
                       float* diagnostic_gate, float* diagnostic_up, cudaStream_t stream);
