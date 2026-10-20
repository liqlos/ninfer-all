#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

bool q4_folded_ffn_pack_run(
    const void* gate, const void* up, std::int64_t row_bytes,
    const void* activation, int columns, void* packed_output, int layout,
    float* diagnostic_gate, float* diagnostic_up, __nv_bfloat16* diagnostic_bf16,
    cudaStream_t stream);
