#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ffn_pair_probe {

std::size_t matrix_fixup_bytes(int columns);
void matrix_product(const void* weight, const void* activation, float* output,
                    void* fixup, cudaStream_t stream);
void store_plane_gate_product(const float* up, const float* gate,
                              __nv_bfloat16* output, cudaStream_t stream);

}
