#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cstdint>

namespace ninfer::ops::gguf {

[[nodiscard]] bool ffn_q4_k_mmq_pair(const void* gate, const void* up,
                                    std::int64_t row_bytes, const void* activation,
                                    int columns, __nv_bfloat16* out, cudaStream_t stream);

}
