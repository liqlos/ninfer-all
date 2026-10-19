#include "ops/linear/gguf/ggml_bridge_ffn_pair.h"

bool run_candidate(const void* gate, const void* up, std::int64_t row_bytes,
                   const void* activation, int columns, __nv_bfloat16* out,
                   cudaStream_t stream) {
    return ninfer::ops::gguf::ffn_q4_k_mmq_pair(gate, up, row_bytes, activation, columns, out, stream);
}

#include "../lab/mmq_swiglu_probe.cuh"
