#include "probe_contract.cuh"
#if NINFER_Q4_SHARED_INPUT_PROBE != 0
#error Native_reference_requires_OFF
#endif
#include "ops/linear/gguf/ggml_bridge_vec.cuh"
#include "instantiate.cuh"
