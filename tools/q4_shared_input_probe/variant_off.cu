#include "probe_contract.cuh"
#if NINFER_Q4_SHARED_INPUT_PROBE != 0
#error OFF_reference_requires_default_switch
#endif
#include "q4_shared_input_generated.cuh"
#include "instantiate.cuh"
