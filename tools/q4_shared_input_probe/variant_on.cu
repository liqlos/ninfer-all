#if defined(NINFER_Q4_SHARED_INPUT_PROBE)
#error Select_ON_only_in_this_translation_unit
#endif
#define NINFER_Q4_SHARED_INPUT_PROBE 1
#include "probe_contract.cuh"
#include "q4_shared_input_generated.cuh"
#include "instantiate.cuh"
