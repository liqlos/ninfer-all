#pragma once

#ifndef NINFER_Q4_SHARED_INPUT_PROBE
#define NINFER_Q4_SHARED_INPUT_PROBE 0
#endif
#if NINFER_Q4_SHARED_INPUT_PROBE != 0 && NINFER_Q4_SHARED_INPUT_PROBE != 1
#error Invalid_shared_input_probe_switch
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 860
#error Closed_SM86_compile_probe
#endif
#include "q4_shared_input_flags_generated.cuh"

namespace ninfer::ops::gguf::detail::q4_shared_input_probe {
inline constexpr int columns = 4;
inline constexpr int k = 5120;
inline constexpr int rows = 17408;
inline constexpr int weight_row_bytes = 2880;
inline constexpr int grid_blocks = 328;
inline constexpr int block_threads = 128;
inline constexpr int dynamic_shared_bytes = NINFER_Q4_SHARED_INPUT_PROBE ? 23040 : 0;
}
