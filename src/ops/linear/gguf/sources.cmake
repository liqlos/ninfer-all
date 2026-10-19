# ggml's block-quant kernels (third_party/ggml-quants) behind a plain bridge. They build as their own
# archive: the ggml headers define CUDA_CHECK and friends, so no NInfer translation unit may see them.
set(NINFER_GGML_QUANTS_ROOT "${PROJECT_SOURCE_DIR}/third_party/ggml-quants")
option(NINFER_GGUF_Q4_STREAMING_LOADS "Use streaming cache loads for GGUF Q4_K vector weights" OFF)
option(NINFER_GGUF_Q6_STREAMING_LOADS "Use streaming cache loads for GGUF Q6_K vector weights" OFF)
option(NINFER_GGUF_Q6_STATIC_K "Specialize SM86 GGUF Q6_K K5120 decode and T4 verification" OFF)
option(NINFER_GGUF_Q6_COOPERATIVE_DECODE "Cooperatively load packed Q6_K blocks across SM86 warp lanes" OFF)
option(NINFER_GGUF_Q6_PACKED_SCALE_LOAD "Load each Q6_K scale pair with one aligned SM86 word transaction" OFF)
option(NINFER_GGUF_Q6_T4_SINGLE_ROW "Assign one non-fused SM86 Q6_K T4 output row to each warp" OFF)
option(NINFER_GGUF_Q6_T4_ROW_TOKEN_STREAM "Stream SM86 Q6_K T4 activations token by token within each output row" OFF)
option(NINFER_GGUF_Q6_T4_IMMA "Use exact-chain SM86 Q6_K T4 INT8 MMA for non-fused K5120 and K17408 vector products" OFF)
option(NINFER_GGUF_Q4_COOPERATIVE_DECODE "Cooperatively load packed Q4_K blocks across SM86 warp lanes" OFF)
option(NINFER_GGUF_Q4_T4_FUSED_WARP_PAIR "Assign adjacent SM86 Q4_K T4 fused gate/up rows to separate warps" OFF)
option(NINFER_GGUF_Q4_T4_IMMA "Use exact-chain SM86 Q4_K T4 INT8 MMA for non-fused K5120 vector products" OFF)
option(NINFER_GGUF_Q4_T4_FUSED_IMMA "Use exact-chain SM86 Q4_K T4 INT8 MMA for fused K5120 SwiGLU" OFF)
option(NINFER_GGUF_Q4_T4_SINGLE_ROW "Assign one non-fused SM86 Q4_K T4 output row to each warp" OFF)
option(NINFER_GGUF_Q4_T4_ROW_TOKEN_STREAM "Stream SM86 Q4_K T4 activations token by token within each output row" OFF)
option(NINFER_GGUF_Q4_VECTOR_HEADER_LOAD "Load the aligned Q4_K vector block header in one SM86 vector transaction" OFF)
option(NINFER_GGUF_Q4_SPLIT_DP4A "Split SM86 Q4_K vector dots into independent DP4A chains" OFF)
option(NINFER_GGUF_Q4_ASYNC_PIPELINE "Pipeline SM86 GGUF Q4_K K5120 vector weight loads" OFF)
option(NINFER_GGUF_VECTOR_PREFETCH2 "Prefetch one SM86 Q4_K/Q6_K vector weight slice in registers" OFF)
option(NINFER_GGUF_Q4_MMQ_ASYNC_Y "Use SM86 cp.async activation staging for GGUF Q4_K MMQ" OFF)
option(NINFER_GGUF_Q4_MMQ_ASYNC_X "Use SM86 cp.async raw-weight staging for GGUF Q4_K MMQ" OFF)
option(NINFER_GGUF_Q4_MMQ_K_STREAM "Stream SM86 Q4_K MMQ K slices to shorten live A fragments" OFF)
option(NINFER_GGUF_Q6_MMQ_ASYNC_Y "Use SM86 cp.async activation staging for GGUF Q6_K MMQ" OFF)
option(NINFER_GGUF_SM86_MMQ_TILE_CAP "Limit the column tile of wide SM86 Q4_K/Q6_K MMQ products" OFF)
set(NINFER_GGUF_SM86_MMQ_MAX_J "64" CACHE STRING "Maximum column tile for the SM86 MMQ experiment")
set_property(CACHE NINFER_GGUF_SM86_MMQ_MAX_J PROPERTY STRINGS 32 64 128)
if(NOT NINFER_GGUF_SM86_MMQ_MAX_J MATCHES "^(32|64|128)$")
  message(FATAL_ERROR "SM86 MMQ maximum column tile must be 32, 64 or 128")
endif()
if(NINFER_GGUF_SM86_MMQ_TILE_CAP)
  if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "86")
    message(FATAL_ERROR "SM86 MMQ column tile experiment requires sm_86")
  endif()
  set_property(SOURCE "${CMAKE_CURRENT_LIST_DIR}/ggml_mmq_q4_k.cu"
    "${CMAKE_CURRENT_LIST_DIR}/ggml_mmq_q6_k.cu"
    "${CMAKE_CURRENT_LIST_DIR}/ggml_mmq_dual_q4_k.cu"
    APPEND PROPERTY COMPILE_DEFINITIONS NINFER_GGUF_SM86_MMQ_TILE_CAP=1
      NINFER_GGUF_SM86_MMQ_MAX_J=${NINFER_GGUF_SM86_MMQ_MAX_J})
endif()
option(NINFER_GGUF_DUAL_LAYOUT_MMQ_PACK "Produce exact D4 and DS4 GGUF MMQ activations in one pass" OFF)
if(NINFER_GGUF_DUAL_LAYOUT_MMQ_PACK)
  if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "86")
    message(FATAL_ERROR "GGUF dual-layout MMQ activation packing requires sm_86")
  endif()
  set_property(SOURCE "${CMAKE_CURRENT_LIST_DIR}/ggml_bridge.cu"
    "${CMAKE_CURRENT_LIST_DIR}/gguf_linear.cpp"
    APPEND PROPERTY COMPILE_DEFINITIONS NINFER_GGUF_DUAL_LAYOUT_MMQ_PACK=1)
endif()
option(NINFER_GGUF_SWIGLU_Q8_FUSION "Fuse GGUF SwiGLU output with exact BF16-rounded q8 activation packing" OFF)
option(NINFER_GGUF_PERSISTENT_FFN "Use a residency-bounded cooperative SM86 GGUF Q4/Q6 FFN kernel" OFF)
option(NINFER_GGUF_SWIGLU_Q8_SEQUENTIAL_DECODE "Decode one SwiGLU operand at a time before q8 packing" OFF)
set(NINFER_GGUF_SWIGLU_Q8_WARPS "4" CACHE STRING "Warps per fused SwiGLU q8 producer block")
set_property(CACHE NINFER_GGUF_SWIGLU_Q8_WARPS PROPERTY STRINGS 4 8 16)
if(NOT NINFER_GGUF_SWIGLU_Q8_WARPS MATCHES "^(4|8|16)$")
  message(FATAL_ERROR "GGUF SwiGLU q8 warps must be 4, 8 or 16")
endif()
if(NINFER_GGUF_SWIGLU_Q8_SEQUENTIAL_DECODE)
  if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "86")
    message(FATAL_ERROR "GGUF SwiGLU sequential decode requires sm_86")
  endif()
  set_property(SOURCE "${CMAKE_CURRENT_LIST_DIR}/ggml_vec_swiglu_q8.cu" APPEND
    PROPERTY COMPILE_DEFINITIONS NINFER_GGUF_SWIGLU_Q8_SEQUENTIAL_DECODE=1)
endif()
option(NINFER_GGUF_RMSNORM_Q8_FUSION "Fuse Qwen3.5 hidden RMSNorm with exact GGUF q8_1 packing" OFF)
if(NINFER_GGUF_RMSNORM_Q8_FUSION)
  if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "86")
    message(FATAL_ERROR "GGUF RMSNorm q8 fusion requires sm_86")
  endif()
endif()
option(NINFER_GGUF_RMSNORM_MMQ_FUSION "Fuse Qwen3.5 hidden RMSNorm with exact GGUF MMQ q8_1 packing" OFF)
if(NINFER_GGUF_RMSNORM_MMQ_FUSION)
  if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "86")
    message(FATAL_ERROR "GGUF RMSNorm MMQ fusion requires sm_86")
  endif()
endif()
option(NINFER_GGUF_DUAL_MMQ_FUSION "Reuse SM86 Q4_K MMQ activation tiles across paired projections" OFF)
if(NINFER_GGUF_DUAL_MMQ_FUSION)
  if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "86")
    message(FATAL_ERROR "GGUF dual MMQ fusion requires sm_86")
  endif()
  set_property(SOURCE "${CMAKE_CURRENT_LIST_DIR}/gguf_linear.cpp" APPEND
    PROPERTY COMPILE_DEFINITIONS NINFER_GGUF_DUAL_MMQ_FUSION=1)
endif()
option(NINFER_GGUF_MMQ_STORE_FUSION "Fuse native SM86 Q4_K/Q6_K MMQ writeback with BF16 output store" OFF)
if(NINFER_GGUF_MMQ_STORE_FUSION)
  if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "86")
    message(FATAL_ERROR "GGUF MMQ BF16 store fusion requires sm_86")
  endif()
  set_property(SOURCE "${CMAKE_CURRENT_LIST_DIR}/gguf_linear.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/ggml_bridge.cu"
    "${CMAKE_CURRENT_LIST_DIR}/ggml_mmq_q4_k.cu"
    "${CMAKE_CURRENT_LIST_DIR}/ggml_mmq_q6_k.cu"
    APPEND PROPERTY COMPILE_DEFINITIONS NINFER_GGUF_MMQ_STORE_FUSION=1)
endif()
option(NINFER_GGUF_MMQ_GATE_EPILOGUE_FUSION "Fuse SM86 Q4_K MMQ GateProduct into BF16 writeback" OFF)
if(NINFER_GGUF_MMQ_GATE_EPILOGUE_FUSION)
  if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "86")
    message(FATAL_ERROR "GGUF MMQ gate epilogue fusion requires sm_86")
  endif()
  set_property(SOURCE "${CMAKE_CURRENT_LIST_DIR}/gguf_linear.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/ggml_bridge.cu"
    "${CMAKE_CURRENT_LIST_DIR}/ggml_mmq_q4_k.cu"
    APPEND PROPERTY COMPILE_DEFINITIONS NINFER_GGUF_MMQ_GATE_EPILOGUE_FUSION=1)
endif()
option(NINFER_GGUF_FFN_RMSNORM_Q8_FUSION "Fuse Qwen3.5 FFN input RMSNorm with exact GGUF q8_1 packing" OFF)
if(NINFER_GGUF_FFN_RMSNORM_Q8_FUSION)
  if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "86")
    message(FATAL_ERROR "GGUF FFN RMSNorm q8 fusion requires sm_86")
  endif()
endif()
if(NINFER_GGUF_SWIGLU_Q8_FUSION)
  if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "86")
    message(FATAL_ERROR "GGUF SwiGLU q8 fusion requires sm_86")
  endif()
  set_property(SOURCE "${CMAKE_CURRENT_LIST_DIR}/gguf_linear.cpp" APPEND
    PROPERTY COMPILE_DEFINITIONS NINFER_GGUF_SWIGLU_Q8_FUSION=1
      NINFER_GGUF_SWIGLU_Q8_WARPS=${NINFER_GGUF_SWIGLU_Q8_WARPS})
endif()
if(NINFER_GGUF_PERSISTENT_FFN)
  if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "86")
    message(FATAL_ERROR "Persistent GGUF FFN requires sm_86")
  endif()
  set_property(SOURCE "${CMAKE_CURRENT_LIST_DIR}/gguf_linear.cpp" APPEND
    PROPERTY COMPILE_DEFINITIONS NINFER_GGUF_PERSISTENT_FFN=1)
  set_property(SOURCE "${CMAKE_CURRENT_LIST_DIR}/ggml_vec_persistent_ffn.cu" APPEND
    PROPERTY COMPILE_DEFINITIONS NINFER_GGUF_PERSISTENT_FFN=1)
endif()
if(NINFER_GGUF_PERSISTENT_FFN AND NINFER_GGUF_Q4_COOPERATIVE_DECODE)
  set_property(SOURCE "${CMAKE_CURRENT_LIST_DIR}/ggml_vec_persistent_ffn.cu" APPEND
    PROPERTY COMPILE_DEFINITIONS NINFER_GGUF_Q4_COOPERATIVE_DECODE=1)
endif()
if(NINFER_GGUF_PERSISTENT_FFN AND NINFER_GGUF_Q6_COOPERATIVE_DECODE)
  set_property(SOURCE "${CMAKE_CURRENT_LIST_DIR}/ggml_vec_persistent_ffn.cu" APPEND
    PROPERTY COMPILE_DEFINITIONS NINFER_GGUF_Q6_COOPERATIVE_DECODE=1)
endif()
if(NINFER_GGUF_PERSISTENT_FFN AND NINFER_GGUF_Q4_ASYNC_PIPELINE)
  message(STATUS "Persistent FFN producer supersedes the generic Q4 async vector pipeline")
endif()
if(NINFER_GGUF_Q6_COOPERATIVE_DECODE)
  if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "86")
    message(FATAL_ERROR "GGUF cooperative Q6 decode requires sm_86")
  endif()
  if(NINFER_GGUF_Q6_STREAMING_LOADS)
    message(FATAL_ERROR "Cooperative Q6 decode replaces the streaming-load experiment")
  endif()
  set_property(SOURCE "${CMAKE_CURRENT_LIST_DIR}/ggml_vec_q6_k.cu" APPEND
    PROPERTY COMPILE_DEFINITIONS NINFER_GGUF_Q6_COOPERATIVE_DECODE=1)
  set_property(SOURCE "${CMAKE_CURRENT_LIST_DIR}/ggml_vec_swiglu_q8.cu" APPEND
    PROPERTY COMPILE_DEFINITIONS NINFER_GGUF_Q6_COOPERATIVE_DECODE=1)
endif()
if(NINFER_GGUF_Q6_PACKED_SCALE_LOAD)
  if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "86")
    message(FATAL_ERROR "Packed GGUF Q6_K scale loads require sm_86")
  endif()
  set_property(SOURCE "${CMAKE_CURRENT_LIST_DIR}/ggml_vec_q6_k.cu"
    "${CMAKE_CURRENT_LIST_DIR}/ggml_vec_swiglu_q8.cu"
    "${CMAKE_CURRENT_LIST_DIR}/ggml_vec_persistent_ffn.cu"
    "${CMAKE_CURRENT_LIST_DIR}/ggml_gdn_persistent.cu"
    APPEND PROPERTY COMPILE_DEFINITIONS NINFER_GGUF_Q6_PACKED_SCALE_LOAD=1)
endif()
if(NINFER_GGUF_Q6_T4_SINGLE_ROW)
  if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "86")
    message(FATAL_ERROR "Q6_K T4 single-row vector scheduling requires sm_86")
  endif()
  set_property(SOURCE "${CMAKE_CURRENT_LIST_DIR}/ggml_vec_q6_k.cu" APPEND
    PROPERTY COMPILE_DEFINITIONS NINFER_GGUF_Q6_T4_SINGLE_ROW=1)
endif()
if(NINFER_GGUF_Q6_T4_ROW_TOKEN_STREAM)
  if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "86")
    message(FATAL_ERROR "Q6_K T4 row-token streaming requires sm_86")
  endif()
  if(NINFER_GGUF_Q6_T4_SINGLE_ROW OR NINFER_GGUF_Q6_T4_IMMA OR
     NINFER_GGUF_VECTOR_PREFETCH2)
    message(FATAL_ERROR "Q6_K T4 row-token streaming cannot be combined with single-row, T4 IMMA, or vector prefetch")
  endif()
  set_property(SOURCE "${CMAKE_CURRENT_LIST_DIR}/ggml_vec_q6_k.cu" APPEND
    PROPERTY COMPILE_DEFINITIONS NINFER_GGUF_Q6_T4_ROW_TOKEN_STREAM=1)
endif()
if(NINFER_GGUF_Q4_COOPERATIVE_DECODE)
  if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "86")
    message(FATAL_ERROR "GGUF cooperative Q4 decode requires sm_86")
  endif()
  if(NINFER_GGUF_Q4_STREAMING_LOADS)
    message(FATAL_ERROR "Cooperative Q4 decode replaces the streaming-load experiment")
  endif()
  set_property(SOURCE "${CMAKE_CURRENT_LIST_DIR}/ggml_vec_q4_k.cu" APPEND
    PROPERTY COMPILE_DEFINITIONS NINFER_GGUF_Q4_COOPERATIVE_DECODE=1)
  set_property(SOURCE "${CMAKE_CURRENT_LIST_DIR}/ggml_vec_swiglu_q8.cu" APPEND
    PROPERTY COMPILE_DEFINITIONS NINFER_GGUF_Q4_COOPERATIVE_DECODE=1)
endif()
if(NINFER_GGUF_Q4_T4_SINGLE_ROW)
  if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "86")
    message(FATAL_ERROR "Q4_K T4 single-row vector scheduling requires sm_86")
  endif()
  set_property(SOURCE "${CMAKE_CURRENT_LIST_DIR}/ggml_vec_q4_k.cu" APPEND
    PROPERTY COMPILE_DEFINITIONS NINFER_GGUF_Q4_T4_SINGLE_ROW=1)
endif()
if(NINFER_GGUF_Q4_T4_ROW_TOKEN_STREAM)
  if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "86")
    message(FATAL_ERROR "Q4_K T4 row-token streaming requires sm_86")
  endif()
  if(NINFER_GGUF_Q4_T4_SINGLE_ROW OR NINFER_GGUF_Q4_T4_IMMA OR
     NINFER_GGUF_Q4_ASYNC_PIPELINE OR NINFER_GGUF_VECTOR_PREFETCH2)
    message(FATAL_ERROR "Q4_K T4 row-token streaming cannot be combined with single-row, T4 IMMA, async pipeline, or vector prefetch")
  endif()
  set_property(SOURCE "${CMAKE_CURRENT_LIST_DIR}/ggml_vec_q4_k.cu" APPEND
    PROPERTY COMPILE_DEFINITIONS NINFER_GGUF_Q4_T4_ROW_TOKEN_STREAM=1)
endif()
if(NINFER_GGUF_Q4_T4_FUSED_WARP_PAIR)
  if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "86")
    message(FATAL_ERROR "Q4_K T4 fused warp-pair scheduling requires sm_86")
  endif()
  set_property(SOURCE "${CMAKE_CURRENT_LIST_DIR}/ggml_vec_q4_k.cu" APPEND
    PROPERTY COMPILE_DEFINITIONS NINFER_GGUF_Q4_T4_FUSED_WARP_PAIR=1)
endif()
if(NINFER_GGUF_Q4_T4_IMMA)
  if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "86")
    message(FATAL_ERROR "Q4_K T4 IMMA vector products require sm_86")
  endif()
  set_property(SOURCE "${CMAKE_CURRENT_LIST_DIR}/ggml_vec_q4_k.cu" APPEND
    PROPERTY COMPILE_DEFINITIONS NINFER_GGUF_Q4_T4_IMMA=1
      NINFER_GGUF_Q4_T4_IMMA_TEST=1)
  set_property(SOURCE "${CMAKE_CURRENT_LIST_DIR}/ggml_vec_q4_k.cu" APPEND
    PROPERTY INCLUDE_DIRECTORIES "${PROJECT_SOURCE_DIR}/src")
endif()
if(NINFER_GGUF_Q6_T4_IMMA)
  if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "86")
    message(FATAL_ERROR "Q6_K T4 IMMA vector products require sm_86")
  endif()
  set_property(SOURCE "${CMAKE_CURRENT_LIST_DIR}/ggml_vec_q6_k.cu" APPEND
    PROPERTY COMPILE_DEFINITIONS NINFER_GGUF_Q6_T4_IMMA=1 NINFER_GGUF_Q6_T4_IMMA_TEST=1)
  set_property(SOURCE "${CMAKE_CURRENT_LIST_DIR}/ggml_vec_q6_k.cu" APPEND
    PROPERTY INCLUDE_DIRECTORIES "${PROJECT_SOURCE_DIR}/src")
endif()
if(NINFER_GGUF_Q4_T4_FUSED_IMMA)
  if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "86")
    message(FATAL_ERROR "Q4_K T4 fused IMMA vector SwiGLU requires sm_86")
  endif()
  set_property(SOURCE "${CMAKE_CURRENT_LIST_DIR}/ggml_vec_q4_k.cu" APPEND
    PROPERTY COMPILE_DEFINITIONS NINFER_GGUF_Q4_T4_FUSED_IMMA=1
      NINFER_GGUF_Q4_T4_FUSED_IMMA_TEST=1)
  set_property(SOURCE "${CMAKE_CURRENT_LIST_DIR}/ggml_vec_q4_k.cu" APPEND
    PROPERTY INCLUDE_DIRECTORIES "${PROJECT_SOURCE_DIR}/src")
endif()
if(NINFER_GGUF_Q6_STATIC_K)
  if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "86")
    message(FATAL_ERROR "GGUF static Q6 experiment requires sm_86")
  endif()
  set_property(SOURCE "${CMAKE_CURRENT_LIST_DIR}/ggml_vec_q6_k.cu" APPEND
    PROPERTY COMPILE_DEFINITIONS NINFER_GGUF_Q6_STATIC_K=1)
endif()
if(NINFER_GGUF_Q6_STREAMING_LOADS)
  if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "86")
    message(FATAL_ERROR "GGUF streaming weight experiments require sm_86")
  endif()
  set_property(SOURCE "${CMAKE_CURRENT_LIST_DIR}/ggml_vec_q6_k.cu" APPEND
    PROPERTY COMPILE_DEFINITIONS NINFER_GGUF_Q6_STREAMING_LOADS=1)
endif()
if(NINFER_GGUF_Q4_VECTOR_HEADER_LOAD)
  if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "86")
    message(FATAL_ERROR "GGUF Q4_K vector header loads require sm_86")
  endif()
  set_property(SOURCE "${CMAKE_CURRENT_LIST_DIR}/ggml_vec_q4_k.cu"
    "${CMAKE_CURRENT_LIST_DIR}/ggml_vec_swiglu_q8.cu"
    "${CMAKE_CURRENT_LIST_DIR}/ggml_vec_persistent_ffn.cu"
    "${CMAKE_CURRENT_LIST_DIR}/ggml_gdn_persistent.cu"
    APPEND PROPERTY COMPILE_DEFINITIONS NINFER_GGUF_Q4_VECTOR_HEADER_LOAD=1)
endif()
if(NINFER_GGUF_Q4_STREAMING_LOADS)
  if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "86")
    message(FATAL_ERROR "GGUF streaming weight experiments require sm_86")
  endif()
  set_property(SOURCE "${CMAKE_CURRENT_LIST_DIR}/ggml_vec_q4_k.cu" APPEND
    PROPERTY COMPILE_DEFINITIONS NINFER_GGUF_Q4_STREAMING_LOADS=1)
endif()
if(NINFER_GGUF_Q4_SPLIT_DP4A)
  if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "86")
    message(FATAL_ERROR "GGUF Q4_K split DP4A requires sm_86")
  endif()
  set_property(SOURCE "${CMAKE_CURRENT_LIST_DIR}/ggml_vec_q4_k.cu"
    "${CMAKE_CURRENT_LIST_DIR}/ggml_vec_swiglu_q8.cu"
    "${CMAKE_CURRENT_LIST_DIR}/ggml_vec_persistent_ffn.cu"
    "${CMAKE_CURRENT_LIST_DIR}/ggml_gdn_persistent.cu"
    APPEND PROPERTY COMPILE_DEFINITIONS NINFER_GGUF_Q4_SPLIT_DP4A=1)
endif()
if(NINFER_GGUF_Q4_ASYNC_PIPELINE)
  if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "86")
    message(FATAL_ERROR "GGUF Q4 async pipeline requires sm_86")
  endif()
  if(NINFER_GGUF_Q4_STREAMING_LOADS)
    message(FATAL_ERROR "GGUF Q4 async pipeline cannot use streaming global loads")
  endif()
  set_property(SOURCE "${CMAKE_CURRENT_LIST_DIR}/ggml_vec_q4_k.cu" APPEND
    PROPERTY COMPILE_DEFINITIONS NINFER_GGUF_Q4_ASYNC_PIPELINE=1)
endif()
if(NINFER_GGUF_VECTOR_PREFETCH2)
  if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "86")
    message(FATAL_ERROR "GGUF vector weight prefetch requires sm_86")
  endif()
  if(NINFER_GGUF_Q4_ASYNC_PIPELINE)
    message(FATAL_ERROR "GGUF vector weight prefetch cannot be combined with the Q4 async pipeline")
  endif()
  if(NINFER_GGUF_Q4_T4_IMMA)
    message(FATAL_ERROR "GGUF vector weight prefetch cannot be combined with the Q4 T4 IMMA route")
  endif()
  if(NINFER_GGUF_Q4_T4_FUSED_WARP_PAIR)
    message(FATAL_ERROR "GGUF vector weight prefetch cannot be combined with the Q4 T4 fused warp-pair route")
  endif()
  set_property(SOURCE "${CMAKE_CURRENT_LIST_DIR}/ggml_vec_q4_k.cu"
    "${CMAKE_CURRENT_LIST_DIR}/ggml_vec_q6_k.cu" APPEND
    PROPERTY COMPILE_DEFINITIONS NINFER_GGUF_VECTOR_PREFETCH2=1
      NINFER_GGUF_VECTOR_PREFETCH2_TEST=1)
  set_property(SOURCE "${CMAKE_CURRENT_LIST_DIR}/ggml_vec_q4_k.cu"
    "${CMAKE_CURRENT_LIST_DIR}/ggml_vec_q6_k.cu" APPEND
    PROPERTY INCLUDE_DIRECTORIES "${PROJECT_SOURCE_DIR}/src")
endif()
option(NINFER_GGUF_Q4_ASYNC_EXTENDED_K "Extend the SM86 Q4 async pipeline to long K rows" OFF)
if(NINFER_GGUF_Q4_ASYNC_EXTENDED_K)
  if(NOT NINFER_GGUF_Q4_ASYNC_PIPELINE)
    message(FATAL_ERROR "GGUF Q4 extended K requires the Q4 async pipeline")
  endif()
  if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "86")
    message(FATAL_ERROR "GGUF Q4 extended K requires sm_86")
  endif()
  set_property(SOURCE "${CMAKE_CURRENT_LIST_DIR}/ggml_vec_q4_k.cu" APPEND
    PROPERTY COMPILE_DEFINITIONS NINFER_GGUF_Q4_ASYNC_EXTENDED_K=1)
endif()
if(NINFER_GGUF_Q4_MMQ_ASYNC_Y)
  if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "86")
    message(FATAL_ERROR "GGUF Q4 MMQ async activation staging requires sm_86")
  endif()
  set_property(SOURCE "${CMAKE_CURRENT_LIST_DIR}/ggml_mmq_q4_k.cu" APPEND
    PROPERTY COMPILE_DEFINITIONS NINFER_GGUF_Q4_MMQ_ASYNC_Y=1)
endif()
if(NINFER_GGUF_Q6_MMQ_ASYNC_Y)
  if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "86")
    message(FATAL_ERROR "GGUF Q6 MMQ async activation staging requires sm_86")
  endif()
  set_property(SOURCE "${CMAKE_CURRENT_LIST_DIR}/ggml_mmq_q6_k.cu" APPEND
    PROPERTY COMPILE_DEFINITIONS NINFER_GGUF_Q6_MMQ_ASYNC_Y=1)
endif()
if(NINFER_GGUF_Q4_MMQ_ASYNC_X)
  if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "86")
    message(FATAL_ERROR "GGUF Q4 MMQ async weight staging requires sm_86")
  endif()
  set_property(SOURCE "${CMAKE_CURRENT_LIST_DIR}/ggml_mmq_q4_k.cu" APPEND
    PROPERTY COMPILE_DEFINITIONS NINFER_GGUF_Q4_MMQ_ASYNC_X=1)
endif()
if(NINFER_GGUF_Q4_MMQ_K_STREAM)
  if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "86")
    message(FATAL_ERROR "GGUF Q4 MMQ K-slice streaming requires sm_86")
  endif()
  set_property(SOURCE "${CMAKE_CURRENT_LIST_DIR}/ggml_mmq_q4_k.cu"
    "${CMAKE_CURRENT_LIST_DIR}/ggml_mmq_dual_q4_k.cu" APPEND
    PROPERTY COMPILE_DEFINITIONS NINFER_GGUF_Q4_MMQ_K_STREAM=1)
endif()
add_library(ninfer_ggml_quants STATIC
  "${CMAKE_CURRENT_LIST_DIR}/ggml_vec_swiglu_q8.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_rmsnorm_q8.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_vec_persistent_ffn.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_bridge.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_mmq_dual_q4_k.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_mmq_iq1_s.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_mmq_iq2_s.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_mmq_iq2_xs.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_mmq_iq2_xxs.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_mmq_iq3_s.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_mmq_iq3_xxs.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_mmq_iq4_nl.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_mmq_iq4_xs.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_mmq_q2_k.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_mmq_q3_k.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_mmq_q4_k.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_mmq_q5_k.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_mmq_q6_k.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_mmq_q8_0.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_vec_q8_0.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_vec_q2_k.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_vec_q3_k.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_vec_q4_k.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_vec_q5_k.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_vec_q6_k.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_vec_iq2_xxs.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_vec_iq2_xs.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_vec_iq2_s.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_vec_iq3_xxs.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_vec_iq3_s.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_vec_iq1_s.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_vec_iq1_m.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_vec_iq4_nl.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_vec_iq4_xs.cu")
target_include_directories(ninfer_ggml_quants PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}"
  "${NINFER_GGML_QUANTS_ROOT}/include"
  "${NINFER_GGML_QUANTS_ROOT}/src")
ninfer_cuda_non_rdc_archive(ninfer_ggml_quants)
target_compile_options(ninfer_ggml_quants PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe=--diag_suppress=177>)
target_link_libraries(ninfer_ggml_quants PRIVATE CUDA::cudart CUDA::cublas)
target_link_libraries(ninfer_ops PRIVATE ninfer_ggml_quants)
target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/gguf_linear.cpp")
