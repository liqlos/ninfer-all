# ggml's block-quant kernels (third_party/ggml-quants) behind a plain bridge. They build as their own
# archive: the ggml headers define CUDA_CHECK and friends, so no NInfer translation unit may see them.
set(NINFER_GGML_QUANTS_ROOT "${PROJECT_SOURCE_DIR}/third_party/ggml-quants")
option(NINFER_GGUF_Q4_STREAMING_LOADS "Use streaming cache loads for GGUF Q4_K vector weights" OFF)
option(NINFER_GGUF_Q6_STREAMING_LOADS "Use streaming cache loads for GGUF Q6_K vector weights" OFF)
option(NINFER_GGUF_Q6_STATIC_K "Specialize SM86 GGUF Q6_K K5120 decode and T4 verification" OFF)
option(NINFER_GGUF_Q6_COOPERATIVE_DECODE "Cooperatively load packed Q6_K blocks across SM86 warp lanes" OFF)
option(NINFER_GGUF_Q4_COOPERATIVE_DECODE "Cooperatively load packed Q4_K blocks across SM86 warp lanes" OFF)
option(NINFER_GGUF_Q4_ASYNC_PIPELINE "Pipeline SM86 GGUF Q4_K K5120 vector weight loads" OFF)
option(NINFER_GGUF_SWIGLU_Q8_FUSION "Fuse GGUF SwiGLU output with exact BF16-rounded q8 activation packing" OFF)
if(NINFER_GGUF_SWIGLU_Q8_FUSION)
  if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "86")
    message(FATAL_ERROR "GGUF SwiGLU q8 fusion requires sm_86")
  endif()
  set_property(SOURCE "${CMAKE_CURRENT_LIST_DIR}/gguf_linear.cpp" APPEND
    PROPERTY COMPILE_DEFINITIONS NINFER_GGUF_SWIGLU_Q8_FUSION=1)
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
if(NINFER_GGUF_Q4_STREAMING_LOADS)
  if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "86")
    message(FATAL_ERROR "GGUF streaming weight experiments require sm_86")
  endif()
  set_source_files_properties("${CMAKE_CURRENT_LIST_DIR}/ggml_vec_q4_k.cu"
    PROPERTIES COMPILE_DEFINITIONS NINFER_GGUF_Q4_STREAMING_LOADS=1)
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
add_library(ninfer_ggml_quants STATIC
  "${CMAKE_CURRENT_LIST_DIR}/ggml_vec_swiglu_q8.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_bridge.cu"
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
