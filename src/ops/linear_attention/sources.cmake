target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/gated_delta_net/gated_delta_net.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/gated_delta_net/replay.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/gated_delta_net/recurrent.cu"
  "${CMAKE_CURRENT_LIST_DIR}/gated_delta_net/state_convert.cu"
  "${CMAKE_CURRENT_LIST_DIR}/gated_delta_net/chunked/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/gated_delta_net/chunked/prepare_wy_wu.cu"
  "${CMAKE_CURRENT_LIST_DIR}/gated_delta_net/chunked/state_passing.cu"
  "${CMAKE_CURRENT_LIST_DIR}/gated_delta_net/chunked/output.cu"
)

option(NINFER_GDN_PERSISTENT_MIXER "Enable the SM86 persistent GDN main-region mixer" OFF)
option(NINFER_GDN_PERSISTENT_ROW_REUSE "Reuse q8 activations across adjacent GDN projection rows" OFF)
if(NINFER_GDN_PERSISTENT_ROW_REUSE AND NOT NINFER_GDN_PERSISTENT_MIXER)
  message(FATAL_ERROR "NINFER_GDN_PERSISTENT_ROW_REUSE requires NINFER_GDN_PERSISTENT_MIXER")
endif()
if(NINFER_GDN_PERSISTENT_MIXER)
  if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "86")
    message(FATAL_ERROR "The persistent GDN mixer requires SM86")
  endif()
  target_sources(ninfer_ops PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/gated_delta_net/persistent_mixer.cpp")
  target_sources(ninfer_ggml_quants PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/../linear/gguf/ggml_gdn_persistent.cu")
  set_property(SOURCE "${CMAKE_CURRENT_LIST_DIR}/../linear/gguf/ggml_gdn_persistent.cu"
    APPEND PROPERTY INCLUDE_DIRECTORIES "${PROJECT_SOURCE_DIR}/include" "${PROJECT_SOURCE_DIR}/src")
  if(NINFER_GDN_PERSISTENT_ROW_REUSE)
    set_property(SOURCE "${CMAKE_CURRENT_LIST_DIR}/../linear/gguf/ggml_gdn_persistent.cu"
      APPEND PROPERTY COMPILE_DEFINITIONS NINFER_GDN_PERSISTENT_ROW_REUSE=1)
  endif()
  if(NINFER_GGUF_Q4_COOPERATIVE_DECODE)
    set_property(SOURCE "${CMAKE_CURRENT_LIST_DIR}/../linear/gguf/ggml_gdn_persistent.cu"
      APPEND PROPERTY COMPILE_DEFINITIONS NINFER_GGUF_Q4_COOPERATIVE_DECODE=1)
  endif()
  if(NINFER_GGUF_Q6_COOPERATIVE_DECODE)
    set_property(SOURCE "${CMAKE_CURRENT_LIST_DIR}/../linear/gguf/ggml_gdn_persistent.cu"
      APPEND PROPERTY COMPILE_DEFINITIONS NINFER_GGUF_Q6_COOPERATIVE_DECODE=1)
  endif()
endif()
