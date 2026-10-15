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
if(NINFER_GDN_PERSISTENT_MIXER)
  if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "86")
    message(FATAL_ERROR "The persistent GDN mixer requires SM86")
  endif()
  target_sources(ninfer_ops PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/gated_delta_net/persistent_mixer.cpp")
  target_sources(ninfer_ggml_quants PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/../linear/gguf/ggml_gdn_persistent.cu")
endif()
