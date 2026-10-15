# Immutable loading and mutable execution have separate compile owners.
add_library(ninfer_model_loading STATIC registry.cpp)
ninfer_internal_includes(ninfer_model_loading)
target_link_libraries(ninfer_model_loading PUBLIC ninfer_artifact ninfer_text ninfer_ops)

add_library(ninfer_model_runtime STATIC)
ninfer_internal_includes(ninfer_model_runtime)
if(NINFER_SPECULATIVE_SELECT_SCATTER_FUSION)
  set_property(SOURCE "${CMAKE_CURRENT_LIST_DIR}/qwen3_5/program/speculative/target_verification.cpp"
    APPEND PROPERTY COMPILE_DEFINITIONS NINFER_SPECULATIVE_SELECT_SCATTER_FUSION=1)
endif()
if(NINFER_MTP_LONGEST_SUFFIX_LOOKUP)
  set_property(SOURCE "${CMAKE_CURRENT_LIST_DIR}/qwen3_5/program/prefill.cpp"
    APPEND PROPERTY COMPILE_DEFINITIONS NINFER_MTP_LONGEST_SUFFIX_LOOKUP=1)
endif()
if(NINFER_GGUF_RMSNORM_Q8_FUSION)
  set_property(SOURCE
    "${CMAKE_CURRENT_LIST_DIR}/qwen3_5/execution/attention.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/qwen3_5/execution/text.cpp"
    APPEND PROPERTY COMPILE_DEFINITIONS NINFER_GGUF_RMSNORM_Q8_FUSION=1)
endif()
if(NINFER_GGUF_RMSNORM_MMQ_FUSION)
  set_property(SOURCE
    "${CMAKE_CURRENT_LIST_DIR}/qwen3_5/execution/attention.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/qwen3_5/execution/text.cpp"
    APPEND PROPERTY COMPILE_DEFINITIONS NINFER_GGUF_RMSNORM_MMQ_FUSION=1)
endif()
if(NINFER_GGUF_FFN_RMSNORM_Q8_FUSION)
  set_property(SOURCE
    "${CMAKE_CURRENT_LIST_DIR}/qwen3_5/execution/ffn.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/qwen3_5/execution/text.cpp"
    APPEND PROPERTY COMPILE_DEFINITIONS NINFER_GGUF_FFN_RMSNORM_Q8_FUSION=1)
endif()
if(NINFER_GDN_PERSISTENT_MIXER)
  set_property(SOURCE "${CMAKE_CURRENT_LIST_DIR}/qwen3_5/execution/text.cpp"
    APPEND PROPERTY COMPILE_DEFINITIONS NINFER_GDN_PERSISTENT_MIXER=1)
endif()
target_link_libraries(ninfer_model_runtime
  PUBLIC ninfer_model_loading ninfer_ops ninfer_runtime_support
  PRIVATE ninfer_media_decode ninfer_jinja)

include(qwen3_5/loading_sources.cmake)
include(qwen3_5/frontend_sources.cmake)
include(qwen3_5/execution_sources.cmake)
include(qwen3_5/program_sources.cmake)
