add_library(ninfer_linear_test_support STATIC
  "${CMAKE_CURRENT_LIST_DIR}/linear_test_common.cpp")
ninfer_test_includes(ninfer_linear_test_support)
ninfer_op_oracle_options(ninfer_linear_test_support)
target_link_libraries(ninfer_linear_test_support PUBLIC ninfer_ops)

ninfer_add_op_test(ninfer_linear_q4_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q4_a16.cpp"
  LIBRARIES ninfer_linear_test_support)

ninfer_add_op_test(ninfer_linear_q5_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q5_a16.cpp"
  LIBRARIES ninfer_linear_test_support)

ninfer_add_op_test(ninfer_linear_q6_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q6_a16.cpp"
  LIBRARIES ninfer_linear_test_support)

ninfer_add_op_test(ninfer_linear_q8_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q8_a16.cpp"
  LIBRARIES ninfer_linear_test_support)

ninfer_add_op_test(ninfer_linear_t2_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_t2_a16.cpp"
  LIBRARIES ninfer_linear_test_support)

ninfer_add_op_test(ninfer_linear_t2_a8_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_t2_a8.cpp"
  LIBRARIES ninfer_linear_test_support)

ninfer_add_op_test(ninfer_linear_nvfp4_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_nvfp4_a16.cpp"
  LIBRARIES ninfer_linear_test_support)

ninfer_add_op_test(ninfer_linear_nvfp4_a4_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_nvfp4_a4.cpp"
  LIBRARIES ninfer_linear_test_support)

ninfer_add_op_test(ninfer_linear_fp8_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_fp8_a16.cpp"
  LIBRARIES ninfer_linear_test_support)

ninfer_add_op_test(ninfer_linear_fp8_a8_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_fp8_a8.cpp"
  LIBRARIES ninfer_linear_test_support)

ninfer_add_op_test(ninfer_linear_bf16_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_bf16_a16.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_linear_gguf_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_gguf.cpp"
  LIBRARIES ninfer_ops)
if(NINFER_GGUF_DUAL_MMQ_FUSION)
  set_property(SOURCE "${CMAKE_CURRENT_LIST_DIR}/test_gguf.cpp" APPEND
    PROPERTY COMPILE_DEFINITIONS NINFER_GGUF_DUAL_MMQ_FUSION=1)
  add_test(NAME ninfer_linear_gguf_dual_mmq_test
    COMMAND ninfer_tests ninfer_linear_gguf_test --dual-mmq-fusion)
  set_tests_properties(ninfer_linear_gguf_dual_mmq_test PROPERTIES SKIP_RETURN_CODE 77)
endif()

if(NINFER_GGUF_PERSISTENT_FFN OR NINFER_GGUF_FFN_RMSNORM_Q8_FUSION)
  ninfer_add_op_test(ninfer_linear_ffn_q8_test
    SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_persistent_ffn.cpp"
    LIBRARIES ninfer_ops ninfer_model_runtime)
  if(NINFER_GGUF_PERSISTENT_FFN)
    set_property(SOURCE "${CMAKE_CURRENT_LIST_DIR}/test_persistent_ffn.cpp" APPEND
      PROPERTY COMPILE_DEFINITIONS NINFER_GGUF_PERSISTENT_FFN=1)
  endif()
  if(NINFER_GGUF_FFN_RMSNORM_Q8_FUSION)
    set_property(SOURCE "${CMAKE_CURRENT_LIST_DIR}/test_persistent_ffn.cpp" APPEND
      PROPERTY COMPILE_DEFINITIONS NINFER_GGUF_FFN_RMSNORM_Q8_FUSION=1)
  endif()
endif()
