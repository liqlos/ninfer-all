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
if(NINFER_GGUF_Q4_T4_FUSED_WARP_PAIR)
  set_property(SOURCE "${CMAKE_CURRENT_LIST_DIR}/test_gguf.cpp" APPEND
    PROPERTY COMPILE_DEFINITIONS NINFER_GGUF_Q4_T4_FUSED_WARP_PAIR_TEST=1)
  add_test(NAME ninfer_linear_gguf_q4_t4_fused_warp_pair_test
    COMMAND ninfer_tests ninfer_linear_gguf_test --q4-t4-fused-warp-pair-regression)
  set_tests_properties(ninfer_linear_gguf_q4_t4_fused_warp_pair_test PROPERTIES SKIP_RETURN_CODE 77)
endif()
if(NINFER_GGUF_Q4_T4_IMMA)
  set_property(SOURCE "${CMAKE_CURRENT_LIST_DIR}/test_gguf.cpp" APPEND
    PROPERTY COMPILE_DEFINITIONS NINFER_GGUF_Q4_T4_IMMA_TEST=1)
  add_test(NAME ninfer_linear_gguf_q4_t4_imma_test
    COMMAND ninfer_tests ninfer_linear_gguf_test --q4-t4-imma-regression)
  set_tests_properties(ninfer_linear_gguf_q4_t4_imma_test PROPERTIES SKIP_RETURN_CODE 77)
endif()
if(NINFER_GGUF_DUAL_LAYOUT_MMQ_PACK)
  ninfer_add_op_test(ninfer_linear_gguf_dual_layout_pack_test
    SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_dual_layout_mmq_pack.cu"
    LIBRARIES ninfer_ops)
endif()
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

if(NINFER_GGUF_MMQ_STORE_FUSION)
  ninfer_add_op_test(ninfer_linear_gguf_mmq_store_test
    SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_gguf_mmq_store.cu"
    LIBRARIES ninfer_ops)
endif()
if(NINFER_GGUF_MMQ_GATE_EPILOGUE_FUSION)
  ninfer_add_op_test(ninfer_linear_gguf_mmq_gate_epilogue_test
    SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_gguf_mmq_gate_epilogue.cu"
    LIBRARIES ninfer_ops)
endif()
if(NINFER_GGUF_SM86_MMQ_TILE_CAP AND NINFER_GGUF_MMQ_STORE_FUSION)
  set_property(SOURCE "${CMAKE_CURRENT_LIST_DIR}/test_gguf_mmq_store.cu" APPEND
    PROPERTY COMPILE_DEFINITIONS NINFER_GGUF_SM86_MMQ_MAX_J=${NINFER_GGUF_SM86_MMQ_MAX_J})
  ninfer_add_op_test(ninfer_linear_mmq_tile_cap_test
    SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_mmq_tile_cap.cu"
    LIBRARIES ninfer_ops)
  set_property(SOURCE "${CMAKE_CURRENT_LIST_DIR}/test_mmq_tile_cap.cu" APPEND
    PROPERTY COMPILE_DEFINITIONS NINFER_GGUF_SM86_MMQ_MAX_J=${NINFER_GGUF_SM86_MMQ_MAX_J})
endif()
