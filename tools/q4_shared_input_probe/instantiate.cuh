namespace ninfer::ops::gguf::detail::vec {
static_assert(kVecWarps == 4);
static_assert(vector_rows_per_warp<GGML_TYPE_Q4_K, 4, true>() == 2);
template __global__ void kernel<GGML_TYPE_Q4_K, 4, true, 5120, true, false, 2>(VecArgs);
}
