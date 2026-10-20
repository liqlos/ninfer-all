#ifndef NINFER_Q6_FULL_WARP_PROBE_ENTRY
#error "select the OFF or ON translation unit"
#endif
#ifndef NINFER_Q6_FULL_WARP_SLICES
#define NINFER_Q6_FULL_WARP_SLICES 0
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 860
#error "this compile gate only supports SM86"
#endif
#define NINFER_Q6_FULL_WARP_PROBE_INCLUDE 1
#include "q6_full_warp_generated.cuh"
#if !defined(NINFER_Q6_FULL_WARP_GENERATED_HEADER) || NINFER_Q6_FULL_WARP_GENERATED_HEADER != 1
#error "the pinned generated vector header was not included"
#endif

namespace {

using namespace ninfer::ops::gguf::detail;

template <int T>
void launch_selected(const VecArgs& args, cudaStream_t stream) {
    constexpr bool full_warp = NINFER_Q6_FULL_WARP_SLICES != 0;
    static const int resident = [] {
        int blocks = 0;
        check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                  &blocks, vec::kernel<GGML_TYPE_Q6_K, T, false, 0, true, false, 2, full_warp>,
                  kVecWarps * 32, 0),
              "FullWarpSlices selected occupancy");
        return std::max(blocks, 1);
    }();
    constexpr int per_block = kVecWarps * 2;
    const int groups = (args.rows + per_block - 1) / per_block;
    const int blocks = std::min(groups, resident * device_facts().sm_count);
    vec::kernel<GGML_TYPE_Q6_K, T, false, 0, true, false, 2, full_warp>
        <<<blocks, kVecWarps * 32, 0, stream>>>(args);
    check(cudaGetLastError(), "FullWarpSlices selected launch");
}

}

extern "C" bool NINFER_Q6_FULL_WARP_PROBE_ENTRY(
    const ninfer::ops::gguf::detail::VecArgs* args, int columns, bool fused, cudaStream_t stream) {
    using namespace ninfer::ops::gguf::detail;
    if (args == nullptr) { return false; }
    using D = vec::Decoder<GGML_TYPE_Q6_K>;
    if (args->k <= 0 || args->k % D::kBlockElems != 0 ||
        args->row_bytes % D::kBlockBytes != 0 ||
        args->row_bytes < std::int64_t(args->k / D::kBlockElems) * D::kBlockBytes ||
        args->rows <= 0) {
        throw std::invalid_argument("gguf vector product: unsupported geometry");
    }
    if (columns < 1 || columns > 8) {
        throw std::invalid_argument("gguf vector product: columns must be 1..8");
    }
    if (fused || (columns != 1 && columns != 4) || args->k % 1024 != 0 ||
        device_facts().cc != 860) {
        return false;
    }
    if (columns == 1) { launch_selected<1>(*args, stream); }
    else { launch_selected<4>(*args, stream); }
    return true;
}
