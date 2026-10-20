#ifndef NINFER_GGUF_Q4_MMQ_COMPACT_A
#define NINFER_GGUF_Q4_MMQ_COMPACT_A 0
#endif
#if defined(NINFER_GGUF_Q4_MMQ_ASYNC_X) || defined(NINFER_GGUF_Q4_MMQ_ASYNC_Y) || \
    defined(NINFER_GGUF_Q4_MMQ_CHILD_CTA) || defined(NINFER_GGUF_Q4_MMQ_K_STREAM) || \
    defined(NINFER_GGUF_Q4_MMQ_PACKED_DM) || defined(NINFER_GGUF_Q4_MMQ_BIASED_DOT) || \
    defined(NINFER_GGUF_Q4_MMQ_OUTPUT_CS) || defined(NINFER_GGUF_Q6_MMQ_ASYNC_Y) || \
    defined(NINFER_GGUF_MMQ_STORE_FUSION) || defined(NINFER_GGUF_MMQ_GATE_EPILOGUE_FUSION)
#error "compact-A probe requires the other MMQ candidates to be undefined"
#endif
#if NINFER_GGUF_Q4_MMQ_COMPACT_A && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ != 860)
#error "compact-A probe targets sm_86"
#endif
#define NINFER_PROBE_VALIDATE_FP32 1
#define mul_mat_q q4_compact_a_mul_mat_q
#define mul_mat_q_stream_k_fixup q4_compact_a_fixup
#define launch_matrix q4_compact_a_launch_matrix
#define matrix_product_impl q4_compact_a_matrix_product_impl
#define matrix_fixup_bytes_impl q4_compact_a_matrix_fixup_bytes_impl
#define select_matrix_launch q4_compact_a_select_matrix_launch

#include "ggml-cuda/mmq.cuh"
#if NINFER_GGUF_Q4_MMQ_COMPACT_A && \
    (!defined(NINFER_Q4_COMPACT_A_GENERATED_MMQ) || \
     !defined(NINFER_Q4_COMPACT_A_GENERATED_LOAD) || \
     !defined(NINFER_Q4_COMPACT_A_GENERATED_DOT))
#error "compact-A probe requires all three generated sibling headers first on the include path"
#endif
#include "ops/linear/gguf/ggml_bridge_mmq.cuh"
#include "../ffn_pair_probe/probe_api.h"

#include <vector>
#include <cstring>

struct Scratch {
    float* gate = nullptr;
    float* up = nullptr;
    Scratch() {
        ninfer::ops::gguf::detail::check(cudaMalloc(&gate, std::size_t(17408) * 1280 * sizeof(float)), "probe gate allocation");
        ninfer::ops::gguf::detail::check(cudaMalloc(&up, std::size_t(17408) * 1280 * sizeof(float)), "probe up allocation");
    }
    ~Scratch() { cudaFree(up); cudaFree(gate); }
};

Scratch& probe_scratch() { static Scratch scratch; return scratch; }

void candidate_validate(const float* gate, const float* up) {
    constexpr std::size_t n = std::size_t(17408) * 1280;
    std::vector<float> expected(n), actual(n);
    const auto compare = [&](const float* a, const float* b) {
        ninfer::ops::gguf::detail::check(cudaMemcpy(expected.data(), a, n * sizeof(float), cudaMemcpyDeviceToHost), "copy baseline FP32");
        ninfer::ops::gguf::detail::check(cudaMemcpy(actual.data(), b, n * sizeof(float), cudaMemcpyDeviceToHost), "copy candidate FP32");
        if (std::memcmp(expected.data(), actual.data(), n * sizeof(float))) { throw std::runtime_error("Q4 MMQ FP32 mismatch"); }
    };
    compare(gate, probe_scratch().gate);
    compare(up, probe_scratch().up);
}

bool run_candidate(const void* gate, const void* up, std::int64_t row_bytes,
                   const void* activation, int columns, __nv_bfloat16* out,
                   cudaStream_t stream) {
    constexpr std::size_t weight_bytes = std::size_t(17408) * 2880;
    const auto start = reinterpret_cast<std::uintptr_t>(out);
    const auto overlaps = [&](const void* p, std::size_t bytes) {
        const auto other = reinterpret_cast<std::uintptr_t>(p);
        return start <= other ? other - start < std::size_t(17408) * 1280 * 2 : start - other < bytes;
    };
    if (!gate || !up || !activation || !out || columns != 1280 || row_bytes != 2880 ||
        overlaps(gate, weight_bytes) || overlaps(up, weight_bytes) ||
        overlaps(activation, std::size_t(5120 / 128) * 1280 * sizeof(block_q8_1_mmq))) {
        return false;
    }
    using namespace ninfer::ops::gguf::detail;
    const auto plan = q4_compact_a_select_matrix_launch<GGML_TYPE_Q4_K>(17408, 1280);
    constexpr std::size_t expected_shared = NINFER_GGUF_Q4_MMQ_COMPACT_A ? 41472 : 57856;
    if (device_facts().cc != 860 || plan.J != 128 || plan.I != 128 || plan.threads != 256 ||
        plan.shared != expected_shared || !plan.stream_k || plan.fixup ||
        plan.row_tiles != 136 || plan.column_tiles != 10 ||
        plan.blocks != plan.row_tiles * plan.column_tiles) {
        return false;
    }
    auto& scratch = probe_scratch();
    q4_compact_a_launch_matrix<GGML_TYPE_Q4_K, 128>(plan, static_cast<const char*>(gate),
        20, 17408, 5120, static_cast<const int*>(activation), 1280, scratch.gate,
        nullptr, 17408, nullptr, stream);
    q4_compact_a_launch_matrix<GGML_TYPE_Q4_K, 128>(plan, static_cast<const char*>(up),
        20, 17408, 5120, static_cast<const int*>(activation), 1280, scratch.up,
        nullptr, 17408, nullptr, stream);
    ffn_pair_probe::store_plane_gate_product(scratch.up, scratch.gate, out, stream);
    return true;
}

#include "../lab/mmq_swiglu_probe.cuh"
