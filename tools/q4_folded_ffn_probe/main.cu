#include "probe_api.h"
#include "../ffn_pair_probe/probe_api.h"
#include "ops/linear/gguf/ggml_bridge_internal.cuh"

#include <cstring>
#include <stdexcept>
#include <vector>

#define NINFER_PROBE_VALIDATE_FP32 1

namespace {
struct LastCall {
    const void* gate = nullptr;
    const void* up = nullptr;
    const void* activation = nullptr;
    __nv_bfloat16* output = nullptr;
    std::int64_t row_bytes = 0;
    int columns = 0;
};
LastCall last_call;
}

bool run_candidate(const void* gate, const void* up, std::int64_t row_bytes,
                   const void* activation, int columns, __nv_bfloat16* output,
                   cudaStream_t stream) {
    if (!q4_folded_ffn_run(gate, up, row_bytes, activation, columns, output,
                           nullptr, nullptr, stream)) {
        return false;
    }
    last_call = {gate, up, activation, output, row_bytes, columns};
    return true;
}

void candidate_validate(const float* gate, const float* up) {
    using ninfer::ops::gguf::detail::check;
    constexpr std::size_t elements = std::size_t(17408) * 1280;
    constexpr std::size_t bytes = elements * sizeof(float);
    float* diagnostic_gate = nullptr;
    float* diagnostic_up = nullptr;
    check(cudaMalloc(&diagnostic_gate, bytes), "allocate diagnostic gate");
    try {
        check(cudaMalloc(&diagnostic_up, bytes), "allocate diagnostic up");
        check(cudaMemset(diagnostic_gate, 0xff, bytes), "initialize diagnostic gate");
        check(cudaMemset(diagnostic_up, 0xff, bytes), "initialize diagnostic up");
        std::vector<__nv_bfloat16> before(elements + 16), after(elements + 16);
        check(cudaMemcpy(before.data(), last_call.output - 8, before.size() * sizeof(__nv_bfloat16),
                         cudaMemcpyDeviceToHost), "copy timed output and guards");
        if (!q4_folded_ffn_run(last_call.gate, last_call.up, last_call.row_bytes,
                               last_call.activation, last_call.columns, last_call.output,
                               diagnostic_gate, diagnostic_up, nullptr)) {
            throw std::runtime_error("folded diagnostic route declined");
        }
        check(cudaDeviceSynchronize(), "synchronize folded diagnostic");
        std::vector<float> expected(elements), actual(elements);
        const auto compare = [&](const float* a, const float* b) {
            check(cudaMemcpy(expected.data(), a, bytes, cudaMemcpyDeviceToHost), "copy baseline FP32");
            check(cudaMemcpy(actual.data(), b, bytes, cudaMemcpyDeviceToHost), "copy folded FP32");
            if (std::memcmp(expected.data(), actual.data(), bytes)) {
                throw std::runtime_error("folded FP32 plane mismatch");
            }
        };
        compare(gate, diagnostic_gate);
        compare(up, diagnostic_up);
        check(cudaMemcpy(after.data(), last_call.output - 8, after.size() * sizeof(__nv_bfloat16),
                         cudaMemcpyDeviceToHost), "copy diagnostic output and guards");
        if (std::memcmp(before.data(), after.data(), before.size() * sizeof(__nv_bfloat16))) {
            throw std::runtime_error("folded diagnostic BF16 output or guards differ from timed kernel");
        }
        if (q4_folded_ffn_run(last_call.gate, last_call.up, last_call.row_bytes,
                              last_call.activation, last_call.columns, last_call.output,
                              diagnostic_gate, nullptr, nullptr)) {
            throw std::runtime_error("folded route accepted mixed diagnostic pointers");
        }
    } catch (...) {
        cudaFree(diagnostic_up);
        cudaFree(diagnostic_gate);
        throw;
    }
    cudaFree(diagnostic_up);
    cudaFree(diagnostic_gate);
}

#include "../lab/mmq_swiglu_probe.cuh"
