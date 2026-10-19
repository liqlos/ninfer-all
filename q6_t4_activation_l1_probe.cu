#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

#if defined(NINFER_GGUF_Q6_T4_HALF_DECODE_CONSUME) || defined(NINFER_GGUF_Q6_T4_ACTIVATION_L1)
#error Probe requires the half-decode and activation L1 flags initially disabled
#endif
#define NINFER_GGUF_Q6_COOPERATIVE_DECODE 1
#include "src/ops/linear/gguf/activation_l1_off.cuh"
#define NINFER_GGUF_Q6_T4_ACTIVATION_L1 1
#include "src/ops/linear/gguf/activation_l1_on.cuh"

using namespace ninfer::ops::gguf::detail;

static void ck(cudaError_t e) {
    if (e != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(e)); }
}

template <bool On>
static void launch(VecArgs args, int grid, cudaStream_t stream) {
    if constexpr (On) {
        activation_l1_on_probe::vec::kernel<GGML_TYPE_Q6_K, 4, false, 0>
            <<<grid, 128, 0, stream>>>(args);
    } else {
        activation_l1_off_probe::vec::kernel<GGML_TYPE_Q6_K, 4, false, 0>
            <<<grid, 128, 0, stream>>>(args);
    }
}

static double ref_at(const std::vector<std::uint8_t>& weights, int row, int k, int column,
                     const std::vector<std::int8_t>& qs, const std::vector<half2>& ds) {
    double total = 0.0;
    for (int i = 0; i < k; ++i) {
        const auto* b = weights.data() + std::int64_t(row) * (k / 256) * 210 + (i / 256) * 210;
        const int n = i % 256;
        const int u = n / 32;
        const int h = u / 4;
        const int q = u % 4;
        const int z = n % 32;
        const int v = ((b[64 * h + 32 * (q & 1) + z] >> (4 * (q / 2))) & 15) |
                      (((b[128 + 32 * h + z] >> (2 * q)) & 3) << 4);
        half d;
        std::memcpy(&d, b + 208, sizeof(d));
        const int scale = std::int8_t(b[192 + 8 * h + 2 * q + z / 16]);
        const float2 activation = __half22float2(ds[column * (k / 32) + i / 32]);
        total += double(__half2float(d)) * scale * (v - 32) *
                 double(activation.x) * int(qs[column * k + i]);
    }
    return total;
}

static void run(const char* file, int k, int rows, const char* tag) {
    constexpr int columns = 4;
    std::ifstream input(file, std::ios::binary | std::ios::ate);
    if (!input) { throw std::runtime_error("real Q6 fixture missing"); }
    const auto bytes = input.tellg();
    if (bytes != std::streamoff(std::int64_t(rows) * (k / 256) * 210)) {
        throw std::runtime_error("real Q6 fixture byte count differs");
    }
    std::vector<std::uint8_t> weights(static_cast<std::size_t>(bytes));
    input.seekg(0);
    input.read(reinterpret_cast<char*>(weights.data()), bytes);
    if (!input) { throw std::runtime_error("real Q6 fixture read failed"); }
    std::vector<std::int8_t> qs(columns * k);
    std::vector<half2> ds(columns * (k / 32));
    for (int j = 0; j < columns; ++j) {
        for (int i = 0; i < k; ++i) { qs[j * k + i] = std::int8_t(((i * 37 + j * 13 + 17) % 255) - 127); }
        for (int i = 0; i < k / 32; ++i) {
            const float d = __half2float(__float2half(0.003f + 0.0001f * ((i + j) % 29)));
            int sum = 0;
            for (int z = 0; z < 32; ++z) { sum += qs[j * k + 32 * i + z]; }
            ds[j * (k / 32) + i] = __floats2half2_rn(d, d * sum);
        }
    }
    std::uint8_t* dw;
    std::int8_t* dq;
    half2* dd;
    __nv_bfloat16* off_output;
    __nv_bfloat16* on_output;
    const std::size_t count = std::size_t(rows) * columns;
    const std::size_t guarded_count = count + 32;
    ck(cudaMalloc(&dw, weights.size()));
    ck(cudaMalloc(&dq, qs.size()));
    ck(cudaMalloc(&dd, ds.size() * sizeof(half2)));
    ck(cudaMalloc(&off_output, guarded_count * sizeof(__nv_bfloat16)));
    ck(cudaMalloc(&on_output, guarded_count * sizeof(__nv_bfloat16)));
    ck(cudaMemcpy(dw, weights.data(), weights.size(), cudaMemcpyHostToDevice));
    ck(cudaMemcpy(dq, qs.data(), qs.size(), cudaMemcpyHostToDevice));
    ck(cudaMemcpy(dd, ds.data(), ds.size() * sizeof(half2), cudaMemcpyHostToDevice));
    ck(cudaMemset(off_output, 0xa5, guarded_count * sizeof(__nv_bfloat16)));
    ck(cudaMemset(on_output, 0xa5, guarded_count * sizeof(__nv_bfloat16)));
    cudaFuncAttributes off_attributes, on_attributes;
    ck(cudaFuncGetAttributes(&off_attributes,
        activation_l1_off_probe::vec::kernel<GGML_TYPE_Q6_K, 4, false, 0>));
    ck(cudaFuncGetAttributes(&on_attributes,
        activation_l1_on_probe::vec::kernel<GGML_TYPE_Q6_K, 4, false, 0>));
    int off_resident, on_resident;
    ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&off_resident,
        activation_l1_off_probe::vec::kernel<GGML_TYPE_Q6_K, 4, false, 0>, 128, 0));
    ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&on_resident,
        activation_l1_on_probe::vec::kernel<GGML_TYPE_Q6_K, 4, false, 0>, 128, 0));
    cudaDeviceProp properties;
    ck(cudaGetDeviceProperties(&properties, 0));
    if (properties.major != 8 || properties.minor != 6) { throw std::runtime_error("SM86 required"); }
    if (off_resident <= 0 || on_resident <= 0) { throw std::runtime_error("kernel has no resident blocks"); }
    const int groups = (rows + 7) / 8;
    const int off_grid = std::min(groups, off_resident * properties.multiProcessorCount);
    const int on_grid = std::min(groups, on_resident * properties.multiProcessorCount);
    std::printf("{\"kind\":\"native_grid\",\"shape\":\"%s\",\"off_grid\":%d,\"on_grid\":%d,\"off_resident\":%d,\"on_resident\":%d}\n",
                tag, off_grid, on_grid, off_resident, on_resident);
    VecArgs off{dw, std::int64_t(k / 256) * 210, rows, k, dq, dd,
                {off_output, nullptr, rows, ninfer::ops::gguf::Epilogue::Store, nullptr, 0}};
    VecArgs on = off;
    on.out.bf16 = on_output;
    launch<false>(off, off_grid, 0);
    launch<true>(on, on_grid, 0);
    ck(cudaGetLastError());
    ck(cudaDeviceSynchronize());
    std::vector<std::uint16_t> expected(guarded_count), actual(guarded_count);
    ck(cudaMemcpy(expected.data(), off_output, guarded_count * 2, cudaMemcpyDeviceToHost));
    ck(cudaMemcpy(actual.data(), on_output, guarded_count * 2, cudaMemcpyDeviceToHost));
    if (expected != actual) { throw std::runtime_error("exact OFF/ON BF16 mismatch"); }
    for (std::size_t i = count; i < guarded_count; ++i) {
        if (expected[i] != 0xa5a5) { throw std::runtime_error("output guard overwritten"); }
    }
    double max_error = 0.0;
    for (int row : {0, rows / 2, rows - 1}) {
        for (int j = 0; j < columns; ++j) {
            const double reference = ref_at(weights, row, k, j, qs, ds);
            __nv_bfloat16 value;
            std::memcpy(&value, &expected[j * rows + row], sizeof(value));
            const double result = __bfloat162float(value);
            const double error = std::abs(result - reference);
            max_error = std::max(max_error, error);
            if (!std::isfinite(result) || error > 0.008 * std::abs(reference) + 0.003) {
                throw std::runtime_error("independent FP64 oracle failure");
            }
        }
    }
    std::printf("{\"kind\":\"identity\",\"shape\":\"%s\",\"rows\":%d,\"k\":%d,\"t\":4,\"off_regs\":%d,\"on_regs\":%d,\"off_local\":%zu,\"on_local\":%zu,\"exact_bf16\":true,\"oracle_samples\":12,\"oracle_max_abs\":%.9g}\n",
                tag, rows, k, off_attributes.numRegs, on_attributes.numRegs,
                off_attributes.localSizeBytes, on_attributes.localSizeBytes, max_error);
    for (int i = 0; i < 8; ++i) {
        launch<false>(off, off_grid, 0);
        launch<true>(on, on_grid, 0);
    }
    ck(cudaDeviceSynchronize());
    cudaEvent_t start, end;
    ck(cudaEventCreate(&start));
    ck(cudaEventCreate(&end));
    for (int block = 0; block < 20; ++block) {
        for (int position = 0; position < 4; ++position) {
            const bool enabled = block % 2 == 0 ? position == 1 || position == 2 : position == 0 || position == 3;
            ck(cudaEventRecord(start));
            for (int repetition = 0; repetition < 12; ++repetition) {
                if (enabled) { launch<true>(on, on_grid, 0); }
                else { launch<false>(off, off_grid, 0); }
            }
            ck(cudaGetLastError());
            ck(cudaEventRecord(end));
            ck(cudaEventSynchronize(end));
            float ms;
            ck(cudaEventElapsedTime(&ms, start, end));
            std::printf("{\"kind\":\"sample\",\"shape\":\"%s\",\"block\":%d,\"position\":%d,\"on\":%s,\"repetitions\":12,\"ms_per_call\":%.9g}\n",
                        tag, block, position, enabled ? "true" : "false", ms / 12);
        }
    }
    for (auto* output : {off_output, on_output}) {
        ck(cudaMemcpy(actual.data(), output, guarded_count * 2, cudaMemcpyDeviceToHost));
        if (expected != actual) { throw std::runtime_error("post-timing parity/guard mismatch"); }
    }
    ck(cudaEventDestroy(start));
    ck(cudaEventDestroy(end));
    ck(cudaFree(dw));
    ck(cudaFree(dq));
    ck(cudaFree(dd));
    ck(cudaFree(off_output));
    ck(cudaFree(on_output));
}

int main() {
    try {
        run("/extra/q6-full-k17408.bin", 17408, 5120, "q6_t4_activation_l1_down_t4");
        run("/fixtures/q6-head-full-k5120.bin", 5120, 248320, "q6_t4_activation_l1_head_t4");
        std::puts("{\"kind\":\"complete\",\"status\":\"PASS\",\"scope\":\"exploratory operator only; warm resident real Q6 weights; synthetic activation\"}");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
