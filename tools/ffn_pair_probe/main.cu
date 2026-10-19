#include "ops/linear/gguf/ggml_bridge_ffn_pair.h"
#include "ops/linear/gguf/ggml_bridge_internal.cuh"
#include "ggml-cuda/common.cuh"
#include "ggml-cuda/mmq.cuh"
#include "probe_api.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdarg>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::ops::gguf::detail {
const DeviceFacts& device_facts() {
    static const DeviceFacts facts = [] {
        int device = 0;
        if (cudaGetDevice(&device) != cudaSuccess) { throw std::runtime_error("get CUDA device failed"); }
        cudaDeviceProp properties{};
        if (cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
            throw std::runtime_error("get CUDA properties failed");
        }
        return DeviceFacts{100 * properties.major + 10 * properties.minor,
                           properties.multiProcessorCount,
                           properties.sharedMemPerBlockOptin};
    }();
    return facts;
}
}

extern "C" void ggml_abort(const char* file, int line, const char* format, ...) {
    std::cerr << "ggml abort " << file << ':' << line << ' ' << format << '\n';
    std::abort();
}

void ggml_cuda_error(const char* statement, const char* function, const char* file,
                     int line, const char* message) {
    std::cerr << "CUDA " << statement << ' ' << function << ' ' << file << ':' << line
              << ' ' << message << '\n';
    std::abort();
}

namespace {

namespace gguf = ninfer::ops::gguf;
constexpr int kRows = 17408;
constexpr int kK = 5120;
constexpr int kColumns = 1280;
constexpr std::size_t kFixtureBytes = 100270080;
constexpr std::size_t kRowBytes = 20 * 144;
constexpr std::uint16_t kCanaryBits = 0x5A3C;
constexpr std::size_t kGuardElements = 8;

void check(cudaError_t status, const char* label) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(label) + ": " + cudaGetErrorString(status));
    }
}

struct DeviceBuffer {
    void* data = nullptr;
    std::size_t size = 0;

    explicit DeviceBuffer(std::size_t bytes) : size(bytes) {
        if (bytes != 0) { check(cudaMalloc(&data, bytes), "cudaMalloc"); }
    }

    ~DeviceBuffer() {
        if (data != nullptr) { cudaFree(data); }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void from_host(const void* source) {
        check(cudaMemcpy(data, source, size, cudaMemcpyHostToDevice), "copy host to device");
    }

    void to_host(void* target) const {
        check(cudaMemcpy(target, data, size, cudaMemcpyDeviceToHost), "copy device to host");
    }
};

std::vector<std::uint8_t> read_fixture(const std::filesystem::path& path) {
    if (!std::filesystem::is_regular_file(path) || std::filesystem::file_size(path) != kFixtureBytes) {
        throw std::runtime_error("the full pinned q4-full-k5120.bin fixture is required; synthetic fallback is disabled");
    }
    std::vector<std::uint8_t> bytes(kFixtureBytes);
    std::ifstream input(path, std::ios::binary);
    input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (input.gcount() != static_cast<std::streamsize>(bytes.size())) {
        throw std::runtime_error("fixture read was incomplete");
    }
    return bytes;
}

float half_value(const std::uint8_t* bytes) {
    std::uint16_t bits = 0;
    std::memcpy(&bits, bytes, sizeof(bits));
    __half value;
    std::memcpy(&value, &bits, sizeof(bits));
    return __half2float(value);
}

double q4k_q8_1_dot(const std::uint8_t* weight, const block_q8_1_mmq* activation,
                    int column, int columns) {
    double result = 0.0;
    for (int block_index = 0; block_index < kK / 256; ++block_index) {
        const auto* block = weight + std::size_t(block_index) * 144;
        const double d = half_value(block);
        const double dmin = half_value(block + 2);
        const auto* scales = block + 4;
        const auto* quants = block + 16;
        for (int subblock = 0; subblock < 8; ++subblock) {
            int scale = 0;
            int minimum = 0;
            if (subblock < 4) {
                scale = scales[subblock] & 63;
                minimum = scales[subblock + 4] & 63;
            } else {
                scale = (scales[subblock + 4] & 15) | ((scales[subblock - 4] >> 6) << 4);
                minimum = (scales[subblock + 4] >> 4) | ((scales[subblock] >> 6) << 4);
            }
            const auto* packed = quants + (subblock / 2) * 32;
            const int shift = 4 * (subblock & 1);
            const int activation_block = block_index * 2 + subblock / 4;
            const auto& q8 = activation[std::size_t(activation_block) * columns + column];
            const int q8_subblock = subblock % 4;
            const double activation_d = __low2float(q8.ds4[q8_subblock]);
            const double activation_sum = __high2float(q8.ds4[q8_subblock]);
            int integer_dot = 0;
            for (int index = 0; index < 32; ++index) {
                integer_dot += ((packed[index] >> shift) & 15) *
                               int(q8.qs[q8_subblock * 32 + index]);
            }
            result += double(integer_dot) * d * scale * activation_d -
                      dmin * minimum * activation_sum;
        }
    }
    return result;
}

std::vector<__nv_bfloat16> make_input(std::uint64_t seed) {
    std::mt19937_64 random(seed);
    std::uniform_real_distribution<float> distribution(-0.45f, 0.45f);
    std::vector<__nv_bfloat16> input(std::size_t(kK) * kColumns);
    for (auto& value : input) { value = __float2bfloat16(distribution(random)); }
    return input;
}

bool check_canaries(const std::vector<__nv_bfloat16>& output) {
    for (std::size_t i = 0; i < kGuardElements; ++i) {
        std::uint16_t before = 0;
        std::uint16_t after = 0;
        std::memcpy(&before, &output[i], sizeof(before));
        std::memcpy(&after, &output[output.size() - kGuardElements + i], sizeof(after));
        if (before != kCanaryBits || after != kCanaryBits) { return false; }
    }
    return true;
}

__global__ void quantize_q8_ds4(const __nv_bfloat16* input, block_q8_1_mmq* output) {
    const int i0 = (int(blockDim.x) * blockIdx.y + threadIdx.x) * 4;
    if (i0 >= kK) { return; }
    const int column = blockIdx.x;
    const __nv_bfloat16* source = input + std::int64_t(column) * kK;
    const uint2 words = *reinterpret_cast<const uint2*>(source + i0);
    const float2 lo = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&words.x));
    const float2 hi = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&words.y));
    const float4 values = make_float4(lo.x, lo.y, hi.x, hi.y);
    float maximum = fmaxf(fmaxf(fabsf(values.x), fabsf(values.y)),
                          fmaxf(fabsf(values.z), fabsf(values.w)));
#pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
        maximum = fmaxf(maximum, __shfl_xor_sync(0xFFFFFFFF, maximum, offset, 32));
    }
    float sum = ((values.x + values.y) + values.z) + values.w;
#pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, 32);
    }
    const float inverse = 127.0f / maximum;
    char4 quantized;
    quantized.x = roundf(values.x * inverse);
    quantized.y = roundf(values.y * inverse);
    quantized.z = roundf(values.z * inverse);
    quantized.w = roundf(values.w * inverse);
    const float scale = 1.0f / inverse;
    const int block_index = (i0 / 128) * kColumns + column;
    const int in_block = i0 % 128;
    reinterpret_cast<char4*>(output[block_index].qs)[in_block / 4] = quantized;
    if (in_block % 32 == 0) {
        output[block_index].ds4[in_block / 32] = make_half2(scale, sum);
    }
}

void baseline(const std::uint8_t* gate, const std::uint8_t* up, const void* activation,
              float* gate_plane, float* up_plane, void* gate_fixup, void* up_fixup,
              __nv_bfloat16* output, cudaStream_t stream) {
    ffn_pair_probe::matrix_product(gate, activation, gate_plane, gate_fixup, stream);
    ffn_pair_probe::matrix_product(up, activation, up_plane, up_fixup, stream);
    ffn_pair_probe::store_plane_gate_product(up_plane, gate_plane, output, stream);
}

double time_arm(bool candidate, const std::uint8_t* gate, const std::uint8_t* up,
                const void* activation, float* gate_plane, float* up_plane,
                void* gate_fixup, void* up_fixup, __nv_bfloat16* output,
                cudaStream_t stream, int repeats) {
    cudaEvent_t start = nullptr;
    cudaEvent_t end = nullptr;
    check(cudaEventCreate(&start), "create start event");
    check(cudaEventCreate(&end), "create end event");
    check(cudaEventRecord(start, stream), "record start event");
    for (int i = 0; i < repeats; ++i) {
        if (candidate) {
            if (!gguf::ffn_q4_k_mmq_pair(gate, up, kRowBytes, activation, kColumns,
                                          output, stream)) {
                throw std::runtime_error("FFN pair route declined the fixed SM86 screen geometry");
            }
        } else {
            baseline(gate, up, activation, gate_plane, up_plane, gate_fixup, up_fixup,
                     output, stream);
        }
    }
    check(cudaEventRecord(end, stream), "record end event");
    check(cudaEventSynchronize(end), "synchronize timing event");
    float milliseconds = 0.0f;
    check(cudaEventElapsedTime(&milliseconds, start, end), "elapsed timing");
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    return double(milliseconds) / repeats;
}

void verify_oracle(const std::vector<std::uint8_t>& weights,
                   const std::vector<block_q8_1_mmq>& activation,
                   const std::vector<__nv_bfloat16>& output,
                   double& max_abs, double& max_rel) {
    const std::uint8_t* gate = weights.data();
    const std::uint8_t* up = weights.data() + std::size_t(kRows) * kRowBytes;
    const std::array<int, 8> rows{0, 1, 7, kRows / 4, kRows / 3, kRows / 2, kRows - 2, kRows - 1};
    const std::array<int, 5> columns{0, 1, kColumns / 3, kColumns / 2, kColumns - 1};
    max_abs = 0.0;
    max_rel = 0.0;
    for (const int column : columns) {
        for (const int row : rows) {
            const auto* gate_row = gate + std::size_t(row) * kRowBytes;
            const auto* up_row = up + std::size_t(row) * kRowBytes;
            const double gate_dot = q4k_q8_1_dot(gate_row, activation.data(), column, kColumns);
            const double up_dot = q4k_q8_1_dot(up_row, activation.data(), column, kColumns);
            const double expected = gate_dot / (1.0 + std::exp(-gate_dot)) * up_dot;
            const double actual = __bfloat162float(output[std::size_t(column) * kRows + row]);
            const double error = std::abs(actual - expected);
            const double relative = error / std::max(std::abs(expected), 1.0e-12);
            max_abs = std::max(max_abs, error);
            max_rel = std::max(max_rel, relative);
            if (!std::isfinite(actual) || error > std::max(0.005, std::abs(expected) * 0.006)) {
                throw std::runtime_error("independent FP64 Q4_K/Q8_1 SwiGLU oracle failed");
            }
        }
    }
}

}

int main(int argc, char** argv) {
    try {
        if ((argc != 3 && argc != 4) || std::string(argv[2]).empty()) {
            throw std::runtime_error("usage: ninfer-ffn-pair-probe <pinned-q4-full-k5120.bin> <fresh-launch-id> [seed]");
        }
        const std::filesystem::path fixture_path(argv[1]);
        const std::string launch_id(argv[2]);
        const auto host_weights = read_fixture(fixture_path);
        int device = 0;
        check(cudaGetDevice(&device), "get device");
        cudaDeviceProp properties{};
        check(cudaGetDeviceProperties(&properties, device), "device properties");
        if (properties.major != 8 || properties.minor != 6) { return 77; }
        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;
        check(cudaMemGetInfo(&free_bytes, &total_bytes), "memory info");
        if (free_bytes < 8ull * 1024 * 1024 * 1024) {
            throw std::runtime_error("standalone FFN pair probe requires at least 8 GiB free VRAM");
        }
        const std::uint64_t seed = argc == 4 ? std::stoull(argv[3]) : std::random_device{}();
        const auto host_input = make_input(seed);
        DeviceBuffer weights_device(host_weights.size());
        DeviceBuffer input_device(host_input.size() * sizeof(__nv_bfloat16));
        constexpr std::size_t activation_elements = std::size_t(kK / 128) * kColumns;
        constexpr std::size_t activation_bytes = activation_elements * sizeof(block_q8_1_mmq) +
                                                 128 * sizeof(block_q8_1_mmq);
        DeviceBuffer activation_device(activation_bytes);
        weights_device.from_host(host_weights.data());
        input_device.from_host(host_input.data());
        quantize_q8_ds4<<<dim3(kColumns, kK / 512, 1), 128>>>(
            static_cast<const __nv_bfloat16*>(input_device.data),
            static_cast<block_q8_1_mmq*>(activation_device.data));
        check(cudaGetLastError(), "launch common Q8_1 DS4 packing");
        check(cudaDeviceSynchronize(), "synchronize common Q8_1 packing");
        std::vector<block_q8_1_mmq> host_activation(activation_elements);
        check(cudaMemcpy(host_activation.data(), activation_device.data,
                         host_activation.size() * sizeof(block_q8_1_mmq),
                         cudaMemcpyDeviceToHost), "copy common packed Q8_1 activation");
        const std::size_t output_elements = std::size_t(kRows) * kColumns;
        const std::size_t output_bytes = (output_elements + 2 * kGuardElements) * sizeof(__nv_bfloat16);
        const std::size_t plane_bytes = output_elements * sizeof(float);
        const std::size_t fixup_bytes = ffn_pair_probe::matrix_fixup_bytes(kColumns);
        DeviceBuffer gate_plane(plane_bytes);
        DeviceBuffer up_plane(plane_bytes);
        DeviceBuffer gate_fixup(fixup_bytes);
        DeviceBuffer up_fixup(fixup_bytes);
        DeviceBuffer baseline_output(output_bytes);
        DeviceBuffer candidate_output(output_bytes);
        __nv_bfloat16 canary = __float2bfloat16(0.0f);
        std::uint16_t canary_bits = kCanaryBits;
        std::memcpy(&canary, &canary_bits, sizeof(canary_bits));
        std::vector<__nv_bfloat16> guards(output_elements + 2 * kGuardElements, canary);
        baseline_output.from_host(guards.data());
        candidate_output.from_host(guards.data());
        const auto* gate = static_cast<const std::uint8_t*>(weights_device.data);
        const auto* up = gate + std::size_t(kRows) * kRowBytes;
        auto* baseline_out = static_cast<__nv_bfloat16*>(baseline_output.data) + kGuardElements;
        auto* candidate_out = static_cast<__nv_bfloat16*>(candidate_output.data) + kGuardElements;
        const auto* activation = static_cast<const void*>(activation_device.data);
        const auto rejected = [&](const void* bad_gate, const void* bad_up,
                                  std::int64_t bad_row_bytes, const void* bad_activation,
                                  int bad_columns, __nv_bfloat16* bad_output) {
            if (gguf::ffn_q4_k_mmq_pair(bad_gate, bad_up, bad_row_bytes, bad_activation,
                                         bad_columns, bad_output, nullptr)) {
                throw std::runtime_error("FFN pair route accepted invalid API arguments");
            }
        };
        rejected(gate, up, kRowBytes, activation, kColumns - 1, candidate_out);
        rejected(gate, up, kRowBytes - 144, activation, kColumns, candidate_out);
        rejected(nullptr, up, kRowBytes, activation, kColumns, candidate_out);
        rejected(gate, nullptr, kRowBytes, activation, kColumns, candidate_out);
        rejected(gate, up, kRowBytes, nullptr, kColumns, candidate_out);
        rejected(gate, up, kRowBytes, activation, kColumns, nullptr);
        rejected(gate, up, kRowBytes, activation, kColumns,
                 reinterpret_cast<__nv_bfloat16*>(const_cast<std::uint8_t*>(gate)));
        std::vector<__nv_bfloat16> rejected_output(output_elements + 2 * kGuardElements);
        candidate_output.to_host(rejected_output.data());
        if (std::memcmp(rejected_output.data(), guards.data(), output_bytes) != 0) {
            throw std::runtime_error("rejected FFN pair API call modified sentinel output");
        }
        if (!gguf::ffn_q4_k_mmq_pair(gate, up, kRowBytes, activation, kColumns,
                                      candidate_out, nullptr)) {
            throw std::runtime_error("FFN pair route declined the fixed SM86 screen geometry");
        }
        baseline(gate, up, activation, static_cast<float*>(gate_plane.data),
                 static_cast<float*>(up_plane.data), gate_fixup.data, up_fixup.data,
                 baseline_out, nullptr);
        check(cudaDeviceSynchronize(), "synchronize baseline and candidate correctness");
        std::vector<__nv_bfloat16> baseline_host(output_elements + 2 * kGuardElements);
        std::vector<__nv_bfloat16> candidate_host(output_elements + 2 * kGuardElements);
        baseline_output.to_host(baseline_host.data());
        candidate_output.to_host(candidate_host.data());
        if (!check_canaries(baseline_host) || !check_canaries(candidate_host)) {
            throw std::runtime_error("output guard canary was overwritten");
        }
        if (std::memcmp(baseline_host.data() + kGuardElements,
                        candidate_host.data() + kGuardElements,
                        output_elements * sizeof(__nv_bfloat16)) != 0) {
            throw std::runtime_error("candidate BF16 output differs from two-MMQ plus GateProduct baseline");
        }
        double oracle_max_abs = 0.0;
        double oracle_max_rel = 0.0;
        std::vector<__nv_bfloat16> compact_candidate(
            candidate_host.begin() + kGuardElements,
            candidate_host.end() - kGuardElements);
        verify_oracle(host_weights, host_activation, compact_candidate,
                      oracle_max_abs, oracle_max_rel);
        std::cout << std::setprecision(10)
                  << "{\"kind\":\"correctness\",\"status\":\"PASS\",\"fixture\":\"q4-full-k5120.bin\",\"fixture_bytes\":"
                  << host_weights.size() << ",\"rows\":" << kRows << ",\"k\":" << kK
                  << ",\"columns\":" << kColumns
                  << ",\"baseline\":\"two Q4_K FP32 MMQ products plus store_plane GateProduct\""
                  << ",\"common_activation_pack\":\"Q8_1 DS4 outside timing\",\"exact_bf16_all_outputs\":true"
                  << ",\"output_canaries\":true,\"independent_fp64_samples\":40,\"fp64_max_abs_error\":"
                  << oracle_max_abs << ",\"fp64_max_rel_error\":" << oracle_max_rel
                  << ",\"input_seed\":" << seed << ",\"invalid_api_guards\":\"PASS\""
                  << ",\"fresh_launch_id\":\"" << launch_id << "\"}\n";
        std::vector<std::array<double, 2>> block_times(3, {0.0, 0.0});
        std::uint64_t order_state = seed ^ 0x9E3779B97F4A7C15ull;
        for (int block = 0; block < 3; ++block) {
            order_state ^= order_state << 13;
            order_state ^= order_state >> 7;
            order_state ^= order_state << 17;
            const bool candidate_first = (order_state & 1U) != 0;
            const std::array<int, 4> sequence = candidate_first
                ? std::array<int, 4>{1, 0, 0, 1}
                : std::array<int, 4>{0, 1, 1, 0};
            for (const int arm : sequence) {
                const bool is_candidate = arm != 0;
                const double ms = time_arm(
                    is_candidate, gate, up, activation, static_cast<float*>(gate_plane.data),
                    static_cast<float*>(up_plane.data), gate_fixup.data, up_fixup.data,
                    is_candidate ? candidate_out : baseline_out, nullptr, 4);
                block_times[block][arm] += ms / 2.0;
            }
            std::cout << std::setprecision(10)
                      << "{\"kind\":\"timing\",\"block\":" << block
                      << ",\"order\":\"" << (candidate_first ? "BAAB" : "ABBA")
                      << "\",\"repetitions_per_sample\":4,\"baseline_ms\":"
                      << block_times[block][0] << ",\"candidate_ms\":"
                      << block_times[block][1] << ",\"speedup_percent\":"
                      << 100.0 * (block_times[block][0] - block_times[block][1]) /
                             block_times[block][0]
                      << "}\n";
        }
        std::cout << "{\"kind\":\"complete\",\"status\":\"PASS\",\"scope\":\"exploratory operator screen only; three balanced blocks; no full-request claim\"}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
