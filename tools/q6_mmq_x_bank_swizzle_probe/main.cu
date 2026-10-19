bool q6_mmq_x_bank_swizzle_layout_test();
#include "probe_api.h"

#include "ggml-cuda/common.cuh"
#include "ops/linear/gguf/ggml_bridge_internal.cuh"
#include <cstdio>
#include <cstdlib>

#include <cuda_fp16.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::ops::gguf::detail {
const DeviceFacts& device_facts() {
    static const DeviceFacts facts = [] {
        int device; check(cudaGetDevice(&device), "device");
        cudaDeviceProp p; check(cudaGetDeviceProperties(&p, device), "properties");
        return DeviceFacts{100*p.major+10*p.minor,p.multiProcessorCount,p.sharedMemPerBlockOptin};
    }();
    return facts;
}
}
extern "C" void ggml_abort(const char* file, int line, const char* fmt, ...) {
    std::fprintf(stderr,"ggml abort %s:%d %s\n",file,line,fmt); std::abort();
}
void ggml_cuda_error(const char* stmt,const char* func,const char* file,int line,const char* msg) {
    std::fprintf(stderr,"CUDA %s %s %s:%d %s\n",stmt,func,file,line,msg);std::abort();
}

namespace {

void check(cudaError_t status, const char* label) {
    if (status != cudaSuccess) { throw std::runtime_error(std::string(label) + ": " + cudaGetErrorString(status)); }
}

struct Buffer {
    void* p = nullptr;
    std::size_t bytes = 0;
    explicit Buffer(std::size_t size) : bytes(size) {
        if (size != 0) { check(cudaMalloc(&p, size), "cudaMalloc"); }
    }
    ~Buffer() { if (p != nullptr) { cudaFree(p); } }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};

std::vector<std::uint8_t> read_bytes(const std::filesystem::path& path, std::size_t bytes) {
    std::ifstream input(path, std::ios::binary);
    if (!input) { throw std::runtime_error("cannot open fixture: " + path.string()); }
    std::vector<std::uint8_t> result(bytes);
    input.read(reinterpret_cast<char*>(result.data()), static_cast<std::streamsize>(bytes));
    if (input.gcount() != static_cast<std::streamsize>(bytes)) {
        throw std::runtime_error("fixture is shorter than requested prefix: " + path.string());
    }
    return result;
}

float half_value(const std::uint8_t* bytes) {
    __half value;
    std::memcpy(&value, bytes, sizeof(value));
    return __half2float(value);
}

double q6_value(const std::uint8_t* row, int k_index) {
    constexpr int block_bytes = 210;
    const int block_index = k_index / 256;
    const int in_block = k_index % 256;
    const int ip = in_block / 128;
    const int slot = in_block % 128;
    const int il = slot % 32;
    const int which = slot / 32;
    const int ql_index = 64 * ip + il + ((which & 1) ? 32 : 0);
    const int byte = row[block_index * block_bytes + ql_index];
    const int ql = (which < 2) ? (byte & 0x0F) : (byte >> 4);
    const int qh = row[block_index * block_bytes + 128 + 32 * ip + il];
    const int q = ql | (((qh >> (2 * which)) & 3) << 4);
    const auto* scales = reinterpret_cast<const std::int8_t*>(row + block_index * block_bytes + 192);
    const int scale_index = 8 * ip + il / 16 + 2 * which;
    const float d = half_value(row + block_index * block_bytes + 208);
    return double(d) * double(scales[scale_index]) * double(q - 32);
}

std::string json_string(const std::string& value) {
    std::string result;
    for (const unsigned char ch : value) {
        if (ch == '"' || ch == '\\') { result.push_back('\\'); result.push_back(char(ch)); }
        else if (ch >= 0x20) { result.push_back(char(ch)); }
    }
    return result;
}

struct Case {
    std::string name;
    std::filesystem::path path;
    int rows;
    int k;
};

void call_variant(bool on, const Q6MmqProbePlan& plan, const char* weights, int rows, int k,
                  const int* activation, int columns, float* out, __nv_bfloat16* bf16,
                  std::int64_t stride, float* fixup, cudaStream_t stream) {
    const int row_blocks = k / 256;
    const int stride_row_blocks = row_blocks;
    if (on) {
        q6_mmq_x_bank_swizzle_on_run(&plan, weights, stride_row_blocks, rows, k, activation, columns,
                              out, bf16, stride, fixup, stream);
    } else {
        q6_mmq_a_x4_off_run(&plan, weights, stride_row_blocks, rows, k, activation, columns,
                               out, bf16, stride, fixup, stream);
    }
}

bool exact_float_outputs(const std::vector<float>& off, const std::vector<float>& on,
                         int rows, int columns, int stride) {
    for (int column = 0; column < columns; ++column) {
        for (int row = 0; row < rows; ++row) {
            const std::size_t at = std::size_t(column) * stride + row;
            if (std::memcmp(&off[at], &on[at], sizeof(float)) != 0) { return false; }
        }
        for (int row = rows; row < stride; ++row) {
            const std::size_t at = std::size_t(column) * stride + row;
            if (std::memcmp(&off[at], &on[at], sizeof(float)) != 0) { return false; }
            std::uint32_t bits = 0;
            std::memcpy(&bits, &off[at], sizeof(bits));
            if (bits != 0xA5A5A5A5u) { return false; }
        }
    }
    return true;
}

bool exact_bf16_outputs(const std::vector<__nv_bfloat16>& off,
                        const std::vector<__nv_bfloat16>& on, int rows, int columns, int stride) {
    for (int column = 0; column < columns; ++column) {
        for (int row = 0; row < rows; ++row) {
            const std::size_t at = std::size_t(column) * stride + row;
            if (std::memcmp(&off[at], &on[at], sizeof(__nv_bfloat16)) != 0) { return false; }
        }
        for (int row = rows; row < stride; ++row) {
            const std::size_t at = std::size_t(column) * stride + row;
            std::uint16_t bits = 0;
            std::memcpy(&bits, &off[at], sizeof(bits));
            if (bits != 0xA5A5u || std::memcmp(&off[at], &on[at], sizeof(__nv_bfloat16)) != 0) {
                return false;
            }
        }
    }
    return true;
}

bool check_oracle(const Case& c, const std::vector<std::uint8_t>& weights,
                  const std::vector<block_q8_1_mmq>& activation, const std::vector<float>& output,
                  int columns, int stride, double& max_abs, double& max_rel) {
    const std::array<int, 4> row_ids{0, c.rows / 3, c.rows / 2, c.rows - 1};
    const std::array<int, 4> column_ids{0, columns / 3, columns / 2, columns - 1};
    max_abs = 0.0;
    max_rel = 0.0;
    for (const int row : row_ids) {
        const auto* weight_row = weights.data() + std::size_t(row) * (c.k / 256) * 210;
        for (const int column : column_ids) {
            double expected = 0.0;
            for (int ki = 0; ki < c.k; ++ki) {
                const int qblock = ki / 128;
                const int in_block = ki % 128;
                const auto& block = activation[std::size_t(qblock) * columns + column];
                const double x = double(block.d4[in_block / 32]) * double(block.qs[in_block]);
                expected += q6_value(weight_row, ki) * x;
            }
            const double actual = double(output[std::size_t(column) * stride + row]);
            const double error = std::abs(expected - actual);
            const double relative = error / std::max(std::abs(expected), 1.0e-12);
            max_abs = std::max(max_abs, error);
            max_rel = std::max(max_rel, relative);
            if (error > 0.02 + 2.0e-4 * std::abs(expected)) { return false; }
        }
    }
    return true;
}

double time_four(bool on, const Q6MmqProbePlan& plan, const char* weights, int rows, int k,
                 const int* activation, int columns, float* out, float* fixup,
                 std::int64_t stride, cudaStream_t stream) {
    cudaEvent_t start = nullptr;
    cudaEvent_t end = nullptr;
    check(cudaEventCreate(&start), "cudaEventCreate start");
    check(cudaEventCreate(&end), "cudaEventCreate end");
    check(cudaEventRecord(start, stream), "cudaEventRecord start");
    for (int repeat = 0; repeat < 4; ++repeat) {
        call_variant(on, plan, weights, rows, k, activation, columns, out,
                     plan.fixup ? nullptr : reinterpret_cast<__nv_bfloat16*>(out),
                     stride, fixup, stream);
    }
    check(cudaEventRecord(end, stream), "cudaEventRecord end");
    check(cudaEventSynchronize(end), "cudaEventSynchronize");
    float milliseconds = 0.0f;
    check(cudaEventElapsedTime(&milliseconds, start, end), "cudaEventElapsedTime");
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    return double(milliseconds) / 4.0;
}

void run_case(const Case& c, int columns, std::ostream& jsonl) {
    const std::size_t row_bytes = std::size_t(c.k / 256) * 210;
    auto host_weights = read_bytes(c.path, std::size_t(c.rows) * row_bytes);
    const std::size_t activation_count = std::size_t(columns) * (c.k / 128);
    std::vector<block_q8_1_mmq> host_activation(activation_count + 128);
    for (std::size_t b = 0; b < activation_count; ++b) {
        for (int j=0;j<4;++j) host_activation[b].d4[j] = 1.0f/64;
        for (int j=0;j<128;++j) host_activation[b].qs[j] = std::int8_t(int((b*37+j*17)%127)-63);
    }
    Buffer device_weights(host_weights.size());
    Buffer device_activation(host_activation.size()*sizeof(block_q8_1_mmq));
    check(cudaMemcpy(device_weights.p, host_weights.data(), device_weights.bytes, cudaMemcpyHostToDevice), "copy weights");
    check(cudaMemcpy(device_activation.p,host_activation.data(),device_activation.bytes,cudaMemcpyHostToDevice),"copy activation");

    Q6MmqProbePlan off_plan{};
    Q6MmqProbePlan on_plan{};
    if (!q6_mmq_a_x4_off_plan(c.rows, columns, &off_plan) ||
        !q6_mmq_x_bank_swizzle_on_plan(c.rows, columns, &on_plan)) {
        throw std::runtime_error("native Q6 MMQ plan unavailable");
    }
    if (off_plan.J != 128 || on_plan.J != 128 || off_plan.I != on_plan.I ||
        off_plan.fixup != on_plan.fixup || off_plan.blocks != on_plan.blocks) {
        throw std::runtime_error("native selector mismatch or unexpected J/I plan");
    }
    const int stride = c.rows + 13;
    const std::size_t output_count = std::size_t(stride) * columns;
    Buffer off_f32(std::size_t(output_count) * sizeof(float));
    Buffer on_f32(std::size_t(output_count) * sizeof(float));
    const std::size_t fixup_bytes = off_plan.fixup
        ? std::size_t(off_plan.blocks) * off_plan.J * off_plan.I * sizeof(float) : 0;
    Buffer off_fixup(fixup_bytes);
    Buffer on_fixup(fixup_bytes);
    Buffer off_bf16(std::size_t(output_count) * sizeof(__nv_bfloat16));
    Buffer on_bf16(std::size_t(output_count) * sizeof(__nv_bfloat16));
    const auto* weight_data = static_cast<const char*>(device_weights.p);
    const auto* activation_data = static_cast<const int*>(device_activation.p);
    const auto initialize = [&](void* target, std::size_t bytes) {
        check(cudaMemset(target, 0xA5, bytes), "initialize output guards");
    };
    initialize(off_f32.p, off_f32.bytes);
    initialize(on_f32.p, on_f32.bytes);
    call_variant(false, off_plan, weight_data, c.rows, c.k, activation_data, columns,
                 static_cast<float*>(off_f32.p), nullptr, stride, static_cast<float*>(off_fixup.p), nullptr);
    call_variant(true, on_plan, weight_data, c.rows, c.k, activation_data, columns,
                 static_cast<float*>(on_f32.p), nullptr, stride, static_cast<float*>(on_fixup.p), nullptr);
    check(cudaDeviceSynchronize(), "FP32 correctness sync");
    std::vector<float> host_off_f32(output_count);
    std::vector<float> host_on_f32(output_count);
    check(cudaMemcpy(host_off_f32.data(), off_f32.p, off_f32.bytes, cudaMemcpyDeviceToHost), "copy OFF FP32");
    check(cudaMemcpy(host_on_f32.data(), on_f32.p, on_f32.bytes, cudaMemcpyDeviceToHost), "copy ON FP32");
    const bool exact_f32 = exact_float_outputs(host_off_f32, host_on_f32, c.rows, columns, stride);
    double max_abs = 0.0;
    double max_rel = 0.0;
    const bool oracle = check_oracle(c, host_weights, host_activation, host_on_f32,
                                     columns, stride, max_abs, max_rel);
    if (!exact_f32 || !oracle) { throw std::runtime_error("FP32 equality/oracle/guard check failed"); }

    bool bf16_checked = false;
    bool exact_bf16 = false;
    if (!off_plan.fixup) {
        initialize(off_bf16.p, off_bf16.bytes);
        initialize(on_bf16.p, on_bf16.bytes);
        call_variant(false, off_plan, weight_data, c.rows, c.k, activation_data, columns,
                     nullptr, static_cast<__nv_bfloat16*>(off_bf16.p), stride, nullptr, nullptr);
        call_variant(true, on_plan, weight_data, c.rows, c.k, activation_data, columns,
                     nullptr, static_cast<__nv_bfloat16*>(on_bf16.p), stride, nullptr, nullptr);
        check(cudaDeviceSynchronize(), "BF16 correctness sync");
        std::vector<__nv_bfloat16> host_off_bf16(output_count);
        std::vector<__nv_bfloat16> host_on_bf16(output_count);
        check(cudaMemcpy(host_off_bf16.data(), off_bf16.p, off_bf16.bytes, cudaMemcpyDeviceToHost), "copy OFF BF16");
        check(cudaMemcpy(host_on_bf16.data(), on_bf16.p, on_bf16.bytes, cudaMemcpyDeviceToHost), "copy ON BF16");
        exact_bf16 = exact_bf16_outputs(host_off_bf16, host_on_bf16, c.rows, columns, stride);
        for (int column = 0; column < columns && exact_bf16; ++column) {
            for (int row = 0; row < c.rows; ++row) {
                const std::size_t at = std::size_t(column) * stride + row;
                const __nv_bfloat16 expected = __float2bfloat16(host_on_f32[at]);
                if (std::memcmp(&expected, &host_on_bf16[at], sizeof(expected)) != 0) {
                    exact_bf16 = false;
                    break;
                }
            }
        }
        bf16_checked = true;
        if (!exact_bf16) { throw std::runtime_error("BF16 equality/guard check failed"); }
    }

    jsonl << std::setprecision(10)
          << "{\"kind\":\"correctness\",\"fixture\":\"" << json_string(c.name)
          << "\",\"rows\":" << c.rows << ",\"k\":" << c.k << ",\"columns\":" << columns
          << ",\"activation_layout\":\"Q8_1_D4\",\"activation_slack_bytes\":"
          << 128 * sizeof(block_q8_1_mmq)
          << ",\"J\":" << off_plan.J << ",\"I\":" << off_plan.I
          << ",\"stream_k\":" << (off_plan.stream_k ? "true" : "false")
          << ",\"fixup\":" << (off_plan.fixup ? "true" : "false")
          << ",\"fp32_off_on_bitwise_equal\":" << (exact_f32 ? "true" : "false")
          << ",\"bf16_checked\":" << (bf16_checked ? "true" : "false")
          << ",\"bf16_off_on_bitwise_equal\":" << (exact_bf16 ? "true" : "false")
          << ",\"fp64_oracle_samples\":16,\"fp64_max_abs_error\":" << max_abs
          << ",\"fp64_max_rel_error\":" << max_rel << "}\n";

    std::vector<std::array<double, 2>> block_times(20, {0.0, 0.0});
    const std::array<int, 4> abba{0, 1, 1, 0};
    const std::array<int, 4> baab{1, 0, 0, 1};
    for (int block = 0; block < 20; ++block) {
        const auto& order = (block % 2 == 0) ? abba : baab;
        for (int position = 0; position < 4; ++position) {
            const bool on = order[position] != 0;
            const double milliseconds = time_four(on, on ? on_plan : off_plan, weight_data,
                c.rows, c.k, activation_data, columns,
                static_cast<float*>(on ? on_f32.p : off_f32.p),
                static_cast<float*>(on ? on_fixup.p : off_fixup.p), stride, nullptr);
            block_times[block][on ? 1 : 0] += milliseconds / 2.0;
            jsonl << std::setprecision(10)
                  << "{\"kind\":\"timing\",\"fixture\":\"" << json_string(c.name)
                  << "\",\"rows\":" << c.rows << ",\"k\":" << c.k << ",\"columns\":" << columns
                  << ",\"block\":" << block << ",\"order\":\"" << ((block % 2 == 0) ? "ABBA" : "BAAB")
                  << "\",\"position\":" << position << ",\"variant\":\"" << (on ? "ON" : "OFF")
                  << "\",\"repetitions\":4,\"milliseconds_per_call\":" << milliseconds << "}\n";
        }
    }
    std::vector<double> paired_improvements;
    paired_improvements.reserve(block_times.size());
    for (int block = 0; block < 20; ++block) {
        const double off = block_times[block][0];
        const double on = block_times[block][1];
        paired_improvements.push_back(100.0 * (off - on) / off);
    }
    std::sort(paired_improvements.begin(), paired_improvements.end());
    jsonl << std::setprecision(10)
          << "{\"kind\":\"summary\",\"fixture\":\"" << json_string(c.name)
          << "\",\"rows\":" << c.rows << ",\"k\":" << c.k << ",\"columns\":" << columns
          << ",\"blocks\":20,\"repetitions_per_sample\":4,\"median_paired_improvement_percent\":"
          << (paired_improvements[9] + paired_improvements[10]) / 2.0
          << ",\"exploratory_only\":true}\n";
}

}

int main(int argc, char** argv) {
    try {
        std::ostream& jsonl = std::cout;
        int device = 0;
        check(cudaGetDevice(&device), "cudaGetDevice");
        cudaDeviceProp properties{};
        check(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
        if (properties.major != 8 || properties.minor != 6) { return 77; }
        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;
        check(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
        jsonl << "{\"kind\":\"environment\",\"device\":\"" << json_string(properties.name)
              << "\",\"compute_capability\":\"8.6\",\"sm_count\":" << properties.multiProcessorCount
              << ",\"free_bytes_before\":" << free_bytes << ",\"total_bytes\":" << total_bytes
              << ",\"feature\":\"Q6_MMQ_X_BANK_SWIZZLE\",\"store_fusion\":true}\n";
        if (!q6_mmq_x_bank_swizzle_layout_test()) throw std::runtime_error("Q6 shared X bank-swizzle layout mismatch");
        jsonl << "{\"kind\":\"layout_mapping\",\"status\":\"PASS\",\"lanes\":32,\"row_groups\":8,\"k_groups\":16}\n";
        const std::array<Case, 2> cases{{
            {"q6-full-k17408", "/extra/q6-full-k17408.bin", 5120, 17408},
            {"q6-head-k5120", "/fixtures/q6-head-full-k5120.bin", 1024, 5120}
        }};
        for (const auto& c : cases) {
            for (const int columns : {1280, 1281}) { run_case(c, columns, jsonl); }
        }
        jsonl.flush();
        jsonl << "{\"kind\":\"complete\",\"status\":\"PASS\",\"scope\":\"exploratory real-weight operator; synthetic Q8 D4 activations\"}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
