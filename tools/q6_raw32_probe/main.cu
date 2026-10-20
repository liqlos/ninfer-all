#include "probe_api.h"
#include "ops/linear/gguf/ggml_bridge_internal.cuh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
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
        check(cudaGetDevice(&device), "device");
        cudaDeviceProp properties{};
        check(cudaGetDeviceProperties(&properties, device), "properties");
        return DeviceFacts{100 * properties.major + 10 * properties.minor,
                           properties.multiProcessorCount, properties.sharedMemPerBlockOptin};
    }();
    return facts;
}
}

extern "C" void ggml_abort(const char* file, int line, const char* format, ...) {
    std::fprintf(stderr, "ggml abort %s:%d %s\n", file, line, format);
    std::abort();
}

void ggml_cuda_error(const char* statement, const char* function, const char* file,
                     int line, const char* message) {
    std::fprintf(stderr, "CUDA %s %s %s:%d %s\n", statement, function, file, line, message);
    std::abort();
}

namespace {

#include "q6_native_producer.cuh"
#include "q6_independent_oracle.cuh"

constexpr int kColumns = 1280;
constexpr int kBlocks = 3;
constexpr int kRepetitions = 4;
constexpr std::size_t kGuardBytes = 64;

void check(cudaError_t status, const char* label) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(label) + ": " + cudaGetErrorString(status));
    }
}

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

struct Buffer {
    std::uint8_t* base = nullptr;
    void* p = nullptr;
    std::size_t bytes;
    explicit Buffer(std::size_t count) : bytes(count) {
        if (bytes) {
            check(cudaMalloc(&base, bytes + 2 * kGuardBytes), "cudaMalloc");
            p = base + kGuardBytes;
            check(cudaMemset(base, 0xA5, bytes + 2 * kGuardBytes), "initialize guards");
        }
    }
    ~Buffer() { if (base) { cudaFree(base); } }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void from_host(const void* source) {
        check(cudaMemcpy(p, source, bytes, cudaMemcpyHostToDevice), "copy to device");
    }
    void to_host(void* destination) const {
        check(cudaMemcpy(destination, p, bytes, cudaMemcpyDeviceToHost), "copy to host");
    }
    void guards() const {
        if (!base) { return; }
        std::array<std::uint8_t, 2 * kGuardBytes> actual{};
        check(cudaMemcpy(actual.data(), base, kGuardBytes, cudaMemcpyDeviceToHost), "prefix guard");
        check(cudaMemcpy(actual.data() + kGuardBytes, base + kGuardBytes + bytes,
                         kGuardBytes, cudaMemcpyDeviceToHost), "suffix guard");
        require(std::all_of(actual.begin(), actual.end(),
                            [](std::uint8_t value) { return value == 0xA5; }),
                "device allocation guard changed");
    }
};

std::string json_string(const std::string& value) {
    std::string result;
    for (const unsigned char ch : value) {
        if (ch == '"' || ch == '\\') { result.push_back('\\'); result.push_back(char(ch)); }
        else if (ch >= 0x20) { result.push_back(char(ch)); }
    }
    return result;
}

struct Case {
    const char* name;
    const char* origin;
    std::filesystem::path path;
    std::size_t file_bytes;
    int rows;
    int k;
    bool accumulate;
};

std::vector<std::uint8_t> read_prefix(const Case& c) {
    require(std::filesystem::is_regular_file(c.path) &&
            std::filesystem::file_size(c.path) == c.file_bytes,
            "required real Q6 fixture is missing or has the wrong exact size");
    std::vector<std::uint8_t> bytes(std::size_t(c.rows) * (c.k / 256) * 210);
    require(bytes.size() <= c.file_bytes, "fixture prefix exceeds the real object");
    std::ifstream input(c.path, std::ios::binary);
    input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    require(input.gcount() == static_cast<std::streamsize>(bytes.size()), "incomplete Q6 fixture read");
    return bytes;
}

struct HostQ8D4 {
    float scales[4];
    std::int8_t quants[128];
};
static_assert(sizeof(HostQ8D4) == 144);

int check_oracle(const Case& c, const std::vector<std::uint8_t>& weights,
                 const std::vector<__nv_bfloat16>& input, const std::vector<HostQ8D4>& packed,
                 const std::vector<float>& actual, int stride) {
    const std::array<int, 8> rows{0, 1, 127, 128, c.rows / 3, c.rows / 2, c.rows - 2, c.rows - 1};
    const std::array<int, 8> columns{0, 1, 127, 128, kColumns / 3, kColumns / 2,
                                   kColumns - 2, kColumns - 1};
    constexpr double epsilon = 8.0 * std::numeric_limits<float>::epsilon();
    for (const int column : columns) {
        for (int start = 0; start < c.k; start += 32) {
            double maximum = 0.0;
            for (int i = 0; i < 32; ++i) {
                maximum = std::max(maximum, std::abs(double(__bfloat162float(
                    input[std::size_t(column) * c.k + start + i]))));
            }
            const auto& block = packed[std::size_t(start / 128) * kColumns + column];
            const double scale = block.scales[(start % 128) / 32];
            const double expected_scale = maximum / 127.0;
            require(std::isfinite(scale) && scale >= 0.0 &&
                    std::abs(scale - expected_scale) <= epsilon * expected_scale,
                    "native Q6 D4 scale differs from independent BF16 oracle");
            for (int i = 0; i < 32; ++i) {
                const double value = __bfloat162float(input[std::size_t(column) * c.k + start + i]);
                const int quant = block.quants[start % 128 + i];
                require(quant >= -127 && std::abs(scale * quant - value) <=
                        0.5 * scale + epsilon * maximum, "native Q6 D4 quant differs from BF16 oracle");
            }
        }
        for (const int row : rows) {
            const auto* weight = weights.data() + std::size_t(row) * (c.k / 256) * 210;
            double expected = 0.0;
            for (int i = 0; i < c.k; ++i) {
                const auto& block = packed[std::size_t(i / 128) * kColumns + column];
                const double value = double(block.scales[(i % 128) / 32]) * double(block.quants[i % 128]);
                expected += q6_runtime_weight_value(weight, i) * value;
            }
            const double got = actual[std::size_t(column) * stride + row];
            require(std::isfinite(expected) && std::isfinite(got) &&
                    std::abs(got - expected) <= 0.02 + 2.0e-4 * std::abs(expected),
                    "Q6 output differs from independent Q6/Q8 FP64 oracle");
        }
    }
    return int(rows.size() * columns.size());
}

bool invoke(Q6Raw32RunFunction run, const Q6Raw32Plan& plan, const Case& c,
            const Buffer& weight, const Buffer& activation, float* fp32,
            __nv_bfloat16* bf16, bool accumulate, int stride, float* fixup) {
    return run(&plan, static_cast<const char*>(weight.p), c.k / 256, c.rows, c.k,
        static_cast<const int*>(activation.p), MMQ_Q8_1_DS_LAYOUT_D4, kColumns, fp32,
        bf16, accumulate, stride, fixup, nullptr);
}

void check_api(Q6Raw32RunFunction run, const Q6Raw32Plan& plan, const Case& c,
               const Buffer& weight, const Buffer& activation, Buffer& fp32, Buffer& bf16,
               int stride, const std::vector<float>& expected_fp32,
               const std::vector<__nv_bfloat16>& expected_bf16) {
    const auto* w = static_cast<const char*>(weight.p);
    const auto* a = static_cast<const int*>(activation.p);
    auto* f = static_cast<float*>(fp32.p);
    auto* b = static_cast<__nv_bfloat16*>(bf16.p);
    const auto rejected = [&](const Q6Raw32Plan* p, const char* wx, int row_blocks, int k,
                              const int* ax, int layout, float* fx, __nv_bfloat16* bx,
                              bool accumulate, int output_stride) {
        require(!run(p, wx, row_blocks, c.rows, k, ax, layout, kColumns, fx, bx,
                     accumulate, output_stride, nullptr, nullptr), "invalid probe API was admitted");
    };
    rejected(nullptr, w, c.k / 256, c.k, a, 0, nullptr, b, false, stride);
    rejected(&plan, nullptr, c.k / 256, c.k, a, 0, nullptr, b, false, stride);
    rejected(&plan, w, c.k / 256, c.k, nullptr, 0, nullptr, b, false, stride);
    rejected(&plan, w, c.k / 256, c.k, a, 0, nullptr, nullptr, false, stride);
    rejected(&plan, w, c.k / 256, c.k, a, 0, f, b, false, stride);
    rejected(&plan, w, c.k / 256, c.k - 256, a, 0, nullptr, b, false, stride);
    rejected(&plan, w, c.k / 256 + 1, c.k, a, 0, nullptr, b, false, stride);
    rejected(&plan, w, c.k / 256, c.k, a, 0, nullptr, b, false, c.rows - 1);
    rejected(&plan, w, c.k / 256, c.k, a, MMQ_Q8_1_DS_LAYOUT_DS4, nullptr, b, false, stride);
    rejected(&plan, w, c.k / 256, c.k, a, 0, f, nullptr, true, stride);
    rejected(&plan, w + 1, c.k / 256, c.k, a, 0, nullptr, b, false, stride);
    rejected(&plan, w, c.k / 256, c.k, a, 0, nullptr,
             reinterpret_cast<__nv_bfloat16*>(weight.p), false, stride);
    rejected(&plan, w, c.k / 256, c.k, a, 0, nullptr,
             reinterpret_cast<__nv_bfloat16*>(activation.p), false, stride);
    auto bad_plan = plan;
    bad_plan.J = 64;
    rejected(&bad_plan, w, c.k / 256, c.k, a, 0, nullptr, b, false, stride);
    bad_plan = plan;
    bad_plan.fixup = true;
    rejected(&bad_plan, w, c.k / 256, c.k, a, 0, nullptr, b, false, stride);
    check(cudaDeviceSynchronize(), "API guard sync");
    std::vector<float> actual_fp32(expected_fp32.size());
    std::vector<__nv_bfloat16> actual_bf16(expected_bf16.size());
    fp32.to_host(actual_fp32.data());
    bf16.to_host(actual_bf16.data());
    require(std::memcmp(actual_fp32.data(), expected_fp32.data(), fp32.bytes) == 0 &&
            std::memcmp(actual_bf16.data(), expected_bf16.data(), bf16.bytes) == 0,
            "rejected API call modified an output");
}

double time_sample(Q6Raw32RunFunction run, const Q6Raw32Plan& plan, const Case& c,
                   const Buffer& weight, const Buffer& activation, Buffer& output,
                   const Buffer& initial, int stride) {
    check(cudaMemcpyAsync(output.p, initial.p, output.bytes, cudaMemcpyDeviceToDevice, nullptr),
          "reset carry outside event region");
    cudaEvent_t begin = nullptr, end = nullptr;
    check(cudaEventCreate(&begin), "create begin event");
    check(cudaEventCreate(&end), "create end event");
    check(cudaEventRecord(begin, nullptr), "record begin event");
    for (int repeat = 0; repeat < kRepetitions; ++repeat) {
        require(invoke(run, plan, c, weight, activation, nullptr,
                       static_cast<__nv_bfloat16*>(output.p), c.accumulate, stride, nullptr),
                "timed Q6 call rejected");
    }
    check(cudaEventRecord(end, nullptr), "record end event");
    check(cudaEventSynchronize(end), "synchronize timed sample");
    float milliseconds = 0.0f;
    check(cudaEventElapsedTime(&milliseconds, begin, end), "read elapsed time");
    check(cudaEventDestroy(end), "destroy end event");
    check(cudaEventDestroy(begin), "destroy begin event");
    return double(milliseconds) / kRepetitions;
}

void run_case(const Case& c, const std::string& tag) {
    const auto host_weight = read_prefix(c);
    const auto host_input = make_input(c.k, kColumns, 0xA119u + kColumns + c.rows);
    Buffer weight(host_weight.size());
    Buffer input(host_input.size() * sizeof(__nv_bfloat16));
    const std::size_t activation_count = std::size_t(c.k / 128) * kColumns;
    Buffer activation((activation_count + 128) * sizeof(HostQ8D4));
    weight.from_host(host_weight.data());
    input.from_host(host_input.data());
    check(cudaMemset(activation.p, 0, activation.bytes), "clear native activation slack");
    require(mmq_get_q8_1_ds_layout(GGML_TYPE_Q6_K) == MMQ_Q8_1_DS_LAYOUT_D4 &&
            mmq_get_q8_1_ds_layout(GGML_TYPE_Q4_K) == MMQ_Q8_1_DS_LAYOUT_DS4,
            "source-defined Q6 D4 / Q4 DS4 contract changed");
    quantize_matrix_kernel<MMQ_Q8_1_DS_LAYOUT_D4>
        <<<dim3(kColumns, c.k / (4 * kMatrixQuantizeThreads), 1), kMatrixQuantizeThreads>>>(
            static_cast<const __nv_bfloat16*>(input.p), nullptr,
            static_cast<block_q8_1_mmq*>(activation.p), c.k, kColumns);
    check(cudaGetLastError(), "native BF16 to Q8 D4 producer");
    Q6Raw32Plan off{}, on{};
    require(q6_raw32_off_plan(c.rows, kColumns, &off) && q6_raw32_on_plan(c.rows, kColumns, &on),
            "native Q6 J128 plan unavailable");
    require(off.J == on.J && off.I == on.I && off.threads == on.threads && off.shared == on.shared &&
            off.blocks == on.blocks && off.row_tiles == on.row_tiles &&
            off.column_tiles == on.column_tiles && off.stream_k == on.stream_k &&
            !off.fixup && !on.fixup, "shortscreen requires matching native no-fixup plans");
    const int stride = c.rows + 19;
    const std::size_t count = std::size_t(stride) * kColumns;
    std::vector<float> initial_fp32(count);
    std::memset(initial_fp32.data(), 0xA5, count * sizeof(float));
    std::vector<__nv_bfloat16> initial_bf16(count);
    for (int column = 0; column < kColumns; ++column) {
        for (int row = 0; row < stride; ++row) {
            const std::size_t at = std::size_t(column) * stride + row;
            if (row < c.rows) {
                initial_fp32[at] = std::numeric_limits<float>::quiet_NaN();
                const float carry = (at & 1 ? -1.0f : 1.0f) * float(1 + at % 17) / 32.0f;
                initial_bf16[at] = __float2bfloat16(c.accumulate ? carry : -777.0f);
            } else {
                initial_bf16[at] = __float2bfloat16(123.0f);
            }
        }
    }
    Buffer off_fp32(count * sizeof(float)), on_fp32(count * sizeof(float));
    Buffer off_bf16(count * sizeof(__nv_bfloat16)), on_bf16(count * sizeof(__nv_bfloat16));
    Buffer initial(count * sizeof(__nv_bfloat16));
    off_fp32.from_host(initial_fp32.data());
    on_fp32.from_host(initial_fp32.data());
    off_bf16.from_host(initial_bf16.data());
    on_bf16.from_host(initial_bf16.data());
    initial.from_host(initial_bf16.data());
    require(invoke(q6_raw32_off_run, off, c, weight, activation,
                   static_cast<float*>(off_fp32.p), nullptr, false, stride, nullptr) &&
            invoke(q6_raw32_on_run, on, c, weight, activation,
                   static_cast<float*>(on_fp32.p), nullptr, false, stride, nullptr),
            "FP32 call rejected");
    require(invoke(q6_raw32_off_run, off, c, weight, activation, nullptr,
                   static_cast<__nv_bfloat16*>(off_bf16.p), c.accumulate, stride, nullptr) &&
            invoke(q6_raw32_on_run, on, c, weight, activation, nullptr,
                   static_cast<__nv_bfloat16*>(on_bf16.p), c.accumulate, stride, nullptr),
            "BF16 call rejected");
    check(cudaDeviceSynchronize(), "correctness sync");
    std::vector<float> baseline_fp32(count), candidate_fp32(count);
    std::vector<__nv_bfloat16> baseline_bf16(count), candidate_bf16(count);
    off_fp32.to_host(baseline_fp32.data());
    on_fp32.to_host(candidate_fp32.data());
    off_bf16.to_host(baseline_bf16.data());
    on_bf16.to_host(candidate_bf16.data());
    require(std::memcmp(baseline_fp32.data(), candidate_fp32.data(), off_fp32.bytes) == 0,
            "full FP32 output or padding differs");
    require(std::memcmp(baseline_bf16.data(), candidate_bf16.data(), off_bf16.bytes) == 0,
            "full BF16 output or padding differs");
    std::size_t carry_sensitive_outputs = 0;
    for (int column = 0; column < kColumns; ++column) {
        for (int row = 0; row < stride; ++row) {
            const std::size_t at = std::size_t(column) * stride + row;
            if (row < c.rows) {
                require(std::isfinite(baseline_fp32[at]), "unwritten or non-finite FP32 output");
                const auto expected = __float2bfloat16(baseline_fp32[at] +
                    (c.accumulate ? __bfloat162float(initial_bf16[at]) : 0.0f));
                const auto store_only = __float2bfloat16(baseline_fp32[at]);
                carry_sensitive_outputs += std::memcmp(&expected, &store_only, sizeof(expected)) != 0;
                require(std::isfinite(__bfloat162float(candidate_bf16[at])) &&
                        std::memcmp(&expected, &candidate_bf16[at], sizeof(expected)) == 0,
                        "BF16 does not equal rounded FP32 plus original nonzero carry");
            } else {
                require(std::memcmp(&initial_fp32[at], &candidate_fp32[at], sizeof(float)) == 0 &&
                        std::memcmp(&initial_bf16[at], &candidate_bf16[at], sizeof(__nv_bfloat16)) == 0,
                        "output row canary changed");
            }
        }
    }
    require(!c.accumulate || carry_sensitive_outputs > 0, "Accumulate fixture has no carry-sensitive output");
    std::vector<HostQ8D4> packed(activation_count + 128);
    activation.to_host(packed.data());
    const int samples = check_oracle(c, host_weight, host_input, packed, candidate_fp32, stride);
    check_api(q6_raw32_off_run, off, c, weight, activation, off_fp32, off_bf16,
              stride, baseline_fp32, baseline_bf16);
    check_api(q6_raw32_on_run, on, c, weight, activation, on_fp32, on_bf16,
              stride, candidate_fp32, candidate_bf16);
    for (const Buffer* buffer : {&weight, &input, &activation, &off_fp32, &on_fp32,
                                  &off_bf16, &on_bf16, &initial}) { buffer->guards(); }
    const char* mode = c.accumulate ? "accumulate" : "store";
    std::cout << "{\"kind\":\"correctness\",\"status\":\"PASS\",\"case\":\"" << c.name
              << "\",\"rows\":" << c.rows << ",\"k\":" << c.k << ",\"columns\":" << kColumns
              << ",\"output_mode\":\"" << mode << "\",\"weight_origin\":\"" << c.origin
              << "\",\"fixture_prefix_bytes\":" << host_weight.size()
              << ",\"fixture_file_bytes\":" << c.file_bytes
              << ",\"activation\":\"native_Q6_D4_from_seeded_BF16\",\"producer_oracle\":\"PASS\""
              << ",\"exact_fp32_all_outputs\":true,\"exact_bf16_all_outputs\":true"
              << ",\"accumulate_matches_fp32_carry\":true,\"output_canaries\":true"
              << ",\"carry_sensitive_outputs\":" << carry_sensitive_outputs
              << ",\"independent_fp64_samples\":" << samples << ",\"invalid_api_guards\":\"PASS\""
              << ",\"J\":" << off.J << ",\"I\":" << off.I << ",\"blocks\":" << off.blocks
              << ",\"threads\":" << off.threads << ",\"shared_bytes\":" << off.shared
              << ",\"fixup\":false,\"store_fusion\":true,\"q6_a_x4\":false"
              << ",\"tag\":\"" << json_string(tag) << "\"}\n" << std::flush;
    constexpr std::array<int, 4> abba{0, 1, 1, 0}, baab{1, 0, 0, 1};
    for (int block = 0; block < kBlocks; ++block) {
        const auto& order = block % 2 == 0 ? abba : baab;
        std::array<double, 4> times{};
        double baseline = 0.0, candidate = 0.0;
        for (int position = 0; position < 4; ++position) {
            const bool enabled = order[position] != 0;
            times[position] = time_sample(enabled ? q6_raw32_on_run : q6_raw32_off_run,
                enabled ? on : off, c, weight, activation, enabled ? on_bf16 : off_bf16, initial, stride);
            (enabled ? candidate : baseline) += 0.5 * times[position];
        }
        std::cout << std::setprecision(10)
                  << "{\"kind\":\"timing\",\"case\":\"" << c.name << "\",\"rows\":" << c.rows
                  << ",\"k\":" << c.k << ",\"columns\":" << kColumns << ",\"output_mode\":\"" << mode
                  << "\",\"block\":" << block << ",\"order\":\"" << (block % 2 == 0 ? "ABBA" : "BAAB")
                  << "\",\"repetitions_per_sample\":" << kRepetitions
                  << ",\"baseline_ms\":" << baseline << ",\"candidate_ms\":" << candidate
                  << ",\"samples_ms\":[" << times[0] << ',' << times[1] << ',' << times[2] << ',' << times[3]
                  << "],\"operator_time_decrease_percent\":" << 100.0 * (baseline - candidate) / baseline
                  << ",\"carry_reset_outside_events\":true,\"exploratory_only\":true}\n" << std::flush;
    }
    off_bf16.guards();
    on_bf16.guards();
}

}

int main(int argc, char** argv) {
    try {
        require(argc == 3 || argc == 4, "usage: q6-raw32-probe <operators-dir> <weights-dir> [tag]");
        const auto& device = ninfer::ops::gguf::detail::device_facts();
        require(device.cc == 860, "Q6 raw32 probe requires SM86");
        const std::filesystem::path operators(argv[1]), weights(argv[2]);
        const std::string tag = argc == 4 ? argv[3] : "screen";
        const std::array<Case, 3> cases{{
            {"down-k17408-n5120", "actual_full_weight_000143", operators / "q6-full-k17408.bin",
             73113600, 5120, 17408, true},
            {"head-prefix-at-gdn-geometry", "actual_head_000131_prefix_not_GDN_tensor",
             weights / "q6-head-full-k5120.bin", 1042944000, 10240, 5120, false},
            {"head-prefix-at-attention-v-geometry", "actual_head_000131_prefix_not_attention_V_tensor",
             weights / "q6-head-full-k5120.bin", 1042944000, 1024, 5120, false}
        }};
        for (const auto& c : cases) { run_case(c, tag); }
        std::cout << "{\"kind\":\"complete\",\"status\":\"PASS\",\"cases\":3,\"timing_blocks\":9,"
                     "\"scope\":\"short_real_weight_operator_screen\",\"performance_claim\":false}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
