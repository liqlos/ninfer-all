#include "probe_api.h"
#include "codec.h"
#include "q6_word_group_guards.cuh"
#include "q6wg_host_receipt.h"
#include "ops/linear/gguf/ggml_bridge_internal.cuh"

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

static_assert(sizeof(void*) == 8 && sizeof(std::size_t) == 8);
static_assert(sizeof(Q6WGView) == 40 && alignof(Q6WGView) == 8);
static_assert(offsetof(Q6WGView, row_begin) == 24 && offsetof(Q6WGView, storage) == 36);
static_assert(sizeof(Q6WGOutput) == 32 && offsetof(Q6WGOutput, mode) == 24);
static_assert(sizeof(Q6WGMatrixPlan) == 72);
static_assert(offsetof(Q6WGMatrixPlan, shared) == 24 &&
              offsetof(Q6WGMatrixPlan, activation_bytes) == 56);
static_assert(sizeof(block_q8_1_mmq) == 144);

extern "C" void ggml_abort(const char* file, int line, const char* fmt, ...) {
    std::fprintf(stderr, "ggml abort at %s:%d: ", file, line);
    va_list args;
    va_start(args, fmt);
    std::vfprintf(stderr, fmt, args);
    va_end(args);
    std::fputc('\n', stderr);
    std::abort();
}

namespace ninfer::ops::gguf::detail {

const DeviceFacts& device_facts() {
    static const DeviceFacts facts = [] {
        int device = 0;
        check(cudaGetDevice(&device), "host device");
        cudaDeviceProp properties{};
        check(cudaGetDeviceProperties(&properties, device), "host device properties");
        return DeviceFacts{100 * properties.major + 10 * properties.minor,
                           properties.multiProcessorCount,
                           properties.sharedMemPerBlockOptin};
    }();
    return facts;
}

}

namespace q6wg_host_native {
#include "q6wg_native_vector_producer.cuh"
#include "q6wg_native_matrix_producer.cuh"
}

namespace {

using ninfer::ops::gguf::detail::check;
constexpr std::size_t kGuard = 64;
constexpr int kRepeats = 32;
bool capture_launches = false;
struct LaunchRecord {
    cudaKernel_t kernel;
    dim3 grid;
    dim3 block;
    std::size_t shared;
};
std::array<LaunchRecord, 8> launches{};
std::size_t launch_count = 0;

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

std::string quoted(const std::string& value) {
    std::ostringstream out;
    out << '"';
    for (const unsigned char c : value) {
        if (c == '"' || c == '\\') { out << '\\' << char(c); }
        else if (c < 32) {
            out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
        } else { out << char(c); }
    }
    out << '"';
    return out.str();
}

struct Json {
    std::ostringstream stream;
    bool first = true;
    Json() { stream << std::setprecision(12) << '{'; }
    void key(const char* name) {
        if (!first) { stream << ','; }
        first = false;
        stream << quoted(name) << ':';
    }
    void raw(const char* name, const std::string& value) { key(name); stream << value; }
    void add(const char* name, const std::string& value) { key(name); stream << quoted(value); }
    void add(const char* name, const char* value) { add(name, std::string(value)); }
    void add(const char* name, bool value) { key(name); stream << (value ? "true" : "false"); }
    template <typename T, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
    void add(const char* name, T value) { key(name); stream << value; }
    std::string finish() { stream << '}'; return stream.str(); }
};

void emit(Json& record) { std::cout << record.finish() << '\n' << std::flush; }

template <typename T>
T load(const std::uint8_t* source) {
    T result;
    std::memcpy(&result, source, sizeof(T));
    return result;
}

template <typename T>
void store(std::uint8_t* target, T value) { std::memcpy(target, &value, sizeof(T)); }

float bf16_float(std::uint16_t bits) {
    const std::uint32_t word = std::uint32_t(bits) << 16;
    float result;
    std::memcpy(&result, &word, sizeof(result));
    return result;
}

std::uint16_t rounded_bf16(float value) {
    std::uint32_t word;
    std::memcpy(&word, &value, sizeof(word));
    require(std::isfinite(value), "BF16 semantic check requires finite FP32");
    return std::uint16_t((word + 0x7fffU + ((word >> 16) & 1U)) >> 16);
}

void rounding_self_test() {
    const std::array<std::pair<std::uint32_t, std::uint16_t>, 12> cases{{
        {0x3f807fffU, 0x3f80U}, {0x3f808000U, 0x3f80U}, {0x3f808001U, 0x3f81U},
        {0x3f817fffU, 0x3f81U}, {0x3f818000U, 0x3f82U}, {0x3f818001U, 0x3f82U},
        {0xbf807fffU, 0xbf80U}, {0xbf808000U, 0xbf80U}, {0xbf808001U, 0xbf81U},
        {0xbf817fffU, 0xbf81U}, {0xbf818000U, 0xbf82U}, {0xbf818001U, 0xbf82U}}};
    for (const auto& item : cases) {
        const float input = load<float>(reinterpret_cast<const std::uint8_t*>(&item.first));
        require(rounded_bf16(input) == item.second, "independent BF16 tie oracle failed");
        const auto native = __float2bfloat16_rn(input);
        require(load<std::uint16_t>(reinterpret_cast<const std::uint8_t*>(&native)) == item.second,
                "native host BF16 rounding differs at a semantic boundary");
    }
}

#include "q6wg_original_oracle.cuh"
#include "q6wg_make_input.cuh"
#include "q6wg_sha256.cuh"

struct Guarded {
    void* allocation = nullptr;
    void* data = nullptr;
    std::size_t bytes;
    explicit Guarded(std::size_t count) : bytes(count) {
        require(count > 0 && count <= std::numeric_limits<std::size_t>::max() - 2 * kGuard,
                "invalid guarded allocation extent");
        check(cudaMalloc(&allocation, count + 2 * kGuard), "allocate guarded buffer");
        data = static_cast<std::uint8_t*>(allocation) + kGuard;
        check(cudaMemset(allocation, 0xa5, count + 2 * kGuard), "initialize canaries");
    }
    ~Guarded() { if (allocation) { cudaFree(allocation); } }
    Guarded(const Guarded&) = delete;
    Guarded& operator=(const Guarded&) = delete;
    void from_host(const void* source, std::size_t count) {
        require(count == bytes, "guarded host input extent differs");
        check(cudaMemcpy(data, source, bytes, cudaMemcpyHostToDevice), "upload guarded payload");
    }
    void reset_from(const Guarded& source) {
        require(source.bytes == bytes, "reset extent differs");
        check(cudaMemcpy(data, source.data, bytes, cudaMemcpyDeviceToDevice), "reset output before event");
    }
    std::vector<std::uint8_t> read() const {
        std::vector<std::uint8_t> result(bytes);
        check(cudaMemcpy(result.data(), data, bytes, cudaMemcpyDeviceToHost), "read guarded payload");
        return result;
    }
    void guards() const {
        std::array<std::uint8_t, 2 * kGuard> actual{};
        check(cudaMemcpy(actual.data(), allocation, kGuard, cudaMemcpyDeviceToHost), "read prefix canary");
        check(cudaMemcpy(actual.data() + kGuard, static_cast<const std::uint8_t*>(data) + bytes,
                         kGuard, cudaMemcpyDeviceToHost), "read suffix canary");
        require(std::all_of(actual.begin(), actual.end(), [](auto value) { return value == 0xa5; }),
                "device buffer canary changed");
    }
};

struct Fixture {
    const char* name;
    const char* relative;
    int rows;
    int k;
    std::size_t bytes;
    const char* sha;
};

constexpr Fixture kHead{"head", "weights/q6-head-full-k5120.bin", 248320, 5120, 1042944000,
    "2155d6e1f7c5549077438c4760e6e150945dd5e406484b29162dff7b03d05da0"};
constexpr Fixture kDown{"down", "operators/q6-full-k17408.bin", 5120, 17408, 73113600,
    "a381c6be36f572170ccc9c7e4a3f8b25e5061e4eb633649399f3ec9759cb43e2"};

std::vector<std::uint8_t> read_fixture(const std::filesystem::path& root, const Fixture& spec) {
    const auto path = root / spec.relative;
    require(std::filesystem::is_regular_file(path) && std::filesystem::file_size(path) == spec.bytes,
            "full real fixture is missing or has a different extent: " + path.string());
    std::vector<std::uint8_t> result(spec.bytes);
    std::ifstream input(path, std::ios::binary);
    input.read(reinterpret_cast<char*>(result.data()), static_cast<std::streamsize>(result.size()));
    require(input.gcount() == static_cast<std::streamsize>(result.size()), "incomplete full fixture read");
    require(sha256(result.data(), result.size()) == spec.sha, "full real fixture SHA256 differs");
    return result;
}

struct Resident {
    Fixture spec;
    std::vector<std::uint8_t> original;
    Guarded native;
    Guarded candidate;
    Resident(const std::filesystem::path& root, Fixture fixture)
        : spec(fixture), original(read_fixture(root, fixture)), native(original.size()),
          candidate(q6wg_codec::packed_bytes(original.size() / 210)) {
        auto packed = q6wg_codec::pack(original);
        q6wg_codec::verify_inverse(original, packed);
        native.from_host(original.data(), original.size());
        candidate.from_host(packed.data(), packed.size());
        native.guards();
        candidate.guards();
        Json record;
        record.add("event", "full_fixture_literal_roundtrip");
        record.add("fixture", spec.name);
        record.add("sha256", spec.sha);
        record.add("original_bytes", original.size());
        record.add("packed_bytes", packed.size());
        record.add("blocks_compared", original.size() / 210);
        record.add("scales_and_d_bits_exact", true);
        record.add("encoded_file_written", false);
        record.add("lab_gpu_backings", 2);
        emit(record);
    }
    Q6WGView view(bool alternate, int begin, int rows) const {
        const auto blocks = std::uint64_t(spec.rows) * (spec.k / 256);
        const auto& buffer = alternate ? candidate : native;
        return {static_cast<const std::uint8_t*>(buffer.data), buffer.bytes,
                unsigned(spec.rows), unsigned(spec.k / 256), unsigned(begin), unsigned(rows),
                alternate ? unsigned((blocks - 1) % 16 + 1) : 0U,
                alternate ? Q6WGStorage::WordGroupSlab16 : Q6WGStorage::Native210};
    }
};

const char* mode_name(Q6WGOutputMode mode) {
    if (mode == Q6WGOutputMode::PlainFP32) { return "PlainFP32"; }
    if (mode == Q6WGOutputMode::StoreBF16) { return "StoreBF16"; }
    return "AccumulateBF16";
}

struct Case {
    const Resident& resident;
    std::string name;
    int begin;
    int rows;
    int columns;
    bool matrix;
    bool timed;
    bool proxy;
    Q6WGOutputMode primary;
    Q6WGMatrixPlan plan{};
    std::int64_t stride;
    Case(const Resident& weights, std::string label, int first, int count, int tokens,
         bool mmq, bool timing, bool is_proxy, Q6WGOutputMode writer)
        : resident(weights), name(std::move(label)), begin(first), rows(count), columns(tokens),
          matrix(mmq), timed(timing), proxy(is_proxy), primary(writer), stride(count + 19) {
        require(first >= 0 && count > 0 && first + count <= resident.spec.rows, "case exceeds parent");
        require(q6wg_validate_view(view(false)) && q6wg_validate_view(view(true)), "valid view rejected");
        if (matrix) {
            require(q6wg_matrix_plan(rows, k(), columns, &plan), "selected matrix plan unsupported");
            require(!plan.fixup && plan.fixup_bytes == 0, "positive cases require no fixup");
        }
        if (begin) {
            const auto origin = std::uint64_t(begin) * (k() / 256);
            require((origin % 16) != 0 && (origin / 16) != ((origin + k() / 256 - 1) / 16),
                    "odd view must begin inside a slab and cross it in the first row");
        }
    }
    int k() const { return resident.spec.k; }
    std::size_t count() const { return std::size_t(stride) * columns; }
    Q6WGView view(bool alternate) const { return resident.view(alternate, begin, rows); }
    std::size_t activation_bytes() const {
        return matrix ? plan.activation_bytes : q6wg_vector_activation_bytes(k(), columns);
    }
    bool invoke(bool alternate, Q6WGReadBuffer activation, const Q6WGOutput& output) const {
        return matrix ? q6wg_matrix_run(plan, view(alternate), activation,
                                        q6wg_matrix_activation_layout(), output, {}, nullptr)
                      : q6wg_vector_run(view(alternate), activation, columns, output, nullptr);
    }
};

struct Activation {
    std::vector<__nv_bfloat16> input;
    Guarded source;
    Guarded packed;
    std::vector<std::uint8_t> host;
    Activation(const Case& c)
        : input(make_input(c.k(), c.columns, 0x6a119U + c.rows + c.columns)),
          source(input.size() * sizeof(__nv_bfloat16)), packed(c.activation_bytes()) {
        source.from_host(input.data(), source.bytes);
        check(cudaMemset(packed.data, 0, packed.bytes), "zero common activation and slack");
        if (c.matrix) {
            require(c.k() % 512 == 0, "native selected matrix producer requires whole 512 groups");
            q6wg_host_native::quantize_matrix_kernel<MMQ_Q8_1_DS_LAYOUT_D4>
                <<<dim3(c.columns, c.k() / 512, 1), 128>>>(
                    static_cast<const __nv_bfloat16*>(source.data), nullptr,
                    static_cast<block_q8_1_mmq*>(packed.data), c.k(), c.columns);
        } else {
            q6wg_host_native::quantize_vector_kernel
                <<<dim3((c.k() + 255) / 256, c.columns, 1), 256>>>(
                    static_cast<const __nv_bfloat16*>(source.data), nullptr,
                    static_cast<std::int8_t*>(packed.data),
                    reinterpret_cast<half2*>(static_cast<std::uint8_t*>(packed.data) +
                        ninfer::ops::gguf::detail::vec_scales_offset(c.k(), c.columns)), c.k());
        }
        check(cudaGetLastError(), "common native activation producer");
        check(cudaDeviceSynchronize(), "finish common activation producer");
        host = packed.read();
        source.guards();
        packed.guards();
        if (c.matrix) {
            const auto payload = std::size_t(c.k() / 128) * c.columns * 144;
            require(std::all_of(host.begin() + payload, host.end(), [](auto v) { return v == 0; }),
                    "common producer wrote matrix overread slack");
        }
    }
    Q6WGReadBuffer read_buffer() const { return {packed.data, packed.bytes}; }
    double scale(const Case& c, int col, int i) const {
        if (c.matrix) {
            const auto offset = (std::size_t(i / 128) * c.columns + col) * 144 + 4 * ((i % 128) / 32);
            return load<float>(host.data() + offset);
        }
        const auto offset = ninfer::ops::gguf::detail::vec_scales_offset(c.k(), c.columns) +
                            4 * (std::size_t(col) * (c.k() / 32) + i / 32);
        return __half2float(load<__half>(host.data() + offset));
    }
    int quant(const Case& c, int col, int i) const {
        const auto offset = c.matrix ? (std::size_t(i / 128) * c.columns + col) * 144 + 16 + i % 128
                                     : std::size_t(col) * c.k() + i;
        return load<std::int8_t>(host.data() + offset);
    }
    void unchanged() const {
        require(packed.read() == host, "consumer mutated common packed activation or slack");
        source.guards();
        packed.guards();
    }
};

std::vector<int> sample_indices(int extent, int count) {
    require(extent >= count && count > 0, "sample dimensions too small");
    std::vector<int> result;
    const auto append = [&](int value) {
        if (value >= 0 && value < extent && int(result.size()) < count &&
            std::find(result.begin(), result.end(), value) == result.end()) { result.push_back(value); }
    };
    for (int value : {0, 1, 127, 128, extent / 3, extent / 2, extent - 2, extent - 1,
                     7, 15, 16, 31, 32, 63, 64, extent / 4}) { append(value); }
    for (int i = 0; int(result.size()) < count && i < count; ++i) {
        append(int(std::int64_t(i) * (extent - 1) / (count - 1)));
    }
    for (int i = 0; int(result.size()) < count; ++i) { append(i); }
    return result;
}

std::vector<std::size_t> oracle_positions(const Case& c) {
    const int cols = c.matrix ? 8 : c.columns;
    const auto rs = sample_indices(c.rows, 64 / cols);
    const auto cs = sample_indices(c.columns, cols);
    std::vector<std::size_t> result;
    for (int col : cs) { for (int row : rs) { result.push_back(std::size_t(col) * c.stride + row); } }
    require(result.size() == 64, "oracle must have exactly 64 distinct points");
    return result;
}

struct OracleStats {
    double max_absolute = 0;
    double max_bound_fraction = 0;
    int samples = 0;
};

OracleStats independent_oracle(const Case& c, const Activation& activation,
                              const std::vector<std::uint8_t>& result) {
    const auto positions = oracle_positions(c);
    const auto cols = sample_indices(c.columns, c.matrix ? 8 : c.columns);
    constexpr double epsilon = 8.0 * std::numeric_limits<float>::epsilon();
    for (int col : cols) {
        for (int start = 0; start < c.k(); start += 32) {
            double maximum = 0;
            for (int j = 0; j < 32; ++j) {
                maximum = std::max(maximum, std::abs(double(__bfloat162float(
                    activation.input[std::size_t(col) * c.k() + start + j]))));
            }
            const double expected = maximum / 127.0;
            const double scale = activation.scale(c, col, start);
            const double rounding = c.matrix ? 0.0 :
                std::abs(double(__half2float(__float2half_rn(float(expected)))) - expected);
            require(std::isfinite(scale) && scale >= 0 &&
                    std::abs(scale - expected) <= rounding + epsilon * expected,
                    "common native activation scale differs from BF16 input");
            for (int j = 0; j < 32; ++j) {
                const int q = activation.quant(c, col, start + j);
                const double input = __bfloat162float(
                    activation.input[std::size_t(col) * c.k() + start + j]);
                require(q >= -127 && std::abs(scale * q - input) <=
                        0.5 * expected + rounding * 127.0 + epsilon * maximum,
                        "common native activation quant differs from BF16 input");
            }
        }
    }
    OracleStats stats;
    for (const auto at : positions) {
        const int col = int(at / c.stride), row = int(at % c.stride);
        const auto* original = c.resident.original.data() +
                               std::size_t(c.begin + row) * (c.k() / 256) * 210;
        double reference = 0;
        for (int i = 0; i < c.k(); ++i) {
            reference += q6_runtime_weight_value(original, i) *
                         (activation.scale(c, col, i) * activation.quant(c, col, i));
        }
        const double got = load<float>(result.data() + at * 4);
        const double difference = std::abs(got - reference);
        const double bound = 0.02 + 2.0e-4 * std::abs(reference);
        require(std::isfinite(reference) && std::isfinite(got) && difference <= bound,
                "result differs from independent original-byte FP64 oracle");
        stats.max_absolute = std::max(stats.max_absolute, difference);
        stats.max_bound_fraction = std::max(stats.max_bound_fraction, difference / bound);
        ++stats.samples;
    }
    return stats;
}

std::vector<std::uint8_t> initial_output(const Case& c, bool fp32) {
    const unsigned size = fp32 ? 4 : 2;
    std::vector<std::uint8_t> result(c.count() * size, 0xa5);
    for (int col = 0; col < c.columns; ++col) {
        for (int row = 0; row < c.rows; ++row) {
            const auto at = std::size_t(col) * c.stride + row;
            if (fp32) { store(result.data() + 4 * at, std::uint32_t(0x7fc0deaf)); }
            else {
                const float magnitude = float(1 + (row * 17 + col * 7) % 61) / 128.0f;
                store(result.data() + 2 * at, rounded_bf16((row + col) & 1 ? -magnitude : magnitude));
            }
        }
    }
    return result;
}

void output_padding(const Case& c, const std::vector<std::uint8_t>& result, unsigned size) {
    require(result.size() == c.count() * size, "output read extent differs");
    for (int col = 0; col < c.columns; ++col) {
        const auto first = (std::size_t(col) * c.stride + c.rows) * size;
        const auto last = std::size_t(col + 1) * c.stride * size;
        require(std::all_of(result.begin() + first, result.begin() + last,
                            [](auto value) { return value == 0xa5; }), "output column padding changed");
    }
}

struct SemanticStats {
    std::size_t points = 0;
    std::size_t carry_sensitive = 0;
    std::size_t halfway_even = 0;
    std::size_t halfway_odd = 0;
    std::size_t adjacent_halfway = 0;
    std::uint32_t minimum_halfway_distance = 65536;
};

SemanticStats semantic_output(const Case& c, Q6WGOutputMode mode,
                             const std::vector<std::uint8_t>& fp32,
                             const std::vector<std::uint8_t>& carry,
                             const std::vector<std::uint8_t>& actual, int repeats = 1,
                             bool only_samples = false) {
    SemanticStats stats;
    const auto samples = only_samples ? oracle_positions(c) : std::vector<std::size_t>{};
    const auto check_at = [&](std::size_t at) {
        const float value = load<float>(fp32.data() + 4 * at);
        require(std::isfinite(value), "non-finite FP32 element");
        std::uint16_t expected = load<std::uint16_t>(carry.data() + 2 * at);
        float before = value;
        for (int n = 0; n < repeats; ++n) {
            before = mode == Q6WGOutputMode::AccumulateBF16 ? value + bf16_float(expected) : value;
            expected = rounded_bf16(before);
        }
        require(load<std::uint16_t>(actual.data() + 2 * at) == expected,
                "BF16 writer differs from FP32 Store/Accumulate semantics");
        require(std::isfinite(bf16_float(expected)), "non-finite BF16 element");
        const auto bits = load<std::uint32_t>(reinterpret_cast<const std::uint8_t*>(&before));
        const auto low = bits & 65535U;
        if (low == 32768) { ((bits >> 16) & 1 ? stats.halfway_odd : stats.halfway_even)++; }
        if (low == 32767 || low == 32769) { ++stats.adjacent_halfway; }
        stats.minimum_halfway_distance = std::min(stats.minimum_halfway_distance,
            unsigned(std::abs(int(low) - 32768)));
        stats.carry_sensitive += expected != rounded_bf16(value);
        ++stats.points;
    };
    if (only_samples) { for (auto at : samples) { check_at(at); } }
    else {
        for (int col = 0; col < c.columns; ++col) {
            for (int row = 0; row < c.rows; ++row) { check_at(std::size_t(col) * c.stride + row); }
        }
    }
    if (mode == Q6WGOutputMode::AccumulateBF16) {
        require(stats.carry_sensitive > 0, "nonzero carry did not exercise Accumulate semantics");
    }
    output_padding(c, actual, 2);
    return stats;
}

std::string diagnostic_launch(const Case& c, const Activation& activation,
                              bool alternate, const Q6WGOutput& output) {
    launch_count = 0;
    capture_launches = true;
    const bool accepted = c.invoke(alternate, activation.read_buffer(), output);
    capture_launches = false;
    require(accepted, "positive dispatch rejected");
    check(cudaDeviceSynchronize(), "complete diagnostic dispatch");
    require(launch_count == 1, "expected exactly one operator kernel without fixup");
    const auto& actual = launches[0];
    const char* raw_name = nullptr;
    check(cudaFuncGetName(&raw_name, reinterpret_cast<const void*>(actual.kernel)), "diagnostic kernel name");
    require(raw_name, "kernel name is unavailable");
    const std::string name(raw_name);
    const std::string candidate_name = c.matrix ? "q6_word_group_mul_mat_q" : "q6_word_group_vector_kernel";
    if (alternate) { require(name.find(candidate_name) != std::string::npos, "candidate dispatch fell back"); }
    else {
        require(name.find("q6_word_group") == std::string::npos &&
                name.find(c.matrix ? "mul_mat_q" : "6kernel") != std::string::npos,
                "native dispatch did not select the pinned control");
    }
    require(actual.grid.x > 0 && actual.grid.y == 1 && actual.grid.z == 1, "unexpected operator grid");
    require(actual.block.x == unsigned(c.matrix ? 32 : 128) &&
            actual.block.y == unsigned(c.matrix ? 8 : 1) && actual.block.z == 1,
            "unexpected selected operator thread geometry");
    require(actual.shared == (c.matrix ? c.plan.shared : 0), "unexpected selected dynamic shared bytes");
    if (c.matrix) { require(actual.grid.x == unsigned(c.plan.blocks), "matrix grid differs from plan"); }
    Json record;
    record.add("kernel", name);
    record.add("grid_x", actual.grid.x);
    record.add("block_x", actual.block.x);
    record.add("block_y", actual.block.y);
    record.add("dynamic_shared_bytes", actual.shared);
    record.add("outside_timing_events", true);
    return record.finish();
}

}

extern "C" cudaError_t CUDARTAPI __real___cudaLaunchKernel(
    cudaKernel_t kernel, dim3 grid, dim3 block, void** args, std::size_t shared, cudaStream_t stream);

extern "C" cudaError_t CUDARTAPI __wrap___cudaLaunchKernel(
    cudaKernel_t kernel, dim3 grid, dim3 block, void** args, std::size_t shared, cudaStream_t stream) {
    if (capture_launches) {
        if (launch_count < launches.size()) { launches[launch_count] = {kernel, grid, block, shared}; }
        ++launch_count;
    }
    return __real___cudaLaunchKernel(kernel, grid, block, args, shared, stream);
}

namespace {

std::string rejection_checks(const Case& c, const Activation& activation,
                             Guarded& output_buffer, Q6WGOutputMode mode) {
    const unsigned size = mode == Q6WGOutputMode::PlainFP32 ? 4 : 2;
    const Q6WGOutput output{{output_buffer.data, output_buffer.bytes}, c.stride, mode};
    const auto original_output = output_buffer.read();
    std::vector<std::string> checks;
    const auto rejected = [&](const std::string& name, auto&& operation) {
        launch_count = 0;
        capture_launches = true;
        const bool accepted = operation();
        capture_launches = false;
        require(!accepted && launch_count == 0, "ABI reject launched or accepted: " + name);
        checks.push_back(name);
    };
    const auto call = [&](const Q6WGView& view, Q6WGReadBuffer read, const Q6WGOutput& out) {
        return c.matrix ? q6wg_matrix_run(c.plan, view, read, q6wg_matrix_activation_layout(),
                                          out, {}, nullptr)
                        : q6wg_vector_run(view, read, c.columns, out, nullptr);
    };
    for (bool alternate : {false, true}) {
        const auto view = c.view(alternate);
        const auto prefix = alternate ? std::string("candidate.") : std::string("native.");
        const auto bad_view = [&](const char* name, auto&& edit) {
            auto invalid = view;
            edit(invalid);
            rejected(prefix + "view." + name, [&] { return q6wg_validate_view(invalid); });
            rejected(prefix + "dispatch." + name, [&] { return call(invalid, activation.read_buffer(), output); });
        };
        bad_view("null_base", [](auto& v) { v.base = nullptr; });
        bad_view("zero_parent_rows", [](auto& v) { v.parent_rows = 0; });
        bad_view("zero_parent_B", [](auto& v) { v.parent_B = 0; });
        bad_view("excess_parent_B", [](auto& v) { v.parent_B = INT_MAX / 256 + 1U; });
        bad_view("zero_rows", [](auto& v) { v.rows = 0; });
        bad_view("signed_rows_overflow", [](auto& v) { v.rows = unsigned(INT_MAX) + 1U; });
        bad_view("origin_outside_parent", [](auto& v) { v.row_begin = v.parent_rows + 1; });
        bad_view("view_past_parent", [](auto& v) { v.row_begin = v.parent_rows; v.rows = 1; });
        bad_view("inexact_backing_extent", [](auto& v) { --v.bytes; });
        bad_view("bad_tail", [](auto& v) { ++v.final_slab_blocks; });
        bad_view("unaligned_base", [](auto& v) { ++v.base; });
        bad_view("wrapping_backing", [](auto& v) {
            v.base = reinterpret_cast<const std::uint8_t*>(std::numeric_limits<std::uintptr_t>::max() & ~std::uintptr_t(15));
        });
        bad_view("unknown_storage", [](auto& v) { v.storage = static_cast<Q6WGStorage>(77); });
        const auto bad_activation = [&](const char* name, Q6WGReadBuffer invalid) {
            rejected(prefix + "activation." + name, [&] { return call(view, invalid, output); });
        };
        bad_activation("short", {activation.packed.data, activation.packed.bytes - 1});
        bad_activation("null", {nullptr, activation.packed.bytes});
        bad_activation("unaligned", {static_cast<const std::uint8_t*>(activation.packed.data) + 1,
                                     activation.packed.bytes});
        bad_activation("wrap", {reinterpret_cast<const void*>(std::numeric_limits<std::uintptr_t>::max() &
                                 ~std::uintptr_t(15)), activation.packed.bytes});
        const auto bad_output = [&](const char* name, auto&& edit) {
            auto invalid = output;
            edit(invalid);
            rejected(prefix + "output." + name, [&] { return call(view, activation.read_buffer(), invalid); });
        };
        bad_output("short", [&](auto& o) {
            o.buffer.bytes = (std::size_t(c.columns - 1) * c.stride + c.rows) * size - 1;
        });
        bad_output("null", [](auto& o) { o.buffer.data = nullptr; });
        bad_output("unaligned", [](auto& o) { o.buffer.data = static_cast<std::uint8_t*>(o.buffer.data) + 1; });
        bad_output("wrap", [](auto& o) {
            o.buffer.data = reinterpret_cast<void*>(std::numeric_limits<std::uintptr_t>::max() & ~std::uintptr_t(15));
        });
        bad_output("stride_short", [&](auto& o) { o.column_stride = c.rows - 1; });
        bad_output("stride_negative", [](auto& o) { o.column_stride = -1; });
        bad_output("stride_overflow", [](auto& o) { o.column_stride = std::int64_t(INT_MAX) + 1; });
        if (c.columns > 1) {
            bad_output("element_overflow", [](auto& o) { o.column_stride = INT_MAX; });
        }
        bad_output("unknown_mode", [](auto& o) { o.mode = static_cast<Q6WGOutputMode>(77); });
        bad_output("alias_parent", [&](auto& o) { o.buffer.data = const_cast<std::uint8_t*>(view.base); });
        bad_output("alias_activation", [&](auto& o) { o.buffer.data = activation.packed.data; });
        if (!c.matrix) {
            rejected(prefix + "columns", [&] {
                return q6wg_vector_run(view, activation.read_buffer(), 2, output, nullptr);
            });
            auto huge = view;
            huge.parent_rows = INT_MAX;
            huge.rows = INT_MAX;
            huge.row_begin = 0;
            const auto blocks = std::uint64_t(huge.parent_rows) * huge.parent_B;
            huge.bytes = alternate ? ((blocks + 15) / 16) * 3360 : blocks * 210;
            huge.final_slab_blocks = alternate ? (blocks - 1) % 16 + 1 : 0;
            rejected(prefix + "rounded_vector_rows_overflow", [&] {
                return q6wg_vector_run(huge, activation.read_buffer(), c.columns, output, nullptr);
            });
        } else {
            const auto bad_plan = [&](const char* name, auto&& edit) {
                auto invalid = c.plan;
                edit(invalid);
                rejected(prefix + "plan." + name, [&] {
                    return q6wg_matrix_run(invalid, view, activation.read_buffer(),
                                           q6wg_matrix_activation_layout(), output, {}, nullptr);
                });
            };
            bad_plan("rows", [](auto& p) { p.rows += 128; });
            bad_plan("k", [](auto& p) { p.k += 256; });
            bad_plan("columns", [](auto& p) { ++p.columns; });
            bad_plan("J", [](auto& p) { p.J = 64; });
            bad_plan("I", [](auto& p) { p.I = 64; });
            bad_plan("threads", [](auto& p) { p.threads = 128; });
            bad_plan("shared", [](auto& p) { --p.shared; });
            bad_plan("row_tiles", [](auto& p) { ++p.row_tiles; });
            bad_plan("column_tiles", [](auto& p) { ++p.column_tiles; });
            bad_plan("stream_k", [](auto& p) { p.stream_k = !p.stream_k; });
            bad_plan("blocks", [](auto& p) { ++p.blocks; });
            bad_plan("fixup", [](auto& p) { p.fixup = !p.fixup; });
            bad_plan("activation_bytes", [](auto& p) { --p.activation_bytes; });
            bad_plan("fixup_bytes", [](auto& p) { ++p.fixup_bytes; });
            rejected(prefix + "matrix_layout", [&] {
                return q6wg_matrix_run(c.plan, view, activation.read_buffer(), -1, output, {}, nullptr);
            });
            for (Q6WGWriteBuffer scratch : {Q6WGWriteBuffer{output_buffer.data, 0}, Q6WGWriteBuffer{nullptr, 1}}) {
                rejected(prefix + "unexpected_fixup", [&] {
                    return q6wg_matrix_run(c.plan, view, activation.read_buffer(),
                                           q6wg_matrix_activation_layout(), output, scratch, nullptr);
                });
            }
            auto huge = view;
            huge.row_begin = 0;
            huge.rows = (std::uint64_t(INT_MAX) / huge.parent_B / 128 + 1) * 128;
            huge.parent_rows = huge.rows;
            const auto blocks = std::uint64_t(huge.rows) * huge.parent_B;
            huge.bytes = alternate ? ((blocks + 15) / 16) * 3360 : blocks * 210;
            huge.final_slab_blocks = alternate ? (blocks - 1) % 16 + 1 : 0;
            auto huge_plan = c.plan;
            huge_plan.rows = int(huge.rows);
            rejected(prefix + "matrix_block_index_overflow", [&] {
                return q6wg_matrix_run(huge_plan, huge, activation.read_buffer(),
                                       q6wg_matrix_activation_layout(), output, {}, nullptr);
            });
            Q6WGMatrixPlan fixup_plan{};
            require(q6wg_matrix_plan(128, c.k(), c.columns, &fixup_plan) && fixup_plan.fixup,
                    "this target must provide a fixup-required negative-test plan");
            auto narrow = view;
            narrow.rows = 128;
            const auto fixup_call = [&](Q6WGWriteBuffer scratch, Q6WGOutput out) {
                return q6wg_matrix_run(fixup_plan, narrow, activation.read_buffer(),
                                       q6wg_matrix_activation_layout(), out, scratch, nullptr);
            };
            if (mode != Q6WGOutputMode::PlainFP32) {
                rejected(prefix + "BF16_fixup_forbidden", [&] { return fixup_call({}, output); });
            } else {
                rejected(prefix + "fixup_missing", [&] { return fixup_call({}, output); });
                rejected(prefix + "fixup_short", [&] {
                    return fixup_call({output_buffer.data, fixup_plan.fixup_bytes - 1}, output);
                });
                rejected(prefix + "fixup_null", [&] {
                    return fixup_call({nullptr, fixup_plan.fixup_bytes}, output);
                });
                rejected(prefix + "fixup_unaligned", [&] {
                    return fixup_call({static_cast<std::uint8_t*>(output_buffer.data) + 1,
                                       fixup_plan.fixup_bytes}, output);
                });
                rejected(prefix + "fixup_wrap", [&] {
                    return fixup_call({reinterpret_cast<void*>(std::numeric_limits<std::uintptr_t>::max() &
                        ~std::uintptr_t(15)), fixup_plan.fixup_bytes}, output);
                });
                for (void* alias : {const_cast<std::uint8_t*>(view.base),
                                    static_cast<std::uint8_t*>(activation.packed.data),
                                    static_cast<std::uint8_t*>(output_buffer.data)}) {
                    rejected(prefix + "fixup_alias", [&] {
                        return fixup_call({alias, fixup_plan.fixup_bytes}, output);
                    });
                }
            }
        }
    }
    if (!c.matrix) {
        for (auto pair : {std::pair<int, int>{0, 1}, {257, 1}, {c.k(), 0}, {c.k(), 2}, {c.k(), 8}}) {
            rejected("vector_activation_extent", [&] {
                return q6wg_vector_activation_bytes(pair.first, pair.second) != 0;
            });
        }
    } else {
        Q6WGMatrixPlan unused{};
        rejected("matrix_plan_null", [&] { return q6wg_matrix_plan(c.rows, c.k(), c.columns, nullptr); });
        for (auto shape : {std::array<int, 3>{0, c.k(), c.columns}, {129, c.k(), c.columns},
                           {INT_MAX - 127, c.k(), c.columns}, {c.rows, 0, c.columns},
                           {c.rows, 257, c.columns}, {c.rows, c.k(), 64},
                           {c.rows, c.k(), INT_MAX}, {INT_MAX - 255, c.k(), INT_MAX - 255}}) {
            rejected("matrix_plan_extent", [&] {
                return q6wg_matrix_plan(shape[0], shape[1], shape[2], &unused);
            });
        }
    }
    require(output_buffer.read() == original_output, "a rejected dispatch changed output");
    output_buffer.guards();
    require(!capture_launches, "diagnostic launch capture leaked into timing");
    std::ostringstream names;
    names << '[';
    for (std::size_t i = 0; i < checks.size(); ++i) { names << (i ? "," : "") << quoted(checks[i]); }
    names << ']';
    Json result;
    result.add("count", checks.size());
    result.raw("checks", names.str());
    result.add("rejected_without_launch", true);
    result.add("output_unchanged", true);
    result.add("unsupported_device_branch_exercised", false);
    result.add("default_off_binary_branch_exercised", false);
    result.add("positive_fixup_qualified", false);
    return result.finish();
}

struct Events {
    cudaEvent_t first{}, last{};
    Events() {
        check(cudaEventCreate(&first), "create timing start");
        check(cudaEventCreate(&last), "create timing stop");
    }
    ~Events() { cudaEventDestroy(first); cudaEventDestroy(last); }
};

double timing_sample(const Case& c, const Activation& activation, bool alternate,
                     Q6WGOutputMode mode, const Guarded& seed, Guarded& output, Events& events) {
    require(!capture_launches, "launch diagnostics must be disabled during timing");
    output.reset_from(seed);
    const Q6WGOutput destination{{output.data, output.bytes}, c.stride, mode};
    const auto input = activation.read_buffer();
    check(cudaEventRecord(events.first), "record start");
    bool accepted = true;
    for (int repeat = 0; repeat < kRepeats; ++repeat) { accepted &= c.invoke(alternate, input, destination); }
    check(cudaEventRecord(events.last), "record stop");
    check(cudaEventSynchronize(events.last), "finish timing sample");
    require(accepted, "timing dispatch rejected");
    float elapsed = 0;
    check(cudaEventElapsedTime(&elapsed, events.first, events.last), "read event time");
    require(elapsed > 0 && std::isfinite(elapsed), "invalid CUDA event duration");
    return double(elapsed) / kRepeats;
}

void short_screen(const Case& c, const Activation& activation, Q6WGOutputMode mode,
                  const std::vector<std::uint8_t>& fp32,
                  const std::vector<std::uint8_t>& carry,
                  const std::vector<std::uint8_t>& expected_once,
                  Guarded& native, Guarded& candidate, const std::string& launch_id) {
    const auto initial = mode == Q6WGOutputMode::PlainFP32 ? initial_output(c, true) : carry;
    Guarded seed(initial.size());
    seed.from_host(initial.data(), initial.size());
    Events events;
    std::array<double, 4> warmups{};
    const auto run = [&](bool alternate) {
        return timing_sample(c, activation, alternate, mode, seed, alternate ? candidate : native, events);
    };
    for (int i = 0; i < 4; ++i) { warmups[i] = run((i & 1) != 0); }
    std::vector<double> a, b;
    std::ostringstream raw;
    raw << std::setprecision(12) << '[';
    for (int block = 0; block < 3; ++block) {
        const bool reversed = (block & 1) != 0;
        std::array<double, 4> samples{};
        std::array<bool, 4> order{reversed, !reversed, !reversed, reversed};
        for (int sample = 0; sample < 4; ++sample) {
            samples[sample] = run(order[sample]);
            (order[sample] ? b : a).push_back(samples[sample]);
        }
        const auto native_result = native.read();
        const auto candidate_result = candidate.read();
        require(native_result == candidate_result, "32-call evolved outputs differ between native and candidate");
        if (mode == Q6WGOutputMode::AccumulateBF16) {
            semantic_output(c, mode, fp32, carry, native_result, kRepeats, true);
            semantic_output(c, mode, fp32, carry, candidate_result, kRepeats, true);
        } else { require(native_result == expected_once, "32-call store differs from one-call reference"); }
        output_padding(c, native_result, mode == Q6WGOutputMode::PlainFP32 ? 4 : 2);
        native.guards();
        candidate.guards();
        Json item;
        item.add("block", block);
        item.add("order", reversed ? "BAAB" : "ABBA");
        std::ostringstream values;
        values << std::setprecision(12) << '[' << samples[0] << ',' << samples[1] << ','
               << samples[2] << ',' << samples[3] << ']';
        item.raw("ms_per_call", values.str());
        raw << (block ? "," : "") << item.finish();
    }
    raw << ']';
    seed.guards();
    const double native_mean = std::accumulate(a.begin(), a.end(), 0.0) / a.size();
    const double candidate_mean = std::accumulate(b.begin(), b.end(), 0.0) / b.size();
    Json result;
    result.add("event", "exploratory_short_screen");
    result.add("launch_id", launch_id);
    result.add("case", c.name);
    result.add("mode", mode_name(mode));
    result.add("primary_endpoint", !c.proxy);
    result.add("head_prefix_geometry_weights_proxy", c.proxy);
    result.add("calls_per_sample", kRepeats);
    result.add("warmup_samples", 4);
    result.add("warmup_order", "ABAB");
    std::ostringstream warm;
    warm << std::setprecision(12) << '[' << warmups[0] << ',' << warmups[1] << ','
         << warmups[2] << ',' << warmups[3] << ']';
    result.raw("warmup_ms_per_call", warm.str());
    result.raw("blocks", raw.str());
    result.add("native_mean_ms", native_mean);
    result.add("candidate_mean_ms", candidate_mean);
    result.add("time_reduction_percent", 100.0 * (native_mean - candidate_mean) / native_mean);
    result.add("common_activation_packed_before_events", true);
    result.add("carry_reset_before_each_event", true);
    result.add("evolved_32_call_outputs_bit_exact", true);
    result.add("evolved_carry_FP32_semantic_samples_per_block",
               mode == Q6WGOutputMode::AccumulateBF16 ? 64 : 0);
    result.add("diagnostic_name_lookup_during_events", false);
    result.add("statistical_confirmation", false);
    result.add("end_to_end_claim", false);
    emit(result);
}

std::string semantic_json(const SemanticStats& stats) {
    Json result;
    result.add("elements", stats.points);
    result.add("carry_sensitive", stats.carry_sensitive);
    result.add("halfway_even", stats.halfway_even);
    result.add("halfway_odd", stats.halfway_odd);
    result.add("adjacent_halfway", stats.adjacent_halfway);
    result.add("minimum_halfway_distance_in_FP32_bits", stats.minimum_halfway_distance);
    result.add("unobserved_boundary_categories_not_qualified", true);
    return result.finish();
}

struct QualifiedCase {
    Case c;
    Activation activation;
    std::vector<std::uint8_t> fp32;
    std::vector<std::uint8_t> carry;
    std::vector<std::uint8_t> stored;
    explicit QualifiedCase(Case value) : c(std::move(value)), activation(c) {}
};

void run_case(QualifiedCase& qualified, const std::string& launch_id) {
    const auto& c = qualified.c;
    const auto& activation = qualified.activation;
    const auto fp32_seed = initial_output(c, true);
    const auto carry = initial_output(c, false);
    Guarded native_fp32(fp32_seed.size()), candidate_fp32(fp32_seed.size());
    Guarded native_bf16(carry.size()), candidate_bf16(carry.size());
    native_fp32.from_host(fp32_seed.data(), fp32_seed.size());
    candidate_fp32.from_host(fp32_seed.data(), fp32_seed.size());
    const Q6WGOutput native_output{{native_fp32.data, native_fp32.bytes}, c.stride, Q6WGOutputMode::PlainFP32};
    const Q6WGOutput candidate_output{{candidate_fp32.data, candidate_fp32.bytes}, c.stride, Q6WGOutputMode::PlainFP32};
    const auto fp32_native_route = diagnostic_launch(c, activation, false, native_output);
    const auto fp32_candidate_route = diagnostic_launch(c, activation, true, candidate_output);
    const auto reference = native_fp32.read(), alternate = candidate_fp32.read();
    require(reference == alternate, "full FP32 outputs are not bit exact");
    output_padding(c, reference, 4);
    for (int col = 0; col < c.columns; ++col) {
        for (int row = 0; row < c.rows; ++row) {
            require(std::isfinite(load<float>(reference.data() + 4 * (std::size_t(col) * c.stride + row))),
                    "unwritten or non-finite FP32 output");
        }
    }
    const auto oracle_native = independent_oracle(c, activation, reference);
    const auto oracle_candidate = independent_oracle(c, activation, alternate);
    const auto fp32_guards = rejection_checks(c, activation, native_fp32, Q6WGOutputMode::PlainFP32);
    std::ostringstream modes;
    modes << '[';
    Json plain;
    plain.add("mode", "PlainFP32");
    plain.add("full_output_bit_exact", true);
    plain.raw("native_route", fp32_native_route);
    plain.raw("candidate_route", fp32_candidate_route);
    plain.raw("public_guard_checks", fp32_guards);
    modes << plain.finish();
    for (auto mode : {Q6WGOutputMode::StoreBF16, Q6WGOutputMode::AccumulateBF16}) {
        native_bf16.from_host(carry.data(), carry.size());
        candidate_bf16.from_host(carry.data(), carry.size());
        const Q6WGOutput out_a{{native_bf16.data, native_bf16.bytes}, c.stride, mode};
        const Q6WGOutput out_b{{candidate_bf16.data, candidate_bf16.bytes}, c.stride, mode};
        const auto route_a = diagnostic_launch(c, activation, false, out_a);
        const auto route_b = diagnostic_launch(c, activation, true, out_b);
        const auto result_a = native_bf16.read(), result_b = candidate_bf16.read();
        require(result_a == result_b, "full BF16 outputs are not bit exact");
        const auto semantic_a = semantic_output(c, mode, reference, carry, result_a);
        const auto semantic_b = semantic_output(c, mode, alternate, carry, result_b);
        if (mode == Q6WGOutputMode::StoreBF16) { qualified.stored = result_a; }
        const auto guards = rejection_checks(c, activation, native_bf16, mode);
        Json item;
        item.add("mode", mode_name(mode));
        item.add("full_output_bit_exact", true);
        item.raw("native_route", route_a);
        item.raw("candidate_route", route_b);
        item.raw("native_FP32_writer_semantics", semantic_json(semantic_a));
        item.raw("candidate_FP32_writer_semantics", semantic_json(semantic_b));
        item.raw("public_guard_checks", guards);
        modes << ',' << item.finish();
    }
    modes << ']';
    activation.unchanged();
    native_fp32.guards();
    candidate_fp32.guards();
    native_bf16.guards();
    candidate_bf16.guards();
    c.resident.native.guards();
    c.resident.candidate.guards();
    Json result;
    result.add("event", "case_correctness");
    result.add("launch_id", launch_id);
    result.add("case", c.name);
    result.add("fixture_sha256", c.resident.spec.sha);
    result.add("actual_full_parent", true);
    result.add("full_output_geometry", c.begin == 0 && c.rows == c.resident.spec.rows);
    result.add("parent_rows", c.resident.spec.rows);
    result.add("parent_B", c.k() / 256);
    result.add("row_begin", c.begin);
    result.add("rows", c.rows);
    result.add("k", c.k());
    result.add("columns", c.columns);
    result.add("column_stride", c.stride);
    result.add("matrix", c.matrix);
    result.add("head_prefix_geometry_weights_proxy", c.proxy);
    result.add("primary_vector_writer", mode_name(c.primary));
    result.add("odd_start_crossing_slab", c.begin != 0);
    result.add("original_parent_base_retained", true);
    result.add("native_FP64_samples", oracle_native.samples);
    result.add("candidate_FP64_samples", oracle_candidate.samples);
    result.add("FP64_max_absolute_error", std::max(oracle_native.max_absolute, oracle_candidate.max_absolute));
    result.add("FP64_max_bound_fraction", std::max(oracle_native.max_bound_fraction, oracle_candidate.max_bound_fraction));
    result.add("FP64_bound_absolute", 0.02);
    result.add("FP64_bound_relative", 2.0e-4);
    result.add("oracle_uses_original_native_bytes", true);
    result.add("oracle_uses_candidate_codec", false);
    result.add("common_native_activation_unchanged", true);
    result.add("all_buffer_canaries_intact", true);
    result.add("positive_fixup_qualified", false);
    result.add("generic_consumers_qualified", false);
    result.add("vision_qualified", false);
    result.raw("output_modes", modes.str());
    emit(result);
    qualified.fp32 = reference;
    qualified.carry = carry;
}

int screen_qualified(const QualifiedCase& qualified, const std::string& launch_id) {
    const auto& c = qualified.c;
    if (!c.timed) { return 0; }
    int screens = 0;
    for (auto mode : {Q6WGOutputMode::PlainFP32, Q6WGOutputMode::StoreBF16, Q6WGOutputMode::AccumulateBF16}) {
        if (!c.matrix && mode != c.primary) { continue; }
        const auto bytes = c.count() * (mode == Q6WGOutputMode::PlainFP32 ? 4 : 2);
        Guarded native(bytes), candidate(bytes);
        short_screen(c, qualified.activation, mode, qualified.fp32, qualified.carry,
                     mode == Q6WGOutputMode::PlainFP32 ? qualified.fp32 : qualified.stored,
                     native, candidate, launch_id);
        ++screens;
    }
    qualified.activation.unchanged();
    c.resident.native.guards();
    c.resident.candidate.guards();
    return screens;
}

struct Options {
    std::filesystem::path fixtures;
    std::string launch_id;
    bool correctness_only = false;
    bool head_prefix = false;
};

Options options(int argc, char** argv) {
    Options result;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--fixtures-root" && i + 1 < argc) { result.fixtures = argv[++i]; }
        else if (arg == "--launch-id" && i + 1 < argc) { result.launch_id = argv[++i]; }
        else if (arg == "--correctness-only") { result.correctness_only = true; }
        else if (arg == "--head-prefix-mmq") { result.head_prefix = true; }
        else { throw std::invalid_argument("unknown or incomplete argument: " + arg); }
    }
    require(!result.fixtures.empty() && !result.launch_id.empty(),
            "requires --fixtures-root PATH --launch-id ID; optional --correctness-only --head-prefix-mmq");
    return result;
}

}

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--help") {
        std::cout << "--fixtures-root PATH --launch-id ID [--correctness-only] [--head-prefix-mmq]\n";
        return 0;
    }
    try {
        const auto args = options(argc, argv);
        require(q6wg_candidate_enabled(), "selected candidate object is default-OFF");
        require(q6wg_matrix_activation_layout() == MMQ_Q8_1_DS_LAYOUT_D4, "compiled activation ABI differs");
        rounding_self_test();
        require(sha256(nullptr, 0) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                "pinned SHA256 empty-vector check failed");
        const std::array<std::uint8_t, 3> abc{'a', 'b', 'c'};
        require(sha256(abc.data(), abc.size()) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
                "pinned SHA256 abc-vector check failed");
        q6wg_codec::self_test();
        check(cudaSetDevice(0), "select device");
        cudaDeviceProp properties{};
        check(cudaGetDeviceProperties(&properties, 0), "read device properties");
        const auto& facts = ninfer::ops::gguf::detail::device_facts();
        require(facts.cc == 860, "isolated probe requires SM86");
        require(facts.shared_per_block_optin >= 57856, "selected MMQ shared memory is unavailable");
        Json start;
        start.add("event", "start");
        start.add("launch_id", args.launch_id);
        start.add("device", properties.name);
        start.add("cc", facts.cc);
        start.add("SM_count", facts.sm_count);
        start.add("total_global_memory", properties.totalGlobalMem);
        start.add("shared_per_block_optin", facts.shared_per_block_optin);
        start.add("correctness_only", args.correctness_only);
        start.add("codec_onehot_bits_checked", 1680);
        start.add("codec_odd_block_counts_checked", 7);
        start.add("host_BF16_boundary_vectors_checked", 12);
        start.add("head_vector_primary_writer", "StoreBF16");
        start.add("down_vector_primary_writer", "AccumulateBF16");
        start.add("native_Q8_producer_is_common", true);
        start.add("runtime_product_one_backing_integrated", false);
        start.add("kernel_objects_recompiled", false);
        start.raw("source_receipt", kQ6WGHostReceiptJson);
        emit(start);
        int screens = 0;
        Resident head(args.fixtures, kHead);
        Resident down(args.fixtures, kDown);
        std::vector<std::unique_ptr<QualifiedCase>> cases;
        const auto execute = [&](Case c) {
            auto qualified = std::make_unique<QualifiedCase>(std::move(c));
            run_case(*qualified, args.launch_id);
            cases.push_back(std::move(qualified));
        };
        {
            for (int columns : {1, 4}) {
                execute(Case(head, "FULL_head_vector_T" + std::to_string(columns), 0, kHead.rows,
                             columns, false, !args.correctness_only, false, Q6WGOutputMode::StoreBF16));
                execute(Case(head, "odd_head_view_vector_T" + std::to_string(columns), 1, 257,
                             columns, false, false, false, Q6WGOutputMode::StoreBF16));
            }
            if (args.head_prefix) {
                execute(Case(head, "head_prefix6144_MMQ_T1280_GEOMETRY_WEIGHTS_PROXY", 0, 6144,
                             1280, true, !args.correctness_only, true, Q6WGOutputMode::StoreBF16));
            }
        }
        {
            for (int columns : {1, 4}) {
                execute(Case(down, "FULL_down_vector_T" + std::to_string(columns), 0, kDown.rows,
                             columns, false, !args.correctness_only, false, Q6WGOutputMode::AccumulateBF16));
                execute(Case(down, "odd_down_view_vector_T" + std::to_string(columns), 1, 257,
                             columns, false, false, false, Q6WGOutputMode::AccumulateBF16));
            }
            execute(Case(down, "FULL_down_MMQ_T1280", 0, kDown.rows, 1280, true,
                         !args.correctness_only, false, Q6WGOutputMode::AccumulateBF16));
            execute(Case(down, "odd_down_view_MMQ_T1280", 1, 1024, 1280, true,
                         false, false, Q6WGOutputMode::AccumulateBF16));
        }
        require(cases.size() == (args.head_prefix ? 11 : 10), "correctness case count differs");
        Json qualification;
        qualification.add("event", "qualification_complete");
        qualification.add("launch_id", args.launch_id);
        qualification.add("correctness_pass", true);
        qualification.add("full_fixture_roundtrips", 2);
        qualification.add("cases", cases.size());
        qualification.add("output_modes_per_case", 3);
        qualification.add("FP64_samples_per_case_per_route", 64);
        qualification.add("timing_has_started", false);
        qualification.add("positive_fixup_qualified", false);
        emit(qualification);
        for (const auto& qualified : cases) { screens += screen_qualified(*qualified, args.launch_id); }
        require(screens == (args.correctness_only ? 0 : args.head_prefix ? 10 : 7),
                "short screen endpoint count differs");
        check(cudaDeviceSynchronize(), "complete isolated run");
        Json complete;
        complete.add("event", "complete");
        complete.add("launch_id", args.launch_id);
        complete.add("correctness_pass", true);
        complete.add("cases", cases.size());
        complete.add("short_screen_endpoints", screens);
        complete.add("primary_short_screen_endpoints",
                     args.correctness_only ? 0 : 7);
        complete.add("all_correctness_passed_before_timing", true);
        complete.add("calls_per_sample", kRepeats);
        complete.add("warmup_samples", 4);
        complete.add("warmup_order", "ABAB");
        complete.add("measurement_blocks_per_endpoint", 3);
        complete.add("samples_per_block", 4);
        complete.raw("block_orders", "[\"ABBA\",\"BAAB\",\"ABBA\"]");
        complete.add("all_three_output_modes_checked_per_case", true);
        complete.add("full_head_T1_T4_StoreBF16", true);
        complete.add("full_down_T1_T4_AccumulateBF16", true);
        complete.add("full_down_T1280_three_modes", true);
        complete.add("actual_parent_odd_start_views_checked", true);
        complete.add("positive_fixup_qualified", false);
        complete.add("generic_consumers_qualified", false);
        complete.add("vision_qualified", false);
        complete.add("runtime_integration_qualified", false);
        complete.add("statistical_confirmation", false);
        complete.add("end_to_end_claim", false);
        emit(complete);
        return 0;
    } catch (const std::exception& error) {
        capture_launches = false;
        Json failure;
        failure.add("event", "failure");
        failure.add("error", error.what());
        failure.add("correctness_pass", false);
        failure.add("end_to_end_claim", false);
        emit(failure);
        return 1;
    }
}

