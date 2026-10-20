#include "whole_api.h"
#include "pack_api.h"
#include "whole_common.cuh"

#include <sstream>
#include <utility>

namespace whole_ffn_native {
#include "whole_native_producer.cuh"
}

namespace {

#include "whole_q6_decode.cuh"

constexpr int kDownRows = 5120;
constexpr int kDownK = 17408;
constexpr int kRepetitions = 32;
constexpr std::size_t kIntermediateCount = std::size_t(kRows) * kColumns;
constexpr std::size_t kDownCount = std::size_t(kDownRows) * kColumns;
constexpr std::size_t kPackedBytes = std::size_t(kDownK / 128) * kColumns * 144;
constexpr std::size_t kPackSlack = 128 * 144;
constexpr std::size_t kGuardBytes = 64;
constexpr const char* kGateSha =
    "be10db58b96b5614830c01afe1e28f9d55c4581a6c481827fa7b9d885a6e2790";
constexpr const char* kDownQ4Sha =
    "8a5bbd71b5016c0382895b156c554ff9909e7b7f1dccf1158a7de191b94ef3ce";
constexpr const char* kDownQ6Sha =
    "a381c6be36f572170ccc9c7e4a3f8b25e5061e4eb633649399f3ec9759cb43e2";

static_assert(kPackedBytes == 25067520);
static_assert(kDownK / 512 == 34 && kDownK / 256 == 68);

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

std::string json_string(const std::string& value) {
    std::ostringstream out;
    for (const unsigned char c : value) {
        if (c == '"' || c == '\\') { out << '\\' << char(c); }
        else if (c < 32) {
            out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
        } else { out << char(c); }
    }
    return out.str();
}

std::string sha256(const std::uint8_t* data, std::size_t size) {
    constexpr std::array<std::uint32_t, 64> constants{
        0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
        0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
        0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
        0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
        0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
        0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
        0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
        0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
        0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
        0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
        0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
        0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
        0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
        0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
        0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
        0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};
    std::array<std::uint32_t, 8> state{
        0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
        0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
    const auto rotate = [](std::uint32_t x, int n) { return (x >> n) | (x << (32 - n)); };
    const std::size_t padded = ((size + 9 + 63) / 64) * 64;
    const std::uint64_t bits = std::uint64_t(size) * 8;
    for (std::size_t offset = 0; offset < padded; offset += 64) {
        std::array<std::uint32_t, 64> words{};
        for (std::size_t i = 0; i < 64; ++i) {
            const std::size_t at = offset + i;
            const std::uint8_t byte = at < size ? data[at] :
                at == size ? 0x80 : at >= padded - 8 ?
                std::uint8_t(bits >> (8 * (padded - 1 - at))) : 0;
            words[i / 4] |= std::uint32_t(byte) << (24 - 8 * (i % 4));
        }
        for (int i = 16; i < 64; ++i) {
            const auto x = words[i - 15], y = words[i - 2];
            words[i] = words[i - 16] + (rotate(x, 7) ^ rotate(x, 18) ^ (x >> 3)) +
                       words[i - 7] + (rotate(y, 17) ^ rotate(y, 19) ^ (y >> 10));
        }
        auto a = state[0], b = state[1], c = state[2], d = state[3];
        auto e = state[4], f = state[5], g = state[6], h = state[7];
        for (int i = 0; i < 64; ++i) {
            const auto first = h + (rotate(e, 6) ^ rotate(e, 11) ^ rotate(e, 25)) +
                               ((e & f) ^ (~e & g)) + constants[i] + words[i];
            const auto second = (rotate(a, 2) ^ rotate(a, 13) ^ rotate(a, 22)) +
                                ((a & b) ^ (a & c) ^ (b & c));
            h = g; g = f; f = e; e = d + first;
            d = c; c = b; b = a; a = first + second;
        }
        state[0] += a; state[1] += b; state[2] += c; state[3] += d;
        state[4] += e; state[5] += f; state[6] += g; state[7] += h;
    }
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (auto value : state) { out << std::setw(8) << value; }
    return out.str();
}

std::vector<std::uint8_t> pinned_fixture(const std::filesystem::path& path,
                                        std::size_t bytes, const char* expected_sha) {
    require(std::filesystem::is_regular_file(path) && std::filesystem::file_size(path) == bytes,
            "whole FFN requires the exact full real fixture size");
    std::vector<std::uint8_t> result(bytes);
    std::ifstream input(path, std::ios::binary);
    input.read(reinterpret_cast<char*>(result.data()), static_cast<std::streamsize>(bytes));
    require(input.gcount() == static_cast<std::streamsize>(bytes), "incomplete fixture read");
    require(sha256(result.data(), result.size()) == expected_sha, "whole FFN fixture SHA256 mismatch");
    return result;
}

struct Guarded {
    DeviceBuffer storage;
    std::size_t bytes;
    void* data;

    explicit Guarded(std::size_t count)
        : storage(count + 2 * kGuardBytes), bytes(count),
          data(static_cast<std::uint8_t*>(storage.data) + kGuardBytes) {
        check(cudaMemset(storage.data, 0xa5, storage.size), "initialize buffer canaries");
        check(cudaMemset(data, 0xff, bytes), "initialize unwritten payload");
    }

    void from_host(const void* source, std::size_t count) {
        require(count <= bytes, "host input exceeds guarded payload");
        check(cudaMemcpy(data, source, count, cudaMemcpyHostToDevice), "copy guarded input");
    }

    template <typename T>
    std::vector<T> read(std::size_t count) const {
        require(count * sizeof(T) <= bytes, "host output exceeds guarded payload");
        std::vector<T> result(count);
        check(cudaMemcpy(result.data(), data, count * sizeof(T), cudaMemcpyDeviceToHost),
              "copy guarded output");
        return result;
    }

    void guards() const {
        std::array<std::uint8_t, 2 * kGuardBytes> actual{};
        check(cudaMemcpy(actual.data(), storage.data, kGuardBytes, cudaMemcpyDeviceToHost),
              "read leading canary");
        check(cudaMemcpy(actual.data() + kGuardBytes,
                         static_cast<const std::uint8_t*>(data) + bytes,
                         kGuardBytes, cudaMemcpyDeviceToHost), "read trailing canary");
        require(std::all_of(actual.begin(), actual.end(), [](auto x) { return x == 0xa5; }),
                "whole FFN buffer canary changed");
    }

    void untouched_tail(std::size_t begin) const {
        require(begin <= bytes, "tail outside payload");
        std::vector<std::uint8_t> tail(bytes - begin);
        check(cudaMemcpy(tail.data(), static_cast<const std::uint8_t*>(data) + begin,
                         tail.size(), cudaMemcpyDeviceToHost), "read activation slack");
        require(std::all_of(tail.begin(), tail.end(), [](auto x) { return x == 0xff; }),
                "whole FFN wrote activation slack");
    }
};

template <typename T>
void exact(const std::vector<T>& a, const std::vector<T>& b, const char* label) {
    require(a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0,
            label);
}

void compare_planes(const Guarded& baseline_plane, const Guarded& diagnostic_plane) {
    const auto expected = baseline_plane.read<float>(kIntermediateCount);
    const auto actual = diagnostic_plane.read<float>(kIntermediateCount);
    exact(expected, actual, "folded packed FP32 gate/up diagnostic differs from native MMQ");
    require(std::all_of(actual.begin(), actual.end(), [](float x) { return std::isfinite(x); }),
            "unwritten or nonfinite gate/up diagnostic");
}

void native_pack(const __nv_bfloat16* input, int k, int layout, void* output) {
    const dim3 grid(kColumns, k / 512, 1);
    auto* target = static_cast<block_q8_1_mmq*>(output);
    if (layout == 0) {
        whole_ffn_native::quantize_matrix_kernel<MMQ_Q8_1_DS_LAYOUT_D4>
            <<<grid, 128>>>(input, nullptr, target, k, kColumns);
    } else if (layout == 1) {
        whole_ffn_native::quantize_matrix_kernel<MMQ_Q8_1_DS_LAYOUT_DS4>
            <<<grid, 128>>>(input, nullptr, target, k, kColumns);
    } else { throw std::runtime_error("invalid whole FFN pack layout"); }
    check(cudaGetLastError(), "native whole FFN matrix pack");
}

struct Invocation {
    int layout;
    const void* gate;
    const void* up;
    const void* input_pack;
    const void* down_weight;
    float* gate_plane;
    float* up_plane;
    __nv_bfloat16* intermediate;
    void* packed;
    __nv_bfloat16* residual;
};

void invoke(bool candidate, const Invocation& args) {
    if (candidate) {
        require(q4_folded_ffn_pack_run(args.gate, args.up, kRowBytes, args.input_pack,
                                      kColumns, args.packed, args.layout,
                                      nullptr, nullptr, nullptr, nullptr),
                "folded pack declined the fixed whole FFN shape");
    } else {
        baseline(static_cast<const std::uint8_t*>(args.gate),
                 static_cast<const std::uint8_t*>(args.up), args.input_pack,
                 args.gate_plane, args.up_plane, nullptr, nullptr, args.intermediate, nullptr);
        native_pack(args.intermediate, kDownK, args.layout, args.packed);
    }
    require(whole_ffn_down_run(args.layout, args.down_weight, args.packed, args.residual, nullptr),
            "native whole FFN down Accumulate declined");
}

double time_sample(bool candidate, const Invocation& args, const Guarded& initial) {
    check(cudaMemcpyAsync(args.residual, initial.data, kDownCount * sizeof(__nv_bfloat16),
                          cudaMemcpyDeviceToDevice, nullptr), "reset carry outside events");
    cudaEvent_t begin = nullptr, end = nullptr;
    check(cudaEventCreate(&begin), "create whole FFN begin event");
    check(cudaEventCreate(&end), "create whole FFN end event");
    check(cudaEventRecord(begin, nullptr), "record whole FFN begin");
    for (int repeat = 0; repeat < kRepetitions; ++repeat) { invoke(candidate, args); }
    check(cudaEventRecord(end, nullptr), "record whole FFN end");
    check(cudaEventSynchronize(end), "synchronize whole FFN sample");
    float ms = 0.0f;
    check(cudaEventElapsedTime(&ms, begin, end), "read whole FFN elapsed time");
    check(cudaEventDestroy(end), "destroy whole FFN end event");
    check(cudaEventDestroy(begin), "destroy whole FFN begin event");
    return double(ms) / kRepetitions;
}

double half_round_bound(double value) {
    return std::abs(value) * std::ldexp(1.0, -11) + std::ldexp(1.0, -25);
}

struct OracleStats {
    int samples = 0;
    double max_abs = 0.0;
    double max_bound_ratio = 0.0;
};

OracleStats gate_oracle(const std::vector<std::uint8_t>& weights,
                        const std::vector<block_q8_1_mmq>& input,
                        const std::vector<__nv_bfloat16>& output) {
    constexpr std::array<int, 16> rows{
        0, 1, 7, 15, 16, 31, 32, 63, 64, 127, 128,
        kRows / 4, kRows / 3, kRows / 2, kRows - 2, kRows - 1};
    constexpr std::array<int, 5> columns{0, 1, kColumns / 3, kColumns / 2, kColumns - 1};
    OracleStats stats;
    for (int column : columns) {
        for (int row : rows) {
            const auto* gate = weights.data() + std::size_t(row) * kRowBytes;
            const auto* up = weights.data() + std::size_t(kRows + row) * kRowBytes;
            const double g = q4k_q8_1_dot(gate, input.data(), column, kColumns);
            const double u = q4k_q8_1_dot(up, input.data(), column, kColumns);
            const double expected = g / (1.0 + std::exp(-g)) * u;
            const double actual = __bfloat162float(output[std::size_t(column) * kRows + row]);
            const double bound = std::max(0.005, std::abs(expected) * 0.006);
            const double error = std::abs(actual - expected);
            require(std::isfinite(actual) && std::isfinite(expected) && error <= bound,
                    "whole FFN gate/up differs from independent FP64 SwiGLU oracle");
            ++stats.samples;
            stats.max_abs = std::max(stats.max_abs, error);
            stats.max_bound_ratio = std::max(stats.max_bound_ratio, error / bound);
        }
    }
    return stats;
}

int producer_oracle(int layout, const std::vector<__nv_bfloat16>& input,
                    const std::vector<block_q8_1_mmq>& packed) {
    constexpr std::array<int, 8> columns{0, 1, 127, 128, kColumns / 3, kColumns / 2,
                                        kColumns - 2, kColumns - 1};
    constexpr double eps = 8.0 * std::numeric_limits<float>::epsilon();
    int groups = 0;
    for (int column : columns) {
        for (int start = 0; start < kDownK; start += 32) {
            double maximum = 0.0, total = 0.0, magnitude = 0.0;
            for (int i = 0; i < 32; ++i) {
                const double x = __bfloat162float(input[std::size_t(column) * kDownK + start + i]);
                require(std::isfinite(x), "nonfinite BF16 input to native down pack");
                maximum = std::max(maximum, std::abs(x));
                total += x;
                magnitude += std::abs(x);
            }
            const auto& block = packed[std::size_t(start / 128) * kColumns + column];
            const int group = (start % 128) / 32;
            const double ideal_scale = maximum / 127.0;
            const double scale = layout == 0 ? block.d4[group] : __low2float(block.ds4[group]);
            const double scale_bound = eps * ideal_scale +
                (layout == 1 ? half_round_bound(ideal_scale) : 0.0);
            require(std::isfinite(scale) && scale >= 0.0 &&
                    std::abs(scale - ideal_scale) <= scale_bound,
                    "native down pack scale differs from independent BF16 quantization oracle");
            if (layout == 1) {
                const double sum = __high2float(block.ds4[group]);
                const double sum_bound = half_round_bound(total) + eps * magnitude * 1.001;
                require(std::isfinite(sum) && std::abs(sum - total) <= sum_bound,
                        "native DS4 sum differs from independent BF16 sum oracle");
            }
            for (int i = 0; i < 32; ++i) {
                const double x = __bfloat162float(input[std::size_t(column) * kDownK + start + i]);
                const int q = block.qs[start % 128 + i];
                require(q >= -127 && q <= 127 &&
                        std::abs(ideal_scale * q - x) <= 0.5 * ideal_scale + eps * maximum,
                        "native down pack code differs from independent quantization oracle");
            }
            ++groups;
        }
    }
    return groups;
}

std::pair<double, double> q4_down_oracle(const std::uint8_t* weight,
                                        const std::vector<block_q8_1_mmq>& packed,
                                        int column) {
    double result = 0.0, coefficient_error = 0.0, magnitude = 0.0;
    for (int kb = 0; kb < kDownK / 256; ++kb) {
        const auto* block = weight + std::size_t(kb) * 144;
        const double d = half_value(block), dmin = half_value(block + 2);
        const auto* scales = block + 4;
        for (int group = 0; group < 8; ++group) {
            const int sc = group < 4 ? scales[group] & 63 :
                (scales[group + 4] & 15) | ((scales[group - 4] >> 6) << 4);
            const int mn = group < 4 ? scales[group + 4] & 63 :
                (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4);
            const auto& input = packed[std::size_t(2 * kb + group / 4) * kColumns + column];
            const double dy = __low2float(input.ds4[group % 4]);
            const double sy = __high2float(input.ds4[group % 4]);
            int dot = 0;
            for (int i = 0; i < 32; ++i) {
                const int q = (block[16 + 32 * (group / 2) + i] >> (4 * (group % 2))) & 15;
                dot += q * int(input.qs[32 * (group % 4) + i]);
            }
            const double a = d * sc, b = dmin * mn;
            require(std::isfinite(a) && std::isfinite(b) &&
                    std::abs(a) <= 65504.0 && std::abs(b) <= 65504.0,
                    "Q4 public coefficient exceeds finite native half lowering");
            const double first = a * dy * dot, second = b * sy;
            result += first - second;
            magnitude += std::abs(first) + std::abs(second);
            coefficient_error += half_round_bound(a) * std::abs(dy * dot) +
                                 half_round_bound(b) * std::abs(sy);
        }
    }
    constexpr double unit = std::numeric_limits<float>::epsilon() / 2.0;
    constexpr double operations = 3.0 * (kDownK / 32);
    constexpr double gamma = operations * unit / (1.0 - operations * unit);
    const double bound = coefficient_error + gamma * (magnitude + coefficient_error) +
                         16.0 * std::numeric_limits<double>::epsilon() * magnitude;
    return {result, bound};
}

OracleStats down_oracle(int layout, const std::vector<std::uint8_t>& weights,
                        const std::vector<block_q8_1_mmq>& packed,
                        const std::vector<float>& actual) {
    constexpr std::array<int, 8> rows{0, 1, 127, 128, kDownRows / 3, kDownRows / 2,
                                     kDownRows - 2, kDownRows - 1};
    constexpr std::array<int, 8> columns{0, 1, 127, 128, kColumns / 3, kColumns / 2,
                                        kColumns - 2, kColumns - 1};
    OracleStats stats;
    for (int column : columns) {
        for (int row : rows) {
            double expected = 0.0, bound = 0.0;
            if (layout == 0) {
                const auto* weight = weights.data() + std::size_t(row) * 14280;
                for (int i = 0; i < kDownK; ++i) {
                    const auto& block = packed[std::size_t(i / 128) * kColumns + column];
                    const double input = double(block.d4[(i % 128) / 32]) * int(block.qs[i % 128]);
                    expected += q6_runtime_weight_value(weight, i) * input;
                }
                bound = 0.02 + 2.0e-4 * std::abs(expected);
            } else {
                const auto reference = q4_down_oracle(
                    weights.data() + std::size_t(row) * 9792, packed, column);
                expected = reference.first;
                bound = reference.second;
            }
            const double got = actual[std::size_t(column) * kDownRows + row];
            const double error = std::abs(got - expected);
            require(std::isfinite(got) && std::isfinite(expected) && std::isfinite(bound) &&
                    error <= bound, "whole FFN down differs from independent FP64 oracle");
            ++stats.samples;
            stats.max_abs = std::max(stats.max_abs, error);
            stats.max_bound_ratio = std::max(stats.max_bound_ratio, bound == 0.0 ? 0.0 : error / bound);
        }
    }
    return stats;
}

void run_case(int layout, const std::vector<std::uint8_t>& gate_weights,
              const Guarded& gate_device, const Guarded& input_pack,
              const std::vector<block_q8_1_mmq>& host_input_pack,
              const std::vector<std::uint8_t>& down_weights, const std::string& tag,
              std::uint64_t seed) {
    const char* name = layout == 0 ? "Q6_D4" : "Q4_DS4";
    WholeFfnDownPlan plan{};
    require(whole_ffn_down_plan(layout, &plan), "native down plan is not full-K I128/J128 no-fixup");
    require(ffn_pair_probe::matrix_fixup_bytes(kColumns) == 0, "native gate/up unexpectedly needs fixup");
    Guarded down_weight(down_weights.size());
    down_weight.from_host(down_weights.data(), down_weights.size());
    Guarded gate_plane(kIntermediateCount * sizeof(float)), up_plane(kIntermediateCount * sizeof(float));
    Guarded diagnostic_gate(kIntermediateCount * sizeof(float));
    Guarded diagnostic_up(kIntermediateCount * sizeof(float));
    Guarded intermediate(kIntermediateCount * sizeof(__nv_bfloat16));
    Guarded diagnostic_bf16(kIntermediateCount * sizeof(__nv_bfloat16));
    Guarded packed_a(kPackedBytes + kPackSlack), packed_b(kPackedBytes + kPackSlack);
    Guarded output_a(kDownCount * sizeof(__nv_bfloat16));
    Guarded output_b(kDownCount * sizeof(__nv_bfloat16));
    Guarded initial(kDownCount * sizeof(__nv_bfloat16));
    Guarded down_fp32_a(kDownCount * sizeof(float)), down_fp32_b(kDownCount * sizeof(float));
    std::vector<__nv_bfloat16> carry(kDownCount);
    for (std::size_t i = 0; i < carry.size(); ++i) {
        carry[i] = __float2bfloat16((i & 1 ? -1.0f : 1.0f) * float(1 + i % 17) / 32.0f);
    }
    initial.from_host(carry.data(), initial.bytes);
    output_a.from_host(carry.data(), initial.bytes);
    output_b.from_host(carry.data(), initial.bytes);
    const auto* gate = static_cast<const std::uint8_t*>(gate_device.data);
    const auto* up = gate + std::size_t(kRows) * kRowBytes;
    Invocation baseline_args{layout, gate, up, input_pack.data, down_weight.data,
        static_cast<float*>(gate_plane.data), static_cast<float*>(up_plane.data),
        static_cast<__nv_bfloat16*>(intermediate.data), packed_a.data,
        static_cast<__nv_bfloat16*>(output_a.data)};
    Invocation candidate_args = baseline_args;
    candidate_args.packed = packed_b.data;
    candidate_args.residual = static_cast<__nv_bfloat16*>(output_b.data);
    invoke(false, baseline_args);
    invoke(true, candidate_args);
    check(cudaDeviceSynchronize(), "synchronize single-call whole FFN correctness");
    const auto host_packed_a = packed_a.read<block_q8_1_mmq>(kPackedBytes / 144);
    const auto host_packed_b = packed_b.read<block_q8_1_mmq>(kPackedBytes / 144);
    exact(host_packed_a, host_packed_b, "full D4/DS4 packed payload differs");
    const auto host_output_a = output_a.read<__nv_bfloat16>(kDownCount);
    const auto host_output_b = output_b.read<__nv_bfloat16>(kDownCount);
    exact(host_output_a, host_output_b, "whole FFN single-call BF16 residual differs");
    require(q4_folded_ffn_pack_run(gate, up, kRowBytes, input_pack.data, kColumns,
                                  packed_b.data, layout, static_cast<float*>(diagnostic_gate.data),
                                  static_cast<float*>(diagnostic_up.data),
                                  static_cast<__nv_bfloat16*>(diagnostic_bf16.data), nullptr),
            "folded diagnostic pack declined");
    check(cudaDeviceSynchronize(), "synchronize whole FFN diagnostics");
    exact(host_packed_b, packed_b.read<block_q8_1_mmq>(kPackedBytes / 144),
          "diagnostic and timed pack payload differ");
    compare_planes(gate_plane, diagnostic_gate);
    compare_planes(up_plane, diagnostic_up);
    const auto host_intermediate = intermediate.read<__nv_bfloat16>(kIntermediateCount);
    exact(host_intermediate, diagnostic_bf16.read<__nv_bfloat16>(kIntermediateCount),
          "folded diagnostic BF16 semantic boundary differs");
    require(whole_ffn_down_fp32(layout, down_weight.data, packed_a.data,
                               static_cast<float*>(down_fp32_a.data), nullptr) &&
            whole_ffn_down_fp32(layout, down_weight.data, packed_b.data,
                               static_cast<float*>(down_fp32_b.data), nullptr),
            "native FP32 down diagnostics declined");
    check(cudaDeviceSynchronize(), "synchronize native down FP32 diagnostics");
    const auto fp32_a = down_fp32_a.read<float>(kDownCount);
    const auto fp32_b = down_fp32_b.read<float>(kDownCount);
    exact(fp32_a, fp32_b, "whole FFN down FP32 diagnostic differs");
    std::size_t carry_sensitive = 0;
    for (std::size_t i = 0; i < kDownCount; ++i) {
        require(std::isfinite(fp32_a[i]), "unwritten or nonfinite native down FP32 output");
        const float sum = fp32_a[i] + __bfloat162float(carry[i]);
        const auto expected = __float2bfloat16(sum);
        const auto without_carry = __float2bfloat16(fp32_a[i]);
        require(std::memcmp(&expected, &host_output_a[i], sizeof(expected)) == 0 &&
                std::memcmp(&expected, &host_output_b[i], sizeof(expected)) == 0 &&
                std::isfinite(__bfloat162float(expected)),
                "native down BF16 Accumulate differs from FP32 plus original nonzero residual");
        carry_sensitive += std::memcmp(&expected, &without_carry, sizeof(expected)) != 0;
    }
    require(carry_sensitive > 0, "whole FFN input has no carry-sensitive output");
    const auto gate_stats = gate_oracle(gate_weights, host_input_pack, host_intermediate);
    const auto producer_groups = producer_oracle(layout, host_intermediate, host_packed_b);
    const auto down_stats = down_oracle(layout, down_weights, host_packed_b, fp32_b);
    require(!q4_folded_ffn_pack_run(gate, up, kRowBytes, input_pack.data, kColumns,
                                   packed_b.data, 2, nullptr, nullptr, nullptr, nullptr) &&
            !q4_folded_ffn_pack_run(gate, up, kRowBytes, input_pack.data, kColumns,
                                   packed_b.data, layout, static_cast<float*>(diagnostic_gate.data),
                                   nullptr, nullptr, nullptr) &&
            !q4_folded_ffn_pack_run(gate, up, kRowBytes, input_pack.data, kColumns - 1,
                                   packed_b.data, layout, nullptr, nullptr, nullptr, nullptr),
            "whole FFN pack accepted an invalid API call");
    check(cudaDeviceSynchronize(), "synchronize invalid whole FFN API calls");
    exact(host_packed_b, packed_b.read<block_q8_1_mmq>(kPackedBytes / 144),
          "rejected pack API call modified packed output");
    const std::array<const Guarded*, 14> buffers{
        &gate_device, &input_pack, &down_weight, &gate_plane, &up_plane, &diagnostic_gate,
        &diagnostic_up, &intermediate, &diagnostic_bf16, &packed_a, &packed_b,
        &output_a, &output_b, &initial};
    for (const auto* buffer : buffers) { buffer->guards(); }
    down_fp32_a.guards();
    down_fp32_b.guards();
    packed_a.untouched_tail(kPackedBytes);
    packed_b.untouched_tail(kPackedBytes);
    std::cout << std::setprecision(10)
              << "{\"kind\":\"correctness\",\"status\":\"PASS\",\"case\":\"" << name
              << "\",\"gate_k\":5120,\"intermediate_rows\":17408,\"down_rows\":5120,\"columns\":1280"
              << ",\"fixture_gate_sha256\":\"" << kGateSha << "\",\"fixture_down_sha256\":\""
              << (layout == 0 ? kDownQ6Sha : kDownQ4Sha)
              << "\",\"exact_packed_bytes\":" << kPackedBytes
              << ",\"exact_gate_up_fp32\":true,\"exact_intermediate_bf16\":true"
              << ",\"exact_down_fp32\":true,\"exact_down_bf16\":true"
              << ",\"diagnostic_matches_timed_pack\":true,\"nonzero_carry_expected\":true"
              << ",\"carry_sensitive_outputs\":" << carry_sensitive
              << ",\"gate_fp64_samples\":" << gate_stats.samples
              << ",\"gate_fp64_max_abs\":" << gate_stats.max_abs
              << ",\"producer_oracle_groups\":" << producer_groups
              << ",\"down_fp64_samples\":" << down_stats.samples
              << ",\"down_fp64_max_abs\":" << down_stats.max_abs
              << ",\"down_fp64_max_bound_ratio\":" << down_stats.max_bound_ratio
              << ",\"down_oracle_bound\":\""
              << (layout == 0 ? "0.02+2e-4*abs(reference)" : "half_coefficient_bound+gamma1632*L1")
              << "\",\"down_J\":" << plan.J << ",\"down_I\":" << plan.I
              << ",\"down_blocks\":" << plan.blocks << ",\"down_shared_bytes\":" << plan.shared
              << ",\"down_fixup\":false,\"down_store_fusion\":true,\"canaries\":true"
              << ",\"activation_slack_untouched\":true,\"invalid_pack_api_guards\":true"
              << ",\"input_seed\":" << seed << ",\"fresh_launch_id\":\"" << json_string(tag)
              << "\"}\n" << std::flush;
    const bool candidate_first = tag.find("candidate-first") != std::string::npos;
    for (int warmup = 0; warmup < 4; ++warmup) {
        const bool candidate = (warmup + int(candidate_first)) % 2 != 0;
        time_sample(candidate, candidate ? candidate_args : baseline_args, initial);
    }
    constexpr std::array<int, 4> abba{0, 1, 1, 0}, baab{1, 0, 0, 1};
    for (int block = 0; block < 3; ++block) {
        const bool baseline_outer = (block + int(candidate_first)) % 2 == 0;
        const auto& order = baseline_outer ? abba : baab;
        std::array<double, 4> samples{};
        double a = 0.0, b = 0.0;
        for (int position = 0; position < 4; ++position) {
            const bool candidate = order[position] != 0;
            samples[position] = time_sample(candidate, candidate ? candidate_args : baseline_args, initial);
            (candidate ? b : a) += samples[position] / 2.0;
        }
        exact(output_a.read<__nv_bfloat16>(kDownCount), output_b.read<__nv_bfloat16>(kDownCount),
              "32-call evolving residual differs between whole FFN arms");
        std::cout << std::setprecision(10)
                  << "{\"kind\":\"timing\",\"case\":\"" << name << "\",\"block\":" << block
                  << ",\"order\":\"" << (baseline_outer ? "ABBA" : "BAAB")
                  << "\",\"repetitions_per_sample\":32,\"warmup_samples\":4"
                  << ",\"baseline_ms\":" << a << ",\"candidate_ms\":" << b
                  << ",\"samples_ms\":[" << samples[0] << ',' << samples[1] << ','
                  << samples[2] << ',' << samples[3] << ']'
                  << ",\"operator_time_decrease_percent\":" << 100.0 * (a - b) / a
                  << ",\"scope\":\"prequantized_gate_input_to_accumulated_residual\""
                  << ",\"common_input_pack_outside_events\":true"
                  << ",\"carry_reset_outside_events\":true,\"evolving_carry_calls\":32"
                  << ",\"evolving_carry_exact\":true,\"exploratory_only\":true"
                  << ",\"fresh_launch_id\":\"" << json_string(tag) << "\"}\n" << std::flush;
    }
    for (const auto* buffer : buffers) { buffer->guards(); }
    packed_a.untouched_tail(kPackedBytes);
    packed_b.untouched_tail(kPackedBytes);
}

}

int main(int argc, char** argv) {
    try {
        require((argc == 3 || argc == 4) && std::string(argv[2]).size() > 0,
                "usage: whole-ffn-probe <operators-fixture-dir> <fresh-launch-id> [seed]");
        require(sha256(nullptr, 0) ==
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                "SHA256 empty-input self-test failed");
        const std::array<std::uint8_t, 3> abc{97, 98, 99};
        require(sha256(abc.data(), abc.size()) ==
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
                "SHA256 abc self-test failed");
        const std::filesystem::path fixtures = argv[1];
        const std::string tag = argv[2];
        const std::uint64_t seed = argc == 4 ? std::stoull(argv[3]) : 309086;
        const auto gate = pinned_fixture(fixtures / "q4-full-k5120.bin", kFixtureBytes, kGateSha);
        const auto down_q4 = pinned_fixture(fixtures / "q4-full-k17408.bin", 50135040, kDownQ4Sha);
        const auto down_q6 = pinned_fixture(fixtures / "q6-full-k17408.bin", 73113600, kDownQ6Sha);
        require(ninfer::ops::gguf::detail::device_facts().cc == 860, "whole FFN requires SM86");
        const auto input = make_input(seed);
        Guarded gate_device(gate.size()), input_device(input.size() * sizeof(__nv_bfloat16));
        constexpr std::size_t input_pack_count = std::size_t(kK / 128) * kColumns;
        Guarded input_pack(input_pack_count * 144 + kPackSlack);
        gate_device.from_host(gate.data(), gate.size());
        input_device.from_host(input.data(), input_device.bytes);
        native_pack(static_cast<const __nv_bfloat16*>(input_device.data), kK, 1, input_pack.data);
        check(cudaDeviceSynchronize(), "synchronize common native input DS4 pack");
        const auto host_input_pack = input_pack.read<block_q8_1_mmq>(input_pack_count);
        input_pack.untouched_tail(input_pack_count * 144);
        run_case(0, gate, gate_device, input_pack, host_input_pack, down_q6, tag, seed);
        run_case(1, gate, gate_device, input_pack, host_input_pack, down_q4, tag, seed);
        input_device.guards();
        input_pack.untouched_tail(input_pack_count * 144);
        std::cout << "{\"kind\":\"complete\",\"status\":\"PASS\",\"cases\":2,\"timing_blocks\":6,"
                     "\"scope\":\"exploratory whole FFN only\",\"full_request_claim\":false}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
