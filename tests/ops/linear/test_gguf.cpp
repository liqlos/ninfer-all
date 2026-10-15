// GGUF block matrices through the public linear, linear_add and linear_swiglu Ops and the embedding
// gather, against an FP64 product of the exactly dequantized weights. The products quantize the
// activation to q8_1 as llama.cpp does, so they are held to that arithmetic's error, not to A16's.

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ops/linear/gguf/ggml_bridge.h"
#include "ops/linear/gguf/gguf_linear.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

using namespace ninfer;

struct TypeCase {
    QType qtype;
    const char* name;
    // Byte offsets of the binary16 multipliers inside one block, set to finite values.
    std::vector<int> half_fields;
};

std::vector<TypeCase> cases() {
    return {
        {QType::GGUF_Q8_0, "q8_0", {0}},
        {QType::GGUF_Q2_K, "q2_k", {80, 82}},
        {QType::GGUF_Q3_K, "q3_k", {108}},
        {QType::GGUF_Q4_K, "q4_k", {0, 2}},
        {QType::GGUF_Q5_K, "q5_k", {0, 2}},
        {QType::GGUF_Q6_K, "q6_k", {208}},
        {QType::GGUF_IQ2_XXS, "iq2_xxs", {0}},
        {QType::GGUF_IQ2_XS, "iq2_xs", {0}},
        {QType::GGUF_IQ2_S, "iq2_s", {0}},
        {QType::GGUF_IQ3_XXS, "iq3_xxs", {0}},
        {QType::GGUF_IQ3_S, "iq3_s", {0}},
        {QType::GGUF_IQ1_S, "iq1_s", {0}},
        {QType::GGUF_IQ1_M, "iq1_m", {}},
        {QType::GGUF_IQ4_NL, "iq4_nl", {0}},
        {QType::GGUF_IQ4_XS, "iq4_xs", {0}},
    };
}

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

std::uint16_t half_bits(float value) {
    const __half h = __float2half(value);
    std::uint16_t bits;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
}

// Random blocks with every binary16 multiplier finite; IQ1_M spreads its multiplier over the top
// nibbles of its four scale words.
std::vector<std::uint8_t> random_blocks(const TypeCase& c, std::int32_t rows, std::int32_t k,
                                        std::mt19937& rng, bool mixed_global_scale_signs = false) {
    const auto block = gguf_block_shape(c.qtype);
    const std::int64_t count = std::int64_t(rows) * (k / block.elements);
    std::vector<std::uint8_t> bytes(count * block.bytes);
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> scale(0.002f, 0.02f);
    for (auto& b : bytes) { b = static_cast<std::uint8_t>(byte(rng)); }
    for (std::int64_t i = 0; i < count; ++i) {
        std::uint8_t* base = bytes.data() + i * block.bytes;
        for (const int offset : c.half_fields) {
            float value = scale(rng);
            if (mixed_global_scale_signs && (i & 1) != 0) { value = -value; }
            const std::uint16_t bits = half_bits(value);
            std::memcpy(base + offset, &bits, sizeof(bits));
        }
        if (c.qtype == QType::GGUF_IQ1_M) {
            const std::uint16_t d = half_bits(scale(rng));
            auto* sc              = reinterpret_cast<std::uint16_t*>(base + 48);
            for (int j = 0; j < 4; ++j) {
                sc[j] = static_cast<std::uint16_t>((sc[j] & 0x0fff) | (((d >> (4 * j)) & 0xf) << 12));
            }
        }
    }
    return bytes;
}

bool load_blocks(const std::string& directory, const std::string& filename,
                 std::size_t expected_bytes, std::vector<std::uint8_t>& bytes) {
    const std::string path = directory + "/" + filename;
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        if (std::filesystem::exists(path)) {
            throw std::runtime_error("could not open weight file: " + path);
        }
        return false;
    }
    const std::streampos size = input.tellg();
    if (size < 0 || static_cast<std::uintmax_t>(size) != expected_bytes) {
        throw std::runtime_error("weight file has wrong size: " + path);
    }
    bytes.resize(expected_bytes);
    input.seekg(0);
    input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!input || static_cast<std::size_t>(input.gcount()) != bytes.size()) {
        throw std::runtime_error("could not read weight file: " + path);
    }
    return true;
}

Weight gguf_weight(QType qtype, const void* data, std::int32_t rows, std::int32_t k) {
    Weight w;
    w.qtype   = qtype;
    w.layout  = QuantLayout::GgufBlocks;
    w.qdata   = data;
    w.payload = data;
    w.n = w.shape[0] = w.padded_shape[0] = rows;
    w.k = w.shape[1] = w.padded_shape[1] = k;
    w.ndim       = 2;
    w.group_size = static_cast<std::uint32_t>(gguf_block_shape(qtype).elements);
    w.group      = gguf_block_shape(qtype).elements;
    return w;
}

std::string fixture_prefix(QType qtype) {
    if (qtype == QType::GGUF_Q4_K) { return "q4"; }
    if (qtype == QType::GGUF_Q6_K) { return "q6"; }
    throw std::logic_error("unsupported real-shape GGUF fixture type");
}

std::vector<float> to_float(const std::vector<__nv_bfloat16>& values) {
    std::vector<float> out(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { out[i] = __bfloat162float(values[i]); }
    return out;
}

struct Error {
    double relative_rms = 0;
    double relative_max = 0;
};

Error compare(const std::vector<float>& got, const std::vector<double>& want) {
    double num = 0, den = 0, worst = 0, scale = 0;
    for (std::size_t i = 0; i < want.size(); ++i) {
        const double d = double(got[i]) - want[i];
        num += d * d;
        den += want[i] * want[i];
        worst = std::max(worst, std::abs(d));
        scale = std::max(scale, std::abs(want[i]));
    }
    return {std::sqrt(num / std::max(den, 1e-30)), worst / std::max(scale, 1e-30)};
}

int run_type(const TypeCase& c, std::uint32_t seed) {
    constexpr std::int32_t kRows = 256;
    constexpr std::int32_t kK    = 1536;
    const std::int32_t widths[]  = {1, 2, 3, 5, 8, 9, 17, 40, 70, 130};
    const std::int32_t max_t     = 130;
    std::mt19937 rng(seed);
    int failures = 0;

    const auto host_blocks = random_blocks(c, 2 * kRows, kK, rng);
    DeviceBuffer blocks(host_blocks.size());
    blocks.copy_from_host(host_blocks.data(), host_blocks.size());
    const auto block        = gguf_block_shape(c.qtype);
    const std::int64_t row_bytes = std::int64_t(kK / block.elements) * block.bytes;
    const Weight gate = gguf_weight(c.qtype, blocks.p, kRows, kK);
    Weight up         = gate;
    up.qdata          = static_cast<const std::byte*>(blocks.p) + kRows * row_bytes;
    const Weight both = gguf_weight(c.qtype, blocks.p, 2 * kRows, kK);

    // Exact weights.
    DeviceBuffer exact_device(std::size_t(2) * kRows * kK * sizeof(float));
    ops::gguf::dequantize_rows(ops::detail::gguf_type(c.qtype), blocks.p, row_bytes, kK,
                               nullptr, 2 * kRows, static_cast<float*>(exact_device.p), kK,
                               nullptr);
    check(cudaDeviceSynchronize(), "dequantize");
    std::vector<float> exact(std::size_t(2) * kRows * kK);
    exact_device.copy_to_host(exact.data(), exact.size() * sizeof(float));
    for (const float v : exact) {
        if (!std::isfinite(v)) {
            std::cerr << c.name << ": non-finite dequantized weight\n";
            return 1;
        }
    }

    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<__nv_bfloat16> x_host(std::size_t(kK) * max_t);
    for (auto& v : x_host) { v = __float2bfloat16(normal(rng)); }
    const auto x_float = to_float(x_host);
    DeviceBuffer x_device(x_host.size() * sizeof(__nv_bfloat16));
    x_device.copy_from_host(x_host.data(), x_host.size() * sizeof(__nv_bfloat16));

    std::vector<std::int32_t> columns(kK);
    for (std::int32_t i = 0; i < kK; ++i) { columns[i] = i; }
    std::shuffle(columns.begin(), columns.end(), rng);
    DeviceBuffer columns_device(columns.size() * sizeof(std::int32_t));
    columns_device.copy_from_host(columns.data(), columns.size() * sizeof(std::int32_t));
    Weight gathered        = gate;
    gathered.input_columns = static_cast<const std::int32_t*>(columns_device.p);

    const auto product = [&](std::int32_t row0, std::int32_t t, bool permuted) {
        std::vector<double> out(std::size_t(kRows) * t);
        for (std::int32_t col = 0; col < t; ++col) {
            for (std::int32_t r = 0; r < kRows; ++r) {
                double acc = 0;
                const float* w = exact.data() + std::size_t(row0 + r) * kK;
                const float* x = x_float.data() + std::size_t(col) * kK;
                for (std::int32_t i = 0; i < kK; ++i) {
                    acc += double(w[i]) * x[permuted ? columns[i] : i];
                }
                out[std::size_t(col) * kRows + r] = acc;
            }
        }
        return out;
    };

    const std::size_t capacity = std::max(
        {ops::linear_workspace_capacity_bytes(c.qtype, kRows, kK, ops::LinearPolicy::A16Only, 1,
                                              max_t),
         ops::linear_swiglu_workspace_capacity_bytes(c.qtype, 2 * kRows, kK,
                                                     ops::LinearPolicy::A16Only, 1, max_t),
         ops::linear_swiglu_pair_workspace_capacity_bytes(c.qtype, c.qtype, kRows, kK,
                                                          ops::LinearPolicy::A16Only, 1, max_t)});
    WorkspaceArena workspace(capacity + (1U << 20));

    // q8_1 activations cost one product about 1 % RMS against the exact weights; a SwiGLU output
    // carries two such products, so it gets twice the budget.
    const auto run_check = [&](const std::string& label, const std::vector<float>& got,
                               const std::vector<double>& want) {
        const Error e     = compare(got, want);
        const double gain = label.starts_with("swiglu") ? 2.0 : 1.0;
        const bool finite = std::all_of(got.begin(), got.end(),
                                        [](float value) { return std::isfinite(value); });
        const bool ok = finite && e.relative_rms < 2e-2 * gain && e.relative_max < 8e-2 * gain;
        if (!ok) {
            std::cerr << "FAIL " << c.name << " " << label << " rms=" << e.relative_rms
                      << " max=" << e.relative_max << '\n';
        }
        return ok ? 0 : 1;
    };

    for (const std::int32_t t : widths) {
        Tensor x(x_device.p, DType::BF16, {kK, t});
        DeviceBuffer out_device(std::size_t(kRows) * t * sizeof(__nv_bfloat16));
        Tensor out(out_device.p, DType::BF16, {kRows, t});
        std::vector<__nv_bfloat16> host(std::size_t(kRows) * t);

        ops::linear(x, gate, out, ops::LinearPolicy::A16Only, workspace, nullptr);
        check(cudaDeviceSynchronize(), "linear");
        out_device.copy_to_host(host.data(), host.size() * sizeof(__nv_bfloat16));
        const auto plain = product(0, t, false);
        failures += run_check("linear T=" + std::to_string(t), to_float(host), plain);

        ops::linear(x, gathered, out, ops::LinearPolicy::A16Only, workspace, nullptr);
        check(cudaDeviceSynchronize(), "gathered linear");
        out_device.copy_to_host(host.data(), host.size() * sizeof(__nv_bfloat16));
        failures += run_check("gathered T=" + std::to_string(t), to_float(host),
                              product(0, t, true));

        std::vector<__nv_bfloat16> residual(std::size_t(kRows) * t);
        for (auto& v : residual) { v = __float2bfloat16(normal(rng)); }
        out_device.copy_from_host(residual.data(), residual.size() * sizeof(__nv_bfloat16));
        ops::linear_add(x, gate, out, ops::LinearPolicy::A16Only, workspace, nullptr);
        check(cudaDeviceSynchronize(), "linear_add");
        out_device.copy_to_host(host.data(), host.size() * sizeof(__nv_bfloat16));
        std::vector<double> sum = plain;
        for (std::size_t i = 0; i < sum.size(); ++i) { sum[i] += __bfloat162float(residual[i]); }
        failures += run_check("linear_add T=" + std::to_string(t), to_float(host), sum);

        const auto upper = product(kRows, t, false);
        std::vector<double> swiglu(plain.size());
        for (std::size_t i = 0; i < swiglu.size(); ++i) {
            swiglu[i] = plain[i] / (1.0 + std::exp(-plain[i])) * upper[i];
        }
        ops::linear_swiglu(x, both, out, ops::LinearPolicy::A16Only, workspace, nullptr);
        check(cudaDeviceSynchronize(), "linear_swiglu");
        out_device.copy_to_host(host.data(), host.size() * sizeof(__nv_bfloat16));
        failures += run_check("swiglu T=" + std::to_string(t), to_float(host), swiglu);

        ops::linear_swiglu(x, gate, up, out, ops::LinearPolicy::A16Only, workspace, nullptr);
        check(cudaDeviceSynchronize(), "linear_swiglu pair");
        out_device.copy_to_host(host.data(), host.size() * sizeof(__nv_bfloat16));
        failures += run_check("swiglu pair T=" + std::to_string(t), to_float(host), swiglu);
    }

    // The token-table gather is the exact weights rounded to BF16.
    const std::vector<std::int32_t> ids{3, 0, 255, 17, 3};
    DeviceBuffer ids_device(ids.size() * sizeof(std::int32_t));
    ids_device.copy_from_host(ids.data(), ids.size() * sizeof(std::int32_t));
    DeviceBuffer rows_device(ids.size() * kK * sizeof(__nv_bfloat16));
    Tensor ids_tensor(ids_device.p, DType::I32, {static_cast<std::int32_t>(ids.size())});
    Tensor rows(rows_device.p, DType::BF16, {kK, static_cast<std::int32_t>(ids.size())});
    ops::embedding(ids_tensor, gate, rows, nullptr);
    check(cudaDeviceSynchronize(), "embedding");
    std::vector<__nv_bfloat16> gathered_rows(ids.size() * kK);
    rows_device.copy_to_host(gathered_rows.data(), gathered_rows.size() * sizeof(__nv_bfloat16));
    for (std::size_t i = 0; i < ids.size(); ++i) {
        for (std::int32_t j = 0; j < kK; ++j) {
            const float want = __bfloat162float(__float2bfloat16(exact[std::size_t(ids[i]) * kK + j]));
            if (__bfloat162float(gathered_rows[i * kK + j]) != want) {
                std::cerr << "FAIL " << c.name << " embedding row " << ids[i] << " column " << j
                          << '\n';
                ++failures;
                i = ids.size();
                break;
            }
        }
    }
    return failures;
}

void update_checksum(std::uint64_t& checksum, const std::vector<__nv_bfloat16>& values) {
    for (const __nv_bfloat16 value : values) {
        std::uint16_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        for (unsigned shift = 0; shift < 16; shift += 8) {
            checksum ^= (bits >> shift) & 0xffU;
            checksum *= 1099511628211ULL;
        }
    }
}

float half_to_float(const std::uint8_t* bytes) {
    std::uint16_t bits = 0;
    std::memcpy(&bits, bytes, sizeof(bits));
    __half value;
    std::memcpy(&value, &bits, sizeof(bits));
    return __half2float(value);
}

void decode_k_cpu(QType qtype, const std::vector<std::uint8_t>& blocks, std::int32_t rows,
                  std::int32_t k, std::vector<float>& output) {
    const auto shape = gguf_block_shape(qtype);
    const std::int32_t blocks_per_row = k / shape.elements;
    output.resize(std::size_t(rows) * k);
    for (std::int32_t row = 0; row < rows; ++row) {
        for (std::int32_t bi = 0; bi < blocks_per_row; ++bi) {
            const std::uint8_t* block = blocks.data() +
                                        (std::size_t(row) * blocks_per_row + bi) * shape.bytes;
            float* decoded = output.data() + std::size_t(row) * k + bi * shape.elements;
            if (qtype == QType::GGUF_Q4_K) {
                const float d = half_to_float(block);
                const float dmin = half_to_float(block + 2);
                const std::uint8_t* scales = block + 4;
                const std::uint8_t* quants = block + 16;
                for (int subblock = 0; subblock < 8; ++subblock) {
                    int scale;
                    int minimum;
                    if (subblock < 4) {
                        scale = scales[subblock] & 63;
                        minimum = scales[subblock + 4] & 63;
                    } else {
                        scale = (scales[subblock + 4] & 0x0f) |
                                ((scales[subblock - 4] >> 6) << 4);
                        minimum = (scales[subblock + 4] >> 4) |
                                  ((scales[subblock] >> 6) << 4);
                    }
                    const float scaled = d * static_cast<float>(scale);
                    const float offset = dmin * static_cast<float>(minimum);
                    const std::uint8_t* packed = quants + (subblock / 2) * 32;
                    const int shift = 4 * (subblock & 1);
                    for (int i = 0; i < 32; ++i) {
                        const int quant = (packed[i] >> shift) & 0x0f;
                        decoded[subblock * 32 + i] =
                            std::fma(scaled, static_cast<float>(quant), -offset);
                    }
                }
            } else if (qtype == QType::GGUF_Q6_K) {
                const std::uint8_t* ql = block;
                const std::uint8_t* qh = block + 128;
                const float d = half_to_float(block + 208);
                for (int part = 0; part < 2; ++part) {
                    for (int lane = 0; lane < 32; ++lane) {
                        const int scale_index = 8 * part + lane / 16;
                        const int high = qh[32 * part + lane];
                        const std::uint8_t* low = ql + 64 * part + lane;
                        const auto scale_value = [&](int index) {
                            const int bits = block[192 + index];
                            return bits < 128 ? bits : bits - 256;
                        };
                        const float scale0 = d * static_cast<float>(scale_value(scale_index));
                        const float scale2 = d * static_cast<float>(scale_value(scale_index + 2));
                        const float scale4 = d * static_cast<float>(scale_value(scale_index + 4));
                        const float scale6 = d * static_cast<float>(scale_value(scale_index + 6));
                        const int q0 = ((low[0] & 0x0f) | ((high & 3) << 4)) - 32;
                        const int q2 = ((low[32] & 0x0f) | (((high >> 2) & 3) << 4)) - 32;
                        const int q4 = ((low[0] >> 4) | (((high >> 4) & 3) << 4)) - 32;
                        const int q6 = ((low[32] >> 4) | (((high >> 6) & 3) << 4)) - 32;
                        const int base = 128 * part + lane;
                        decoded[base] = scale0 * static_cast<float>(q0);
                        decoded[base + 32] = scale2 * static_cast<float>(q2);
                        decoded[base + 64] = scale4 * static_cast<float>(q4);
                        decoded[base + 96] = scale6 * static_cast<float>(q6);
                    }
                }
            } else {
                throw std::invalid_argument("CPU GGUF oracle supports only Q4_K and Q6_K");
            }
        }
    }
}

int compare_dequantized(const std::vector<float>& gpu, const std::vector<float>& cpu,
                        const std::string& label) {
    if (gpu.size() != cpu.size()) {
        std::cerr << label << ": CPU/GPU dequantized sizes differ\n";
        return 1;
    }
    for (std::size_t i = 0; i < gpu.size(); ++i) {
        std::uint32_t gpu_bits = 0;
        std::uint32_t cpu_bits = 0;
        std::memcpy(&gpu_bits, &gpu[i], sizeof(gpu_bits));
        std::memcpy(&cpu_bits, &cpu[i], sizeof(cpu_bits));
        if (gpu_bits != cpu_bits) {
            std::cerr << label << ": CPU/GPU dequantization differs at " << i
                      << " gpu=0x" << std::hex << gpu_bits << " cpu=0x" << cpu_bits << std::dec
                      << '\n';
            return 1;
        }
    }
    return 0;
}

void write_outputs(std::ofstream* output, const std::vector<__nv_bfloat16>& values) {
    if (output == nullptr) { return; }
    for (const __nv_bfloat16 value : values) {
        std::uint16_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        const std::uint8_t bytes[2] = {static_cast<std::uint8_t>(bits & 0xffU),
                                       static_cast<std::uint8_t>((bits >> 8) & 0xffU)};
        output->write(reinterpret_cast<const char*>(bytes), sizeof(bytes));
    }
    if (!*output) { throw std::runtime_error("failed writing real-shape output file"); }
}

class GuardedOutput {
public:
    explicit GuardedOutput(std::size_t elements)
        : payload_bytes_(elements * sizeof(__nv_bfloat16)), storage_(payload_bytes_ + 512) {}

    void* data() const { return static_cast<std::uint8_t*>(storage_.p) + 256; }

    void poison() {
        check(cudaMemset(storage_.p, 0xa5, storage_.bytes), "initialize output guards");
        check(cudaMemset(data(), 0xff, payload_bytes_), "poison output");
        check(cudaDeviceSynchronize(), "finish output poisoning");
    }

    std::vector<__nv_bfloat16> read(const char* label) const {
        std::vector<std::uint8_t> bytes(storage_.bytes);
        check(cudaMemcpy(bytes.data(), storage_.p, storage_.bytes, cudaMemcpyDeviceToHost), label);
        for (std::size_t i = 0; i < 256; ++i) {
            if (bytes[i] != 0xa5 || bytes[256 + payload_bytes_ + i] != 0xa5) {
                throw std::runtime_error(std::string(label) + ": output guard was overwritten");
            }
        }
        std::vector<__nv_bfloat16> values(payload_bytes_ / sizeof(__nv_bfloat16));
        std::memcpy(values.data(), bytes.data() + 256, payload_bytes_);
        if (std::any_of(values.begin(), values.end(), [](const __nv_bfloat16 value) {
                std::uint16_t bits = 0;
                std::memcpy(&bits, &value, sizeof(bits));
                return bits == 0xffffU;
            })) {
            throw std::runtime_error(std::string(label) + ": output retained poison values");
        }
        return values;
    }

private:
    std::size_t payload_bytes_;
    DeviceBuffer storage_;
};

template <typename Launch>
void check_graph_parity(Launch&& launch, GuardedOutput& guarded,
                        const std::vector<__nv_bfloat16>& eager, const std::string& label,
                        std::ofstream* output_file) {
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    check(cudaStreamCreate(&stream), "create graph test stream");
    try {
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "begin graph capture");
        launch(stream);
        check(cudaStreamEndCapture(stream, &graph), "end graph capture");
        check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0),
              "instantiate graph");
        guarded.poison();
        check(cudaGraphLaunch(executable, stream), "launch graph");
        check(cudaStreamSynchronize(stream), "synchronize graph");
        const auto replay = guarded.read("read graph output");
        write_outputs(output_file, replay);
        if (replay.size() != eager.size() ||
            std::memcmp(replay.data(), eager.data(), replay.size() * sizeof(__nv_bfloat16)) != 0) {
            throw std::runtime_error(label + ": graph replay differs from eager output");
        }
    } catch (...) {
        if (executable != nullptr) { cudaGraphExecDestroy(executable); }
        if (graph != nullptr) { cudaGraphDestroy(graph); }
        cudaStreamDestroy(stream);
        throw;
    }
    check(cudaGraphExecDestroy(executable), "destroy graph executable");
    check(cudaGraphDestroy(graph), "destroy graph");
    check(cudaStreamDestroy(stream), "destroy graph test stream");
}

std::uint64_t checksum_value(std::uint64_t value) {
    std::uint64_t checksum = 14695981039346656037ULL;
    for (unsigned shift = 0; shift < 64; shift += 8) {
        checksum ^= (value >> shift) & 0xffU;
        checksum *= 1099511628211ULL;
    }
    return checksum;
}

int run_real_shape(const TypeCase& c, std::int32_t k, std::uint32_t seed,
                   std::uint64_t& total_checksum, std::ofstream* output_file,
                   const std::string& weight_directory, std::size_t& files_used) {
    constexpr std::int32_t rows   = 257;
    constexpr std::int32_t max_t  = 8;
    const std::int32_t widths[]   = {1, 3, 4, 8};
    std::mt19937 rng(seed);
    int failures = 0;

    const auto block = gguf_block_shape(c.qtype);
    const std::size_t block_bytes = std::size_t(rows) * (k / block.elements) * block.bytes;
    const std::string filename = fixture_prefix(c.qtype) + "-k" + std::to_string(k) + ".bin";
    std::vector<std::uint8_t> host_blocks;
    const bool loaded = !weight_directory.empty() &&
                        load_blocks(weight_directory, filename, block_bytes, host_blocks);
    if (!loaded) { host_blocks = random_blocks(c, rows, k, rng, true); }
    if (loaded) { ++files_used; }
    const std::string source = loaded ? filename : "synthetic";
    std::vector<float> cpu_exact;
    decode_k_cpu(c.qtype, host_blocks, rows, k, cpu_exact);
    DeviceBuffer blocks(host_blocks.size());
    blocks.copy_from_host(host_blocks.data(), host_blocks.size());
    const std::int64_t row_bytes = std::int64_t(k / block.elements) * block.bytes;
    const Weight weight = gguf_weight(c.qtype, blocks.p, rows, k);

    DeviceBuffer exact_device(std::size_t(rows) * k * sizeof(float));
    ops::gguf::dequantize_rows(ops::detail::gguf_type(c.qtype), blocks.p, row_bytes, k, nullptr,
                               rows, static_cast<float*>(exact_device.p), k, nullptr);
    check(cudaDeviceSynchronize(), "real-shape dequantize");
    std::vector<float> exact(std::size_t(rows) * k);
    exact_device.copy_to_host(exact.data(), exact.size() * sizeof(float));
    failures += compare_dequantized(exact, cpu_exact, std::string(c.name) + " K=" + std::to_string(k));
    for (const float value : exact) {
        if (!std::isfinite(value)) {
            std::cerr << c.name << " K=" << k << ": non-finite dequantized weight\n";
            return 1;
        }
    }

    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<__nv_bfloat16> x_host(std::size_t(k) * max_t);
    for (std::int32_t col = 0; col < max_t; ++col) {
        for (std::int32_t i = 0; i < k; ++i) {
            const float value = i % 32 == 0 ? (((i / 32 + col) & 1) == 0 ? 8192.0f : -8192.0f)
                                             : normal(rng);
            x_host[std::size_t(col) * k + i] = __float2bfloat16(value);
        }
    }
    std::vector<float> x_float(x_host.size());
    for (std::size_t i = 0; i < x_host.size(); ++i) {
        x_float[i] = __bfloat162float(x_host[i]);
    }
    DeviceBuffer x_device(x_host.size() * sizeof(__nv_bfloat16));
    x_device.copy_from_host(x_host.data(), x_host.size() * sizeof(__nv_bfloat16));

    const std::size_t capacity = ops::linear_workspace_capacity_bytes(
        c.qtype, rows, k, ops::LinearPolicy::A16Only, 1, max_t);
    WorkspaceArena workspace(capacity + (1U << 20));
    std::uint64_t checksum = 14695981039346656037ULL;

    for (const std::int32_t t : widths) {
        GuardedOutput guarded(std::size_t(rows) * t);
        Tensor x(x_device.p, DType::BF16, {k, t});
        Tensor out(guarded.data(), DType::BF16, {rows, t});
        std::vector<__nv_bfloat16> host(std::size_t(rows) * t);
        guarded.poison();
        ops::linear(x, weight, out, ops::LinearPolicy::A16Only, workspace, nullptr);
        check(cudaDeviceSynchronize(), "real-shape linear");
        host = guarded.read("read real-shape linear output");

        std::vector<double> want(std::size_t(rows) * t);
        for (std::int32_t col = 0; col < t; ++col) {
            const float* activation = x_float.data() + std::size_t(col) * k;
            for (std::int32_t row = 0; row < rows; ++row) {
                const float* row_weights = cpu_exact.data() + std::size_t(row) * k;
                double acc = 0.0;
                for (std::int32_t i = 0; i < k; ++i) {
                    acc += double(row_weights[i]) * activation[i];
                }
                want[std::size_t(col) * rows + row] = acc;
            }
        }

        const std::vector<float> got = to_float(host);
        const Error error             = compare(got, want);
        const bool finite = std::all_of(got.begin(), got.end(),
                                        [](float value) { return std::isfinite(value); });
        if (!finite || !(error.relative_rms < 2e-2 && error.relative_max < 8e-2)) {
            std::cerr << "FAIL " << c.name << " K=" << k << " T=" << t
                      << " rms=" << error.relative_rms << " max=" << error.relative_max << '\n';
            ++failures;
        }
        update_checksum(checksum, host);
        write_outputs(output_file, host);
        if (t == 4) {
            check_graph_parity(
                [&](cudaStream_t stream) {
                    ops::linear(x, weight, out, ops::LinearPolicy::A16Only, workspace, stream);
                },
                guarded, host, std::string(c.name) + " K=" + std::to_string(k) + " T=4", output_file);
        }
    }

    if (c.qtype == QType::GGUF_Q4_K && k == 5120) {
        constexpr std::int32_t t = 4;
        const std::size_t swiglu_block_bytes =
            std::size_t(2 * rows) * (k / block.elements) * block.bytes;
        const std::string swiglu_filename = "q4-swiglu-k5120.bin";
        std::vector<std::uint8_t> swiglu_blocks;
        const bool swiglu_loaded = !weight_directory.empty() &&
                                   load_blocks(weight_directory, swiglu_filename,
                                               swiglu_block_bytes, swiglu_blocks);
        if (!swiglu_loaded) { swiglu_blocks = random_blocks(c, 2 * rows, k, rng, true); }
        if (swiglu_loaded) { ++files_used; }
        const std::string swiglu_source = swiglu_loaded ? swiglu_filename : "synthetic";
        DeviceBuffer swiglu_device(swiglu_blocks.size());
        swiglu_device.copy_from_host(swiglu_blocks.data(), swiglu_blocks.size());
        const std::int64_t swiglu_row_bytes =
            std::int64_t(k / block.elements) * block.bytes;
        const Weight swiglu_weight = gguf_weight(c.qtype, swiglu_device.p, 2 * rows, k);
        DeviceBuffer swiglu_exact_device(std::size_t(2) * rows * k * sizeof(float));
        ops::gguf::dequantize_rows(ops::detail::gguf_type(c.qtype), swiglu_device.p,
                                   swiglu_row_bytes, k, nullptr, 2 * rows,
                                   static_cast<float*>(swiglu_exact_device.p), k, nullptr);
        check(cudaDeviceSynchronize(), "real-shape SwiGLU dequantize");
        std::vector<float> swiglu_exact(std::size_t(2) * rows * k);
        swiglu_exact_device.copy_to_host(swiglu_exact.data(), swiglu_exact.size() * sizeof(float));
        std::vector<float> swiglu_cpu_exact;
        decode_k_cpu(c.qtype, swiglu_blocks, 2 * rows, k, swiglu_cpu_exact);
        failures += compare_dequantized(swiglu_exact, swiglu_cpu_exact,
                                        std::string(c.name) + " K=" + std::to_string(k) + " SwiGLU");
        for (const float value : swiglu_exact) {
            if (!std::isfinite(value)) {
                std::cerr << c.name << " K=" << k << ": non-finite SwiGLU dequantized weight\n";
                return failures + 1;
            }
        }
        const std::size_t swiglu_capacity = ops::linear_swiglu_workspace_capacity_bytes(
            c.qtype, 2 * rows, k, ops::LinearPolicy::A16Only, 1, t);
        WorkspaceArena swiglu_workspace(swiglu_capacity + (1U << 20));
        GuardedOutput guarded(std::size_t(rows) * t);
        Tensor x(x_device.p, DType::BF16, {k, t});
        Tensor out(guarded.data(), DType::BF16, {rows, t});
        guarded.poison();
        ops::linear_swiglu(x, swiglu_weight, out, ops::LinearPolicy::A16Only, swiglu_workspace,
                           nullptr);
        check(cudaDeviceSynchronize(), "real-shape SwiGLU");
        const auto host = guarded.read("read real-shape SwiGLU output");
        std::vector<double> want(std::size_t(rows) * t);
        for (std::int32_t col = 0; col < t; ++col) {
            const float* activation = x_float.data() + std::size_t(col) * k;
            for (std::int32_t row = 0; row < rows; ++row) {
                double gate = 0.0;
                double up = 0.0;
                const float* gate_weights = swiglu_cpu_exact.data() + std::size_t(row) * k;
                const float* up_weights = swiglu_cpu_exact.data() + std::size_t(rows + row) * k;
                for (std::int32_t i = 0; i < k; ++i) {
                    gate += double(gate_weights[i]) * activation[i];
                    up += double(up_weights[i]) * activation[i];
                }
                want[std::size_t(col) * rows + row] = gate / (1.0 + std::exp(-gate)) * up;
            }
        }
        const auto got = to_float(host);
        const Error error = compare(got, want);
        const bool finite = std::all_of(got.begin(), got.end(),
                                        [](float value) { return std::isfinite(value); });
        if (!finite || !(error.relative_rms < 4e-2 && error.relative_max < 1.6e-1)) {
            std::cerr << "FAIL " << c.name << " K=" << k << " SwiGLU T=" << t
                      << " rms=" << error.relative_rms << " max=" << error.relative_max << '\n';
            ++failures;
        }
        update_checksum(checksum, host);
        write_outputs(output_file, host);
        check_graph_parity(
            [&](cudaStream_t stream) {
                ops::linear_swiglu(x, swiglu_weight, out, ops::LinearPolicy::A16Only,
                                   swiglu_workspace, stream);
            },
            guarded, host, std::string(c.name) + " K=" + std::to_string(k) + " SwiGLU T=4", output_file);
        std::cout << "gguf real-shape " << c.name << " K=" << k << " SwiGLU source="
                  << swiglu_source << '\n';
    }

    std::cout << (failures == 0 ? "OK   " : "FAIL ") << "gguf real-shape " << c.name
              << " K=" << k << " rows=" << rows << " source=" << source << " checksum=0x"
              << std::hex << std::setw(16)
              << std::setfill('0') << checksum << std::dec << '\n';
    total_checksum ^= checksum_value(checksum);
    total_checksum *= 1099511628211ULL;
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    bool real_shapes = false;
    bool output_specified = false;
    bool weight_directory_specified = false;
    std::string output_path;
    std::string weight_directory;
    for (int i = 1; i < argc; ++i) {
        const std::string argument(argv[i]);
        if (argument == "--real-shapes" && !real_shapes) {
            real_shapes = true;
        } else if (argument == "--output" && !output_specified && i + 1 < argc) {
            output_specified = true;
            output_path = argv[++i];
        } else if (argument == "--weight-dir" && !weight_directory_specified && i + 1 < argc) {
            weight_directory_specified = true;
            weight_directory = argv[++i];
        } else {
            std::cerr << "usage: ninfer_linear_gguf_test [--real-shapes [--output <file>] "
                         "[--weight-dir <dir>]]\n";
            return 2;
        }
    }
    if ((output_specified && (!real_shapes || output_path.empty())) ||
        (weight_directory_specified && (!real_shapes || weight_directory.empty()))) {
        std::cerr << "usage: ninfer_linear_gguf_test [--real-shapes [--output <file>] "
                     "[--weight-dir <dir>]]\n";
        return 2;
    }
    try {
        if (real_shapes) {
            std::ofstream output_file;
            std::ofstream* output = nullptr;
            if (output_specified) {
                output_file.open(output_path, std::ios::binary | std::ios::trunc);
                if (!output_file) {
                    throw std::runtime_error("could not open real-shape output file: " + output_path);
                }
                output = &output_file;
            }
            const auto all = cases();
            const TypeCase* q4 = nullptr;
            const TypeCase* q6 = nullptr;
            for (const auto& c : all) {
                if (c.qtype == QType::GGUF_Q4_K) { q4 = &c; }
                if (c.qtype == QType::GGUF_Q6_K) { q6 = &c; }
            }
            if (q4 == nullptr || q6 == nullptr) {
                throw std::logic_error("real-shape GGUF cases are not registered");
            }
            constexpr std::int32_t shapes[] = {5120, 6144, 17408};
            if (weight_directory_specified) {
                if (!std::filesystem::is_directory(weight_directory)) {
                    throw std::runtime_error("--weight-dir is not a directory: " + weight_directory);
                }
                bool recognized_file = false;
                for (const TypeCase* c : {q4, q6}) {
                    for (const std::int32_t k : shapes) {
                        const std::string filename =
                            fixture_prefix(c->qtype) + "-k" + std::to_string(k) + ".bin";
                        recognized_file = recognized_file ||
                                          std::filesystem::exists(weight_directory + "/" + filename);
                    }
                }
                recognized_file = recognized_file || std::filesystem::exists(
                    weight_directory + "/q4-swiglu-k5120.bin");
                if (!recognized_file) {
                    throw std::runtime_error(
                        "--weight-dir contains no recognized q4-k5120.bin, q4-k6144.bin, "
                        "q4-k17408.bin, q6-k5120.bin, q6-k6144.bin, q6-k17408.bin, or "
                        "q4-swiglu-k5120.bin files");
                }
            }
            std::uint64_t checksum = 14695981039346656037ULL;
            int failures = 0;
            std::size_t files_used = 0;
            std::uint32_t seed = 19001;
            for (const TypeCase* c : {q4, q6}) {
                for (const std::int32_t k : shapes) {
                    failures += run_real_shape(*c, k, seed++, checksum, output, weight_directory,
                                               files_used);
                }
            }
            if (weight_directory_specified && files_used == 0) {
                throw std::runtime_error("--weight-dir did not provide any recognized weight files");
            }
            if (output != nullptr) {
                output_file.flush();
                if (!output_file) { throw std::runtime_error("failed flushing output file"); }
            }
            std::cout << "real-shapes checksum=0x" << std::hex << std::setw(16)
                      << std::setfill('0') << checksum << std::dec << '\n';
            std::cout << (failures == 0 ? "OK" : "FAIL") << " GGUF real shapes\n";
            return failures == 0 ? 0 : 1;
        }
        int failures      = 0;
        std::uint32_t seed = 9001;
        for (const auto& c : cases()) {
            const int f = run_type(c, seed++);
            std::cout << (f == 0 ? "OK   " : "FAIL ") << "gguf " << c.name << '\n';
            failures += f;
        }
        std::cout << (failures == 0 ? "OK" : "FAIL") << " GGUF Linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "GGUF Linear: " << error.what() << '\n';
        return 1;
    }
}
