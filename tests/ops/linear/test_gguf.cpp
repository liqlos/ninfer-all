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
#include "ninfer/ops/rmsnorm.h"
#include "ops/linear/gguf/ggml_bridge.h"
#include "ops/linear/gguf/ggml_bridge_dual_mmq.h"
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
#include <limits>
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
    double max_absolute = 0;
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
    return {std::sqrt(num / std::max(den, 1e-30)), worst / std::max(scale, 1e-30), worst};
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

    void poison(cudaStream_t stream = nullptr) {
        if (stream == nullptr) {
            check(cudaMemset(storage_.p, 0xa5, storage_.bytes), "initialize output guards");
            check(cudaMemset(data(), 0xff, payload_bytes_), "poison output");
            check(cudaDeviceSynchronize(), "finish output poisoning");
        } else {
            check(cudaMemsetAsync(storage_.p, 0xa5, storage_.bytes, stream),
                  "initialize output guards");
            check(cudaMemsetAsync(data(), 0xff, payload_bytes_, stream), "poison output");
        }
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

class GuardedRaw {
public:
    explicit GuardedRaw(std::size_t bytes) : payload_bytes_(bytes), storage_(bytes + 128) {}

    void* data() const { return static_cast<std::uint8_t*>(storage_.p) + 64; }

    void poison(cudaStream_t stream = nullptr) {
        if (stream == nullptr) {
            check(cudaMemset(storage_.p, 0xa5, storage_.bytes), "initialize raw guards");
            check(cudaMemset(data(), 0xcd, payload_bytes_), "poison raw output");
        } else {
            check(cudaMemsetAsync(storage_.p, 0xa5, storage_.bytes, stream),
                  "initialize raw guards");
            check(cudaMemsetAsync(data(), 0xcd, payload_bytes_, stream), "poison raw output");
        }
    }

    std::vector<std::uint8_t> read(const char* label) const {
        std::vector<std::uint8_t> bytes(storage_.bytes);
        check(cudaMemcpy(bytes.data(), storage_.p, bytes.size(), cudaMemcpyDeviceToHost), label);
        for (std::size_t i = 0; i < 64; ++i) {
            if (bytes[i] != 0xa5 || bytes[64 + payload_bytes_ + i] != 0xa5) {
                throw std::runtime_error(std::string(label) + ": raw output guard was overwritten");
            }
        }
        return {bytes.begin() + 64, bytes.begin() + 64 + payload_bytes_};
    }

private:
    std::size_t payload_bytes_;
    DeviceBuffer storage_;
};

std::vector<std::uint8_t> cpu_quantize_vector_activation(
    const std::vector<__nv_bfloat16>& values, std::int32_t rows, std::int32_t columns) {
    const std::size_t scales = (std::size_t(rows) * columns + 15) / 16 * 16;
    std::vector<std::uint8_t> output(ops::gguf::vector_activation_bytes(rows, columns));
    for (std::int32_t column = 0; column < columns; ++column) {
        for (std::int32_t group = 0; group < rows / 32; ++group) {
            float x[32], sums[32], maxima[32];
            for (int lane = 0; lane < 32; ++lane) {
                const float v = __bfloat162float(values[std::size_t(column) * rows + group * 32 + lane]);
                x[lane] = v;
                sums[lane] = v;
                maxima[lane] = std::fabs(v);
            }
            for (int offset = 16; offset > 0; offset >>= 1) {
                float prior_sum[32], prior_max[32];
                for (int lane = 0; lane < 32; ++lane) {
                    prior_sum[lane] = sums[lane];
                    prior_max[lane] = maxima[lane];
                }
                for (int lane = 0; lane < 32; ++lane) {
                    sums[lane] = prior_sum[lane] + prior_sum[lane ^ offset];
                    maxima[lane] = std::max(prior_max[lane], prior_max[lane ^ offset]);
                }
            }
            const float d = maxima[0] / 127.0f;
            for (int lane = 0; lane < 32; ++lane) {
                const std::int8_t q = maxima[0] == 0.0f
                                          ? 0
                                          : static_cast<std::int8_t>(std::round(x[lane] / d));
                output[std::size_t(column) * rows + group * 32 + lane] =
                    static_cast<std::uint8_t>(q);
            }
            const std::uint16_t d_bits = half_bits(d);
            const std::uint16_t sum_bits = half_bits(sums[0]);
            const std::size_t at = scales +
                                   (std::size_t(column) * (rows / 32) + group) * sizeof(__half2);
            std::memcpy(output.data() + at, &d_bits, sizeof(d_bits));
            std::memcpy(output.data() + at + sizeof(d_bits), &sum_bits, sizeof(sum_bits));
        }
    }
    return output;
}

void run_rmsnorm_q8_fusion(std::uint32_t seed) {
    constexpr std::int32_t k = 5120;
    constexpr std::int32_t widths[] = {1, 3, 4, 8};
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0f, 0.5f);
    cudaStream_t stream = nullptr;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create RMSNorm q8 stream");
    constexpr int kWidth = 5120;
    const auto rejected_alignment = [=](const __nv_bfloat16* x, const __nv_bfloat16* weight,
                                       void* out) {
        try {
            ops::gguf::rmsnorm_q8_1_vector_activation(x, weight, kWidth, 1, 1.0e-6f, true,
                                                      out, nullptr);
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    const auto* misaligned_bf16 = reinterpret_cast<const __nv_bfloat16*>(std::uintptr_t{2});
    void* misaligned_output = reinterpret_cast<void*>(std::uintptr_t{2});
    if (!rejected_alignment(misaligned_bf16, reinterpret_cast<const __nv_bfloat16*>(4),
                            reinterpret_cast<void*>(4)) ||
        !rejected_alignment(reinterpret_cast<const __nv_bfloat16*>(4), misaligned_bf16,
                            reinterpret_cast<void*>(4)) ||
        !rejected_alignment(reinterpret_cast<const __nv_bfloat16*>(4),
                            reinterpret_cast<const __nv_bfloat16*>(4), misaligned_output)) {
        throw std::runtime_error("RMSNorm q8 API accepted a misaligned operand");
    }
    for (int weight_case = 0; weight_case < 2; ++weight_case) {
        std::vector<__nv_bfloat16> host_weight(k);
        for (auto& value : host_weight) {
            value = weight_case == 0 ? __float2bfloat16(1.0f + normal(rng))
                                     : __float2bfloat16(-0.0f);
        }
        DeviceBuffer weight(std::size_t(k) * sizeof(__nv_bfloat16));
        weight.copy_from_host(host_weight.data(), std::size_t(k) * sizeof(__nv_bfloat16));
        for (const std::int32_t t : widths) {
            std::vector<__nv_bfloat16> host_x(std::size_t(k) * t);
            for (auto& value : host_x) { value = __float2bfloat16(normal(rng)); }
            if (t == 1) { std::fill(host_x.begin(), host_x.end(), __float2bfloat16(1.25f)); }
            if (weight_case == 1 && t == 3) {
                std::fill(host_x.begin(), host_x.end(), __float2bfloat16(0.0f));
            }
            DeviceBuffer x(std::size_t(k) * t * sizeof(__nv_bfloat16));
            DeviceBuffer normalized(std::size_t(k) * t * sizeof(__nv_bfloat16));
            x.copy_from_host(host_x.data(), host_x.size() * sizeof(__nv_bfloat16));
            for (const bool unit_offset : {false, true}) {
                const std::size_t bytes = ops::gguf::vector_activation_bytes(k, t);
                DeviceBuffer reference(bytes);
                GuardedRaw actual(bytes);
                actual.poison(stream);
                cudaGraph_t reference_graph = nullptr;
                cudaGraphExec_t reference_exec = nullptr;
                check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
                      "begin RMSNorm q8 reference graph");
                Tensor normalized_view{normalized.p, DType::BF16, {k, t}};
                ops::rmsnorm(Tensor{x.p, DType::BF16, {k, t}}, Tensor{weight.p, DType::BF16, {k}},
                             1.0e-6f, unit_offset,
                             normalized_view, stream);
                ops::gguf::quantize_vector_activation(
                    static_cast<const __nv_bfloat16*>(normalized.p), k, t, nullptr, reference.p,
                    stream);
                check(cudaStreamEndCapture(stream, &reference_graph),
                      "end RMSNorm q8 reference graph");
                check(cudaGraphInstantiate(&reference_exec, reference_graph, nullptr, nullptr, 0),
                      "instantiate RMSNorm q8 reference graph");
                check(cudaGraphLaunch(reference_exec, stream), "launch RMSNorm q8 reference graph");
                check(cudaStreamSynchronize(stream), "wait RMSNorm q8 reference graph");
                std::vector<std::uint8_t> expected(bytes);
                reference.copy_to_host(expected.data(), expected.size());
                check(cudaGraphExecDestroy(reference_exec), "destroy RMSNorm q8 reference executable");
                check(cudaGraphDestroy(reference_graph), "destroy RMSNorm q8 reference graph");
                cudaGraph_t fused_graph = nullptr;
                cudaGraphExec_t fused_exec = nullptr;
                check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
                      "begin fused RMSNorm q8 graph");
                ops::gguf::rmsnorm_q8_1_vector_activation(
                    static_cast<const __nv_bfloat16*>(x.p),
                    static_cast<const __nv_bfloat16*>(weight.p), k, t, 1.0e-6f, unit_offset,
                    actual.data(), stream);
                check(cudaStreamEndCapture(stream, &fused_graph), "end fused RMSNorm q8 graph");
                check(cudaGraphInstantiate(&fused_exec, fused_graph, nullptr, nullptr, 0),
                      "instantiate fused RMSNorm q8 graph");
                check(cudaGraphLaunch(fused_exec, stream), "launch fused RMSNorm q8 graph");
                check(cudaStreamSynchronize(stream), "wait fused RMSNorm q8 graph");
                if (actual.read("read fused RMSNorm q8 graph") != expected) {
                    throw std::runtime_error(
                        "RMSNorm q8 fusion differs from BF16 RMSNorm plus q8_1 graph");
                }
                check(cudaGraphExecDestroy(fused_exec), "destroy fused RMSNorm q8 executable");
                check(cudaGraphDestroy(fused_graph), "destroy fused RMSNorm q8 graph");
            }
        }
    }
    check(cudaStreamDestroy(stream), "destroy RMSNorm q8 stream");
}

void run_rmsnorm_mmq_q8_fusion(std::uint32_t seed) {
    constexpr std::int32_t k = 5120;
    constexpr std::int32_t widths[] = {9, 17, 40, 70, 127, 130, 1024};
    const TypeCase types[] = {{QType::GGUF_Q8_0, "q8_0", {0}},
                              {QType::GGUF_Q2_K, "q2_k", {80, 82}},
                              {QType::GGUF_Q4_K, "q4_k", {0, 2}}};
    const TypeCase q6_type{QType::GGUF_Q6_K, "q6_k", {208}};
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0f, 0.4f);
    const auto projection_blocks = random_blocks(types[2], 256, k, rng);
    const auto projection_q6_blocks = random_blocks(q6_type, 256, k, rng);
    DeviceBuffer projection_weight_buffer(projection_blocks.size());
    DeviceBuffer projection_q6_weight_buffer(projection_q6_blocks.size());
    projection_weight_buffer.copy_from_host(projection_blocks.data(), projection_blocks.size());
    projection_q6_weight_buffer.copy_from_host(projection_q6_blocks.data(),
                                                projection_q6_blocks.size());
    const Weight projection_weight = gguf_weight(QType::GGUF_Q4_K, projection_weight_buffer.p,
                                                  256, k);
    const Weight projection_q6_weight = gguf_weight(QType::GGUF_Q6_K,
                                                    projection_q6_weight_buffer.p, 256, k);
    cudaStream_t stream = nullptr;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
          "create RMSNorm MMQ q8 stream");
    const auto rejected_alignment = [=](const __nv_bfloat16* x, const __nv_bfloat16* weight,
                                        void* out) {
        try {
            ops::gguf::rmsnorm_q8_1_matrix_activation(
                x, weight, k, 9, 1.0e-6f, true, ops::gguf::GgmlType::Q4_K, out, nullptr);
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    const auto* misaligned_bf16 = reinterpret_cast<const __nv_bfloat16*>(std::uintptr_t{2});
    void* misaligned_output = reinterpret_cast<void*>(std::uintptr_t{2});
    if (!rejected_alignment(misaligned_bf16, reinterpret_cast<const __nv_bfloat16*>(4),
                            reinterpret_cast<void*>(4)) ||
        !rejected_alignment(reinterpret_cast<const __nv_bfloat16*>(4), misaligned_bf16,
                            reinterpret_cast<void*>(4)) ||
        !rejected_alignment(reinterpret_cast<const __nv_bfloat16*>(4),
                            reinterpret_cast<const __nv_bfloat16*>(4), misaligned_output)) {
        throw std::runtime_error("RMSNorm MMQ q8 API accepted a misaligned operand");
    }
    for (const TypeCase& type_case : types) {
        const auto type = ops::detail::gguf_type(type_case.qtype);
        for (const std::int32_t t : widths) {
            std::vector<__nv_bfloat16> host_x(std::size_t(k) * t);
            std::vector<__nv_bfloat16> host_weight(k);
            for (auto& value : host_x) { value = __float2bfloat16(normal(rng)); }
            for (auto& value : host_weight) {
                value = __float2bfloat16(normal(rng) < 0.0f ? -0.75f : 1.25f);
            }
            if (t == 9) {
                std::fill(host_x.begin(), host_x.end(), __float2bfloat16(-0.0f));
                std::fill(host_weight.begin(), host_weight.end(), __float2bfloat16(-0.0f));
            } else if (t == 17) {
                const float special[] = {
                    3.0e38f, -3.0e38f, std::numeric_limits<float>::infinity(),
                    -std::numeric_limits<float>::infinity(),
                    std::numeric_limits<float>::quiet_NaN(), 1.0e-38f, -1.0e-38f};
                for (std::size_t i = 0; i < host_x.size(); ++i) {
                    host_x[i] = __float2bfloat16(special[i % (sizeof(special) / sizeof(special[0]))]);
                }
            }
            DeviceBuffer x(std::size_t(k) * t * sizeof(__nv_bfloat16));
            DeviceBuffer weight(std::size_t(k) * sizeof(__nv_bfloat16));
            DeviceBuffer normalized(std::size_t(k) * t * sizeof(__nv_bfloat16));
            x.copy_from_host(host_x.data(), host_x.size() * sizeof(__nv_bfloat16));
            weight.copy_from_host(host_weight.data(), host_weight.size() * sizeof(__nv_bfloat16));
            const std::size_t bytes = ops::gguf::matrix_activation_bytes(k, t);
            DeviceBuffer reference(bytes);
            DeviceBuffer reference_q6(bytes);
            GuardedRaw actual(bytes);
            GuardedRaw actual_q6(bytes);
            actual.poison(stream);
            actual_q6.poison(stream);
            const bool graph_test = t == 9 || t == 40 || t == 130 || t == 1024;
            for (const bool unit_offset : {false, true}) {
                check(cudaMemsetAsync(reference.p, 0, bytes, stream), "zero MMQ reference slack");
                Tensor normalized_view{normalized.p, DType::BF16, {k, t}};
                if (graph_test) {
                    cudaGraph_t reference_graph = nullptr;
                    cudaGraphExec_t reference_exec = nullptr;
                    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
                          "begin RMSNorm MMQ reference graph");
                    ops::rmsnorm(Tensor{x.p, DType::BF16, {k, t}},
                                 Tensor{weight.p, DType::BF16, {k}}, 1.0e-6f, unit_offset,
                                 normalized_view, stream);
                    ops::gguf::quantize_matrix_activation(
                        type, static_cast<const __nv_bfloat16*>(normalized.p), k, t, nullptr,
                        reference.p, stream);
                    check(cudaStreamEndCapture(stream, &reference_graph),
                          "end RMSNorm MMQ reference graph");
                    check(cudaGraphInstantiate(&reference_exec, reference_graph, nullptr, nullptr, 0),
                          "instantiate RMSNorm MMQ reference graph");
                    check(cudaGraphLaunch(reference_exec, stream),
                          "launch RMSNorm MMQ reference graph");
                    check(cudaStreamSynchronize(stream), "wait RMSNorm MMQ reference graph");
                    check(cudaGraphExecDestroy(reference_exec),
                          "destroy RMSNorm MMQ reference executable");
                    check(cudaGraphDestroy(reference_graph), "destroy RMSNorm MMQ reference graph");
                } else {
                    ops::rmsnorm(Tensor{x.p, DType::BF16, {k, t}},
                                 Tensor{weight.p, DType::BF16, {k}}, 1.0e-6f, unit_offset,
                                 normalized_view, stream);
                    ops::gguf::quantize_matrix_activation(
                        type, static_cast<const __nv_bfloat16*>(normalized.p), k, t, nullptr,
                        reference.p, stream);
                    check(cudaStreamSynchronize(stream), "wait RMSNorm MMQ reference");
                }
                std::vector<std::uint8_t> expected(bytes);
                reference.copy_to_host(expected.data(), expected.size());
                if (graph_test) {
                    cudaGraph_t fused_graph = nullptr;
                    cudaGraphExec_t fused_exec = nullptr;
                    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
                          "begin fused RMSNorm MMQ graph");
                    ops::gguf::rmsnorm_q8_1_matrix_activation(
                        static_cast<const __nv_bfloat16*>(x.p),
                        static_cast<const __nv_bfloat16*>(weight.p), k, t, 1.0e-6f, unit_offset,
                        type, actual.data(), stream);
                    check(cudaStreamEndCapture(stream, &fused_graph),
                          "end fused RMSNorm MMQ graph");
                    check(cudaGraphInstantiate(&fused_exec, fused_graph, nullptr, nullptr, 0),
                          "instantiate fused RMSNorm MMQ graph");
                    check(cudaGraphLaunch(fused_exec, stream), "launch fused RMSNorm MMQ graph");
                    check(cudaStreamSynchronize(stream), "wait fused RMSNorm MMQ graph");
                    check(cudaGraphExecDestroy(fused_exec), "destroy fused RMSNorm MMQ executable");
                    check(cudaGraphDestroy(fused_graph), "destroy fused RMSNorm MMQ graph");
                } else {
                    ops::gguf::rmsnorm_q8_1_matrix_activation(
                        static_cast<const __nv_bfloat16*>(x.p),
                        static_cast<const __nv_bfloat16*>(weight.p), k, t, 1.0e-6f, unit_offset,
                        type, actual.data(), stream);
                    check(cudaStreamSynchronize(stream), "wait fused RMSNorm MMQ");
                }
                const auto got = actual.read("read fused RMSNorm MMQ graph");
                if (got != expected) {
                    const auto first = std::mismatch(got.begin(), got.end(), expected.begin());
                    const std::size_t offset = static_cast<std::size_t>(first.first - got.begin());
                    const std::size_t slack_bytes = 128 * (128 + 16);
                    const std::size_t block_bytes =
                        (bytes - slack_bytes) / (std::size_t(k / 128) * t);
                    const std::size_t active_bytes = std::size_t(k / 128) * t * block_bytes;
                    std::cerr << "RMSNorm MMQ first mismatch type=" << type_case.name
                              << " T=" << t << " unit_offset=" << unit_offset
                              << " byte=" << offset;
                    if (offset < active_bytes) {
                        const std::size_t block = offset / block_bytes;
                        const std::size_t in_block = offset % block_bytes;
                        std::cerr << " qblock=" << block << " kblock=" << block / t
                                  << " column=" << block % t << " block_byte=" << in_block
                                  << (in_block < 16 ? " scale_or_sum" : " q8_code");
                    } else {
                        std::cerr << " slack_byte=" << offset - active_bytes;
                    }
                    std::cerr << " got=" << unsigned(got[offset])
                              << " expected=" << unsigned(expected[offset]) << '\n';
                    throw std::runtime_error(std::string(type_case.name) +
                                             " RMSNorm MMQ fusion differs from BF16 RMSNorm plus native packing at T=" +
                                             std::to_string(t));
                }
                if (type_case.qtype == QType::GGUF_Q4_K) {
                    check(cudaMemsetAsync(reference_q6.p, 0, bytes, stream),
                          "zero Q6_K MMQ reference slack");
                    ops::gguf::quantize_matrix_activation(
                        ops::detail::gguf_type(QType::GGUF_Q6_K),
                        static_cast<const __nv_bfloat16*>(normalized.p), k, t, nullptr,
                        reference_q6.p, stream);
                    check(cudaStreamSynchronize(stream), "wait Q6_K matrix activation reference");
                    const auto expected_q6 = [&] {
                        std::vector<std::uint8_t> packed(bytes);
                        reference_q6.copy_to_host(packed.data(), packed.size());
                        return packed;
                    }();
                    if (graph_test) {
                        cudaGraph_t dual_graph = nullptr;
                        cudaGraphExec_t dual_exec = nullptr;
                        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
                              "begin dual-layout RMSNorm MMQ graph");
                        ops::gguf::rmsnorm_q8_1_matrix_activations(
                            static_cast<const __nv_bfloat16*>(x.p),
                            static_cast<const __nv_bfloat16*>(weight.p), k, t, 1.0e-6f,
                            unit_offset, type, actual.data(),
                            ops::gguf::GgmlType::Q6_K, actual_q6.data(), stream);
                        check(cudaStreamEndCapture(stream, &dual_graph),
                              "end dual-layout RMSNorm MMQ graph");
                        check(cudaGraphInstantiate(&dual_exec, dual_graph, nullptr, nullptr, 0),
                              "instantiate dual-layout RMSNorm MMQ graph");
                        check(cudaGraphLaunch(dual_exec, stream),
                              "launch dual-layout RMSNorm MMQ graph");
                        check(cudaStreamSynchronize(stream), "wait dual-layout RMSNorm MMQ graph");
                        check(cudaGraphExecDestroy(dual_exec), "destroy dual-layout RMSNorm MMQ executable");
                        check(cudaGraphDestroy(dual_graph), "destroy dual-layout RMSNorm MMQ graph");
                    } else {
                        ops::gguf::rmsnorm_q8_1_matrix_activations(
                            static_cast<const __nv_bfloat16*>(x.p),
                            static_cast<const __nv_bfloat16*>(weight.p), k, t, 1.0e-6f,
                            unit_offset, type, actual.data(),
                            ops::gguf::GgmlType::Q6_K, actual_q6.data(), stream);
                        check(cudaStreamSynchronize(stream), "wait dual-layout RMSNorm MMQ");
                    }
                    if (actual.read("read dual-layout DS4 activation") != expected ||
                        actual_q6.read("read dual-layout D4 activation") != expected_q6) {
                        throw std::runtime_error("dual-layout RMSNorm MMQ packing differs from native planes");
                    }
                }
            }
            if (type_case.qtype == QType::GGUF_Q4_K) {
                constexpr std::int32_t rows = 256;
                const std::size_t output_bytes = std::size_t(rows) * t * sizeof(__nv_bfloat16);
                GuardedOutput reference_output(std::size_t(rows) * t);
                GuardedOutput reference_q6_output(std::size_t(rows) * t);
                GuardedOutput fused_output(std::size_t(rows) * t);
                GuardedOutput fused_q6_output(std::size_t(rows) * t);
                reference_output.poison(stream);
                reference_q6_output.poison(stream);
                fused_output.poison(stream);
                fused_q6_output.poison(stream);
                Tensor reference_tensor{reference_output.data(), DType::BF16, {rows, t}};
                Tensor reference_q6_tensor{reference_q6_output.data(), DType::BF16, {rows, t}};
                Tensor fused_tensor{fused_output.data(), DType::BF16, {rows, t}};
                Tensor fused_q6_tensor{fused_q6_output.data(), DType::BF16, {rows, t}};
                ops::detail::GgufProduct products[] = {
                    {&projection_weight, &reference_tensor, 0, ops::detail::GgufEpilogue::Store},
                    {&projection_q6_weight, &reference_q6_tensor, 0,
                     ops::detail::GgufEpilogue::Store}};
                const ops::detail::GgufShape shapes[] = {
                    {QType::GGUF_Q4_K, rows, k}, {QType::GGUF_Q6_K, rows, k}};
                const std::size_t workspace_bytes =
                    ops::detail::gguf_project_workspace_bytes(shapes, t, t) + (1U << 20);
                WorkspaceArena reference_workspace(workspace_bytes);
                WorkspaceArena fused_workspace(workspace_bytes);
                ops::detail::gguf_project(Tensor{normalized.p, DType::BF16, {k, t}}, products,
                                          reference_workspace, stream);
                check(cudaStreamSynchronize(stream), "wait reference mixed-layout MMQ projections");
                const auto expected = reference_output.read("read reference Q4_K MMQ projection");
                const auto expected_q6 =
                    reference_q6_output.read("read reference Q6_K MMQ projection");
                products[0].out = &fused_tensor;
                products[1].out = &fused_q6_tensor;
                const ops::gguf::MatrixActivationPlane activations[] = {
                    {ops::gguf::GgmlType::Q4_K, actual.data()},
                    {ops::gguf::GgmlType::Q6_K, actual_q6.data()}};
                if (graph_test) {
                    cudaGraph_t projection_graph = nullptr;
                    cudaGraphExec_t projection_exec = nullptr;
                    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
                          "begin mixed-layout prepared MMQ graph");
                    ops::detail::gguf_project_matrix_q8_1_activations(
                        k, t, activations, products, fused_workspace, stream);
                    check(cudaStreamEndCapture(stream, &projection_graph),
                          "end mixed-layout prepared MMQ graph");
                    check(cudaGraphInstantiate(&projection_exec, projection_graph, nullptr, nullptr, 0),
                          "instantiate mixed-layout prepared MMQ graph");
                    check(cudaGraphLaunch(projection_exec, stream),
                          "launch mixed-layout prepared MMQ graph");
                    check(cudaStreamSynchronize(stream), "wait mixed-layout prepared MMQ graph");
                    check(cudaGraphExecDestroy(projection_exec),
                          "destroy mixed-layout prepared MMQ executable");
                    check(cudaGraphDestroy(projection_graph), "destroy mixed-layout prepared MMQ graph");
                } else {
                    ops::detail::gguf_project_matrix_q8_1_activations(
                        k, t, activations, products, fused_workspace, stream);
                    check(cudaStreamSynchronize(stream), "wait fused mixed-layout MMQ projections");
                }
                check(cudaStreamSynchronize(stream), "wait fused prepared MMQ projection");
                const auto got = fused_output.read("read fused prepared MMQ projection");
                const auto got_q6 = fused_q6_output.read("read fused Q6_K MMQ projection");
                if (got.size() != expected.size() || got_q6.size() != expected_q6.size() ||
                    std::memcmp(got.data(), expected.data(), output_bytes) != 0 ||
                    std::memcmp(got_q6.data(), expected_q6.data(), output_bytes) != 0) {
                    throw std::runtime_error("prepared mixed-layout MMQ projection changed normalized q8 output");
                }
            }
        }
    }
    check(cudaStreamDestroy(stream), "destroy RMSNorm MMQ q8 stream");
}

#if defined(NINFER_GGUF_DUAL_MMQ_FUSION)
void run_dual_mmq_fusion() {
    constexpr std::int32_t rows = 6144;
    constexpr std::int32_t k = 5120;
    constexpr std::int32_t widths[] = {9, 17, 40, 70, 127, 130, 1024};
    constexpr bool expected_dual[] = {true, true, false, false, false, false, false};
    const TypeCase q4{QType::GGUF_Q4_K, "q4_k", {0, 2}};
    std::mt19937 rng(28005);
    std::normal_distribution<float> normal(0.0f, 0.35f);
    const auto weights_a_host = random_blocks(q4, rows, k, rng);
    const auto weights_b_host = random_blocks(q4, rows, k, rng);
    DeviceBuffer weights_a(weights_a_host.size());
    DeviceBuffer weights_b(weights_b_host.size());
    weights_a.copy_from_host(weights_a_host.data(), weights_a_host.size());
    weights_b.copy_from_host(weights_b_host.data(), weights_b_host.size());
    const Weight weight_a = gguf_weight(QType::GGUF_Q4_K, weights_a.p, rows, k);
    const Weight weight_b = gguf_weight(QType::GGUF_Q4_K, weights_b.p, rows, k);
    const std::int64_t row_bytes = std::int64_t(k / 256) * 144;
    const auto dual_available = [&](std::int32_t columns) {
        return ops::gguf::dual_q4_k_matrix_supported(rows, columns);
    };
    cudaStream_t stream = nullptr;
    check(cudaStreamCreate(&stream), "create dual MMQ stream");
    for (std::size_t width_index = 0;
         width_index < sizeof(widths) / sizeof(widths[0]); ++width_index) {
        const std::int32_t columns = widths[width_index];
        if (dual_available(columns) != expected_dual[width_index]) {
            throw std::runtime_error("dual MMQ selector did not follow native tile admission");
        }
        std::vector<__nv_bfloat16> host_x(std::size_t(k) * columns);
        for (auto& value : host_x) { value = __float2bfloat16(normal(rng)); }
        DeviceBuffer x(host_x.size() * sizeof(__nv_bfloat16));
        x.copy_from_host(host_x.data(), host_x.size() * sizeof(__nv_bfloat16));
        DeviceBuffer activation(ops::gguf::matrix_activation_bytes(k, columns));
        ops::gguf::quantize_matrix_activation(
            ops::gguf::GgmlType::Q4_K, static_cast<const __nv_bfloat16*>(x.p), k, columns,
            nullptr, activation.p, stream);

        GuardedOutput reference_a(std::size_t(rows) * columns);
        GuardedOutput reference_b(std::size_t(rows) * columns);
        GuardedOutput paired_a(std::size_t(rows) * columns);
        GuardedOutput paired_b(std::size_t(rows) * columns);
        reference_a.poison();
        reference_b.poison();
        paired_a.poison();
        paired_b.poison();
        DeviceBuffer f32_a(std::size_t(rows) * columns * sizeof(float));
        DeviceBuffer f32_b(std::size_t(rows) * columns * sizeof(float));
        DeviceBuffer fixup_a(std::max<std::size_t>(
            4, ops::gguf::matrix_fixup_bytes(ops::gguf::GgmlType::Q4_K, rows, columns)));
        DeviceBuffer fixup_b(std::max<std::size_t>(
            4, ops::gguf::matrix_fixup_bytes(ops::gguf::GgmlType::Q4_K, rows, columns)));
        ops::gguf::matrix_product(ops::gguf::GgmlType::Q4_K, weight_a.qdata, row_bytes, rows,
                                  k, activation.p, columns, static_cast<float*>(f32_a.p), rows,
                                  fixup_a.p, stream);
        ops::gguf::matrix_product(ops::gguf::GgmlType::Q4_K, weight_b.qdata, row_bytes, rows,
                                  k, activation.p, columns, static_cast<float*>(f32_b.p), rows,
                                  fixup_b.p, stream);
        const auto store_reference = [&](const DeviceBuffer& source, GuardedOutput& destination) {
            ops::gguf::VectorOutput output;
            output.bf16 = static_cast<__nv_bfloat16*>(destination.data());
            output.column_stride = rows;
            output.epilogue = ops::gguf::Epilogue::Store;
            ops::gguf::store_plane(static_cast<const float*>(source.p), rows, rows, columns,
                                   output, stream);
        };
        store_reference(f32_a, reference_a);
        store_reference(f32_b, reference_b);
        check(cudaStreamSynchronize(stream), "wait dual MMQ single-product references");
        const auto expected_a = reference_a.read("read dual MMQ Q4_K reference");
        const auto expected_b = reference_b.read("read dual MMQ second Q4_K reference");

        Tensor paired_tensor_a{paired_a.data(), DType::BF16, {rows, columns}};
        Tensor paired_tensor_b{paired_b.data(), DType::BF16, {rows, columns}};
        const ops::detail::GgufProduct products[] = {
            {&weight_a, &paired_tensor_a, 0, ops::detail::GgufEpilogue::Store},
            {&weight_b, &paired_tensor_b, 0, ops::detail::GgufEpilogue::Store}};
        const ops::detail::GgufShape shapes[] = {
            {QType::GGUF_Q4_K, rows, k}, {QType::GGUF_Q4_K, rows, k}};
        WorkspaceArena workspace(
            ops::detail::gguf_project_workspace_bytes(shapes, columns, columns) + (1U << 20));
        bool dual_enqueued = false;
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t executable = nullptr;
        if (columns == 9 || columns == 130 || columns == 1024) {
            check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
                  "begin dual MMQ projection graph");
            dual_enqueued = ops::detail::gguf_project(
                Tensor{x.p, DType::BF16, {k, columns}}, products, workspace, stream);
            check(cudaStreamEndCapture(stream, &graph), "end dual MMQ projection graph");
            if (dual_enqueued != expected_dual[width_index]) {
                throw std::runtime_error("dual MMQ graph route disagrees with native-tile admission");
            }
            check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0),
                  "instantiate dual MMQ projection graph");
            check(cudaGraphLaunch(executable, stream), "launch dual MMQ projection graph");
            check(cudaStreamSynchronize(stream), "wait dual MMQ projection graph");
        } else {
            dual_enqueued = ops::detail::gguf_project(
                Tensor{x.p, DType::BF16, {k, columns}}, products, workspace, stream);
            if (dual_enqueued != expected_dual[width_index]) {
                throw std::runtime_error("dual MMQ route disagrees with native-tile admission");
            }
            check(cudaStreamSynchronize(stream), "wait dual MMQ projections");
        }
        const auto got_a = paired_a.read("read paired dual MMQ Q4_K output");
        const auto got_b = paired_b.read("read paired dual MMQ second Q4_K output");
        const auto report_mismatch = [&](const std::vector<__nv_bfloat16>& got,
                                         const std::vector<__nv_bfloat16>& expected,
                                         const char* label) {
            const std::size_t count = std::min(got.size(), expected.size());
            std::size_t first = count;
            for (std::size_t i = 0; i < count; ++i) {
                std::uint16_t actual_bits = 0;
                std::uint16_t expected_bits = 0;
                std::memcpy(&actual_bits, &got[i], sizeof(actual_bits));
                std::memcpy(&expected_bits, &expected[i], sizeof(expected_bits));
                if (actual_bits != expected_bits) {
                    first = i;
                    break;
                }
            }
            if (got.size() != expected.size() || first != count) {
                std::uint16_t actual_bits = 0;
                std::uint16_t expected_bits = 0;
                if (first < count) {
                    std::memcpy(&actual_bits, &got[first], sizeof(actual_bits));
                    std::memcpy(&expected_bits, &expected[first], sizeof(expected_bits));
                }
                std::cerr << "dual Q4_K MMQ mismatch T=" << columns << " output=" << label
                          << " first=" << first << " got_size=" << got.size()
                          << " expected_size=" << expected.size() << " got=0x" << std::hex
                          << actual_bits << " expected=0x" << expected_bits << std::dec << '\n';
                return false;
            }
            return true;
        };
        if (!report_mismatch(got_a, expected_a, "A") ||
            !report_mismatch(got_b, expected_b, "B")) {
            throw std::runtime_error("dual Q4_K MMQ changed native single-product outputs");
        }
        if (graph != nullptr) {
            check(cudaGraphExecDestroy(executable), "destroy dual MMQ projection executable");
            check(cudaGraphDestroy(graph), "destroy dual MMQ projection graph");
        }
    }
    check(cudaStreamDestroy(stream), "destroy dual MMQ stream");
}
#endif

#if defined(NINFER_GGUF_Q4_T4_FUSED_WARP_PAIR_TEST)
void run_q4_t4_fused_warp_pair_case(std::int32_t rows, std::int32_t k,
                                   std::uint32_t seed,
                                   const std::vector<std::int32_t>& checked_rows) {
    constexpr std::int32_t columns = 4;
    const TypeCase type{QType::GGUF_Q4_K, "q4_k", {0, 2}};
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0f, 0.4f);
    const auto host_weights = random_blocks(type, 2 * rows, k, rng);
    DeviceBuffer weights(host_weights.size());
    weights.copy_from_host(host_weights.data(), host_weights.size());
    const auto shape = gguf_block_shape(type.qtype);
    const std::int64_t row_bytes = std::int64_t(k / shape.elements) * shape.bytes;

    std::vector<__nv_bfloat16> host_x(std::size_t(k) * columns);
    for (auto& value : host_x) { value = __float2bfloat16(normal(rng)); }
    const auto input_bytes = cpu_quantize_vector_activation(host_x, k, columns);
    DeviceBuffer input(input_bytes.size());
    input.copy_from_host(input_bytes.data(), input_bytes.size());

    for (std::size_t index = 0; index < checked_rows.size(); ++index) {
        const std::int32_t row = checked_rows[index];
        if (row < 0 || row >= rows) {
            throw std::runtime_error("Q4_K fused warp-pair oracle row is out of range");
        }
    }
    const std::size_t activation_scales = (std::size_t(k) * columns + 15) / 16 * 16;
    const auto q8_dot = [&](std::int32_t row, std::int32_t column) {
        const std::uint8_t* row_data = host_weights.data() + std::size_t(row) * row_bytes;
        double result = 0.0;
        for (std::int32_t group = 0; group < k / 32; ++group) {
            const std::int32_t block_index = group / 8;
            const std::int32_t subblock = group % 8;
            const std::uint8_t* quant_block = row_data + std::size_t(block_index) * shape.bytes;
            const std::uint8_t* scales = quant_block + 4;
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
            const std::uint8_t* quants = quant_block + 16 + (subblock / 2) * 32;
            const int shift = 4 * (subblock & 1);
            std::int64_t integer_dot = 0;
            for (std::int32_t lane = 0; lane < 32; ++lane) {
                const int weight_q = (quants[lane] >> shift) & 0x0f;
                const std::uint8_t activation_byte =
                    input_bytes[std::size_t(column) * k + group * 32 + lane];
                const int activation_q = activation_byte < 128
                                             ? activation_byte
                                             : int(activation_byte) - 256;
                integer_dot += std::int64_t(weight_q) * activation_q;
            }
            const std::size_t scale_at = activation_scales +
                (std::size_t(column) * (k / 32) + group) * sizeof(__half2);
            const double activation_d = half_to_float(input_bytes.data() + scale_at);
            const double activation_sum =
                half_to_float(input_bytes.data() + scale_at + sizeof(__half));
            const double weight_d = half_to_float(quant_block);
            const double weight_dmin = half_to_float(quant_block + 2);
            result += (weight_d * scale) * activation_d * double(integer_dot) -
                      (weight_dmin * minimum) * activation_sum;
        }
        return result;
    };
    std::vector<double> want(checked_rows.size() * columns);
    for (std::int32_t column = 0; column < columns; ++column) {
        for (std::size_t index = 0; index < checked_rows.size(); ++index) {
            const double gate = q8_dot(checked_rows[index], column);
            const double up = q8_dot(rows + checked_rows[index], column);
            want[std::size_t(column) * checked_rows.size() + index] =
                gate / (1.0 + std::exp(-gate)) * up;
        }
    }

    GuardedOutput actual(std::size_t(rows) * columns);
    actual.poison();
    ops::gguf::vector_swiglu(ops::gguf::GgmlType::Q4_K, weights.p, row_bytes, rows, k,
                             input.p, columns, static_cast<__nv_bfloat16*>(actual.data()), rows,
                             nullptr);
    check(cudaDeviceSynchronize(), "Q4_K fused warp-pair regression");
    const auto eager = actual.read("read Q4_K fused warp-pair output");
    const bool finite = std::all_of(eager.begin(), eager.end(), [](__nv_bfloat16 value) {
        return std::isfinite(__bfloat162float(value));
    });
    std::vector<float> got(checked_rows.size() * columns);
    for (std::int32_t column = 0; column < columns; ++column) {
        for (std::size_t index = 0; index < checked_rows.size(); ++index) {
            got[std::size_t(column) * checked_rows.size() + index] = __bfloat162float(
                eager[std::size_t(column) * rows + checked_rows[index]]);
        }
    }
    const Error error = compare(got, want);
    if (!finite || !(error.relative_rms < 3.5e-3 && error.relative_max < 5e-3)) {
        throw std::runtime_error("Q4_K fused warp-pair output differs from the FP64 oracle");
    }
    check_graph_parity(
        [&](cudaStream_t stream) {
            ops::gguf::vector_swiglu(ops::gguf::GgmlType::Q4_K, weights.p, row_bytes, rows, k,
                                     input.p, columns,
                                     static_cast<__nv_bfloat16*>(actual.data()), rows, stream);
        },
        actual, eager, "Q4_K fused warp-pair grid-stride graph", nullptr);
}

int run_q4_t4_fused_warp_pair_regression() {
    constexpr std::int32_t rows = 8193;
    std::vector<std::int32_t> all_rows(rows);
    for (std::int32_t row = 0; row < rows; ++row) { all_rows[row] = row; }
    run_q4_t4_fused_warp_pair_case(rows, 256, 87031, all_rows);
    run_q4_t4_fused_warp_pair_case(rows, 5120, 87032,
                                  {0, 1, 257, 1023, 2048, 4095, 4096, 6143, 8191, 8192});
    return 0;
}
#endif

int run_swiglu_q8_fusion(std::uint32_t seed, bool test_route) {
    constexpr std::int32_t k = 5120;
    constexpr std::int32_t rows[] = {32, 64, 17408};
    constexpr std::int32_t widths[] = {1, 3, 4, 8};
    const TypeCase types[] = {{QType::GGUF_Q4_K, "q4_k", {0, 2}},
                              {QType::GGUF_Q6_K, "q6_k", {208}}};
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0f, 0.3f);
    for (const auto& c : types) {
        for (const std::int32_t n : rows) {
            const auto host_weights = random_blocks(c, 2 * n, k, rng);
            DeviceBuffer weights(host_weights.size());
            weights.copy_from_host(host_weights.data(), host_weights.size());
            const Weight gate = gguf_weight(c.qtype, weights.p, 2 * n, k);
            const auto block = gguf_block_shape(c.qtype);
            const std::int64_t row_bytes = std::int64_t(k / block.elements) * block.bytes;
            for (const std::int32_t t : widths) {
                std::vector<__nv_bfloat16> host_x(std::size_t(k) * t);
                for (auto& value : host_x) { value = __float2bfloat16(normal(rng)); }
                DeviceBuffer x(std::size_t(k) * t * sizeof(__nv_bfloat16));
                x.copy_from_host(host_x.data(), std::size_t(k) * t * sizeof(__nv_bfloat16));
                const std::size_t input_bytes = ops::gguf::vector_activation_bytes(k, t);
                DeviceBuffer input(input_bytes);
                ops::gguf::quantize_vector_activation(static_cast<const __nv_bfloat16*>(x.p), k, t,
                                                      gate.input_columns, input.p, nullptr);
                DeviceBuffer reference(std::size_t(n) * t * sizeof(__nv_bfloat16));
                ops::gguf::vector_swiglu(ops::detail::gguf_type(c.qtype), gate.qdata, row_bytes, n,
                                         k, input.p, t,
                                         static_cast<__nv_bfloat16*>(reference.p), n, nullptr);
                const std::size_t output_bytes = ops::gguf::vector_activation_bytes(n, t);
                std::vector<std::uint8_t> want;
                DeviceBuffer expected(output_bytes);
                if (n % 256 == 0) {
                    ops::gguf::quantize_vector_activation(
                        static_cast<const __nv_bfloat16*>(reference.p), n, t, nullptr, expected.p,
                        nullptr);
                    want.resize(output_bytes);
                    expected.copy_to_host(want.data(), want.size());
                } else {
                    std::vector<__nv_bfloat16> reference_host(std::size_t(n) * t);
                    reference.copy_to_host(reference_host.data(), reference_host.size() * sizeof(__nv_bfloat16));
                    want = cpu_quantize_vector_activation(reference_host, n, t);
                }
                GuardedRaw actual(output_bytes);
                actual.poison();
                ops::gguf::vector_swiglu_q8(ops::detail::gguf_type(c.qtype), gate.qdata, row_bytes,
                                            n, k, input.p, t, actual.data(), nullptr);
                check(cudaDeviceSynchronize(), "fused SwiGLU q8 parity");
                const auto got = actual.read("read fused q8 output");
                if (got != want) {
                    throw std::runtime_error(std::string(c.name) + " fused q8 pack differs at N=" +
                                             std::to_string(n) + " T=" + std::to_string(t));
                }
                for (const int warps : {8, 16}) {
                    actual.poison();
                    ops::gguf::vector_swiglu_q8(ops::detail::gguf_type(c.qtype), gate.qdata,
                                                row_bytes, n, k, input.p, t, actual.data(), nullptr,
                                                warps);
                    check(cudaDeviceSynchronize(), "fused q8 alternate row scheduling");
                    if (actual.read("read scheduled fused q8 output") != want) {
                        throw std::runtime_error("alternate fused q8 scheduling changed packed bytes");
                    }
                }
                if (n == 17408) {
                    cudaStream_t timing_stream = nullptr;
                    cudaEvent_t start = nullptr, end = nullptr;
                    check(cudaStreamCreate(&timing_stream), "create fusion timing stream");
                    check(cudaEventCreate(&start), "create fusion timing start");
                    check(cudaEventCreate(&end), "create fusion timing end");
                    for (int sample = 0; sample < 16; ++sample) {
                        constexpr int policies[] = {0, 4, 8, 16};
                        const int pass = sample / 4;
                        const int warps = policies[(pass & 1) ? 3 - sample % 4 : sample % 4];
                        cudaGraph_t timing_graph = nullptr;
                        cudaGraphExec_t timing_exec = nullptr;
                        check(cudaStreamBeginCapture(timing_stream, cudaStreamCaptureModeGlobal),
                              "begin fusion timing graph");
                        if (warps == 0) {
                            ops::gguf::vector_swiglu(ops::detail::gguf_type(c.qtype), gate.qdata,
                                                     row_bytes, n, k, input.p, t,
                                                     static_cast<__nv_bfloat16*>(reference.p), n,
                                                     timing_stream);
                            ops::gguf::quantize_vector_activation(
                                static_cast<const __nv_bfloat16*>(reference.p), n, t, nullptr,
                                expected.p, timing_stream);
                        } else {
                            ops::gguf::vector_swiglu_q8(ops::detail::gguf_type(c.qtype), gate.qdata,
                                                        row_bytes, n, k, input.p, t, actual.data(),
                                                        timing_stream, warps);
                        }
                        check(cudaStreamEndCapture(timing_stream, &timing_graph), "end fusion timing graph");
                        check(cudaGraphInstantiate(&timing_exec, timing_graph, nullptr, nullptr, 0),
                              "instantiate fusion timing graph");
                        for (int i = 0; i < 10; ++i) {
                            check(cudaGraphLaunch(timing_exec, timing_stream), "warm fusion timing graph");
                        }
                        check(cudaEventRecord(start, timing_stream), "start fusion timing");
                        for (int i = 0; i < 100; ++i) {
                            check(cudaGraphLaunch(timing_exec, timing_stream), "replay fusion timing graph");
                        }
                        check(cudaEventRecord(end, timing_stream), "end fusion timing");
                        check(cudaEventSynchronize(end), "wait fusion timing");
                        float elapsed = 0.0f;
                        check(cudaEventElapsedTime(&elapsed, start, end), "read fusion timing");
                        std::cout << "FUSION_TIMING type=" << c.name << " N=" << n << " K=" << k
                                  << " T=" << t << " warps=" << warps << " pass=" << pass
                                  << " graph_us=" << elapsed * 10.0f << std::endl;
                        check(cudaGraphExecDestroy(timing_exec), "destroy fusion timing executable");
                        check(cudaGraphDestroy(timing_graph), "destroy fusion timing graph");
                    }
                    check(cudaEventDestroy(start), "destroy fusion timing start");
                    check(cudaEventDestroy(end), "destroy fusion timing end");
                    check(cudaStreamDestroy(timing_stream), "destroy fusion timing stream");
                }
                if (test_route && n == 17408 &&
                    ((c.qtype == QType::GGUF_Q4_K && t == 4) ||
                     (c.qtype == QType::GGUF_Q6_K && t == 3))) {
                    const std::int32_t down_rows = 32;
                    const auto host_down = random_blocks(c, down_rows, n, rng);
                    DeviceBuffer down_storage(host_down.size());
                    down_storage.copy_from_host(host_down.data(), host_down.size());
                    const Weight down = gguf_weight(c.qtype, down_storage.p, down_rows, n);
                    const std::size_t workspace_bytes = std::max(
                        {ops::linear_swiglu_workspace_capacity_bytes(
                             c.qtype, 2 * n, k, ops::LinearPolicy::A16Only, t, t),
                         ops::linear_add_workspace_capacity_bytes(
                             c.qtype, down_rows, n, ops::LinearPolicy::A16Only, t, t),
                         ops::detail::gguf_swiglu_linear_add_workspace_capacity_bytes(
                             gate, nullptr, down, t, t)});
                    WorkspaceArena workspace(workspace_bytes + (1U << 20));
                    DeviceBuffer activation(std::size_t(n) * t * sizeof(__nv_bfloat16));
                    Tensor x_tensor(x.p, DType::BF16, {k, t});
                    Tensor activation_tensor(activation.p, DType::BF16, {n, t});
                    DeviceBuffer baseline(std::size_t(down_rows) * t * sizeof(__nv_bfloat16));
                    DeviceBuffer fused(baseline.bytes);
                    std::vector<__nv_bfloat16> initial(std::size_t(down_rows) * t);
                    for (auto& value : initial) { value = __float2bfloat16(normal(rng)); }
                    baseline.copy_from_host(initial.data(), initial.size() * sizeof(__nv_bfloat16));
                    fused.copy_from_host(initial.data(), initial.size() * sizeof(__nv_bfloat16));
                    Tensor baseline_tensor(baseline.p, DType::BF16, {down_rows, t});
                    Tensor fused_tensor(fused.p, DType::BF16, {down_rows, t});
                    ops::linear_swiglu(x_tensor, gate, activation_tensor,
                                       ops::LinearPolicy::A16Only, workspace, nullptr);
                    ops::linear_add(activation_tensor, down, baseline_tensor,
                                    ops::LinearPolicy::A16Only, workspace, nullptr);
                    check(cudaDeviceSynchronize(), "baseline fused q8 residual");
                    const bool used = ops::detail::gguf_swiglu_linear_add(
                        x_tensor, gate, nullptr, down, fused_tensor, workspace, nullptr);
                    check(cudaDeviceSynchronize(), "fused q8 residual");
                    if (!used) {
                        throw std::runtime_error("eligible fused q8 down projection was not selected");
                    }
                    std::vector<__nv_bfloat16> baseline_host(initial.size());
                    std::vector<__nv_bfloat16> fused_host(initial.size());
                    baseline.copy_to_host(baseline_host.data(), baseline_host.size() * sizeof(__nv_bfloat16));
                    fused.copy_to_host(fused_host.data(), fused_host.size() * sizeof(__nv_bfloat16));
                    if (std::memcmp(baseline_host.data(), fused_host.data(),
                                    fused_host.size() * sizeof(__nv_bfloat16)) != 0) {
                        throw std::runtime_error("fused q8 down projection differs from GGUF fallback");
                    }
                    if (used) {
                        std::vector<std::int32_t> down_columns(n);
                        for (std::int32_t i = 0; i < n; ++i) { down_columns[i] = i; }
                        DeviceBuffer down_columns_device(down_columns.size() * sizeof(std::int32_t));
                        down_columns_device.copy_from_host(down_columns.data(),
                                                           down_columns.size() * sizeof(std::int32_t));
                        Weight gathered_down = down;
                        gathered_down.input_columns =
                            static_cast<const std::int32_t*>(down_columns_device.p);
                        if (ops::detail::gguf_swiglu_linear_add_workspace_capacity_bytes(
                                gate, nullptr, gathered_down, t, t) != 0 ||
                            ops::detail::gguf_swiglu_linear_add(x_tensor, gate, nullptr,
                                                                gathered_down, fused_tensor,
                                                                workspace, nullptr)) {
                            throw std::runtime_error("fused q8 route admitted a gathered down weight");
                        }
                        if (ops::detail::gguf_swiglu_linear_add_workspace_capacity_bytes(
                                gate, &gate, down, t, t) != 0 ||
                            ops::detail::gguf_swiglu_linear_add(x_tensor, gate, &gate, down,
                                                                fused_tensor, workspace, nullptr)) {
                            throw std::runtime_error("fused q8 route admitted a separate up weight");
                        }
                    }
                }
                if ((n == 17408 && t == 4) || (n == 64 && t == 3)) {
                    cudaStream_t stream = nullptr;
                    cudaGraph_t graph = nullptr;
                    cudaGraphExec_t executable = nullptr;
                    check(cudaStreamCreate(&stream), "create fused q8 graph stream");
                    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
                          "begin fused q8 graph capture");
                    ops::gguf::vector_swiglu_q8(ops::detail::gguf_type(c.qtype), gate.qdata,
                                                row_bytes, n, k, input.p, t, actual.data(), stream);
                    check(cudaStreamEndCapture(stream, &graph), "end fused q8 graph capture");
                    check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0),
                          "instantiate fused q8 graph");
                    actual.poison();
                    check(cudaGraphLaunch(executable, stream), "launch fused q8 graph");
                    check(cudaStreamSynchronize(stream), "synchronize fused q8 graph");
                    if (actual.read("read fused q8 graph output") != want) {
                        throw std::runtime_error("fused q8 graph replay differs from eager output");
                    }
                    check(cudaGraphExecDestroy(executable), "destroy fused q8 graph executable");
                    check(cudaGraphDestroy(graph), "destroy fused q8 graph");
                    check(cudaStreamDestroy(stream), "destroy fused q8 graph stream");
                }
            }
        }
    }
    return 0;
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
        } else {
            std::cout << "OK   " << c.name << " K=" << k << " T=" << t << " rms_rel="
                      << std::setprecision(8) << error.relative_rms << " max_rel="
                      << error.relative_max << " max_abs=" << error.max_absolute << '\n';
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
        } else {
            std::cout << "OK   " << c.name << " K=" << k << " SwiGLU T=" << t
                      << " rms_rel=" << std::setprecision(8) << error.relative_rms
                      << " max_rel=" << error.relative_max << " max_abs=" << error.max_absolute
                      << '\n';
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

int run_mmq_real_shape(const TypeCase& c, std::int32_t k, std::uint32_t seed,
                       std::ofstream* output_file, const std::string& weight_directory,
                       std::size_t& files_used) {
    constexpr std::int32_t rows = 256;
    constexpr std::int32_t fixture_rows = 257;
    constexpr std::int32_t widths[] = {9, 17, 40, 70, 127, 130, 1024};
    const auto type = ops::detail::gguf_type(c.qtype);
    if ((c.qtype != QType::GGUF_Q4_K && c.qtype != QType::GGUF_Q6_K) ||
        !ops::gguf::has_matrix_kernel(type)) {
        throw std::runtime_error("MMQ real-shapes requires the Q4_K or Q6_K matrix route");
    }
    const char* type_name = c.qtype == QType::GGUF_Q4_K ? "Q4_K" : "Q6_K";

    const auto block = gguf_block_shape(c.qtype);
    const std::size_t bytes = std::size_t(rows) * (k / block.elements) * block.bytes;
    const std::string filename = fixture_prefix(c.qtype) + "-k" + std::to_string(k) + ".bin";
    std::vector<std::uint8_t> host_blocks;
    const bool loaded = !weight_directory.empty() &&
                        load_blocks(weight_directory, filename,
                                    std::size_t(fixture_rows) * (k / block.elements) * block.bytes,
                                    host_blocks);
    if (loaded) {
        host_blocks.resize(bytes);
        ++files_used;
    } else {
        std::mt19937 weight_rng(seed);
        host_blocks = random_blocks(c, rows, k, weight_rng, true);
    }
    const std::string source = loaded ? filename : "synthetic";
    std::vector<float> cpu_weights;
    decode_k_cpu(c.qtype, host_blocks, rows, k, cpu_weights);
    DeviceBuffer weight_device(host_blocks.size());
    weight_device.copy_from_host(host_blocks.data(), host_blocks.size());
    const Weight weight = gguf_weight(c.qtype, weight_device.p, rows, k);
    DeviceBuffer unaligned_weight(host_blocks.size() + 4);
    auto* shifted_bytes = static_cast<std::uint8_t*>(unaligned_weight.p) + 4;
    check(cudaMemcpy(shifted_bytes, host_blocks.data(), host_blocks.size(), cudaMemcpyHostToDevice),
          "copy MMQ alignment fallback weights");
    const Weight shifted_weight = gguf_weight(c.qtype, shifted_bytes, rows, k);

    constexpr std::int32_t max_t = 1024;
    std::mt19937 activation_rng(seed + 1907);
    std::normal_distribution<float> normal(0.0f, 0.35f);
    std::vector<__nv_bfloat16> x_host(std::size_t(k) * max_t);
    for (std::int32_t col = 0; col < max_t; ++col) {
        for (std::int32_t i = 0; i < k; ++i) {
            float value = normal(activation_rng);
            if (i % 32 == 0) { value = ((i / 32 + col) & 1) ? 4.0f : -4.0f; }
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
    int failures = 0;

    for (const std::int32_t t : widths) {
        GuardedOutput guarded(std::size_t(rows) * t);
        Tensor x(x_device.p, DType::BF16, {k, t});
        Tensor out(guarded.data(), DType::BF16, {rows, t});
        guarded.poison();
        ops::linear(x, weight, out, ops::LinearPolicy::A16Only, workspace, nullptr);
        check(cudaDeviceSynchronize(), "MMQ real-shape linear");
        const auto host = guarded.read("read MMQ real-shape output");
        const auto got = to_float(host);
        if (!std::all_of(got.begin(), got.end(),
                         [](float value) { return std::isfinite(value); })) {
            throw std::runtime_error("MMQ real-shape output contains non-finite values");
        }

        std::vector<std::int32_t> samples{0, 1, t / 2, t - 2, t - 1};
        std::sort(samples.begin(), samples.end());
        samples.erase(std::unique(samples.begin(), samples.end()), samples.end());
        std::vector<float> sampled_got;
        std::vector<double> sampled_want;
        sampled_got.reserve(samples.size() * rows);
        sampled_want.reserve(samples.size() * rows);
        for (const std::int32_t col : samples) {
            for (std::int32_t row = 0; row < rows; ++row) {
                double acc = 0.0;
                const float* wr = cpu_weights.data() + std::size_t(row) * k;
                const float* xr = x_float.data() + std::size_t(col) * k;
                for (std::int32_t i = 0; i < k; ++i) {
                    acc += double(wr[i]) * xr[i];
                }
                sampled_got.push_back(got[std::size_t(col) * rows + row]);
                sampled_want.push_back(acc);
            }
        }
        const Error error = compare(sampled_got, sampled_want);
        if (!(error.relative_rms < 5e-2 && error.relative_max < 2e-1)) {
            std::cerr << "FAIL MMQ " << type_name << " K=" << k << " T=" << t
                      << " rms_rel=" << error.relative_rms << " max_rel="
                      << error.relative_max << " max_abs=" << error.max_absolute << '\n';
            ++failures;
        } else {
            std::cout << "OK MMQ " << type_name << " K=" << k << " T=" << t << " rms_rel="
                      << std::setprecision(8) << error.relative_rms << " max_rel="
                      << error.relative_max << " max_abs=" << error.max_absolute << '\n';
        }
        write_outputs(output_file, host);
        if (t == 17) {
            guarded.poison();
            ops::linear(x, shifted_weight, out, ops::LinearPolicy::A16Only, workspace, nullptr);
            check(cudaDeviceSynchronize(), "MMQ alignment fallback linear");
            const auto shifted_host = guarded.read("read MMQ alignment fallback output");
            if (std::memcmp(host.data(), shifted_host.data(), host.size() * sizeof(__nv_bfloat16)) != 0) {
                throw std::runtime_error("MMQ alignment fallback changed exact BF16 output");
            }
            write_outputs(output_file, shifted_host);
        }
        if (t == 9 || t == 40 || t == 130 || t == 1024) {
            check_graph_parity(
                [&](cudaStream_t stream) {
                    ops::linear(x, weight, out, ops::LinearPolicy::A16Only, workspace, stream);
                },
                guarded, host,
                "MMQ " + std::string(type_name) + " K=" + std::to_string(k) +
                    " T=" + std::to_string(t), output_file);
        }
    }
    std::cout << "MMQ source " << type_name << " K=" << k << " rows=" << rows
              << " weights=" << source << '\n';
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
    bool mmq_real_shapes = false;
    bool mmq_q6_real_shapes = false;
    bool swiglu_q8_fusion = false;
    bool swiglu_q8_benchmark = false;
    bool rmsnorm_q8_fusion = false;
    bool rmsnorm_mmq_q8_fusion = false;
    bool dual_mmq_fusion = false;
#if defined(NINFER_GGUF_Q4_T4_FUSED_WARP_PAIR_TEST)
    bool q4_t4_fused_warp_pair_regression = false;
#endif
    bool output_specified = false;
    bool weight_directory_specified = false;
    std::string output_path;
    std::string weight_directory;
    for (int i = 1; i < argc; ++i) {
        const std::string argument(argv[i]);
        if (argument == "--real-shapes" && !real_shapes) {
            real_shapes = true;
        } else if (argument == "--mmq-real-shapes" && !mmq_real_shapes) {
            mmq_real_shapes = true;
        } else if (argument == "--mmq-q6-real-shapes" && !mmq_q6_real_shapes) {
            mmq_q6_real_shapes = true;
        } else if (argument == "--swiglu-q8-fusion" && !swiglu_q8_fusion) {
            swiglu_q8_fusion = true;
        } else if (argument == "--swiglu-q8-benchmark" && !swiglu_q8_fusion) {
            swiglu_q8_fusion = true;
            swiglu_q8_benchmark = true;
        } else if (argument == "--rmsnorm-q8-fusion" && !rmsnorm_q8_fusion) {
            rmsnorm_q8_fusion = true;
        } else if (argument == "--rmsnorm-mmq-q8-fusion" && !rmsnorm_mmq_q8_fusion) {
            rmsnorm_mmq_q8_fusion = true;
        } else if (argument == "--dual-mmq-fusion" && !dual_mmq_fusion) {
            dual_mmq_fusion = true;
#if defined(NINFER_GGUF_Q4_T4_FUSED_WARP_PAIR_TEST)
        } else if (argument == "--q4-t4-fused-warp-pair-regression" &&
                   !q4_t4_fused_warp_pair_regression) {
            q4_t4_fused_warp_pair_regression = true;
#endif
        } else if (argument == "--output" && !output_specified && i + 1 < argc) {
            output_specified = true;
            output_path = argv[++i];
        } else if (argument == "--weight-dir" && !weight_directory_specified && i + 1 < argc) {
            weight_directory_specified = true;
            weight_directory = argv[++i];
        } else {
            std::cerr << "usage: ninfer_linear_gguf_test [--rmsnorm-q8-fusion | --rmsnorm-mmq-q8-fusion | --dual-mmq-fusion | --swiglu-q8-fusion | --swiglu-q8-benchmark | --mmq-real-shapes | --mmq-q6-real-shapes | --real-shapes [--output <file>] "
                         "[--weight-dir <dir>]]\n";
            return 2;
        }
    }
    if ((output_specified &&
         (!(real_shapes || mmq_real_shapes || mmq_q6_real_shapes) || output_path.empty())) ||
        (weight_directory_specified &&
         (!(real_shapes || mmq_real_shapes || mmq_q6_real_shapes) || weight_directory.empty())) ||
        (real_shapes && (mmq_real_shapes || mmq_q6_real_shapes)) ||
        (mmq_real_shapes && mmq_q6_real_shapes) ||
        ((real_shapes || mmq_real_shapes || mmq_q6_real_shapes) &&
         (swiglu_q8_fusion || rmsnorm_q8_fusion || rmsnorm_mmq_q8_fusion || dual_mmq_fusion)) ||
        (rmsnorm_q8_fusion && (swiglu_q8_fusion || rmsnorm_mmq_q8_fusion)) ||
        (rmsnorm_mmq_q8_fusion && (swiglu_q8_fusion || dual_mmq_fusion)) ||
        (dual_mmq_fusion && (swiglu_q8_fusion || rmsnorm_q8_fusion))
#if defined(NINFER_GGUF_Q4_T4_FUSED_WARP_PAIR_TEST)
        || (q4_t4_fused_warp_pair_regression &&
            (real_shapes || mmq_real_shapes || mmq_q6_real_shapes || swiglu_q8_fusion ||
             rmsnorm_q8_fusion || rmsnorm_mmq_q8_fusion || dual_mmq_fusion || output_specified ||
             weight_directory_specified))
#endif
        ) {
        std::cerr << "usage: ninfer_linear_gguf_test [--rmsnorm-q8-fusion | --rmsnorm-mmq-q8-fusion | --dual-mmq-fusion | --swiglu-q8-fusion | --swiglu-q8-benchmark | --mmq-real-shapes | --mmq-q6-real-shapes | --real-shapes [--output <file>] "
                     "[--weight-dir <dir>]]\n";
        return 2;
    }
    try {
#if defined(NINFER_GGUF_Q4_T4_FUSED_WARP_PAIR_TEST)
        if (q4_t4_fused_warp_pair_regression) {
            const int failures = run_q4_t4_fused_warp_pair_regression();
            std::cout << (failures == 0 ? "OK" : "FAIL")
                      << " GGUF Q4_K T4 fused warp-pair regression\n";
            return failures == 0 ? 0 : 1;
        }
#endif
        if (swiglu_q8_fusion) {
            const int failures = run_swiglu_q8_fusion(28001, !swiglu_q8_benchmark);
            std::cout << (failures == 0 ? "OK" : "FAIL") << " GGUF SwiGLU q8 fusion\n";
            return failures == 0 ? 0 : 1;
        }
        if (mmq_real_shapes) {
            std::ofstream output_file;
            std::ofstream* output = nullptr;
            if (output_specified) {
                output_file.open(output_path, std::ios::binary | std::ios::trunc);
                if (!output_file) {
                    throw std::runtime_error("could not open MMQ output file: " + output_path);
                }
                output = &output_file;
            }
            if (weight_directory_specified && !std::filesystem::is_directory(weight_directory)) {
                throw std::runtime_error("--weight-dir is not a directory: " + weight_directory);
            }
            const auto all = cases();
            const auto q4 = std::find_if(all.begin(), all.end(), [](const TypeCase& c) {
                return c.qtype == QType::GGUF_Q4_K;
            });
            if (q4 == all.end()) { throw std::logic_error("Q4_K case is not registered"); }
            std::size_t files_used = 0;
            int failures = 0;
            for (const std::int32_t k : {5120, 17408}) {
                failures += run_mmq_real_shape(*q4, k, 29003 + k, output, weight_directory,
                                                files_used);
            }
            if (weight_directory_specified && files_used == 0) {
                throw std::runtime_error("--weight-dir provided no q4-k5120.bin or q4-k17408.bin");
            }
            if (output != nullptr) {
                output_file.flush();
                if (!output_file) { throw std::runtime_error("failed flushing MMQ output file"); }
            }
            std::cout << (failures == 0 ? "OK" : "FAIL") << " GGUF MMQ real shapes\n";
            return failures == 0 ? 0 : 1;
        }
        if (mmq_q6_real_shapes) {
            std::ofstream output_file;
            std::ofstream* output = nullptr;
            if (output_specified) {
                output_file.open(output_path, std::ios::binary | std::ios::trunc);
                if (!output_file) {
                    throw std::runtime_error("could not open Q6 MMQ output file: " +
                                             output_path);
                }
                output = &output_file;
            }
            if (weight_directory_specified && !std::filesystem::is_directory(weight_directory)) {
                throw std::runtime_error("--weight-dir is not a directory: " + weight_directory);
            }
            const auto all = cases();
            const auto q6 = std::find_if(all.begin(), all.end(), [](const TypeCase& c) {
                return c.qtype == QType::GGUF_Q6_K;
            });
            if (q6 == all.end()) { throw std::logic_error("Q6_K case is not registered"); }
            std::size_t files_used = 0;
            int failures = 0;
            for (const std::int32_t k : {5120, 17408}) {
                failures += run_mmq_real_shape(*q6, k, 39003 + k, output, weight_directory,
                                                files_used);
            }
            if (weight_directory_specified && files_used == 0) {
                throw std::runtime_error(
                    "--weight-dir provided no q6-k5120.bin or q6-k17408.bin");
            }
            if (output != nullptr) {
                output_file.flush();
                if (!output_file) { throw std::runtime_error("failed flushing Q6 MMQ output file"); }
            }
            std::cout << (failures == 0 ? "OK" : "FAIL") << " GGUF Q6_K MMQ real shapes\n";
            return failures == 0 ? 0 : 1;
        }
        if (rmsnorm_q8_fusion) {
            run_rmsnorm_q8_fusion(28002);
            std::cout << "OK GGUF RMSNorm q8 fusion\n";
            return 0;
        }
        if (rmsnorm_mmq_q8_fusion) {
            run_rmsnorm_mmq_q8_fusion(28004);
            std::cout << "OK GGUF RMSNorm MMQ q8 fusion\n";
            return 0;
        }
#if defined(NINFER_GGUF_DUAL_MMQ_FUSION)
        if (dual_mmq_fusion) {
            run_dual_mmq_fusion();
            std::cout << "OK GGUF dual Q4_K MMQ fusion\n";
            return 0;
        }
#else
        if (dual_mmq_fusion) {
            throw std::runtime_error("dual MMQ fusion test requires NINFER_GGUF_DUAL_MMQ_FUSION");
        }
#endif
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
