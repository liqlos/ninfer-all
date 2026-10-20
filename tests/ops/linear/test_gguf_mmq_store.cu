#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ops/linear/gguf/ggml_bridge.h"
#include "ops/linear/gguf/gguf_linear.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::ops;
using ninfer::ops::detail::GgufProduct;
using ninfer::ops::detail::GgufShape;
using ninfer::ops::detail::GgufEpilogue;

void check(cudaError_t status, const char* label) {
    if (status != cudaSuccess) { throw std::runtime_error(std::string(label) + ": " + cudaGetErrorString(status)); }
}

struct Case {
    QType qtype;
    gguf::GgmlType type;
    int rows;
    int k;
    int scale_offset;
};

Case q4_case() { return {QType::GGUF_Q4_K, gguf::GgmlType::Q4_K, 6144, 5120, 0}; }
Case q6_case() { return {QType::GGUF_Q6_K, gguf::GgmlType::Q6_K, 5120, 17408, 208}; }

std::vector<std::uint8_t> make_weight(const Case& c, std::uint32_t seed) {
    const auto block = gguf::block_shape(c.type);
    const std::size_t blocks = std::size_t(c.rows) * (c.k / block.elements);
    std::vector<std::uint8_t> bytes(blocks * block.bytes);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> byte(0, 255);
    for (auto& value : bytes) { value = static_cast<std::uint8_t>(byte(rng)); }
    for (std::size_t i = 0; i < blocks; ++i) {
        const std::uint16_t scale = 0x3980;
        std::memcpy(bytes.data() + i * block.bytes + c.scale_offset, &scale, sizeof(scale));
        if (c.qtype == QType::GGUF_Q4_K) {
            std::memcpy(bytes.data() + i * block.bytes + 2, &scale, sizeof(scale));
        }
    }
    return bytes;
}

Weight make_gguf_weight(const Case& c, const void* data) {
    Weight weight;
    weight.qtype = c.qtype;
    weight.layout = QuantLayout::GgufBlocks;
    weight.qdata = data;
    weight.payload = data;
    weight.n = weight.shape[0] = weight.padded_shape[0] = c.rows;
    weight.k = weight.shape[1] = weight.padded_shape[1] = c.k;
    weight.ndim = 2;
    const auto block = gguf::block_shape(c.type);
    weight.group_size = static_cast<std::uint32_t>(block.elements);
    weight.group = block.elements;
    return weight;
}

std::vector<__nv_bfloat16> make_input(int k, int columns, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> value(-0.25f, 0.25f);
    std::vector<__nv_bfloat16> input(std::size_t(k) * columns);
    for (auto& element : input) { element = __float2bfloat16(value(rng)); }
    return input;
}

bool compare_rounding(const std::vector<float>& fp32, const std::vector<__nv_bfloat16>& bf16,
                      int rows, int columns) {
    for (std::size_t i = 0; i < std::size_t(rows) * columns; ++i) {
        const __nv_bfloat16 expected = __float2bfloat16(fp32[i]);
        std::uint16_t expected_bits = 0;
        std::uint16_t actual_bits = 0;
        std::memcpy(&expected_bits, &expected, sizeof(expected_bits));
        std::memcpy(&actual_bits, &bf16[i], sizeof(actual_bits));
        if (expected_bits != actual_bits) { return false; }
    }
    return true;
}

bool run_case(const Case& c, int columns, bool prepared, bool expect_fused) {
    const auto host_weight = make_weight(c, 0xE190u + c.k + columns);
    DeviceBuffer weight(host_weight.size());
    weight.copy_from_host(host_weight.data(), host_weight.size());

    const auto host_input = make_input(c.k, columns, 0xA119u + columns);
    DeviceBuffer input(host_input.size() * sizeof(__nv_bfloat16));
    input.copy_from_host(host_input.data(), input.bytes);

    const std::size_t activation_bytes = gguf::matrix_activation_bytes(c.k, columns);
    DeviceBuffer activation(activation_bytes);
    gguf::quantize_matrix_activation(c.type, static_cast<const __nv_bfloat16*>(input.p), c.k,
                                     columns, nullptr, activation.p, nullptr);

    gguf::MatrixBf16StorePlan plan;
    const bool plan_supported = gguf::matrix_bf16_store_plan(c.type, c.rows, columns, plan);
    int expected_j = 128;
#if defined(NINFER_GGUF_SM86_MMQ_MAX_J)
    if (columns >= 64) { expected_j = NINFER_GGUF_SM86_MMQ_MAX_J; }
#endif
    if (expect_fused && (!plan_supported || plan.J != expected_j || plan.I != 128 || !plan.stream_k)) {
        throw std::runtime_error("native MMQ store plan did not select the expected J/I tile");
    }
    if (!expect_fused && plan_supported) {
        throw std::runtime_error("unsupported width unexpectedly admitted BF16 MMQ store");
    }

    DeviceBuffer actual(std::size_t(c.rows) * columns * sizeof(__nv_bfloat16));
    DeviceBuffer fp32(std::size_t(c.rows) * columns * sizeof(float));
    const std::size_t fixup_bytes = gguf::matrix_fixup_bytes(c.type, c.rows, columns);
    DeviceBuffer fixup(fixup_bytes);
    gguf::matrix_product(c.type, weight.p, std::int64_t(c.k / 256) * gguf::block_shape(c.type).bytes,
                         c.rows, c.k, activation.p, columns, static_cast<float*>(fp32.p), c.rows,
                         fixup.p, nullptr);

    Tensor output_tensor(actual.p, DType::BF16, {c.rows, columns});
    Weight product_weight = make_gguf_weight(c, weight.p);
    GgufProduct product{&product_weight, &output_tensor, 0, GgufEpilogue::Store};
    bool fused = false;
    bool store_enqueued = false;
    if (prepared) {
        const gguf::MatrixActivationPlane plane{c.type, activation.p};
        WorkspaceArena workspace(std::size_t(512) << 20);
        fused = detail::gguf_project_matrix_q8_1_activations(
            c.k, columns, {&plane, 1}, {&product, 1}, workspace, nullptr,
            &store_enqueued);
    } else {
        const GgufShape shape{c.qtype, c.rows, c.k};
        const std::size_t workspace_bytes = detail::gguf_project_workspace_bytes({&shape, 1}, columns, columns);
        WorkspaceArena workspace(workspace_bytes + (1U << 20));
        Tensor x(input.p, DType::BF16, {c.k, columns});
        fused = detail::gguf_project(x, {&product, 1}, workspace, nullptr,
                                     &store_enqueued);
    }
    check(cudaDeviceSynchronize(), "MMQ store test sync");
    if (fused || store_enqueued != expect_fused) {
        throw std::runtime_error("GGUF projector reported the wrong MMQ store route");
    }

    std::vector<float> host_fp32(std::size_t(c.rows) * columns);
    std::vector<__nv_bfloat16> host_bf16(std::size_t(c.rows) * columns);
    fp32.copy_to_host(host_fp32.data(), fp32.bytes);
    actual.copy_to_host(host_bf16.data(), actual.bytes);
    if (!compare_rounding(host_fp32, host_bf16, c.rows, columns)) {
        throw std::runtime_error("fused BF16 MMQ store differs from the native FP32 accumulator rounding");
    }
    return true;
}

bool run_accumulate_graph_case() {
    const Case c = q6_case();
    constexpr int columns = 1024;
    const int stride = c.rows + 19;
    cudaStream_t stream = nullptr;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create graph test stream");
    const auto host_weight = make_weight(c, 0xE19ACCU);
    DeviceBuffer weight(host_weight.size());
    check(cudaMemcpyAsync(weight.p, host_weight.data(), host_weight.size(), cudaMemcpyHostToDevice,
                          stream), "copy graph test weight");
    const auto host_input = make_input(c.k, columns, 0xA19ACCU);
    DeviceBuffer input(host_input.size() * sizeof(__nv_bfloat16));
    check(cudaMemcpyAsync(input.p, host_input.data(), input.bytes, cudaMemcpyHostToDevice, stream),
          "copy graph test input");
    DeviceBuffer activation(gguf::matrix_activation_bytes(c.k, columns));
    gguf::quantize_matrix_activation(c.type, static_cast<const __nv_bfloat16*>(input.p), c.k,
                                     columns, nullptr, activation.p, stream);

    std::vector<__nv_bfloat16> initial(std::size_t(stride) * columns);
    std::mt19937 rng(0xA66U);
    std::uniform_real_distribution<float> residual(-0.5f, 0.5f);
    for (int column = 0; column < columns; ++column) {
        for (int row = 0; row < stride; ++row) {
            initial[std::size_t(column) * stride + row] =
                row < c.rows ? __float2bfloat16(residual(rng)) : __float2bfloat16(123.0f);
        }
    }
    DeviceBuffer actual(initial.size() * sizeof(__nv_bfloat16));
    check(cudaMemcpyAsync(actual.p, initial.data(), actual.bytes, cudaMemcpyHostToDevice, stream),
          "initialize graph test output");
    DeviceBuffer fp32(std::size_t(c.rows) * columns * sizeof(float));
    DeviceBuffer fixup(gguf::matrix_fixup_bytes(c.type, c.rows, columns));
    gguf::matrix_product(c.type, weight.p,
                         std::int64_t(c.k / 256) * gguf::block_shape(c.type).bytes,
                         c.rows, c.k, activation.p, columns, static_cast<float*>(fp32.p), c.rows,
                         fixup.p, stream);
    check(cudaStreamSynchronize(stream), "warm graph test kernels");

    Tensor input_tensor(input.p, DType::BF16, {c.k, columns});
    Tensor output_tensor(actual.p, DType::BF16, {stride, columns});
    Weight product_weight = make_gguf_weight(c, weight.p);
    GgufProduct product{&product_weight, &output_tensor, 0, GgufEpilogue::Accumulate};
    const GgufShape shape{c.qtype, c.rows, c.k};
    WorkspaceArena workspace(detail::gguf_project_workspace_bytes({&shape, 1}, columns, columns) +
                             (1U << 20));
    bool warm_store_enqueued = false;
    if (detail::gguf_project(input_tensor, {&product, 1}, workspace, stream,
                             &warm_store_enqueued) || !warm_store_enqueued) {
        throw std::runtime_error("E19 accumulate warmup did not select its store route");
    }
    check(cudaMemcpyAsync(actual.p, initial.data(), actual.bytes, cudaMemcpyHostToDevice, stream),
          "reset warmed graph output residual");
    check(cudaStreamSynchronize(stream), "synchronize E19 accumulate warmup");
    bool store_enqueued = false;
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin MMQ accumulate capture");
    const bool pair_enqueued = detail::gguf_project(input_tensor, {&product, 1}, workspace, stream,
                                                     &store_enqueued);
    cudaGraph_t graph = nullptr;
    check(cudaStreamEndCapture(stream, &graph), "end MMQ accumulate capture");
    if (pair_enqueued || !store_enqueued) {
        throw std::runtime_error("E19 accumulate graph did not select its store route");
    }
    cudaGraphExec_t graph_exec = nullptr;
    check(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0), "instantiate MMQ accumulate graph");

    const std::size_t total = initial.size();
    std::vector<__nv_bfloat16> actual_host(total);
    std::vector<float> fp32_host(std::size_t(c.rows) * columns);
    fp32.copy_to_host(fp32_host.data(), fp32.bytes);
    for (int replay = 0; replay < 2; ++replay) {
        check(cudaMemcpyAsync(actual.p, initial.data(), actual.bytes, cudaMemcpyHostToDevice, stream),
              "reset graph output residual");
        check(cudaGraphLaunch(graph_exec, stream), "launch MMQ accumulate graph");
        check(cudaStreamSynchronize(stream), "synchronize MMQ accumulate graph");
        actual.copy_to_host(actual_host.data(), actual.bytes);
        for (int column = 0; column < columns; ++column) {
            for (int row = 0; row < stride; ++row) {
                const auto at = std::size_t(column) * stride + row;
                __nv_bfloat16 expected = initial[at];
                if (row < c.rows) {
                    expected = __float2bfloat16(fp32_host[std::size_t(column) * c.rows + row] +
                                               __bfloat162float(initial[at]));
                }
                std::uint16_t expected_bits = 0;
                std::uint16_t actual_bits = 0;
                std::memcpy(&expected_bits, &expected, sizeof(expected_bits));
                std::memcpy(&actual_bits, &actual_host[at], sizeof(actual_bits));
                if (expected_bits != actual_bits) {
                    cudaGraphExecDestroy(graph_exec);
                    cudaGraphDestroy(graph);
                    cudaStreamDestroy(stream);
                    throw std::runtime_error("E19 accumulate graph changed oracle output or guard rows");
                }
            }
        }
    }
    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    return true;
}

std::vector<std::uint8_t> read_fixture_prefix(const std::string& path,
                                             std::size_t expected_file_bytes,
                                             std::size_t prefix_bytes) {
    if (std::filesystem::file_size(path) != expected_file_bytes || prefix_bytes > expected_file_bytes) {
        throw std::runtime_error("pinned Q6 fixture has an unexpected size: " + path);
    }
    std::ifstream input(path, std::ios::binary);
    if (!input) { throw std::runtime_error("pinned Q6 fixture is missing: " + path); }
    std::vector<std::uint8_t> bytes(prefix_bytes);
    input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (input.gcount() != static_cast<std::streamsize>(bytes.size())) {
        throw std::runtime_error("pinned Q6 fixture prefix is truncated: " + path);
    }
    return bytes;
}

void write_dump(std::ofstream& output, const void* data, std::size_t bytes) {
    output.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    if (!output) { throw std::runtime_error("failed writing Q6 runtime output dump"); }
}

double q6_runtime_weight_value(const std::uint8_t* row, int index) {
    const auto* block = row + std::size_t(index / 256) * 210;
    const int part = (index % 256) / 128;
    const int slot = index % 128;
    const int lane = slot % 32;
    const int group = slot / 32;
    const int low_byte = block[64 * part + lane + 32 * (group & 1)];
    const int low = group < 2 ? (low_byte & 15) : (low_byte >> 4);
    const int high = (block[128 + 32 * part + lane] >> (2 * group)) & 3;
    const int scale_bits = block[192 + 8 * part + lane / 16 + 2 * group];
    const int scale = scale_bits < 128 ? scale_bits : scale_bits - 256;
    __half d;
    std::memcpy(&d, block + 208, sizeof(d));
    return double(__half2float(d)) * double(scale) * double((low | (high << 4)) - 32);
}

int check_q6_runtime_oracle(const Case& c, const std::vector<std::uint8_t>& host_weight,
                            const std::vector<__nv_bfloat16>& host_input,
                            const DeviceBuffer& activation, const std::vector<float>& actual,
                            int columns) {
    if (!std::all_of(actual.begin(), actual.end(),
                     [](float value) { return std::isfinite(value); })) {
        throw std::runtime_error("Q6 runtime gate produced non-finite FP32 output");
    }
    struct Q8D4Block {
        float scales[4];
        std::int8_t quants[128];
    };
    static_assert(sizeof(Q8D4Block) == 144);
    std::vector<Q8D4Block> packed(std::size_t(c.k / 128) * columns);
    activation.copy_to_host(packed.data(), packed.size() * sizeof(Q8D4Block));
    const std::array<int, 8> rows{0, 1, 127, 128, c.rows / 3, c.rows / 2, c.rows - 2,
                                c.rows - 1};
    const std::array<int, 8> cols{0, 1, 127, 128, columns / 3, columns / 2, columns - 2,
                                columns - 1};
    constexpr double epsilon = 8.0 * std::numeric_limits<float>::epsilon();
    for (const int column : cols) {
        for (int start = 0; start < c.k; start += 32) {
            double maximum = 0.0;
            for (int i = 0; i < 32; ++i) {
                maximum = std::max(maximum, std::abs(double(__bfloat162float(
                    host_input[std::size_t(column) * c.k + start + i]))));
            }
            const auto& block = packed[std::size_t(start / 128) * columns + column];
            const double scale = block.scales[(start % 128) / 32];
            const double expected_scale = maximum / 127.0;
            if (!std::isfinite(scale) || scale < 0.0 ||
                std::abs(scale - expected_scale) > epsilon * expected_scale) {
                throw std::runtime_error("Q6 runtime Q8 D4 producer scale differs from BF16 input");
            }
            for (int i = 0; i < 32; ++i) {
                const double input = __bfloat162float(
                    host_input[std::size_t(column) * c.k + start + i]);
                const int quant = block.quants[start % 128 + i];
                if (quant < -127 || std::abs(scale * quant - input) >
                                       0.5 * scale + epsilon * maximum) {
                    throw std::runtime_error("Q6 runtime Q8 D4 producer value differs from BF16 input");
                }
            }
        }
        for (const int row : rows) {
            const auto* weight_row = host_weight.data() + std::size_t(row) * (c.k / 256) * 210;
            double expected = 0.0;
            for (int i = 0; i < c.k; ++i) {
                const auto& block = packed[std::size_t(i / 128) * columns + column];
                const double input = double(block.scales[(i % 128) / 32]) *
                                     double(block.quants[i % 128]);
                expected += q6_runtime_weight_value(weight_row, i) * input;
            }
            const double got = actual[std::size_t(column) * c.rows + row];
            if (!std::isfinite(expected) ||
                std::abs(got - expected) > 0.02 + 2.0e-4 * std::abs(expected)) {
                throw std::runtime_error("Q6 runtime output differs from independent FP64 oracle");
            }
        }
    }
    return int(rows.size() * cols.size());
}

void run_q6_store_runtime_case(const Case& c, const std::vector<std::uint8_t>& host_weight,
                               int columns, bool accumulate, bool expect_store,
                               std::ofstream& dump) {
    DeviceBuffer weight(host_weight.size());
    weight.copy_from_host(host_weight.data(), host_weight.size());
    const auto host_input = make_input(c.k, columns, 0xA119u + columns + c.rows);
    DeviceBuffer input(host_input.size() * sizeof(__nv_bfloat16));
    input.copy_from_host(host_input.data(), input.bytes);
    const std::size_t activation_bytes = gguf::matrix_activation_bytes(c.k, columns);
    DeviceBuffer activation(activation_bytes);
    gguf::quantize_matrix_activation(c.type, static_cast<const __nv_bfloat16*>(input.p), c.k,
                                     columns, nullptr, activation.p, nullptr);
    DeviceBuffer fp32(std::size_t(c.rows) * columns * sizeof(float));
    fp32.fill(0xFF);
    DeviceBuffer fixup(gguf::matrix_fixup_bytes(c.type, c.rows, columns));
    const std::int64_t row_bytes = std::int64_t(c.k / 256) * gguf::block_shape(c.type).bytes;
    gguf::matrix_product(c.type, weight.p, row_bytes, c.rows, c.k, activation.p, columns,
                         static_cast<float*>(fp32.p), c.rows, fixup.p, nullptr);

    const int stride = c.rows + 19;
    std::vector<__nv_bfloat16> initial(std::size_t(stride) * columns);
    std::mt19937 rng(0xA66u + columns + c.rows + (accumulate ? 1u : 0u));
    std::uniform_real_distribution<float> residual(-0.5f, 0.5f);
    for (int column = 0; column < columns; ++column) {
        for (int row = 0; row < stride; ++row) {
            initial[std::size_t(column) * stride + row] = row < c.rows
                ? __float2bfloat16(accumulate ? residual(rng) : -777.0f)
                : __float2bfloat16(123.0f);
        }
    }
    DeviceBuffer actual(initial.size() * sizeof(__nv_bfloat16));
    actual.copy_from_host(initial.data(), actual.bytes);
    Tensor input_tensor(input.p, DType::BF16, {c.k, columns});
    Tensor output_tensor(actual.p, DType::BF16, {stride, columns});
    Weight product_weight = make_gguf_weight(c, weight.p);
    GgufProduct product{&product_weight, &output_tensor, 0,
                        accumulate ? GgufEpilogue::Accumulate : GgufEpilogue::Store};
    const GgufShape shape{c.qtype, c.rows, c.k};
    WorkspaceArena workspace(detail::gguf_project_workspace_bytes({&shape, 1}, columns, columns) +
                             (1U << 20));
    bool store_enqueued = false;
    if (detail::gguf_project(input_tensor, {&product, 1}, workspace, nullptr,
                             &store_enqueued) || store_enqueued != expect_store) {
        throw std::runtime_error("Q6 runtime gate selected an unexpected MMQ BF16 store route");
    }
    check(cudaDeviceSynchronize(), "synchronize real Q6 store runtime gate");
    std::vector<__nv_bfloat16> host_actual(initial.size());
    std::vector<float> host_fp32(std::size_t(c.rows) * columns);
    actual.copy_to_host(host_actual.data(), actual.bytes);
    fp32.copy_to_host(host_fp32.data(), fp32.bytes);
    const int oracle_samples = check_q6_runtime_oracle(
        c, host_weight, host_input, activation, host_fp32, columns);
    if (!accumulate) { write_dump(dump, host_fp32.data(), fp32.bytes); }
    for (int column = 0; column < columns; ++column) {
        for (int row = 0; row < stride; ++row) {
            const std::size_t at = std::size_t(column) * stride + row;
            const __nv_bfloat16 expected = row < c.rows
                ? __float2bfloat16(host_fp32[std::size_t(column) * c.rows + row] +
                                   (accumulate ? __bfloat162float(initial[at]) : 0.0f))
                : initial[at];
            std::uint16_t expected_bits = 0;
            std::uint16_t actual_bits = 0;
            std::memcpy(&expected_bits, &expected, sizeof(expected_bits));
            std::memcpy(&actual_bits, &host_actual[at], sizeof(actual_bits));
            if (!std::isfinite(__bfloat162float(host_actual[at])) || expected_bits != actual_bits) {
                throw std::runtime_error("Q6 runtime gate changed the FP32/BF16 result or guard rows");
            }
        }
    }
    write_dump(dump, host_actual.data(), host_actual.size() * sizeof(__nv_bfloat16));
    std::cout << "q6_runtime_gate case=K" << c.k << "_N" << c.rows << "_T" << columns
              << " epilogue=" << (accumulate ? "Accumulate" : "Store")
              << " store_route=" << (store_enqueued ? "FUSED" : "FALLBACK")
              << " fp64_samples=" << oracle_samples << " producer=PASS status=PASS\n";
}

void run_q6_plain_runtime_case(const Case& c, const std::vector<std::uint8_t>& host_weight,
                               int columns, std::ofstream& dump) {
    DeviceBuffer weight(host_weight.size());
    weight.copy_from_host(host_weight.data(), host_weight.size());
    const auto host_input = make_input(c.k, columns, 0xA119u + columns + c.rows);
    DeviceBuffer input(host_input.size() * sizeof(__nv_bfloat16));
    input.copy_from_host(host_input.data(), input.bytes);
    const std::size_t activation_bytes = gguf::matrix_activation_bytes(c.k, columns);
    DeviceBuffer activation(activation_bytes);
    gguf::quantize_matrix_activation(c.type, static_cast<const __nv_bfloat16*>(input.p), c.k,
                                     columns, nullptr, activation.p, nullptr);
    constexpr std::size_t guard_bytes = 64;
    const std::size_t output_bytes = std::size_t(c.rows) * columns * sizeof(float);
    DeviceBuffer actual(output_bytes + 2 * guard_bytes);
    actual.fill(0xA5);
    DeviceBuffer fixup(gguf::matrix_fixup_bytes(c.type, c.rows, columns));
    const std::int64_t row_bytes = std::int64_t(c.k / 256) * gguf::block_shape(c.type).bytes;
    auto* output = reinterpret_cast<float*>(static_cast<std::uint8_t*>(actual.p) + guard_bytes);
    check(cudaMemset(output, 0xFF, output_bytes), "poison plain Q6 runtime output");
    gguf::matrix_product(c.type, weight.p, row_bytes, c.rows, c.k, activation.p, columns,
                         output, c.rows, fixup.p, nullptr);
    check(cudaDeviceSynchronize(), "synchronize plain real Q6 runtime gate");
    std::vector<std::uint8_t> guarded_bytes(actual.bytes);
    actual.copy_to_host(guarded_bytes.data(), actual.bytes);
    if (!std::all_of(guarded_bytes.begin(), guarded_bytes.begin() + guard_bytes,
                     [](std::uint8_t value) { return value == 0xA5; }) ||
        !std::all_of(guarded_bytes.end() - guard_bytes, guarded_bytes.end(),
                     [](std::uint8_t value) { return value == 0xA5; })) {
        throw std::runtime_error("plain Q6 runtime gate modified its output canaries");
    }
    std::vector<float> host_actual(std::size_t(c.rows) * columns);
    std::memcpy(host_actual.data(), guarded_bytes.data() + guard_bytes, output_bytes);
    const int oracle_samples = check_q6_runtime_oracle(
        c, host_weight, host_input, activation, host_actual, columns);
    write_dump(dump, host_actual.data(), host_actual.size() * sizeof(float));
    std::cout << "q6_runtime_gate case=K" << c.k << "_N" << c.rows << "_T" << columns
              << " epilogue=plain_matrix_product fp64_samples=" << oracle_samples
              << " producer=PASS route=PASS\n";
}

void run_q6_runtime_gate(const std::string& operator_fixture_dir,
                         const std::string& weights_fixture_dir,
                         const std::string& output_path) {
    int device = 0;
    check(cudaGetDevice(&device), "get Q6 runtime gate device");
    cudaDeviceProp properties{};
    check(cudaGetDeviceProperties(&properties, device), "Q6 runtime gate device properties");
    if (properties.major != 8 || properties.minor != 6) {
        throw std::runtime_error("Q6 runtime gate requires an SM86 device");
    }
    const auto q6_block = gguf::block_shape(gguf::GgmlType::Q6_K);
    const Case layer{QType::GGUF_Q6_K, gguf::GgmlType::Q6_K, 5120, 17408, 208};
    const std::size_t layer_bytes = std::size_t(layer.rows) *
        (layer.k / q6_block.elements) * q6_block.bytes;
    const auto layer_weight = read_fixture_prefix(operator_fixture_dir + "/q6-full-k17408.bin",
                                                   layer_bytes, layer_bytes);
    const Case head{QType::GGUF_Q6_K, gguf::GgmlType::Q6_K, 10240, 5120, 208};
    const std::size_t full_head_bytes = std::size_t(248320) *
        (head.k / q6_block.elements) * q6_block.bytes;
    const std::size_t head_prefix_bytes = std::size_t(head.rows) *
        (head.k / q6_block.elements) * q6_block.bytes;
    const auto head_weight = read_fixture_prefix(weights_fixture_dir + "/q6-head-full-k5120.bin",
                                                   full_head_bytes, head_prefix_bytes);
    std::ofstream dump(output_path, std::ios::binary | std::ios::trunc);
    if (!dump) { throw std::runtime_error("could not open Q6 runtime output dump: " + output_path); }
#if defined(NINFER_GGUF_Q6_MMQ_A_X4_TEST)
    std::cout << "q6_runtime_gate NINFER_GGUF_Q6_MMQ_A_X4=ON\n";
#else
    std::cout << "q6_runtime_gate NINFER_GGUF_Q6_MMQ_A_X4=OFF\n";
#endif
    for (const int columns : {1280, 1281}) {
        run_q6_store_runtime_case(layer, layer_weight, columns, false, columns == 1280, dump);
        run_q6_store_runtime_case(layer, layer_weight, columns, true, columns == 1280, dump);
        run_q6_plain_runtime_case(head, head_weight, columns, dump);
    }
    run_q6_store_runtime_case(layer, layer_weight, 768, true, true, dump);
    const Case attention{QType::GGUF_Q6_K, gguf::GgmlType::Q6_K, 1024, 5120, 208};
    const std::size_t attention_bytes = std::size_t(attention.rows) *
        (attention.k / q6_block.elements) * q6_block.bytes;
    const std::vector<std::uint8_t> attention_weight(
        head_weight.begin(), head_weight.begin() + attention_bytes);
    for (const int columns : {1280, 768}) {
        run_q6_store_runtime_case(head, head_weight, columns, false, true, dump);
        run_q6_store_runtime_case(attention, attention_weight, columns, false, columns == 1280,
                                  dump);
    }
    dump.flush();
    if (!dump) { throw std::runtime_error("failed flushing Q6 runtime output dump"); }
    std::cout << "q6_runtime_gate fixtures=real_only prefix_rows=10240 status=PASS\n";
}

}

int main(int argc, char** argv) {
    try {
        if (argc == 5 && std::string(argv[1]) == "--q6-runtime-gate") {
            run_q6_runtime_gate(argv[2], argv[3], argv[4]);
            return 0;
        }
        if (argc != 1) {
            throw std::invalid_argument(
                "usage: ninfer_tests ninfer_linear_gguf_mmq_store_test [--q6-runtime-gate <operators-dir> <weights-dir> <output-file>]");
        }
        int device = 0;
        check(cudaGetDevice(&device), "get device");
        cudaDeviceProp properties{};
        check(cudaGetDeviceProperties(&properties, device), "device properties");
        if (properties.major != 8 || properties.minor != 6) { return 77; }
        bool expect_130_fused = false;
#if defined(NINFER_GGUF_SM86_MMQ_MAX_J) && NINFER_GGUF_SM86_MMQ_MAX_J == 32
        expect_130_fused = true;
#endif
        if (!run_case(q4_case(), 1024, false, true) ||
            !run_case(q4_case(), 1024, true, true) ||
            !run_case(q4_case(), 1000, false, true) ||
            !run_case(q6_case(), 1024, false, true) ||
            !run_case(q4_case(), 130, false, expect_130_fused) ||
            !run_accumulate_graph_case()) {
            return 1;
        }
        std::cout << "native MMQ BF16 store exactness and fallback checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
