#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ops/linear/gguf/ggml_bridge.h"
#include "ops/linear/gguf/gguf_linear.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
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

float half_value(const std::uint8_t* bytes) {
    __half value;
    std::memcpy(&value, bytes, sizeof(value));
    return __half2float(value);
}

double q4_k_q8_1_dot(const std::uint8_t* row_data, int k, int columns, int column,
                     const std::vector<std::uint8_t>& input_bytes) {
    const std::size_t activation_scales = (std::size_t(k) * columns + 15) / 16 * 16;
    double result = 0.0;
    for (int group = 0; group < k / 32; ++group) {
        const int block_index = group / 8;
        const int subblock = group % 8;
        const std::uint8_t* quant_block = row_data + std::size_t(block_index) * 144;
        const std::uint8_t* scales = quant_block + 4;
        int scale;
        int minimum;
        if (subblock < 4) {
            scale = scales[subblock] & 63;
            minimum = scales[subblock + 4] & 63;
        } else {
            scale = (scales[subblock + 4] & 0x0f) | ((scales[subblock - 4] >> 6) << 4);
            minimum = (scales[subblock + 4] >> 4) | ((scales[subblock] >> 6) << 4);
        }
        const std::uint8_t* quants = quant_block + 16 + (subblock / 2) * 32;
        const int shift = 4 * (subblock & 1);
        std::int64_t integer_dot = 0;
        for (int lane = 0; lane < 32; ++lane) {
            const int weight_q = (quants[lane] >> shift) & 0x0f;
            const std::uint8_t activation_byte =
                input_bytes[std::size_t(column) * k + group * 32 + lane];
            const int activation_q = activation_byte < 128 ? activation_byte : int(activation_byte) - 256;
            integer_dot += std::int64_t(weight_q) * activation_q;
        }
        const std::size_t scale_at = activation_scales +
            (std::size_t(column) * (k / 32) + group) * sizeof(__half2);
        const double activation_d = half_value(input_bytes.data() + scale_at);
        const double activation_sum = half_value(input_bytes.data() + scale_at + sizeof(__half));
        const double weight_d = half_value(quant_block);
        const double weight_dmin = half_value(quant_block + 2);
        result += (weight_d * scale) * activation_d * double(integer_dot) -
                  (weight_dmin * minimum) * activation_sum;
    }
    return result;
}

bool run_child_cta_public_case(int rows, int columns, std::ofstream* output) {
    const Case c{QType::GGUF_Q4_K, gguf::GgmlType::Q4_K, rows, 5120, 0};
    const auto host_weight = make_weight(c, 0xCC1800u + rows + columns);
    DeviceBuffer weight(host_weight.size());
    weight.copy_from_host(host_weight.data(), host_weight.size());
    const auto host_input = make_input(c.k, columns, 0xCC1900u + rows + columns);
    DeviceBuffer input(host_input.size() * sizeof(__nv_bfloat16));
    input.copy_from_host(host_input.data(), input.bytes);
    DeviceBuffer activation(gguf::matrix_activation_bytes(c.k, columns));
    gguf::quantize_matrix_activation(c.type, static_cast<const __nv_bfloat16*>(input.p), c.k,
                                     columns, nullptr, activation.p, nullptr);
    const std::size_t output_bytes = std::size_t(rows) * columns * sizeof(float);
    DeviceBuffer eager(output_bytes);
    DeviceBuffer graph_output(output_bytes);
    DeviceBuffer fixup(gguf::matrix_fixup_bytes(c.type, rows, columns));
    const std::int64_t row_bytes = std::int64_t(c.k / 256) * gguf::block_shape(c.type).bytes;
    gguf::matrix_product(c.type, weight.p, row_bytes, rows, c.k, activation.p, columns,
                         static_cast<float*>(eager.p), rows, fixup.p, nullptr);
    check(cudaDeviceSynchronize(), "child-CTA public route eager sync");
    cudaStream_t stream = nullptr;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "child-CTA graph stream");
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "child-CTA begin graph capture");
    gguf::matrix_product(c.type, weight.p, row_bytes, rows, c.k, activation.p, columns,
                         static_cast<float*>(graph_output.p), rows, fixup.p, stream);
    cudaGraph_t graph = nullptr;
    check(cudaStreamEndCapture(stream, &graph), "child-CTA end graph capture");
    cudaGraphExec_t graph_exec = nullptr;
    check(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0), "child-CTA graph instantiate");
    check(cudaGraphLaunch(graph_exec, stream), "child-CTA graph launch");
    check(cudaStreamSynchronize(stream), "child-CTA graph sync");
    std::vector<float> eager_host(std::size_t(rows) * columns);
    std::vector<float> graph_host(eager_host.size());
    eager.copy_to_host(eager_host.data(), eager.bytes);
    graph_output.copy_to_host(graph_host.data(), graph_output.bytes);
    for (std::size_t i = 0; i < eager_host.size(); ++i) {
        if (std::memcmp(&eager_host[i], &graph_host[i], sizeof(float)) != 0) {
            cudaGraphExecDestroy(graph_exec);
            cudaGraphDestroy(graph);
            cudaStreamDestroy(stream);
            throw std::runtime_error("child-CTA public route eager/graph output mismatch");
        }
    }
    std::vector<std::uint8_t> activation_host(activation.bytes);
    activation.copy_to_host(activation_host.data(), activation.bytes);
    const int sample_rows[] = {0, rows / 2, rows - 1};
    const int sample_columns[] = {0, columns / 2, columns - 1};
    for (const int column : sample_columns) {
        for (const int row : sample_rows) {
            const double expected = q4_k_q8_1_dot(
                host_weight.data() + std::size_t(row) * row_bytes, c.k, columns, column,
                activation_host);
            const double actual = eager_host[std::size_t(column) * rows + row];
            const double relative = std::abs(actual - expected) / std::max(1.0, std::abs(expected));
            if (relative > 5e-2) {
                cudaGraphExecDestroy(graph_exec);
                cudaGraphDestroy(graph);
                cudaStreamDestroy(stream);
                throw std::runtime_error("child-CTA public route differs from independent Q8_1 oracle");
            }
        }
    }
    gguf::MatrixBf16StorePlan store_plan;
    if (gguf::matrix_bf16_store_plan(c.type, rows, columns, store_plan)) {
        DeviceBuffer bf16(std::size_t(rows) * columns * sizeof(__nv_bfloat16));
        if (!gguf::matrix_product_bf16_store(c.type, weight.p, row_bytes, rows, c.k,
                                              activation.p, columns,
                                              static_cast<__nv_bfloat16*>(bf16.p), rows, nullptr)) {
            cudaGraphExecDestroy(graph_exec);
            cudaGraphDestroy(graph);
            cudaStreamDestroy(stream);
            throw std::runtime_error("child-CTA public BF16 store route was not enqueued");
        }
        std::vector<__nv_bfloat16> bf16_host(std::size_t(rows) * columns);
        bf16.copy_to_host(bf16_host.data(), bf16.bytes);
        if (!compare_rounding(eager_host, bf16_host, rows, columns)) {
            cudaGraphExecDestroy(graph_exec);
            cudaGraphDestroy(graph);
            cudaStreamDestroy(stream);
            throw std::runtime_error("child-CTA public BF16 store differs from FP32 accumulator rounding");
        }
        std::vector<__nv_bfloat16> residual_host(std::size_t(rows) * columns);
        for (std::size_t i = 0; i < residual_host.size(); ++i) {
            residual_host[i] = __float2bfloat16(float(int(i % 17) - 8) * 0.03125f);
        }
        check(cudaMemcpy(bf16.p, residual_host.data(), bf16.bytes, cudaMemcpyHostToDevice),
              "initialize child-CTA accumulate residual");
        if (!gguf::matrix_product_bf16_store(c.type, weight.p, row_bytes, rows, c.k,
                                              activation.p, columns,
                                              static_cast<__nv_bfloat16*>(bf16.p), rows, nullptr, true)) {
            cudaGraphExecDestroy(graph_exec);
            cudaGraphDestroy(graph);
            cudaStreamDestroy(stream);
            throw std::runtime_error("child-CTA public BF16 accumulate route was not enqueued");
        }
        bf16.copy_to_host(bf16_host.data(), bf16.bytes);
        for (std::size_t i = 0; i < bf16_host.size(); ++i) {
            const __nv_bfloat16 expected = __float2bfloat16(
                eager_host[i] + __bfloat162float(residual_host[i]));
            std::uint16_t expected_bits = 0;
            std::uint16_t actual_bits = 0;
            std::memcpy(&expected_bits, &expected, sizeof(expected_bits));
            std::memcpy(&actual_bits, &bf16_host[i], sizeof(actual_bits));
            if (expected_bits != actual_bits) {
                cudaGraphExecDestroy(graph_exec);
                cudaGraphDestroy(graph);
                cudaStreamDestroy(stream);
                throw std::runtime_error("child-CTA public BF16 accumulate changed exact output");
            }
        }
    }
    gguf::MatrixBf16StorePlan gate_plan;
    if (gguf::matrix_bf16_gate_store_plan(c.type, rows, columns, gate_plan)) {
        std::vector<float> gate_host(std::size_t(rows) * columns);
        for (std::size_t i = 0; i < gate_host.size(); ++i) {
            gate_host[i] = float(int(i % 23) - 11) * 0.0625f;
        }
        DeviceBuffer gate(gate_host.size() * sizeof(float));
        check(cudaMemcpy(gate.p, gate_host.data(), gate.bytes, cudaMemcpyHostToDevice),
              "copy child-CTA gate plane");
        DeviceBuffer bf16(std::size_t(rows) * columns * sizeof(__nv_bfloat16));
        if (!gguf::matrix_product_bf16_gate_store(c.type, weight.p, row_bytes, rows, c.k,
                                                   activation.p, columns,
                                                   static_cast<__nv_bfloat16*>(bf16.p), rows,
                                                   static_cast<const float*>(gate.p), rows, nullptr)) {
            cudaGraphExecDestroy(graph_exec);
            cudaGraphDestroy(graph);
            cudaStreamDestroy(stream);
            throw std::runtime_error("child-CTA public BF16 GateProduct route was not enqueued");
        }
        std::vector<__nv_bfloat16> bf16_host(std::size_t(rows) * columns);
        bf16.copy_to_host(bf16_host.data(), bf16.bytes);
        for (std::size_t i = 0; i < bf16_host.size(); ++i) {
            const float gate_value = gate_host[i];
            float expected_value = eager_host[i];
            expected_value *= gate_value / (1.0f + std::exp(-gate_value));
            const __nv_bfloat16 expected = __float2bfloat16(expected_value);
            std::uint16_t expected_bits = 0;
            std::uint16_t actual_bits = 0;
            std::memcpy(&expected_bits, &expected, sizeof(expected_bits));
            std::memcpy(&actual_bits, &bf16_host[i], sizeof(actual_bits));
            if (expected_bits != actual_bits) {
                cudaGraphExecDestroy(graph_exec);
                cudaGraphDestroy(graph);
                cudaStreamDestroy(stream);
                throw std::runtime_error("child-CTA public BF16 GateProduct changed exact output");
            }
        }
    }
    if (output != nullptr) {
        const std::uint32_t header[] = {static_cast<std::uint32_t>(rows),
                                        static_cast<std::uint32_t>(columns)};
        output->write(reinterpret_cast<const char*>(header), sizeof(header));
        output->write(reinterpret_cast<const char*>(eager_host.data()),
                      static_cast<std::streamsize>(eager_host.size() * sizeof(float)));
        if (!*output) {
            cudaGraphExecDestroy(graph_exec);
            cudaGraphDestroy(graph);
            cudaStreamDestroy(stream);
            throw std::runtime_error("child-CTA public route output dump failed");
        }
    }
    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    return true;
}

}

int main(int argc, char** argv) {
    try {
        bool child_cta_regression = false;
        std::string output_path;
        for (int i = 1; i < argc; ++i) {
            const std::string argument = argv[i];
            if (argument == "--q4-mmq-child-cta-regression" && !child_cta_regression) {
                child_cta_regression = true;
            } else if (argument == "--output" && i + 1 < argc && output_path.empty()) {
                output_path = argv[++i];
            } else {
                throw std::invalid_argument("unknown MMQ store test argument: " + argument);
            }
        }
        int device = 0;
        check(cudaGetDevice(&device), "get device");
        cudaDeviceProp properties{};
        check(cudaGetDeviceProperties(&properties, device), "device properties");
        if (properties.major != 8 || properties.minor != 6) { return 77; }
        if (child_cta_regression) {
            std::ofstream output_file;
            std::ofstream* output = nullptr;
            if (!output_path.empty()) {
                if (std::filesystem::exists(output_path)) {
                    throw std::runtime_error("Q4 MMQ child-CTA output already exists: " + output_path);
                }
                output_file.open(output_path, std::ios::binary | std::ios::trunc);
                if (!output_file) { throw std::runtime_error("could not open child-CTA route dump"); }
                output = &output_file;
            }
            for (const int rows : {17408, 384}) {
                for (const int columns : {1280, 1281}) {
                    run_child_cta_public_case(rows, columns, output);
                }
            }
            if (output != nullptr) {
                output_file.flush();
                if (!output_file) { throw std::runtime_error("could not flush child-CTA route dump"); }
            }
            std::cout << "OK Q4 MMQ public bridge eager/graph, Q8_1 oracle, and exact output dump route\n";
            return 0;
        }
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
