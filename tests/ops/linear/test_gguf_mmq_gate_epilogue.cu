#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ops/linear/gguf/ggml_bridge.h"
#include "ops/linear/gguf/gguf_linear.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
namespace gguf = ninfer::ops::gguf;
namespace detail = ninfer::ops::detail;
using detail::GgufEpilogue;
using detail::GgufProduct;
using detail::GgufShape;

struct PackedQ8MmqDs4 {
    __half2 ds4[4];
    std::int8_t qs[128];
};

enum class MatrixActivationDsLayout : int { D4, DS4, D2S6 };

static_assert(offsetof(PackedQ8MmqDs4, qs) == sizeof(__half2) * 4);
static_assert(sizeof(PackedQ8MmqDs4) == 144);

void check(cudaError_t status, const char* label) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(label) + ": " + cudaGetErrorString(status));
    }
}

float half_value(const std::uint8_t* bytes) {
    std::uint16_t bits = 0;
    std::memcpy(&bits, bytes, sizeof(bits));
    __half value;
    std::memcpy(&value, &bits, sizeof(bits));
    return __half2float(value);
}

std::vector<std::uint8_t> make_q4_weight(int rows, int k, std::uint32_t seed) {
    const auto block = gguf::block_shape(gguf::GgmlType::Q4_K);
    std::vector<std::uint8_t> result(std::size_t(rows) * (k / block.elements) * block.bytes);
    std::mt19937 random(seed);
    std::uniform_int_distribution<int> byte(0, 255);
    for (auto& value : result) { value = static_cast<std::uint8_t>(byte(random)); }
    const std::uint16_t scale = 0x3980;
    const std::size_t block_count = result.size() / block.bytes;
    for (std::size_t index = 0; index < block_count; ++index) {
        auto* data = result.data() + index * block.bytes;
        std::memcpy(data, &scale, sizeof(scale));
        std::memcpy(data + 2, &scale, sizeof(scale));
    }
    return result;
}

std::vector<__nv_bfloat16> make_input(int k, int columns, std::uint32_t seed) {
    std::mt19937 random(seed);
    std::uniform_real_distribution<float> values(-0.45f, 0.45f);
    std::vector<__nv_bfloat16> result(std::size_t(k) * columns);
    for (auto& value : result) { value = __float2bfloat16(values(random)); }
    return result;
}

std::vector<float> make_gate(int rows, int columns, std::uint32_t seed) {
    std::mt19937 random(seed);
    std::uniform_real_distribution<float> values(-8.0f, 8.0f);
    std::vector<float> result(std::size_t(rows) * columns);
    for (auto& value : result) { value = values(random); }
    if (rows >= 7) {
        result[0] = 0.0f;
        result[1] = std::copysign(0.0f, -1.0f);
        result[2] = 1000.0f;
        result[3] = -1000.0f;
        result[4] = std::numeric_limits<float>::quiet_NaN();
        result[5] = std::numeric_limits<float>::infinity();
        result[6] = -std::numeric_limits<float>::infinity();
    }
    return result;
}

Weight make_weight(int rows, int k, const void* data) {
    Weight result;
    result.qtype = QType::GGUF_Q4_K;
    result.layout = QuantLayout::GgufBlocks;
    result.qdata = data;
    result.payload = data;
    result.n = result.shape[0] = result.padded_shape[0] = rows;
    result.k = result.shape[1] = result.padded_shape[1] = k;
    result.ndim = 2;
    result.group_size = 256;
    result.group = 256;
    return result;
}

bool same_bf16_output(const __nv_bfloat16* expected, const __nv_bfloat16* actual,
                      std::size_t elements) {
    for (std::size_t i = 0; i < elements; ++i) {
        const float expected_value = __bfloat162float(expected[i]);
        const float actual_value = __bfloat162float(actual[i]);
        if (std::isnan(expected_value) && std::isnan(actual_value)) { continue; }
        std::uint16_t expected_bits = 0;
        std::uint16_t actual_bits = 0;
        std::memcpy(&expected_bits, &expected[i], sizeof(expected_bits));
        std::memcpy(&actual_bits, &actual[i], sizeof(actual_bits));
        if (expected_bits != actual_bits) { return false; }
    }
    return true;
}

void decode_reference_rows(const std::vector<std::uint8_t>& blocks,
                           const std::vector<std::uint8_t>& packed_activation,
                           int rows, int k, int columns,
                           const std::vector<float>& gate, std::vector<float>& reference) {
    const auto shape = gguf::block_shape(gguf::GgmlType::Q4_K);
    const int blocks_per_row = k / shape.elements;
    const int selected_rows[] = {0, 1, 2, 3, 4, 5, 6, 7, rows / 3, rows / 2, rows - 1};
    const int selected_columns[] = {0, columns / 2, columns - 1};
    for (int row : selected_rows) {
        for (int column : selected_columns) {
            double dot = 0.0;
            for (int block_index = 0; block_index < blocks_per_row; ++block_index) {
                const auto* block = blocks.data() +
                    (std::size_t(row) * blocks_per_row + block_index) * shape.bytes;
                const double d = half_value(block);
                const double dmin = half_value(block + 2);
                const auto* scales = block + 4;
                const auto* quants = block + 16;
                for (int subblock = 0; subblock < 8; ++subblock) {
                    int scale;
                    int minimum;
                    if (subblock < 4) {
                        scale = scales[subblock] & 63;
                        minimum = scales[subblock + 4] & 63;
                    } else {
                        scale = (scales[subblock + 4] & 15) |
                                ((scales[subblock - 4] >> 6) << 4);
                        minimum = (scales[subblock + 4] >> 4) |
                                   ((scales[subblock] >> 6) << 4);
                    }
                    const auto* packed = quants + (subblock / 2) * 32;
                    const int shift = 4 * (subblock & 1);
                    const int activation_block = block_index * 2 + subblock / 4;
                    const auto* q8 = packed_activation.data() +
                        (std::size_t(activation_block) * columns + column) *
                            sizeof(PackedQ8MmqDs4);
                    const int q8_subblock = subblock % 4;
                    const std::size_t ds_offset =
                        offsetof(PackedQ8MmqDs4, ds4) + q8_subblock * sizeof(__half2);
                    const double activation_d = half_value(q8 + ds_offset);
                    const double activation_sum = half_value(q8 + ds_offset + sizeof(__half));
                    const auto* q8_codes = q8 + offsetof(PackedQ8MmqDs4, qs);
                    int dot_q4_q8 = 0;
                    for (int index = 0; index < 32; ++index) {
                        const int quant = (packed[index] >> shift) & 15;
                        std::int8_t activation_quant = 0;
                        std::memcpy(&activation_quant, q8_codes + q8_subblock * 32 + index,
                                    sizeof(activation_quant));
                        dot_q4_q8 += quant * activation_quant;
                    }
                    dot += double(dot_q4_q8) * d * scale * activation_d -
                           dmin * minimum * activation_sum;
                }
            }
            const double gate_value = gate[std::size_t(column) * rows + row];
            const double silu = gate_value / (1.0 + std::exp(-gate_value));
            reference.push_back(static_cast<float>(dot * silu));
        }
    }
}

bool compare_case(int rows, int columns, bool prepared, bool graph, int row_offset,
                  bool expect_fused, std::uint32_t seed) {
    constexpr int k = 5120;
    const int output_stride = rows + 13;
    const auto host_weight = make_q4_weight(rows, k, seed + 1);
    const auto host_input = make_input(k, columns, seed + 2);
    const auto host_gate = make_gate(rows, columns, seed + 3);
    DeviceBuffer weight_device(host_weight.size());
    DeviceBuffer input_device(host_input.size() * sizeof(__nv_bfloat16));
    DeviceBuffer gate_device(host_gate.size() * sizeof(float));
    DeviceBuffer activation_device(gguf::matrix_activation_bytes(k, columns));
    DeviceBuffer native_device(std::size_t(rows) * columns * sizeof(float));
    DeviceBuffer fixup_device(gguf::matrix_fixup_bytes(gguf::GgmlType::Q4_K, rows, columns));
    weight_device.copy_from_host(host_weight.data(), weight_device.bytes);
    input_device.copy_from_host(host_input.data(), input_device.bytes);
    gate_device.copy_from_host(host_gate.data(), gate_device.bytes);
    gguf::quantize_matrix_activation(
        gguf::GgmlType::Q4_K, static_cast<const __nv_bfloat16*>(input_device.p), k, columns,
        nullptr, activation_device.p, nullptr);
    if (gguf::matrix_activation_layout(gguf::GgmlType::Q4_K) !=
        static_cast<int>(MatrixActivationDsLayout::DS4)) {
        throw std::runtime_error("Q4_K oracle requires the source-defined Q8_1 DS4 activation layout");
    }
    check(cudaDeviceSynchronize(), "synchronize packed Q8_1 DS4 activation");
    std::vector<std::uint8_t> host_packed_activation(activation_device.bytes);
    activation_device.copy_to_host(host_packed_activation.data(), activation_device.bytes);
    gguf::MatrixBf16StorePlan plan;
    const bool plan_supported =
        gguf::matrix_bf16_gate_store_plan(gguf::GgmlType::Q4_K, rows, columns, plan);
    if (plan_supported != expect_fused) {
        throw std::runtime_error("gate epilogue plan admitted the wrong MMQ geometry");
    }
    if (expect_fused && (plan.I != 128 || plan.J <= 0)) {
        throw std::runtime_error("gate epilogue selected an unexpected MMQ tile");
    }
    gguf::matrix_product(gguf::GgmlType::Q4_K, weight_device.p,
                         std::int64_t(k / 256) * gguf::block_shape(gguf::GgmlType::Q4_K).bytes,
                         rows, k, activation_device.p, columns,
                         static_cast<float*>(native_device.p), rows, fixup_device.p, nullptr);
    DeviceBuffer expected_device(std::size_t(output_stride) * columns * sizeof(__nv_bfloat16));
    DeviceBuffer actual_device(std::size_t(output_stride) * columns * sizeof(__nv_bfloat16));
    std::vector<__nv_bfloat16> canaries(std::size_t(output_stride) * columns,
                                        __float2bfloat16(123.25f));
    expected_device.copy_from_host(canaries.data(), expected_device.bytes);
    actual_device.copy_from_host(canaries.data(), actual_device.bytes);
    gguf::VectorOutput expected_output;
    expected_output.bf16 = static_cast<__nv_bfloat16*>(expected_device.p) + row_offset;
    expected_output.column_stride = output_stride;
    expected_output.epilogue = gguf::Epilogue::GateProduct;
    expected_output.gate = static_cast<const float*>(gate_device.p);
    expected_output.gate_column_stride = rows;
    gguf::store_plane(static_cast<const float*>(native_device.p), rows, rows, columns,
                      expected_output, nullptr);
    check(cudaDeviceSynchronize(), "synchronize native gate reference");
    std::vector<__nv_bfloat16> expected(canaries.size());
    std::vector<__nv_bfloat16> actual(canaries.size());
    expected_device.copy_to_host(expected.data(), expected_device.bytes);

    Tensor input_tensor(input_device.p, DType::BF16, {k, columns});
    Tensor output_tensor(actual_device.p, DType::BF16, {output_stride, columns});
    Weight product_weight = make_weight(rows, k, weight_device.p);
    GgufProduct product{&product_weight, &output_tensor, row_offset, GgufEpilogue::GateProduct,
                        static_cast<const float*>(gate_device.p)};
    const GgufShape shape{QType::GGUF_Q4_K, rows, k};
    const std::size_t workspace_bytes =
        detail::gguf_project_workspace_bytes({&shape, 1}, columns, columns);
    WorkspaceArena workspace(workspace_bytes);
    bool used_gate = false;
    cudaStream_t stream = nullptr;
    if (graph) {
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create gate graph stream");
    }
    auto invoke = [&] {
        if (prepared) {
            const gguf::MatrixActivationPlane activation{gguf::GgmlType::Q4_K,
                                                          activation_device.p};
            return detail::gguf_project_matrix_q8_1_activations(
                k, columns, {&activation, 1}, {&product, 1}, workspace, stream, nullptr,
                &used_gate);
        }
        return detail::gguf_project(input_tensor, {&product, 1}, workspace, stream, nullptr,
                                    nullptr, &used_gate);
    };
    if (graph) {
        invoke();
        check(cudaStreamSynchronize(stream), "warm gate graph route");
        actual_device.copy_from_host(canaries.data(), actual_device.bytes);
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
              "capture gate projection");
        invoke();
        cudaGraph_t captured = nullptr;
        check(cudaStreamEndCapture(stream, &captured), "finish gate projection capture");
        cudaGraphExec_t executable = nullptr;
        check(cudaGraphInstantiate(&executable, captured, nullptr, nullptr, 0),
              "instantiate gate projection graph");
        check(cudaGraphLaunch(executable, stream), "launch gate projection graph");
        check(cudaStreamSynchronize(stream), "synchronize gate projection graph");
        std::vector<__nv_bfloat16> graph_output(canaries.size());
        actual_device.copy_to_host(graph_output.data(), actual_device.bytes);
        if (!same_bf16_output(expected.data(), graph_output.data(), expected.size())) {
            throw std::runtime_error("first MMQ GateProduct graph replay differs from reference");
        }
        actual_device.copy_from_host(canaries.data(), actual_device.bytes);
        check(cudaGraphLaunch(executable, stream), "relaunch gate projection graph");
        check(cudaStreamSynchronize(stream), "resynchronize gate projection graph");
        actual_device.copy_to_host(graph_output.data(), actual_device.bytes);
        if (!same_bf16_output(expected.data(), graph_output.data(), expected.size())) {
            throw std::runtime_error("second MMQ GateProduct graph replay differs from reference");
        }
        cudaGraphExecDestroy(executable);
        cudaGraphDestroy(captured);
    } else {
        invoke();
        check(cudaDeviceSynchronize(), "synchronize gate projection");
    }
    if (stream != nullptr) { cudaStreamDestroy(stream); }
    if (used_gate != expect_fused) {
        throw std::runtime_error("projector selected the wrong Q4 MMQ gate epilogue route");
    }
    actual_device.copy_to_host(actual.data(), actual_device.bytes);
    if (!same_bf16_output(expected.data(), actual.data(), expected.size())) {
        throw std::runtime_error("MMQ GateProduct differs from native FP32 plus vec_silu BF16 store");
    }
    const auto oracle_samples = [&] {
        std::vector<float> result;
        decode_reference_rows(host_weight, host_packed_activation, rows, k, columns, host_gate,
                              result);
        return result;
    }();
    std::size_t sample = 0;
    const int selected_rows[] = {0, 1, 2, 3, 4, 5, 6, 7, rows / 3, rows / 2, rows - 1};
    const int selected_columns[] = {0, columns / 2, columns - 1};
    for (int row : selected_rows) {
        for (int column : selected_columns) {
            const float expected_math = oracle_samples[sample++];
            const float actual_math = __bfloat162float(
                actual[std::size_t(column) * output_stride + row_offset + row]);
            const float tolerance = std::max(0.12f, std::abs(expected_math) * 0.04f);
            const bool invalid = std::isnan(expected_math) ? !std::isnan(actual_math) :
                (std::isinf(expected_math) ? actual_math != expected_math :
                 (!std::isfinite(actual_math) ||
                  std::abs(actual_math - expected_math) > tolerance ||
                  (expected_math == 0.0f && actual_math == 0.0f &&
                   std::signbit(expected_math) != std::signbit(actual_math))));
            if (invalid) {
                const float native_math = __bfloat162float(
                    expected[std::size_t(column) * output_stride + row_offset + row]);
                std::ostringstream message;
                message << std::setprecision(9)
                        << "MMQ GateProduct failed the packed Q4_K/Q8_1 DS4 FP64 oracle"
                        << " at row=" << row << " column=" << column
                        << " oracle=" << expected_math << " actual=" << actual_math
                        << " native_expected=" << native_math << " tolerance=" << tolerance;
                throw std::runtime_error(message.str());
            }
        }
    }
    return true;
}

bool gather_falls_back(int rows, int columns, std::uint32_t seed) {
    constexpr int k = 5120;
    const auto host_weight = make_q4_weight(rows, k, seed + 11);
    const auto host_input = make_input(k, columns, seed + 12);
    const auto host_gate = make_gate(rows, columns, seed + 13);
    std::vector<std::int32_t> indices(k);
    for (int i = 0; i < k; ++i) { indices[i] = k - 1 - i; }
    DeviceBuffer weight_device(host_weight.size());
    DeviceBuffer input_device(host_input.size() * sizeof(__nv_bfloat16));
    DeviceBuffer gate_device(host_gate.size() * sizeof(float));
    DeviceBuffer gather_device(indices.size() * sizeof(std::int32_t));
    DeviceBuffer output_device(std::size_t(rows + 13) * columns * sizeof(__nv_bfloat16));
    DeviceBuffer expected_device(std::size_t(rows + 13) * columns * sizeof(__nv_bfloat16));
    DeviceBuffer activation_device(gguf::matrix_activation_bytes(k, columns));
    DeviceBuffer native_device(std::size_t(rows) * columns * sizeof(float));
    DeviceBuffer fixup_device(gguf::matrix_fixup_bytes(gguf::GgmlType::Q4_K, rows, columns));
    weight_device.copy_from_host(host_weight.data(), weight_device.bytes);
    input_device.copy_from_host(host_input.data(), input_device.bytes);
    gate_device.copy_from_host(host_gate.data(), gate_device.bytes);
    gather_device.copy_from_host(indices.data(), gather_device.bytes);
    std::vector<__nv_bfloat16> canaries(std::size_t(rows + 13) * columns,
                                        __float2bfloat16(123.25f));
    output_device.copy_from_host(canaries.data(), output_device.bytes);
    expected_device.copy_from_host(canaries.data(), expected_device.bytes);
    Tensor input_tensor(input_device.p, DType::BF16, {k, columns});
    Tensor output_tensor(output_device.p, DType::BF16, {rows + 13, columns});
    Weight product_weight = make_weight(rows, k, weight_device.p);
    product_weight.input_columns = static_cast<const std::int32_t*>(gather_device.p);
    GgufProduct product{&product_weight, &output_tensor, 2, GgufEpilogue::GateProduct,
                        static_cast<const float*>(gate_device.p)};
    const GgufShape shape{QType::GGUF_Q4_K, rows, k};
    WorkspaceArena workspace(detail::gguf_project_workspace_bytes({&shape, 1}, columns, columns));
    gguf::quantize_matrix_activation(gguf::GgmlType::Q4_K,
                                     static_cast<const __nv_bfloat16*>(input_device.p), k,
                                     columns, static_cast<const std::int32_t*>(gather_device.p),
                                     activation_device.p, nullptr);
    gguf::matrix_product(gguf::GgmlType::Q4_K, weight_device.p,
                         std::int64_t(k / 256) * gguf::block_shape(gguf::GgmlType::Q4_K).bytes,
                         rows, k, activation_device.p, columns,
                         static_cast<float*>(native_device.p), rows, fixup_device.p, nullptr);
    gguf::VectorOutput expected_output;
    expected_output.bf16 = static_cast<__nv_bfloat16*>(expected_device.p) + 2;
    expected_output.column_stride = rows + 13;
    expected_output.epilogue = gguf::Epilogue::GateProduct;
    expected_output.gate = static_cast<const float*>(gate_device.p);
    expected_output.gate_column_stride = rows;
    gguf::store_plane(static_cast<const float*>(native_device.p), rows, rows, columns,
                      expected_output, nullptr);
    bool used_gate = true;
    detail::gguf_project(input_tensor, {&product, 1}, workspace, nullptr, nullptr, nullptr,
                         &used_gate);
    check(cudaDeviceSynchronize(), "synchronize gather fallback");
    if (used_gate) { throw std::runtime_error("MMQ gate epilogue ignored a gathered input"); }
    std::vector<__nv_bfloat16> expected(canaries.size());
    std::vector<__nv_bfloat16> actual(canaries.size());
    expected_device.copy_to_host(expected.data(), expected_device.bytes);
    output_device.copy_to_host(actual.data(), output_device.bytes);
    if (!same_bf16_output(expected.data(), actual.data(), expected.size())) {
        throw std::runtime_error("gather fallback changed native GateProduct bytes or canaries");
    }
    return true;
}

bool invalid_spans_and_alias_do_not_write(int rows, int columns, std::uint32_t seed) {
    constexpr int k = 5120;
    const auto host_weight = make_q4_weight(rows, k, seed + 21);
    const auto host_input = make_input(k, columns, seed + 22);
    const auto host_gate = make_gate(rows, columns, seed + 23);
    DeviceBuffer weight_device(host_weight.size());
    DeviceBuffer input_device(host_input.size() * sizeof(__nv_bfloat16));
    DeviceBuffer gate_device(host_gate.size() * sizeof(float));
    DeviceBuffer activation_device(gguf::matrix_activation_bytes(k, columns));
    DeviceBuffer output_device(std::size_t(rows + 13) * columns * sizeof(__nv_bfloat16));
    weight_device.copy_from_host(host_weight.data(), weight_device.bytes);
    input_device.copy_from_host(host_input.data(), input_device.bytes);
    gate_device.copy_from_host(host_gate.data(), gate_device.bytes);
    gguf::quantize_matrix_activation(gguf::GgmlType::Q4_K,
                                     static_cast<const __nv_bfloat16*>(input_device.p), k,
                                     columns, nullptr, activation_device.p, nullptr);
    std::vector<__nv_bfloat16> canaries(std::size_t(rows + 13) * columns,
                                        __float2bfloat16(123.25f));
    output_device.copy_from_host(canaries.data(), output_device.bytes);
    const auto* q4 = static_cast<const std::uint8_t*>(weight_device.p);
    const std::int64_t row_bytes =
        std::int64_t(k / 256) * gguf::block_shape(gguf::GgmlType::Q4_K).bytes;
    const bool bad_output_span = gguf::matrix_product_bf16_gate_store(
        gguf::GgmlType::Q4_K, q4, row_bytes, rows, k, activation_device.p, columns,
        static_cast<__nv_bfloat16*>(output_device.p), INT_MAX - 1,
        static_cast<const float*>(gate_device.p), rows, nullptr);
    const bool bad_gate_span = gguf::matrix_product_bf16_gate_store(
        gguf::GgmlType::Q4_K, q4, row_bytes, rows, k, activation_device.p, columns,
        static_cast<__nv_bfloat16*>(output_device.p), rows + 13,
        static_cast<const float*>(gate_device.p), INT_MAX - 1, nullptr);
    const bool overlap = gguf::matrix_product_bf16_gate_store(
        gguf::GgmlType::Q4_K, q4, row_bytes, rows, k, activation_device.p, columns,
        static_cast<__nv_bfloat16*>(output_device.p), rows + 13,
        reinterpret_cast<const float*>(output_device.p), rows + 13, nullptr);
    check(cudaDeviceSynchronize(), "synchronize invalid gate span checks");
    std::vector<__nv_bfloat16> actual(canaries.size());
    output_device.copy_to_host(actual.data(), output_device.bytes);
    if (bad_output_span || bad_gate_span || overlap ||
        std::memcmp(canaries.data(), actual.data(), output_device.bytes) != 0) {
        throw std::runtime_error("invalid MMQ gate spans or alias were admitted or wrote output");
    }
    return true;
}

bool full_swiglu_wrapper_case(std::uint32_t seed) {
    constexpr int k = 5120;
    constexpr int columns = 1024;
    constexpr int parent_rows = 34816;
    constexpr int rows = parent_rows / 2;
    const auto host_weight = make_q4_weight(parent_rows, k, seed + 31);
    const auto host_input = make_input(k, columns, seed + 32);
    DeviceBuffer weight_device(host_weight.size());
    DeviceBuffer input_device(host_input.size() * sizeof(__nv_bfloat16));
    DeviceBuffer activation_device(gguf::matrix_activation_bytes(k, columns));
    DeviceBuffer gate_plane(std::size_t(rows) * columns * sizeof(float));
    DeviceBuffer up_plane(std::size_t(rows) * columns * sizeof(float));
    DeviceBuffer expected_device(std::size_t(rows) * columns * sizeof(__nv_bfloat16));
    DeviceBuffer actual_device(std::size_t(rows) * columns * sizeof(__nv_bfloat16));
    DeviceBuffer fixup_gate(gguf::matrix_fixup_bytes(gguf::GgmlType::Q4_K, rows, columns));
    DeviceBuffer fixup_up(gguf::matrix_fixup_bytes(gguf::GgmlType::Q4_K, rows, columns));
    weight_device.copy_from_host(host_weight.data(), weight_device.bytes);
    input_device.copy_from_host(host_input.data(), input_device.bytes);
    std::vector<__nv_bfloat16> canaries(std::size_t(rows) * columns,
                                        __float2bfloat16(123.25f));
    actual_device.copy_from_host(canaries.data(), actual_device.bytes);
    gguf::quantize_matrix_activation(gguf::GgmlType::Q4_K,
                                     static_cast<const __nv_bfloat16*>(input_device.p), k,
                                     columns, nullptr, activation_device.p, nullptr);
    const std::int64_t row_bytes =
        std::int64_t(k / 256) * gguf::block_shape(gguf::GgmlType::Q4_K).bytes;
    const auto* parent = static_cast<const std::uint8_t*>(weight_device.p);
    const auto* up = parent + std::int64_t(rows) * row_bytes;
    gguf::matrix_product(gguf::GgmlType::Q4_K, parent, row_bytes, rows, k,
                         activation_device.p, columns, static_cast<float*>(gate_plane.p), rows,
                         fixup_gate.p, nullptr);
    gguf::matrix_product(gguf::GgmlType::Q4_K, up, row_bytes, rows, k, activation_device.p,
                         columns, static_cast<float*>(up_plane.p), rows, fixup_up.p, nullptr);
    gguf::VectorOutput reference_output;
    reference_output.bf16 = static_cast<__nv_bfloat16*>(expected_device.p);
    reference_output.column_stride = rows;
    reference_output.epilogue = gguf::Epilogue::GateProduct;
    reference_output.gate = static_cast<const float*>(gate_plane.p);
    reference_output.gate_column_stride = rows;
    gguf::store_plane(static_cast<const float*>(up_plane.p), rows, rows, columns,
                      reference_output, nullptr);
    check(cudaDeviceSynchronize(), "synchronize two-MMQ SwiGLU reference");

    Tensor input_tensor(input_device.p, DType::BF16, {k, columns});
    Tensor output_tensor(actual_device.p, DType::BF16, {rows, columns});
    Weight parent_weight = make_weight(parent_rows, k, weight_device.p);
    const GgufShape parent_shape{QType::GGUF_Q4_K, parent_rows, k};
    WorkspaceArena workspace(detail::gguf_swiglu_workspace_bytes(
        parent_shape, nullptr, columns, columns));
    detail::gguf_swiglu(input_tensor, parent_weight, nullptr, output_tensor, workspace, nullptr);
    check(cudaDeviceSynchronize(), "synchronize full GGUF SwiGLU wrapper");
    std::vector<__nv_bfloat16> expected(std::size_t(rows) * columns);
    std::vector<__nv_bfloat16> actual(expected.size());
    expected_device.copy_to_host(expected.data(), expected_device.bytes);
    actual_device.copy_to_host(actual.data(), actual_device.bytes);
    if (!same_bf16_output(expected.data(), actual.data(), expected.size())) {
        throw std::runtime_error("full GGUF SwiGLU wrapper differs from two-MMQ reference");
    }
    return true;
}

}

int main() {
    try {
        int device = 0;
        check(cudaGetDevice(&device), "get device");
        cudaDeviceProp properties{};
        check(cudaGetDeviceProperties(&properties, device), "device properties");
        if (properties.major != 8 || properties.minor != 6) { return 77; }
        const int synthetic_rows = 128 * properties.multiProcessorCount;
        gguf::MatrixBf16StorePlan unsupported_plan;
        if (gguf::matrix_bf16_gate_store_plan(gguf::GgmlType::Q6_K, 5120, 1024,
                                              unsupported_plan)) {
            throw std::runtime_error("Q6_K was admitted to the Q4-only gate epilogue");
        }
        if (!compare_case(synthetic_rows, 17, false, false, 3, true, 0xE190u) ||
            !compare_case(synthetic_rows, 17, true, false, 5, true, 0xE191u) ||
            !compare_case(17408, 1024, true, false, 7, true, 0xE192u) ||
            !compare_case(synthetic_rows, 17, false, true, 4, true, 0xE193u) ||
            !compare_case(128, 1024, false, false, 2, false, 0xE195u) ||
            !gather_falls_back(synthetic_rows, 17, 0xE194u) ||
            !invalid_spans_and_alias_do_not_write(synthetic_rows, 17, 0xE196u) ||
            !full_swiglu_wrapper_case(0xE197u)) {
            return 1;
        }
        std::cout << "Q4 MMQ GateProduct direct, prepared, graph, and fallback checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
