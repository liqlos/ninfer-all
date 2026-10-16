#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ops/linear/gguf/ggml_bridge.h"
#include "ops/linear/gguf/gguf_linear.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
namespace gguf = ninfer::ops::gguf;
namespace detail = ninfer::ops::detail;
using ninfer::ops::detail::GgufEpilogue;
using ninfer::ops::detail::GgufProduct;
using ninfer::ops::detail::GgufShape;

void check(cudaError_t status, const char* label) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(label) + ": " + cudaGetErrorString(status));
    }
}

std::vector<__nv_bfloat16> make_input(int k, int columns, int pattern, std::uint32_t seed) {
    std::vector<__nv_bfloat16> values(std::size_t(k) * columns);
    std::mt19937 random(seed);
    std::uniform_real_distribution<float> value(-3.0f, 3.0f);
    for (std::size_t i = 0; i < values.size(); ++i) {
        float element = 0.0f;
        if (pattern == 1) {
            element = (i % 4 == 0) ? 1.0e20f : (i % 4 == 1) ? -1.0e20f :
                      (i % 4 == 2) ? 1.0e-15f : -1.0e-15f;
        } else if (pattern == 2) {
            element = value(random);
        }
        values[i] = __float2bfloat16(element);
    }
    return values;
}

std::vector<std::int32_t> reverse_indices(int k) {
    std::vector<std::int32_t> indices(k);
    for (int i = 0; i < k; ++i) { indices[i] = k - 1 - i; }
    return indices;
}

bool equal_bytes(const std::vector<std::uint8_t>& left,
                 const std::vector<std::uint8_t>& right) {
    return left.size() == right.size() && std::memcmp(left.data(), right.data(), left.size()) == 0;
}

void run_pair_oracle(int k, int columns, int pattern, bool gather, bool reverse_order) {
    const auto host_input = make_input(k, columns, pattern, 0x61A7U + k + columns + pattern);
    DeviceBuffer input(host_input.size() * sizeof(__nv_bfloat16));
    input.copy_from_host(host_input.data(), input.bytes);
    std::vector<std::int32_t> host_gather;
    DeviceBuffer gather_device;
    if (gather) {
        host_gather = reverse_indices(k);
        gather_device = DeviceBuffer(std::size_t(k) * sizeof(std::int32_t));
        gather_device.copy_from_host(host_gather.data(), gather_device.bytes);
    }

    const auto type_a = reverse_order ? gguf::GgmlType::Q6_K : gguf::GgmlType::Q4_K;
    const auto type_b = reverse_order ? gguf::GgmlType::Q4_K : gguf::GgmlType::Q6_K;
    const std::size_t bytes = gguf::matrix_activation_bytes(k, columns);
    DeviceBuffer reference_a(bytes);
    DeviceBuffer reference_b(bytes);
    DeviceBuffer actual_a(bytes);
    DeviceBuffer actual_b(bytes);
    reference_a.fill(0xA5);
    reference_b.fill(0xA5);
    actual_a.fill(0xA5);
    actual_b.fill(0xA5);
    const auto* map = gather ? static_cast<const std::int32_t*>(gather_device.p) : nullptr;
    gguf::quantize_matrix_activation(type_a, static_cast<const __nv_bfloat16*>(input.p), k,
                                     columns, map, reference_a.p, nullptr);
    gguf::quantize_matrix_activation(type_b, static_cast<const __nv_bfloat16*>(input.p), k,
                                     columns, map, reference_b.p, nullptr);
    if (!gguf::quantize_matrix_activation_pair(type_a, type_b,
                                                static_cast<const __nv_bfloat16*>(input.p), k,
                                                columns, map, actual_a.p, actual_b.p, nullptr)) {
        throw std::runtime_error("supported Q4_K/Q6_K pair was rejected");
    }
    check(cudaDeviceSynchronize(), "synchronize paired activation oracle");
    std::vector<std::uint8_t> expected_a(bytes), expected_b(bytes), got_a(bytes), got_b(bytes);
    reference_a.copy_to_host(expected_a.data(), bytes);
    reference_b.copy_to_host(expected_b.data(), bytes);
    actual_a.copy_to_host(got_a.data(), bytes);
    actual_b.copy_to_host(got_b.data(), bytes);
    if (!equal_bytes(expected_a, got_a) || !equal_bytes(expected_b, got_b)) {
        throw std::runtime_error("paired matrix activation differs from independent native pack bytes");
    }
}

void run_rejection_oracle() {
    constexpr int k = 5120;
    constexpr int columns = 17;
    const auto host_input = make_input(k, columns, 2, 0xE11U);
    DeviceBuffer input(host_input.size() * sizeof(__nv_bfloat16));
    input.copy_from_host(host_input.data(), input.bytes);
    const std::size_t bytes = gguf::matrix_activation_bytes(k, columns);
    DeviceBuffer output_a(bytes);
    DeviceBuffer output_b(bytes);
    output_a.fill(0x3C);
    output_b.fill(0xC3);
    std::vector<std::uint8_t> before_a(bytes), before_b(bytes), after_a(bytes), after_b(bytes);
    output_a.copy_to_host(before_a.data(), bytes);
    output_b.copy_to_host(before_b.data(), bytes);
    std::vector<std::uint8_t> before_input(input.bytes), after_input(input.bytes);
    input.copy_to_host(before_input.data(), input.bytes);
    const bool unsupported = gguf::quantize_matrix_activation_pair(
        gguf::GgmlType::Q4_K, gguf::GgmlType::Q5_K,
        static_cast<const __nv_bfloat16*>(input.p), k, columns, nullptr, output_a.p, output_b.p,
        nullptr);
    const bool alias_rejected = gguf::quantize_matrix_activation_pair(
        gguf::GgmlType::Q4_K, gguf::GgmlType::Q6_K,
        static_cast<const __nv_bfloat16*>(input.p), k, columns, nullptr, output_a.p, output_a.p,
        nullptr);
    DeviceBuffer overlapping(bytes * 2);
    overlapping.fill(0x69);
    std::vector<std::uint8_t> before_overlap(overlapping.bytes), after_overlap(overlapping.bytes);
    overlapping.copy_to_host(before_overlap.data(), overlapping.bytes);
    auto* overlap_a = overlapping.p;
    auto* overlap_b = static_cast<std::uint8_t*>(overlapping.p) + bytes - sizeof(std::uint32_t);
    const bool overlap_rejected = gguf::quantize_matrix_activation_pair(
        gguf::GgmlType::Q4_K, gguf::GgmlType::Q6_K,
        static_cast<const __nv_bfloat16*>(input.p), k, columns, nullptr, overlap_a, overlap_b,
        nullptr);
    const bool input_alias_rejected = gguf::quantize_matrix_activation_pair(
        gguf::GgmlType::Q4_K, gguf::GgmlType::Q6_K,
        static_cast<const __nv_bfloat16*>(input.p), k, columns, nullptr, input.p, output_b.p,
        nullptr);
    DeviceBuffer gather(std::size_t(k) * sizeof(std::int32_t));
    gather.fill(0);
    std::vector<std::uint8_t> before_gather(gather.bytes), after_gather(gather.bytes);
    gather.copy_to_host(before_gather.data(), gather.bytes);
    const bool gather_alias_rejected = gguf::quantize_matrix_activation_pair(
        gguf::GgmlType::Q4_K, gguf::GgmlType::Q6_K,
        static_cast<const __nv_bfloat16*>(input.p), k, columns,
        static_cast<const std::int32_t*>(gather.p), gather.p, output_b.p, nullptr);
    DeviceBuffer unaligned_input(input.bytes + sizeof(std::uint32_t));
    unaligned_input.fill(0);
    const bool input_alignment_rejected = gguf::quantize_matrix_activation_pair(
        gguf::GgmlType::Q4_K, gguf::GgmlType::Q6_K,
        reinterpret_cast<const __nv_bfloat16*>(static_cast<const std::uint8_t*>(unaligned_input.p) + 2),
        k, columns, nullptr, output_a.p, output_b.p, nullptr);
    const bool output_alignment_rejected = gguf::quantize_matrix_activation_pair(
        gguf::GgmlType::Q4_K, gguf::GgmlType::Q6_K,
        static_cast<const __nv_bfloat16*>(input.p), k, columns, nullptr,
        static_cast<std::uint8_t*>(output_a.p) + 2, output_b.p, nullptr);
    const bool geometry_rejected = gguf::quantize_matrix_activation_pair(
        gguf::GgmlType::Q4_K, gguf::GgmlType::Q6_K,
        static_cast<const __nv_bfloat16*>(input.p), k + 1, columns, nullptr, output_a.p,
        output_b.p, nullptr);
    check(cudaDeviceSynchronize(), "synchronize rejected activation pairs");
    output_a.copy_to_host(after_a.data(), bytes);
    output_b.copy_to_host(after_b.data(), bytes);
    input.copy_to_host(after_input.data(), input.bytes);
    overlapping.copy_to_host(after_overlap.data(), overlapping.bytes);
    gather.copy_to_host(after_gather.data(), gather.bytes);
    if (unsupported || alias_rejected || overlap_rejected || input_alias_rejected ||
        gather_alias_rejected || input_alignment_rejected || output_alignment_rejected ||
        geometry_rejected || !equal_bytes(before_overlap, after_overlap) ||
        !equal_bytes(before_gather, after_gather) || !equal_bytes(before_input, after_input) ||
        !equal_bytes(before_a, after_a) ||
        !equal_bytes(before_b, after_b)) {
        throw std::runtime_error("rejected activation pair enqueued work or changed output bytes");
    }
}

struct WeightBytes {
    QType type;
    int rows;
    int k;
    int block_bytes;
    int scale_offset;
};

std::vector<std::uint8_t> make_weight(const WeightBytes& shape, std::uint32_t seed) {
    const int blocks_per_row = shape.k / 256;
    std::vector<std::uint8_t> result(std::size_t(shape.rows) * blocks_per_row * shape.block_bytes);
    std::mt19937 random(seed);
    std::uniform_int_distribution<int> byte(0, 255);
    for (auto& element : result) { element = static_cast<std::uint8_t>(byte(random)); }
    const std::uint16_t scale = 0x3980;
    const std::size_t block_count = std::size_t(shape.rows) * blocks_per_row;
    for (std::size_t i = 0; i < block_count; ++i) {
        std::memcpy(result.data() + i * shape.block_bytes + shape.scale_offset, &scale,
                    sizeof(scale));
        if (shape.type == QType::GGUF_Q4_K) {
            std::memcpy(result.data() + i * shape.block_bytes + 2, &scale, sizeof(scale));
        }
    }
    return result;
}

Weight make_weight(const WeightBytes& shape, const void* data, const std::int32_t* input_columns) {
    Weight result;
    result.qtype = shape.type;
    result.layout = QuantLayout::GgufBlocks;
    result.qdata = data;
    result.payload = data;
    result.n = result.shape[0] = result.padded_shape[0] = shape.rows;
    result.k = result.shape[1] = result.padded_shape[1] = shape.k;
    result.ndim = 2;
    result.group_size = 256;
    result.group = 256;
    result.input_columns = input_columns;
    return result;
}

std::int64_t row_bytes(const WeightBytes& shape) {
    return std::int64_t(shape.k / 256) * shape.block_bytes;
}

void run_projector_case(int gather_mode, bool graph_replay = false) {
    constexpr int k = 5120;
    constexpr int columns = 17;
    constexpr int rows = 512;
    constexpr int stride = rows + 2;
    const WeightBytes q4_shape{QType::GGUF_Q4_K, rows, k, 144, 0};
    const WeightBytes q6_shape{QType::GGUF_Q6_K, rows, k, 210, 208};
    const auto q4_host = make_weight(q4_shape, 0x4A11U);
    const auto q6_host = make_weight(q6_shape, 0x6A11U);
    DeviceBuffer q4_device(q4_host.size());
    DeviceBuffer q6_device(q6_host.size());
    q4_device.copy_from_host(q4_host.data(), q4_device.bytes);
    q6_device.copy_from_host(q6_host.data(), q6_device.bytes);

    const auto host_input = make_input(k, columns, 2, 0x22U);
    DeviceBuffer input(host_input.size() * sizeof(__nv_bfloat16));
    input.copy_from_host(host_input.data(), input.bytes);
    std::vector<std::int32_t> host_gather_a;
    std::vector<std::int32_t> host_gather_b;
    DeviceBuffer gather_a;
    DeviceBuffer gather_b;
    const std::int32_t* map_a = nullptr;
    const std::int32_t* map_b = nullptr;
    if (gather_mode != 0) {
        host_gather_a = reverse_indices(k);
        gather_a = DeviceBuffer(std::size_t(k) * sizeof(std::int32_t));
        gather_a.copy_from_host(host_gather_a.data(), gather_a.bytes);
        map_a = static_cast<const std::int32_t*>(gather_a.p);
        if (gather_mode == 1) {
            map_b = map_a;
        } else {
            host_gather_b.resize(k);
            for (int i = 0; i < k; ++i) { host_gather_b[i] = (i + 1) % k; }
            gather_b = DeviceBuffer(std::size_t(k) * sizeof(std::int32_t));
            gather_b.copy_from_host(host_gather_b.data(), gather_b.bytes);
            map_b = static_cast<const std::int32_t*>(gather_b.p);
        }
    }
    Weight q4 = make_weight(q4_shape, q4_device.p, map_a);
    Weight q6 = make_weight(q6_shape, q6_device.p, map_b);

    DeviceBuffer actual_q4(std::size_t(stride) * columns * sizeof(__nv_bfloat16));
    DeviceBuffer actual_q6(std::size_t(stride) * columns * sizeof(__nv_bfloat16));
    DeviceBuffer expected_q4(std::size_t(stride) * columns * sizeof(__nv_bfloat16));
    DeviceBuffer expected_q6(std::size_t(stride) * columns * sizeof(__nv_bfloat16));
    std::vector<__nv_bfloat16> initial(std::size_t(stride) * columns,
                                       __float2bfloat16(-7.25f));
    actual_q4.copy_from_host(initial.data(), actual_q4.bytes);
    actual_q6.copy_from_host(initial.data(), actual_q6.bytes);
    expected_q4.copy_from_host(initial.data(), expected_q4.bytes);
    expected_q6.copy_from_host(initial.data(), expected_q6.bytes);
    Tensor actual_q4_tensor(actual_q4.p, DType::BF16, {stride, columns});
    Tensor actual_q6_tensor(actual_q6.p, DType::BF16, {stride, columns});
    Tensor expected_q4_tensor(expected_q4.p, DType::BF16, {stride, columns});
    Tensor expected_q6_tensor(expected_q6.p, DType::BF16, {stride, columns});
    GgufProduct products[2] = {
        {&q4, &actual_q4_tensor, 1, GgufEpilogue::Store},
        {&q6, &actual_q6_tensor, 1, GgufEpilogue::Store},
    };

    const std::size_t plane_bytes = gguf::matrix_activation_bytes(k, columns);
    DeviceBuffer activation_q4(plane_bytes);
    DeviceBuffer activation_q6(plane_bytes);
    gguf::quantize_matrix_activation(gguf::GgmlType::Q4_K,
                                     static_cast<const __nv_bfloat16*>(input.p), k, columns,
                                     map_a, activation_q4.p, nullptr);
    gguf::quantize_matrix_activation(gguf::GgmlType::Q6_K,
                                     static_cast<const __nv_bfloat16*>(input.p), k, columns,
                                     map_b, activation_q6.p, nullptr);
    const std::size_t fixup_bytes = std::max(
        gguf::matrix_fixup_bytes(gguf::GgmlType::Q4_K, rows, columns),
        gguf::matrix_fixup_bytes(gguf::GgmlType::Q6_K, rows, columns));
    DeviceBuffer fixup(fixup_bytes);
    DeviceBuffer fp32_q4(std::size_t(rows) * columns * sizeof(float));
    DeviceBuffer fp32_q6(std::size_t(rows) * columns * sizeof(float));
    gguf::matrix_product(gguf::GgmlType::Q4_K, q4_device.p, row_bytes(q4_shape), rows, k,
                         activation_q4.p, columns, static_cast<float*>(fp32_q4.p), rows,
                         fixup.p, nullptr);
    gguf::matrix_product(gguf::GgmlType::Q6_K, q6_device.p, row_bytes(q6_shape), rows, k,
                         activation_q6.p, columns, static_cast<float*>(fp32_q6.p), rows,
                         fixup.p, nullptr);
    check(cudaDeviceSynchronize(), "synchronize reference matrix products");
    std::vector<float> host_fp32_q4(std::size_t(rows) * columns);
    std::vector<float> host_fp32_q6(std::size_t(rows) * columns);
    fp32_q4.copy_to_host(host_fp32_q4.data(), fp32_q4.bytes);
    fp32_q6.copy_to_host(host_fp32_q6.data(), fp32_q6.bytes);
    for (int column = 0; column < columns; ++column) {
        for (int row = 0; row < rows; ++row) {
            initial[std::size_t(column) * stride + 1 + row] =
                __float2bfloat16(host_fp32_q4[std::size_t(column) * rows + row]);
        }
    }
    expected_q4.copy_from_host(initial.data(), expected_q4.bytes);
    initial.assign(std::size_t(stride) * columns, __float2bfloat16(-7.25f));
    for (int column = 0; column < columns; ++column) {
        for (int row = 0; row < rows; ++row) {
            initial[std::size_t(column) * stride + 1 + row] =
                __float2bfloat16(host_fp32_q6[std::size_t(column) * rows + row]);
        }
    }
    expected_q6.copy_from_host(initial.data(), expected_q6.bytes);

    const GgufShape shapes[2] = {{QType::GGUF_Q4_K, rows, k},
                                 {QType::GGUF_Q6_K, rows, k}};
    const auto workspace_bytes = detail::gguf_project_workspace_bytes(shapes, columns, columns);
    WorkspaceArena workspace(workspace_bytes);
    Tensor x(input.p, DType::BF16, {k, columns});
    bool used_pair = false;
    if (!graph_replay) {
        detail::gguf_project(x, products, workspace, nullptr, nullptr, &used_pair);
        check(cudaDeviceSynchronize(), "synchronize dual-layout projection");
    } else {
        cudaStream_t stream = nullptr;
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t executable = nullptr;
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
              "create dual-layout graph stream");
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
              "begin dual-layout graph capture");
        detail::gguf_project(x, products, workspace, stream, nullptr, &used_pair);
        check(cudaStreamEndCapture(stream, &graph), "end dual-layout graph capture");
        check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0),
              "instantiate dual-layout graph");
        std::vector<__nv_bfloat16> reset_output(std::size_t(stride) * columns,
                                                __float2bfloat16(-7.25f));
        for (int replay = 0; replay < 2; ++replay) {
            check(cudaMemcpyAsync(input.p, host_input.data(), input.bytes,
                                  cudaMemcpyHostToDevice, stream),
                  "reset dual-layout graph input");
            check(cudaMemcpyAsync(actual_q4.p, reset_output.data(), actual_q4.bytes,
                                  cudaMemcpyHostToDevice, stream),
                  "reset Q4 dual-layout graph output");
            check(cudaMemcpyAsync(actual_q6.p, reset_output.data(), actual_q6.bytes,
                                  cudaMemcpyHostToDevice, stream),
                  "reset Q6 dual-layout graph output");
            check(cudaGraphLaunch(executable, stream), "launch dual-layout graph");
            check(cudaStreamSynchronize(stream), "synchronize dual-layout graph replay");
        }
        check(cudaGraphExecDestroy(executable), "destroy dual-layout graph executable");
        check(cudaGraphDestroy(graph), "destroy dual-layout graph");
        check(cudaStreamDestroy(stream), "destroy dual-layout graph stream");
    }
    std::vector<__nv_bfloat16> actual_a(std::size_t(stride) * columns);
    std::vector<__nv_bfloat16> actual_b(std::size_t(stride) * columns);
    std::vector<__nv_bfloat16> expected_a(actual_a.size());
    std::vector<__nv_bfloat16> expected_b(actual_b.size());
    actual_q4.copy_to_host(actual_a.data(), actual_q4.bytes);
    actual_q6.copy_to_host(actual_b.data(), actual_q6.bytes);
    expected_q4.copy_to_host(expected_a.data(), expected_q4.bytes);
    expected_q6.copy_to_host(expected_b.data(), expected_q6.bytes);
    const bool expect_pair = gather_mode != 2;
    if (used_pair != expect_pair ||
        std::memcmp(actual_a.data(), expected_a.data(), actual_q4.bytes) != 0 ||
        std::memcmp(actual_b.data(), expected_b.data(), actual_q6.bytes) != 0) {
        throw std::runtime_error("GGUF projection selected the wrong pair route or changed output bytes");
    }
}

void run_pack_benchmark_case(int k) {
    constexpr int columns = 1024;
    constexpr int repetitions = 50;
    const auto host_input = make_input(k, columns, 2, 0xD00DU + k);
    DeviceBuffer input(host_input.size() * sizeof(__nv_bfloat16));
    input.copy_from_host(host_input.data(), input.bytes);
    const std::size_t bytes = gguf::matrix_activation_bytes(k, columns);
    DeviceBuffer q4(bytes);
    DeviceBuffer q6(bytes);
    DeviceBuffer paired_q4(bytes);
    DeviceBuffer paired_q6(bytes);
    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create pack benchmark stream");
    check(cudaEventCreate(&start), "create pack benchmark start event");
    check(cudaEventCreate(&stop), "create pack benchmark stop event");
    const auto* x = static_cast<const __nv_bfloat16*>(input.p);
    for (int i = 0; i < 10; ++i) {
        if (!gguf::quantize_matrix_activation_pair(gguf::GgmlType::Q4_K,
                                                   gguf::GgmlType::Q6_K, x, k, columns, nullptr,
                                                   paired_q4.p, paired_q6.p, stream)) {
            throw std::runtime_error("paired pack rejected benchmark geometry");
        }
        gguf::quantize_matrix_activation(gguf::GgmlType::Q4_K, x, k, columns, nullptr, q4.p, stream);
        gguf::quantize_matrix_activation(gguf::GgmlType::Q6_K, x, k, columns, nullptr, q6.p, stream);
    }
    check(cudaStreamSynchronize(stream), "warm matrix activation pack benchmark");
    for (int trial = 0; trial < 5; ++trial) {
        float elapsed[2]{};
        for (int route = 0; route < 2; ++route) {
            const bool paired_first = trial % 2 == 0;
            const bool run_pair = route == 0 ? paired_first : !paired_first;
            check(cudaEventRecord(start, stream), "record pack benchmark start");
            for (int iteration = 0; iteration < repetitions; ++iteration) {
                if (run_pair) {
                    if (!gguf::quantize_matrix_activation_pair(
                            gguf::GgmlType::Q4_K, gguf::GgmlType::Q6_K, x, k, columns, nullptr,
                            paired_q4.p, paired_q6.p, stream)) {
                        throw std::runtime_error("paired pack rejected benchmark geometry");
                    }
                } else {
                    gguf::quantize_matrix_activation(gguf::GgmlType::Q4_K, x, k, columns, nullptr,
                                                     q4.p, stream);
                    gguf::quantize_matrix_activation(gguf::GgmlType::Q6_K, x, k, columns, nullptr,
                                                     q6.p, stream);
                }
            }
            check(cudaEventRecord(stop, stream), "record pack benchmark stop");
            check(cudaEventSynchronize(stop), "synchronize pack benchmark trial");
            check(cudaEventElapsedTime(&elapsed[route], start, stop), "read pack benchmark time");
        }
        const float paired_ms = trial % 2 == 0 ? elapsed[0] : elapsed[1];
        const float native_ms = trial % 2 == 0 ? elapsed[1] : elapsed[0];
        std::cout << k << ',' << columns << ',' << trial << ',' << paired_ms / repetitions << ','
                  << native_ms / repetitions << '\n';
    }
    check(cudaEventDestroy(stop), "destroy pack benchmark stop event");
    check(cudaEventDestroy(start), "destroy pack benchmark start event");
    check(cudaStreamDestroy(stream), "destroy pack benchmark stream");
}

}

int main(int argc, char** argv) {
    try {
        int device = 0;
        check(cudaGetDevice(&device), "get CUDA device");
        cudaDeviceProp properties{};
        check(cudaGetDeviceProperties(&properties, device), "CUDA device properties");
        if (properties.major != 8 || properties.minor != 6) { return 77; }
        if (argc == 2 && std::string(argv[1]) == "--benchmark") {
            std::cout << "K,T,trial,paired_ms_per_call,two_native_ms_per_call\n";
            for (int k : {5120, 17408}) { run_pack_benchmark_case(k); }
            return 0;
        }
        if (argc != 1) { throw std::invalid_argument("expected no arguments or --benchmark"); }
        for (int k : {5120, 6144, 17408}) {
            for (int columns : {9, 17, 127, 1024}) {
                run_pair_oracle(k, columns, 0, false, false);
                run_pair_oracle(k, columns, 1, true, true);
                run_pair_oracle(k, columns, 2, false, true);
            }
        }
        run_rejection_oracle();
        run_projector_case(0);
        run_projector_case(0, true);
        run_projector_case(1);
        run_projector_case(2);
        std::cout << "exact dual-layout MMQ pack and projector fallback checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "dual-layout MMQ pack: " << error.what() << '\n';
        return 1;
    }
}
