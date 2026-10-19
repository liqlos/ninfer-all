#include "artifact/q6_fields.h"
#include "core/q6_fields.h"
#include "core/weight_view.h"
#include "ops/linear/gguf/ggml_bridge.h"
#include "ops/linear/gguf/ggml_bridge_mmq.cuh"
#include "ops/linear/gguf/ggml_bridge_q6_mmq.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) { throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status)); }
}

std::vector<std::byte> read_model_bytes(const std::string& path, std::uint64_t offset, std::uint64_t bytes) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { throw std::runtime_error("cannot open model"); }
    file.seekg(std::streamoff(offset));
    std::vector<std::byte> data(bytes);
    file.read(reinterpret_cast<char*>(data.data()), std::streamsize(bytes));
    if (std::uint64_t(file.gcount()) != bytes) { throw std::runtime_error("truncated model tensor"); }
    return data;
}

template <typename T>
T* device_alloc(std::size_t bytes) {
    T* value = nullptr;
    check(cudaMalloc(reinterpret_cast<void**>(&value), bytes), "device allocation");
    return value;
}

bool device_equal(const void* left, const void* right, std::size_t bytes) {
    constexpr std::size_t chunk_bytes = 4u * 1024u * 1024u;
    std::vector<std::byte> a(std::min(bytes, chunk_bytes));
    std::vector<std::byte> b(a.size());
    for (std::size_t offset = 0; offset < bytes; offset += a.size()) {
        const auto count = std::min(a.size(), bytes - offset);
        check(cudaMemcpy(a.data(), static_cast<const std::byte*>(left) + offset, count, cudaMemcpyDeviceToHost), "copy left comparison range");
        check(cudaMemcpy(b.data(), static_cast<const std::byte*>(right) + offset, count, cudaMemcpyDeviceToHost), "copy right comparison range");
        if (std::memcmp(a.data(), b.data(), count) != 0) { return false; }
    }
    return true;
}

template <typename Launch>
cudaGraphExec_t capture_graph(Launch launch, cudaStream_t stream) {
    launch();
    check(cudaStreamSynchronize(stream), "warmup sync");
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin graph capture");
    launch();
    cudaGraph_t graph = nullptr;
    check(cudaStreamEndCapture(stream, &graph), "end graph capture");
    cudaGraphExec_t executable = nullptr;
    check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), "instantiate graph");
    check(cudaGraphDestroy(graph), "destroy graph");
    return executable;
}

double graph_replay_milliseconds(cudaGraphExec_t executable, cudaStream_t stream,
                                 cudaEvent_t start, cudaEvent_t stop) {
    check(cudaEventRecord(start, stream), "record start event");
    for (int replay = 0; replay < 10; ++replay) { check(cudaGraphLaunch(executable, stream), "graph replay"); }
    check(cudaEventRecord(stop, stream), "record stop event");
    check(cudaEventSynchronize(stop), "timing sync");
    float elapsed = 0.0f;
    check(cudaEventElapsedTime(&elapsed, start, stop), "read elapsed time");
    return double(elapsed) / 10.0;
}

template <typename LaunchA, typename LaunchB>
std::pair<std::vector<double>, std::vector<double>> graph_milliseconds_ab(
        LaunchA launch_a, LaunchB launch_b, cudaStream_t stream) {
    auto executable_a = capture_graph(launch_a, stream);
    auto executable_b = capture_graph(launch_b, stream);
    check(cudaGraphLaunch(executable_a, stream), "graph warmup A");
    check(cudaGraphLaunch(executable_b, stream), "graph warmup B");
    check(cudaStreamSynchronize(stream), "graph warmup sync");
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    check(cudaEventCreate(&start), "create start event");
    check(cudaEventCreate(&stop), "create stop event");
    std::vector<double> samples_a;
    std::vector<double> samples_b;
    for (int sample = 0; sample < 5; ++sample) {
        if ((sample & 1) == 0) {
            samples_a.push_back(graph_replay_milliseconds(executable_a, stream, start, stop));
            samples_b.push_back(graph_replay_milliseconds(executable_b, stream, start, stop));
        } else {
            samples_b.push_back(graph_replay_milliseconds(executable_b, stream, start, stop));
            samples_a.push_back(graph_replay_milliseconds(executable_a, stream, start, stop));
        }
    }
    check(cudaEventDestroy(stop), "destroy stop event");
    check(cudaEventDestroy(start), "destroy start event");
    check(cudaGraphExecDestroy(executable_b), "destroy graph executable B");
    check(cudaGraphExecDestroy(executable_a), "destroy graph executable A");
    std::sort(samples_a.begin(), samples_a.end());
    std::sort(samples_b.begin(), samples_b.end());
    return {samples_a, samples_b};
}

bool padding_is_a5(const void* device, std::int64_t stride, int rows, int columns,
                   std::size_t element_bytes) {
    const std::size_t padding_bytes = std::size_t(stride - rows) * element_bytes;
    if (padding_bytes == 0) { return true; }
    std::vector<std::byte> padding(padding_bytes * std::size_t(columns));
    const auto* first = static_cast<const std::byte*>(device) + std::size_t(rows) * element_bytes;
    check(cudaMemcpy2D(padding.data(), padding_bytes, first,
                       std::size_t(stride) * element_bytes, padding_bytes,
                       std::size_t(columns), cudaMemcpyDeviceToHost), "copy output canaries");
    return std::all_of(padding.begin(), padding.end(), [](std::byte value) { return value == std::byte{0xa5}; });
}

std::vector<__nv_bfloat16> make_input(int k, int columns) {
    std::vector<__nv_bfloat16> input(std::size_t(k) * columns);
    for (int column = 0; column < columns; ++column) {
        for (int index = 0; index < k; ++index) {
            const int value = (index * 17 + column * 29) % 509 - 254;
            input[std::size_t(column) * k + index] = __float2bfloat16(float(value) / 256.0f);
        }
    }
    return input;
}

struct MatrixCase {
    bool fp32_exact = false;
    bool bf16_exact = false;
    bool fp32_canary = false;
    bool bf16_canary = false;
    std::vector<double> raw_ms;
    std::vector<double> fields_ms;
};

MatrixCase compare_matrix(const ninfer::Weight& weight, const std::byte* raw, std::int64_t raw_row_bytes,
                          int k, int columns, int rows, const void* activation, int row_origin,
                          cudaStream_t stream) {
    const std::int64_t stride = std::int64_t(rows) + 16;
    const std::size_t f32_bytes = std::size_t(stride) * columns * sizeof(float);
    const std::size_t bf16_bytes = std::size_t(stride) * columns * sizeof(__nv_bfloat16);
    auto* raw_f32 = device_alloc<float>(f32_bytes);
    auto* fields_f32 = device_alloc<float>(f32_bytes);
    auto* raw_bf16 = device_alloc<__nv_bfloat16>(bf16_bytes);
    auto* fields_bf16 = device_alloc<__nv_bfloat16>(bf16_bytes);
    check(cudaMemset(raw_f32, 0xa5, f32_bytes), "initialize raw FP32 output");
    check(cudaMemset(fields_f32, 0xa5, f32_bytes), "initialize fields FP32 output");
    check(cudaMemset(raw_bf16, 0xa5, bf16_bytes), "initialize raw BF16 output");
    check(cudaMemset(fields_bf16, 0xa5, bf16_bytes), "initialize fields BF16 output");
    const auto fixup_bytes = ninfer::ops::gguf::matrix_fixup_bytes(ninfer::ops::gguf::GgmlType::Q6_K, rows, columns);
    void* raw_fixup = fixup_bytes ? device_alloc<std::byte>(fixup_bytes) : nullptr;
    void* fields_fixup = fixup_bytes ? device_alloc<std::byte>(fixup_bytes) : nullptr;
    const auto raw_view = [&] {
        ninfer::ops::gguf::matrix_product(ninfer::ops::gguf::GgmlType::Q6_K,
            raw + std::int64_t(row_origin) * raw_row_bytes, raw_row_bytes, rows, k, activation,
            columns, raw_f32, stride, raw_fixup, stream);
    };
    const auto fields_view = [&] {
        ninfer::ops::gguf::matrix_q6_fields(weight, activation, columns, fields_f32, stride,
                                            fields_fixup, stream);
    };
    raw_view();
    fields_view();
    check(cudaStreamSynchronize(stream), "FP32 operator sync");
    MatrixCase result;
    result.fp32_exact = device_equal(raw_f32, fields_f32, f32_bytes);
    result.fp32_canary = padding_is_a5(raw_f32, stride, rows, columns, sizeof(float)) &&
                         padding_is_a5(fields_f32, stride, rows, columns, sizeof(float));
    ninfer::ops::gguf::store_plane(raw_f32, stride, rows, columns,
        ninfer::ops::gguf::VectorOutput{raw_bf16, nullptr, stride, ninfer::ops::gguf::Epilogue::Store, nullptr, 0}, stream);
    ninfer::ops::gguf::store_plane(fields_f32, stride, rows, columns,
        ninfer::ops::gguf::VectorOutput{fields_bf16, nullptr, stride, ninfer::ops::gguf::Epilogue::Store, nullptr, 0}, stream);
    check(cudaStreamSynchronize(stream), "BF16 operator sync");
    result.bf16_exact = device_equal(raw_bf16, fields_bf16, bf16_bytes);
    result.bf16_canary = padding_is_a5(raw_bf16, stride, rows, columns, sizeof(__nv_bfloat16)) &&
                         padding_is_a5(fields_bf16, stride, rows, columns, sizeof(__nv_bfloat16));
    auto timings = graph_milliseconds_ab(raw_view, fields_view, stream);
    result.raw_ms = std::move(timings.first);
    result.fields_ms = std::move(timings.second);
    check(cudaFree(fields_fixup), "free fields fixup");
    check(cudaFree(raw_fixup), "free raw fixup");
    check(cudaFree(fields_bf16), "free fields BF16 output");
    check(cudaFree(raw_bf16), "free raw BF16 output");
    check(cudaFree(fields_f32), "free fields FP32 output");
    check(cudaFree(raw_f32), "free raw FP32 output");
    return result;
}

void emit(const char* tensor, int n, int k, int columns, const char* view, int row_origin,
          int rows, const ninfer::ops::gguf::detail::MatrixLaunch& launch, const MatrixCase& result) {
    std::cout << "{\"tensor\":\"" << tensor << "\",\"N\":" << n << ",\"K\":" << k
              << ",\"T\":" << columns << ",\"view\":\"" << view
              << "\",\"row_origin\":" << row_origin << ",\"rows\":" << rows
              << ",\"J\":" << launch.J << ",\"I\":" << launch.I
              << ",\"threads\":" << launch.threads << ",\"shared_bytes\":" << launch.shared
              << ",\"stream_k\":" << (launch.stream_k ? "true" : "false")
              << ",\"blocks\":" << launch.blocks << ",\"fixup\":" << (launch.fixup ? "true" : "false")
              << ",\"fp32_exact\":" << (result.fp32_exact ? "true" : "false")
              << ",\"bf16_exact\":" << (result.bf16_exact ? "true" : "false")
              << ",\"fp32_canary\":" << (result.fp32_canary ? "true" : "false")
              << ",\"bf16_canary\":" << (result.bf16_canary ? "true" : "false")
              << ",\"raw_graph_ms\":[";
    for (std::size_t i = 0; i < result.raw_ms.size(); ++i) { if (i) { std::cout << ','; } std::cout << result.raw_ms[i]; }
    std::cout << "],\"fields_graph_ms\":[";
    for (std::size_t i = 0; i < result.fields_ms.size(); ++i) { if (i) { std::cout << ','; } std::cout << result.fields_ms[i]; }
    std::cout << "]}\n";
    if (!result.fp32_exact || !result.bf16_exact || !result.fp32_canary || !result.bf16_canary) {
        throw std::runtime_error("Q6 MMQ fields output or output canary differs from raw MMQ");
    }
}

void run(const std::string& model, std::uint64_t offset, int n, int k, const std::string& identity,
         const std::vector<int>& column_cases, int slice_begin, int slice_rows) {
    const std::int64_t raw_row_bytes = std::int64_t(k / 256) * 210;
    auto raw = read_model_bytes(model, offset, std::uint64_t(n) * raw_row_bytes);
    const std::array<std::uint64_t, 2> shape{std::uint64_t(n), std::uint64_t(k)};
    const auto stored = ninfer::weight_geometry(ninfer::QType::GGUF_Q6_K, ninfer::QuantLayout::GgufBlocks, shape);
    const auto geometry = ninfer::artifact::describe_q6_fields(stored, raw);
    if (!geometry) { throw std::runtime_error("actual artifact describe_q6_fields rejected matrix"); }
    auto encoded = ninfer::artifact::encode_q6_fields(*geometry, raw);
    auto* device_encoded = device_alloc<std::byte>(encoded.size());
    auto* device_raw = device_alloc<std::byte>(raw.size());
    auto* device_reconstructed = device_alloc<std::byte>(raw.size());
    check(cudaMemcpy(device_encoded, encoded.data(), encoded.size(), cudaMemcpyHostToDevice), "upload encoded Q6 matrix");
    check(cudaMemcpy(device_raw, raw.data(), raw.size(), cudaMemcpyHostToDevice), "upload raw Q6 matrix");
    ninfer::WeightParent parent;
    parent.geometry = *geometry;
    parent.data = device_encoded;
    ninfer::WeightView root_view;
    root_view.shape = {std::uint64_t(n), std::uint64_t(k)};
    root_view.parts.push_back({&parent, 0, std::uint64_t(n) * k});
    const auto root_weight = ninfer::native_weight(root_view);
    ninfer::unpack_q6_fields(root_weight, 0, n, device_reconstructed, nullptr);
    check(cudaDeviceSynchronize(), "Q6 device byte reconstruction sync");
    if (!device_equal(device_raw, device_reconstructed, raw.size())) {
        throw std::runtime_error("Q6 full device byte reconstruction differs");
    }
    cudaStream_t stream = nullptr;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
    for (const int columns : column_cases) {
        auto input = make_input(k, columns);
        auto* device_input = device_alloc<__nv_bfloat16>(input.size() * sizeof(__nv_bfloat16));
        check(cudaMemcpy(device_input, input.data(), input.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice), "upload BF16 input");
        const auto activation_bytes = ninfer::ops::gguf::matrix_activation_bytes(k, columns);
        auto* activation = device_alloc<std::byte>(activation_bytes);
        ninfer::ops::gguf::quantize_matrix_activation(ninfer::ops::gguf::GgmlType::Q6_K,
            device_input, k, columns, nullptr, activation, stream);
        check(cudaStreamSynchronize(stream), "quantize activation");
        const auto root_launch = ninfer::ops::gguf::detail::select_matrix_launch<GGML_TYPE_Q6_K>(n, columns);
        const auto root_result = compare_matrix(root_weight, device_raw, raw_row_bytes, k, columns,
                                                n, activation, 0, stream);
        emit(identity.c_str(), n, k, columns, "root", 0, n, root_launch, root_result);
        if (slice_rows > 0 && slice_begin + slice_rows <= n) {
            ninfer::WeightView slice_view;
            slice_view.shape = {std::uint64_t(slice_rows), std::uint64_t(k)};
            slice_view.parts.push_back({&parent, std::uint64_t(slice_begin) * k,
                                        std::uint64_t(slice_begin + slice_rows) * k});
            const auto slice_weight = ninfer::native_weight(slice_view);
            const auto slice_launch = ninfer::ops::gguf::detail::select_matrix_launch<GGML_TYPE_Q6_K>(slice_rows, columns);
            const auto slice_result = compare_matrix(slice_weight, device_raw, raw_row_bytes, k, columns,
                slice_rows, activation, slice_begin, stream);
            emit(identity.c_str(), n, k, columns, "slice", slice_begin, slice_rows, slice_launch, slice_result);
        }
        check(cudaFree(activation), "free matrix activation");
        check(cudaFree(device_input), "free BF16 input");
    }
    check(cudaStreamDestroy(stream), "destroy stream");
    check(cudaFree(device_raw), "free raw Q6 matrix");
    check(cudaFree(device_reconstructed), "free reconstructed Q6 matrix");
    check(cudaFree(device_encoded), "free encoded Q6 matrix");
    std::cout << "{\"tensor\":\"" << identity << "\",\"N\":" << n << ",\"K\":" << k
              << ",\"q6_fields_width\":" << unsigned(geometry->q6_fields.width)
              << ",\"raw_bytes\":" << raw.size() << ",\"encoded_bytes\":" << encoded.size()
              << ",\"full_byte_reconstruction\":true}\n";
}

}

int main(int argc, char** argv) {
    if (argc != 8) {
        std::cerr << "model offset N K tensor T-list slice-begin:slice-rows\n";
        return 2;
    }
    try {
        std::vector<int> columns;
        std::string list = argv[6];
        std::size_t start = 0;
        while (start < list.size()) {
            const auto end = list.find(',', start);
            columns.push_back(std::stoi(list.substr(start, end == std::string::npos ? end : end - start)));
            if (end == std::string::npos) { break; }
            start = end + 1;
        }
        const std::string slice = argv[7];
        const auto separator = slice.find(':');
        if (separator == std::string::npos) { throw std::runtime_error("invalid slice argument"); }
        run(argv[1], std::stoull(argv[2]), std::stoi(argv[3]), std::stoi(argv[4]), argv[5], columns,
            std::stoi(slice.substr(0, separator)), std::stoi(slice.substr(separator + 1)));
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
