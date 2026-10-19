#include "artifact/q4_fields.h"
#include "core/q4_fields.h"
#include "core/weight_view.h"
#include "ops/linear/gguf/ggml_bridge.h"
#include "ops/linear/gguf/ggml_bridge_mmq.cuh"
#include "ops/linear/gguf/ggml_bridge_q4_mmq.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) { throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status)); }
}

std::vector<std::byte> read_bytes(const std::string& path, std::uint64_t offset, std::uint64_t bytes) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { throw std::runtime_error("cannot open input file"); }
    file.seekg(std::streamoff(offset));
    std::vector<std::byte> data(bytes);
    file.read(reinterpret_cast<char*>(data.data()), std::streamsize(bytes));
    if (std::uint64_t(file.gcount()) != bytes) { throw std::runtime_error("input file range is truncated"); }
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

template <typename A, typename B, typename C>
std::array<std::vector<double>, 3> graph_milliseconds_abc(A a, B b, C c, cudaStream_t stream) {
    auto executable_a = capture_graph(a, stream);
    auto executable_b = capture_graph(b, stream);
    auto executable_c = capture_graph(c, stream);
    check(cudaGraphLaunch(executable_a, stream), "graph warmup raw");
    check(cudaGraphLaunch(executable_b, stream), "graph warmup unpack");
    check(cudaGraphLaunch(executable_c, stream), "graph warmup public direct");
    check(cudaStreamSynchronize(stream), "graph warmup sync");
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    check(cudaEventCreate(&start), "create start event");
    check(cudaEventCreate(&stop), "create stop event");
    std::array<std::vector<double>, 3> samples;
    for (int sample = 0; sample < 5; ++sample) {
        for (int rotation = 0; rotation < 3; ++rotation) {
            const int index = (sample + rotation) % 3;
            const auto executable = index == 0 ? executable_a : index == 1 ? executable_b : executable_c;
            samples[index].push_back(graph_replay_milliseconds(executable, stream, start, stop));
        }
    }
    check(cudaEventDestroy(stop), "destroy stop event");
    check(cudaEventDestroy(start), "destroy start event");
    check(cudaGraphExecDestroy(executable_c), "destroy graph executable direct");
    check(cudaGraphExecDestroy(executable_b), "destroy graph executable unpack");
    check(cudaGraphExecDestroy(executable_a), "destroy graph executable raw");
    for (auto& group : samples) { std::sort(group.begin(), group.end()); }
    return samples;
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

std::vector<__nv_bfloat16> make_input(int k, int columns, const std::string& mode,
                                      const std::string& path) {
    if (mode == "real") {
        auto bytes = read_bytes(path, 0, std::uint64_t(k) * columns * sizeof(__nv_bfloat16));
        std::vector<__nv_bfloat16> input(std::size_t(k) * columns);
        std::memcpy(input.data(), bytes.data(), bytes.size());
        return input;
    }
    if (mode != "synthetic") { throw std::invalid_argument("activation mode must be real or synthetic"); }
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
    bool graph_fp32_exact = false;
    bool graph_fp32_canary = false;
    std::array<std::vector<double>, 3> times;
};

MatrixCase compare_matrix(const ninfer::Weight& weight, const std::byte* raw,
                          std::int64_t raw_row_bytes, int k, int columns, int rows,
                          int row_origin, const void* activation, void* unpacked,
                          cudaStream_t stream) {
    const std::int64_t stride = std::int64_t(rows) + 16;
    const std::size_t f32_bytes = std::size_t(stride) * columns * sizeof(float);
    const std::size_t bf16_bytes = std::size_t(stride) * columns * sizeof(__nv_bfloat16);
    auto* raw_f32 = device_alloc<float>(f32_bytes);
    auto* unpack_f32 = device_alloc<float>(f32_bytes);
    auto* direct_f32 = device_alloc<float>(f32_bytes);
    auto* raw_bf16 = device_alloc<__nv_bfloat16>(bf16_bytes);
    auto* unpack_bf16 = device_alloc<__nv_bfloat16>(bf16_bytes);
    auto* direct_bf16 = device_alloc<__nv_bfloat16>(bf16_bytes);
    check(cudaMemset(raw_f32, 0xa5, f32_bytes), "initialize raw FP32 output");
    check(cudaMemset(unpack_f32, 0xa5, f32_bytes), "initialize unpack FP32 output");
    check(cudaMemset(direct_f32, 0xa5, f32_bytes), "initialize direct FP32 output");
    check(cudaMemset(raw_bf16, 0xa5, bf16_bytes), "initialize raw BF16 output");
    check(cudaMemset(unpack_bf16, 0xa5, bf16_bytes), "initialize unpack BF16 output");
    check(cudaMemset(direct_bf16, 0xa5, bf16_bytes), "initialize direct BF16 output");
    const auto fixup_bytes = ninfer::ops::gguf::matrix_fixup_bytes(ninfer::ops::gguf::GgmlType::Q4_K, rows, columns);
    void* raw_fixup = fixup_bytes ? device_alloc<std::byte>(fixup_bytes) : nullptr;
    void* unpack_fixup = fixup_bytes ? device_alloc<std::byte>(fixup_bytes) : nullptr;
    void* direct_fixup = fixup_bytes ? device_alloc<std::byte>(fixup_bytes) : nullptr;
    const auto raw_view = [&] {
        ninfer::ops::gguf::matrix_product(ninfer::ops::gguf::GgmlType::Q4_K,
            raw + std::int64_t(row_origin) * raw_row_bytes, raw_row_bytes, rows, k, activation,
            columns, raw_f32, stride, raw_fixup, stream);
    };
    const auto unpack_view = [&] {
        ninfer::unpack_q4_fields(weight, 0, rows, unpacked, stream);
        ninfer::ops::gguf::matrix_product(ninfer::ops::gguf::GgmlType::Q4_K, unpacked,
            raw_row_bytes, rows, k, activation, columns, unpack_f32, stride, unpack_fixup, stream);
    };
    const auto direct_view = [&] {
        ninfer::ops::gguf::matrix_q4_fields(weight, activation, columns, direct_f32,
                                             stride, direct_fixup, stream);
    };
    raw_view();
    unpack_view();
    direct_view();
    check(cudaStreamSynchronize(stream), "FP32 operator sync");
    MatrixCase result;
    result.fp32_exact = device_equal(raw_f32, unpack_f32, f32_bytes) &&
                        device_equal(raw_f32, direct_f32, f32_bytes);
    result.fp32_canary = padding_is_a5(raw_f32, stride, rows, columns, sizeof(float)) &&
                         padding_is_a5(unpack_f32, stride, rows, columns, sizeof(float)) &&
                         padding_is_a5(direct_f32, stride, rows, columns, sizeof(float));
    ninfer::ops::gguf::store_plane(raw_f32, stride, rows, columns,
        ninfer::ops::gguf::VectorOutput{raw_bf16, nullptr, stride, ninfer::ops::gguf::Epilogue::Store, nullptr, 0}, stream);
    ninfer::ops::gguf::store_plane(unpack_f32, stride, rows, columns,
        ninfer::ops::gguf::VectorOutput{unpack_bf16, nullptr, stride, ninfer::ops::gguf::Epilogue::Store, nullptr, 0}, stream);
    ninfer::ops::gguf::store_plane(direct_f32, stride, rows, columns,
        ninfer::ops::gguf::VectorOutput{direct_bf16, nullptr, stride, ninfer::ops::gguf::Epilogue::Store, nullptr, 0}, stream);
    check(cudaStreamSynchronize(stream), "BF16 operator sync");
    result.bf16_exact = device_equal(raw_bf16, unpack_bf16, bf16_bytes) &&
                        device_equal(raw_bf16, direct_bf16, bf16_bytes);
    result.bf16_canary = padding_is_a5(raw_bf16, stride, rows, columns, sizeof(__nv_bfloat16)) &&
                         padding_is_a5(unpack_bf16, stride, rows, columns, sizeof(__nv_bfloat16)) &&
                         padding_is_a5(direct_bf16, stride, rows, columns, sizeof(__nv_bfloat16));
    result.times = graph_milliseconds_abc(raw_view, unpack_view, direct_view, stream);
    result.graph_fp32_exact = device_equal(raw_f32, unpack_f32, f32_bytes) &&
                              device_equal(raw_f32, direct_f32, f32_bytes);
    result.graph_fp32_canary = padding_is_a5(raw_f32, stride, rows, columns, sizeof(float)) &&
                               padding_is_a5(unpack_f32, stride, rows, columns, sizeof(float)) &&
                               padding_is_a5(direct_f32, stride, rows, columns, sizeof(float));
    check(cudaFree(direct_fixup), "free direct fixup");
    check(cudaFree(unpack_fixup), "free unpack fixup");
    check(cudaFree(raw_fixup), "free raw fixup");
    check(cudaFree(direct_bf16), "free direct BF16 output");
    check(cudaFree(unpack_bf16), "free unpack BF16 output");
    check(cudaFree(raw_bf16), "free raw BF16 output");
    check(cudaFree(direct_f32), "free direct FP32 output");
    check(cudaFree(unpack_f32), "free unpack FP32 output");
    check(cudaFree(raw_f32), "free raw FP32 output");
    return result;
}

void emit(const char* tensor, int n, int k, int columns, const char* view, int row_origin,
          int rows, const char* activation_mode,
          const ninfer::ops::gguf::detail::MatrixLaunch& launch, const MatrixCase& result) {
    std::cout << "{\"tensor\":\"" << tensor << "\",\"N\":" << n << ",\"K\":" << k
              << ",\"T\":" << columns << ",\"view\":\"" << view
              << "\",\"row_origin\":" << row_origin << ",\"rows\":" << rows
              << ",\"activation\":\"" << activation_mode
              << "\",\"J\":" << launch.J << ",\"I\":" << launch.I
              << ",\"threads\":" << launch.threads << ",\"shared_bytes\":" << launch.shared
              << ",\"stream_k\":" << (launch.stream_k ? "true" : "false")
              << ",\"blocks\":" << launch.blocks << ",\"fixup\":" << (launch.fixup ? "true" : "false")
              << ",\"fp32_exact\":" << (result.fp32_exact ? "true" : "false")
              << ",\"bf16_exact\":" << (result.bf16_exact ? "true" : "false")
              << ",\"fp32_canary\":" << (result.fp32_canary ? "true" : "false")
              << ",\"bf16_canary\":" << (result.bf16_canary ? "true" : "false")
              << ",\"graph_fp32_exact\":" << (result.graph_fp32_exact ? "true" : "false")
              << ",\"graph_fp32_canary\":" << (result.graph_fp32_canary ? "true" : "false")
              << ",\"raw_graph_ms\":[";
    for (std::size_t i = 0; i < result.times[0].size(); ++i) { if (i) { std::cout << ','; } std::cout << result.times[0][i]; }
    std::cout << "],\"unpack_raw_graph_ms\":[";
    for (std::size_t i = 0; i < result.times[1].size(); ++i) { if (i) { std::cout << ','; } std::cout << result.times[1][i]; }
    std::cout << "],\"public_direct_graph_ms\":[";
    for (std::size_t i = 0; i < result.times[2].size(); ++i) { if (i) { std::cout << ','; } std::cout << result.times[2][i]; }
    std::cout << "]}\n";
    if (!result.fp32_exact || !result.bf16_exact || !result.fp32_canary || !result.bf16_canary ||
        !result.graph_fp32_exact || !result.graph_fp32_canary) {
        throw std::runtime_error("public Q4 direct MMQ or output canary differs from raw MMQ");
    }
}

void run(const std::string& model, std::uint64_t offset, int n, int k,
         const std::string& identity, const std::vector<int>& column_cases,
         const std::string& activation_mode, const std::string& activation_path) {
    const std::int64_t raw_row_bytes = std::int64_t(k / 256) * 144;
    auto raw = read_bytes(model, offset, std::uint64_t(n) * raw_row_bytes);
    const std::array<std::uint64_t, 2> shape{std::uint64_t(n), std::uint64_t(k)};
    const auto stored = ninfer::weight_geometry(ninfer::QType::GGUF_Q4_K, ninfer::QuantLayout::GgufBlocks, shape);
    const auto geometry = ninfer::artifact::describe_q4_fields(stored, raw);
    if (!geometry) { throw std::runtime_error("actual artifact describe_q4_fields rejected matrix"); }
    auto encoded = ninfer::artifact::encode_q4_fields(*geometry, raw);
    auto* device_encoded = device_alloc<std::byte>(encoded.size());
    auto* device_raw = device_alloc<std::byte>(raw.size());
    auto* device_reconstructed = device_alloc<std::byte>(raw.size());
    auto* device_unpacked = device_alloc<std::byte>(raw.size());
    check(cudaMemcpy(device_encoded, encoded.data(), encoded.size(), cudaMemcpyHostToDevice), "upload encoded Q4 matrix");
    check(cudaMemcpy(device_raw, raw.data(), raw.size(), cudaMemcpyHostToDevice), "upload raw Q4 matrix");
    ninfer::WeightParent parent;
    parent.geometry = *geometry;
    parent.data = device_encoded;
    ninfer::WeightView root_view;
    root_view.shape = {std::uint64_t(n), std::uint64_t(k)};
    root_view.parts.push_back({&parent, 0, std::uint64_t(n) * k});
    const auto root_weight = ninfer::native_weight(root_view);
    auto malformed_weight = root_weight;
    malformed_weight.q4_field_row_begin = std::numeric_limits<std::uint64_t>::max();
    bool malformed_origin_rejected = false;
    try {
        ninfer::ops::gguf::matrix_q4_fields(malformed_weight, device_encoded, 9,
            reinterpret_cast<float*>(device_reconstructed), n, nullptr, nullptr);
    } catch (const std::invalid_argument&) {
        malformed_origin_rejected = true;
    }
    if (!malformed_origin_rejected) { throw std::runtime_error("Q4 MMQ API accepted a malformed row origin"); }
    ninfer::unpack_q4_fields(root_weight, 0, n, device_reconstructed, nullptr);
    check(cudaDeviceSynchronize(), "Q4 device byte reconstruction sync");
    if (!device_equal(device_raw, device_reconstructed, raw.size())) {
        throw std::runtime_error("Q4 full device byte reconstruction differs");
    }
    cudaStream_t stream = nullptr;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
    for (const int columns : column_cases) {
        auto input = make_input(k, columns, activation_mode, activation_path);
        auto* device_input = device_alloc<__nv_bfloat16>(input.size() * sizeof(__nv_bfloat16));
        check(cudaMemcpy(device_input, input.data(), input.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice), "upload BF16 input");
        const auto activation_bytes = ninfer::ops::gguf::matrix_activation_bytes(k, columns);
        auto* activation = device_alloc<std::byte>(activation_bytes);
        ninfer::ops::gguf::quantize_matrix_activation(ninfer::ops::gguf::GgmlType::Q4_K,
            device_input, k, columns, nullptr, activation, stream);
        check(cudaStreamSynchronize(stream), "quantize activation");
        const auto root_launch = ninfer::ops::gguf::detail::select_matrix_launch<GGML_TYPE_Q4_K>(n, columns);
        const auto root_result = compare_matrix(root_weight, device_raw, raw_row_bytes, k, columns,
                                                n, 0, activation, device_unpacked, stream);
        emit(identity.c_str(), n, k, columns, "root", 0, n, activation_mode.c_str(), root_launch, root_result);
        const int slice_begin = 1;
        const int slice_rows = root_launch.I * 2;
        if (slice_begin + slice_rows <= n) {
            ninfer::WeightView slice_view;
            slice_view.shape = {std::uint64_t(slice_rows), std::uint64_t(k)};
            slice_view.parts.push_back({&parent, std::uint64_t(slice_begin) * k,
                                        std::uint64_t(slice_begin + slice_rows) * k});
            const auto slice_weight = ninfer::native_weight(slice_view);
            const auto slice_launch = ninfer::ops::gguf::detail::select_matrix_launch<GGML_TYPE_Q4_K>(slice_rows, columns);
            const auto slice_result = compare_matrix(slice_weight, device_raw, raw_row_bytes, k, columns,
                slice_rows, slice_begin, activation, device_unpacked, stream);
            emit(identity.c_str(), n, k, columns, "odd-origin-slice", slice_begin, slice_rows,
                 activation_mode.c_str(), slice_launch, slice_result);
        }
        if (identity == "weight/000060") {
            const std::array<std::pair<int, int>, 2> gdn_views{{{0, 10240}, {10240, 6144}}};
            const std::array<const char*, 2> gdn_names{"gdn-first-split", "gdn-second-split"};
            for (std::size_t view_index = 0; view_index < gdn_views.size(); ++view_index) {
                const auto [view_begin, view_rows] = gdn_views[view_index];
                ninfer::WeightView merged_view;
                merged_view.shape = {std::uint64_t(view_rows), std::uint64_t(k)};
                merged_view.parts.push_back({&parent, std::uint64_t(view_begin) * k,
                    std::uint64_t(view_begin + view_rows) * k});
                const auto merged_weight = ninfer::native_weight(merged_view);
                const auto merged_launch = ninfer::ops::gguf::detail::select_matrix_launch<GGML_TYPE_Q4_K>(view_rows, columns);
                if (view_rows % merged_launch.I != 0) {
                    throw std::runtime_error("real GDN merged view does not fill MMQ tiles");
                }
                const auto merged_result = compare_matrix(merged_weight, device_raw, raw_row_bytes, k, columns,
                    view_rows, view_begin, activation, device_unpacked, stream);
                emit(identity.c_str(), n, k, columns, gdn_names[view_index], view_begin, view_rows,
                     activation_mode.c_str(), merged_launch, merged_result);
            }
        }
        if (n == 34816 && k == 5120) {
            const int split_rows = n / 2;
            const std::array<std::pair<int, int>, 2> mlp_views{{{0, split_rows}, {split_rows, split_rows}}};
            const std::array<const char*, 2> mlp_names{"mlp-gate-split", "mlp-up-split"};
            for (std::size_t view_index = 0; view_index < mlp_views.size(); ++view_index) {
                const auto [view_begin, view_rows] = mlp_views[view_index];
                ninfer::WeightView mlp_view;
                mlp_view.shape = {std::uint64_t(view_rows), std::uint64_t(k)};
                mlp_view.parts.push_back({&parent, std::uint64_t(view_begin) * k,
                    std::uint64_t(view_begin + view_rows) * k});
                const auto mlp_weight = ninfer::native_weight(mlp_view);
                const auto mlp_launch = ninfer::ops::gguf::detail::select_matrix_launch<GGML_TYPE_Q4_K>(view_rows, columns);
                if (view_rows % mlp_launch.I != 0) {
                    throw std::runtime_error("MLP gate/up split does not fill MMQ tiles");
                }
                const auto mlp_result = compare_matrix(mlp_weight, device_raw, raw_row_bytes, k, columns,
                    view_rows, view_begin, activation, device_unpacked, stream);
                emit(identity.c_str(), n, k, columns, mlp_names[view_index], view_begin, view_rows,
                     activation_mode.c_str(), mlp_launch, mlp_result);
            }
        }
        check(cudaFree(activation), "free matrix activation");
        check(cudaFree(device_input), "free BF16 input");
    }
    check(cudaStreamDestroy(stream), "destroy stream");
    check(cudaFree(device_unpacked), "free current unpack buffer");
    check(cudaFree(device_reconstructed), "free reconstructed Q4 matrix");
    check(cudaFree(device_raw), "free raw Q4 matrix");
    check(cudaFree(device_encoded), "free encoded Q4 matrix");
    std::cout << "{\"tensor\":\"" << identity << "\",\"N\":" << n << ",\"K\":" << k
              << ",\"q4_d_width\":" << unsigned(geometry->q4_fields.d_width)
              << ",\"q4_dmin_width\":" << unsigned(geometry->q4_fields.dmin_width)
              << ",\"raw_bytes\":" << raw.size() << ",\"encoded_bytes\":" << encoded.size()
              << ",\"full_byte_reconstruction\":true}\n";
}

}

int main(int argc, char** argv) {
    if (argc != 9) {
        std::cerr << "model offset N K tensor T-list activation-mode activation-file\n";
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
        run(argv[1], std::stoull(argv[2]), std::stoi(argv[3]), std::stoi(argv[4]), argv[5],
            columns, argv[7], argv[8]);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
