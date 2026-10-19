#include "artifact/q6_fields.h"
#include "core/weight_view.h"
#include "ops/linear/gguf/ggml_bridge.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) { throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status)); }
}

std::uint32_t read_u32(const std::byte* data) {
    return std::to_integer<std::uint32_t>(data[0]) |
           (std::to_integer<std::uint32_t>(data[1]) << 8) |
           (std::to_integer<std::uint32_t>(data[2]) << 16) |
           (std::to_integer<std::uint32_t>(data[3]) << 24);
}

struct Activation {
    int k = 0;
    int columns = 0;
    std::vector<std::byte> payload;
};

Activation load_activation(const std::string& path, int qtype, int k) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) { throw std::runtime_error("cannot open activation snapshot"); }
    const auto size = file.tellg();
    if (size < 32) { throw std::runtime_error("activation snapshot is shorter than its header"); }
    file.seekg(0);
    std::array<std::byte, 32> header{};
    file.read(reinterpret_cast<char*>(header.data()), header.size());
    const int columns = int(read_u32(header.data() + 16));
    if (read_u32(header.data()) != 0x31534156u || read_u32(header.data() + 4) != 1 ||
        read_u32(header.data() + 8) != std::uint32_t(qtype) || read_u32(header.data() + 12) != std::uint32_t(k) ||
        columns <= 0 || columns > 8 || read_u32(header.data() + 20) != 0) {
        throw std::runtime_error("activation snapshot header does not match Q6 vector geometry");
    }
    const auto payload_bytes = read_u32(header.data() + 24);
    const auto expected = std::uint64_t(k) * columns + std::uint64_t(k / 32) * columns * 4;
    if (payload_bytes != expected || std::uint64_t(size) != 32 + expected) {
        throw std::runtime_error("activation snapshot payload length mismatch");
    }
    Activation result{k, columns, std::vector<std::byte>(payload_bytes)};
    file.read(reinterpret_cast<char*>(result.payload.data()), payload_bytes);
    if (!file) { throw std::runtime_error("activation snapshot payload read failed"); }
    return result;
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

void emit_case(const char* view_name, const char* dtype_name, const char* epilogue_name,
               std::uint64_t begin_row, int rows, int columns, bool exact) {
    std::cout << "{\"view\":\"" << view_name << "\",\"dtype\":\"" << dtype_name
              << "\",\"epilogue\":\"" << epilogue_name << "\",\"row_begin\":" << begin_row
              << ",\"rows\":" << rows << ",\"columns\":" << columns
              << ",\"bit_exact\":" << (exact ? "true" : "false") << "}\n";
    if (!exact) { throw std::runtime_error("bridge output differs from original GGUF vector product"); }
}

void compare_case(const ninfer::Weight& encoded_weight, const std::byte* raw, std::int64_t raw_row_bytes,
                  int k, int columns, const void* activation, std::uint64_t begin_row, int rows,
                  bool bf16, bool accumulate, const char* view_name, cudaStream_t stream) {
    const auto count = std::size_t(rows) * columns;
    std::vector<float> seed_f32(count);
    std::vector<__nv_bfloat16> seed_bf16(count);
    for (std::size_t i = 0; i < count; ++i) {
        const float value = float(int(i % 17) - 8) * 0.03125f;
        seed_f32[i] = value;
        seed_bf16[i] = __float2bfloat16(value);
    }
    auto* baseline_f32 = device_alloc<float>(count * sizeof(float));
    auto* candidate_f32 = device_alloc<float>(count * sizeof(float));
    auto* baseline_bf16 = device_alloc<__nv_bfloat16>(count * sizeof(__nv_bfloat16));
    auto* candidate_bf16 = device_alloc<__nv_bfloat16>(count * sizeof(__nv_bfloat16));
    const auto epilogue = accumulate ? ninfer::ops::gguf::Epilogue::Accumulate : ninfer::ops::gguf::Epilogue::Store;
    const auto output_f32 = [&](float* output) {
        return ninfer::ops::gguf::VectorOutput{nullptr, output, rows, epilogue, nullptr, 0};
    };
    const auto output_bf16 = [&]( __nv_bfloat16* output) {
        return ninfer::ops::gguf::VectorOutput{output, nullptr, rows, epilogue, nullptr, 0};
    };
    if (accumulate) {
        check(cudaMemcpy(baseline_f32, seed_f32.data(), count * sizeof(float), cudaMemcpyHostToDevice), "seed FP32 baseline");
        check(cudaMemcpy(candidate_f32, seed_f32.data(), count * sizeof(float), cudaMemcpyHostToDevice), "seed FP32 candidate");
        check(cudaMemcpy(baseline_bf16, seed_bf16.data(), count * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice), "seed BF16 baseline");
        check(cudaMemcpy(candidate_bf16, seed_bf16.data(), count * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice), "seed BF16 candidate");
    }
    if (bf16) {
        ninfer::ops::gguf::vector_product(ninfer::ops::gguf::GgmlType::Q6_K,
            raw + std::int64_t(begin_row) * raw_row_bytes, raw_row_bytes, rows, k, activation, columns,
            output_bf16(baseline_bf16), stream);
        ninfer::ops::gguf::vector_q6_fields(encoded_weight, activation, columns, output_bf16(candidate_bf16), stream);
    } else {
        ninfer::ops::gguf::vector_product(ninfer::ops::gguf::GgmlType::Q6_K,
            raw + std::int64_t(begin_row) * raw_row_bytes, raw_row_bytes, rows, k, activation, columns,
            output_f32(baseline_f32), stream);
        ninfer::ops::gguf::vector_q6_fields(encoded_weight, activation, columns, output_f32(candidate_f32), stream);
    }
    check(cudaStreamSynchronize(stream), "compare sync");
    std::vector<float> host_baseline_f32(count), host_candidate_f32(count);
    std::vector<__nv_bfloat16> host_baseline_bf16(count), host_candidate_bf16(count);
    check(cudaMemcpy(host_baseline_f32.data(), baseline_f32, count * sizeof(float), cudaMemcpyDeviceToHost), "copy FP32 baseline");
    check(cudaMemcpy(host_candidate_f32.data(), candidate_f32, count * sizeof(float), cudaMemcpyDeviceToHost), "copy FP32 candidate");
    check(cudaMemcpy(host_baseline_bf16.data(), baseline_bf16, count * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost), "copy BF16 baseline");
    check(cudaMemcpy(host_candidate_bf16.data(), candidate_bf16, count * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost), "copy BF16 candidate");
    const bool exact = bf16
        ? std::memcmp(host_baseline_bf16.data(), host_candidate_bf16.data(), count * sizeof(__nv_bfloat16)) == 0
        : std::memcmp(host_baseline_f32.data(), host_candidate_f32.data(), count * sizeof(float)) == 0;
    emit_case(view_name, bf16 ? "bf16" : "fp32", accumulate ? "accumulate" : "store", begin_row, rows, columns, exact);
    check(cudaFree(candidate_bf16), "free BF16 candidate");
    check(cudaFree(baseline_bf16), "free BF16 baseline");
    check(cudaFree(candidate_f32), "free FP32 candidate");
    check(cudaFree(baseline_f32), "free FP32 baseline");
}

void run(const std::string& model, std::uint64_t offset, int n, int k, const std::string& identity,
         const std::string& activation_path, std::uint64_t slice_begin, int slice_rows) {
    const auto raw_row_bytes = std::int64_t(k / 256) * 210;
    auto raw = read_model_bytes(model, offset, std::uint64_t(n) * raw_row_bytes);
    const std::array<std::uint64_t, 2> shape{std::uint64_t(n), std::uint64_t(k)};
    const auto stored = ninfer::weight_geometry(ninfer::QType::GGUF_Q6_K, ninfer::QuantLayout::GgufBlocks, shape);
    const auto encoded_geometry = ninfer::artifact::describe_q6_fields(stored, raw);
    if (!encoded_geometry) { throw std::runtime_error("actual artifact describe_q6_fields rejected tensor"); }
    auto encoded = ninfer::artifact::encode_q6_fields(*encoded_geometry, raw);
    if (encoded.size() != encoded_geometry->bytes) { throw std::runtime_error("encoded payload length differs from geometry"); }
    auto* device_encoded = device_alloc<std::byte>(encoded.size());
    check(cudaMemcpy(device_encoded, encoded.data(), encoded.size(), cudaMemcpyHostToDevice), "upload encoded weight");
    ninfer::WeightParent parent;
    parent.geometry = *encoded_geometry;
    parent.data = device_encoded;
    ninfer::WeightView root_view;
    root_view.shape = {std::uint64_t(n), std::uint64_t(k)};
    root_view.parts.push_back({&parent, 0, std::uint64_t(n) * k});
    const auto root_weight = ninfer::native_weight(root_view);
    auto activation = load_activation(activation_path, 14, k);
    const int columns_actual = activation.columns;
    auto* device_activation = device_alloc<std::byte>(activation.payload.size());
    auto* device_raw = device_alloc<std::byte>(raw.size());
    check(cudaMemcpy(device_activation, activation.payload.data(), activation.payload.size(), cudaMemcpyHostToDevice), "upload activation");
    check(cudaMemcpy(device_raw, raw.data(), raw.size(), cudaMemcpyHostToDevice), "upload raw weight");
    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
    const auto run_view = [&](const char* name, std::uint64_t begin, int rows, const ninfer::Weight& weight) {
        compare_case(weight, device_raw, raw_row_bytes, k, columns_actual, device_activation, begin,
                     rows, false, false, name, stream);
        compare_case(weight, device_raw, raw_row_bytes, k, columns_actual, device_activation, begin,
                     rows, false, true, name, stream);
        compare_case(weight, device_raw, raw_row_bytes, k, columns_actual, device_activation, begin,
                     rows, true, false, name, stream);
        compare_case(weight, device_raw, raw_row_bytes, k, columns_actual, device_activation, begin,
                     rows, true, true, name, stream);
    };
    run_view("root", 0, n, root_weight);
    if (slice_begin + std::uint64_t(slice_rows) > std::uint64_t(n)) { throw std::runtime_error("slice exceeds matrix rows"); }
    ninfer::WeightView slice_view;
    slice_view.shape = {std::uint64_t(slice_rows), std::uint64_t(k)};
    slice_view.parts.push_back({&parent, slice_begin * std::uint64_t(k),
                                (slice_begin + std::uint64_t(slice_rows)) * std::uint64_t(k)});
    const auto slice_weight = ninfer::native_weight(slice_view);
    run_view("slice", slice_begin, slice_rows, slice_weight);
    check(cudaStreamDestroy(stream), "destroy stream");
    check(cudaFree(device_raw), "free raw weight");
    check(cudaFree(device_activation), "free activation");
    check(cudaFree(device_encoded), "free encoded weight");
    std::cout << "{\"matrix\":\"" << identity << "\",\"N\":" << n << ",\"K\":" << k
              << ",\"columns\":" << activation.columns << ",\"raw_bytes\":" << raw.size()
              << ",\"encoded_bytes\":" << encoded.size() << ",\"q6_fields_width\":"
              << unsigned(encoded_geometry->q6_fields.width) << ",\"slice_begin\":" << slice_begin
              << ",\"slice_rows\":" << slice_rows << ",\"all_cases_bit_exact\":true}\n";
}

}

int main(int argc, char** argv) {
    if (argc != 9) {
        std::cerr << "model offset N K tensor activation slice_begin slice_rows\n";
        return 2;
    }
    try {
        run(argv[1], std::stoull(argv[2]), std::stoi(argv[3]), std::stoi(argv[4]), argv[5], argv[6],
            std::stoull(argv[7]), std::stoi(argv[8]));
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
