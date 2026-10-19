#include "artifact/q4_fields.h"
#include "core/q4_fields.h"
#include "core/weight_view.h"
#include "ops/linear/gguf/ggml_bridge.h"
#include "q4_gdn_compare_fields.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <array>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using ninfer::ops::gguf::Epilogue;
using ninfer::ops::gguf::VectorOutput;
using ninfer::ops::gguf::detail::VecArgs;
using ninfer::ops::gguf::detail::VecStore;
using ninfer::ops::gguf::detail::q4_gdn_compare::Metadata;

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

Activation load_activation(const std::string& path, int k) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) { throw std::runtime_error("cannot open activation snapshot"); }
    const auto size = file.tellg();
    if (size < 32) { throw std::runtime_error("activation snapshot is shorter than its header"); }
    file.seekg(0);
    std::array<std::byte, 32> header{};
    file.read(reinterpret_cast<char*>(header.data()), header.size());
    const int columns = int(read_u32(header.data() + 16));
    if (read_u32(header.data()) != 0x31534156u || read_u32(header.data() + 4) != 1 ||
        read_u32(header.data() + 8) != 12 || read_u32(header.data() + 12) != std::uint32_t(k) ||
        columns <= 0 || columns > 8 || read_u32(header.data() + 20) != 0) {
        throw std::runtime_error("activation snapshot header does not match Q4 vector geometry");
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

struct OutputBuffer {
    void* data = nullptr;
    std::size_t bytes = 0;
    std::size_t guard_bytes = 64;
};

OutputBuffer alloc_output(bool bf16, std::size_t count) {
    const auto item_bytes = bf16 ? sizeof(__nv_bfloat16) : sizeof(float);
    const auto bytes = count * item_bytes;
    auto* data = device_alloc<std::byte>(bytes + 64);
    std::array<std::byte, 64> guard;
    guard.fill(std::byte{0xA7});
    check(cudaMemcpy(data + bytes, guard.data(), guard.size(), cudaMemcpyHostToDevice), "initialize output canary");
    return {data, bytes, guard.size()};
}

bool canary_ok(const OutputBuffer& output) {
    std::array<std::byte, 64> guard{};
    check(cudaMemcpy(guard.data(), static_cast<const std::byte*>(output.data) + output.bytes,
                     guard.size(), cudaMemcpyDeviceToHost), "read output canary");
    for (const auto value : guard) { if (value != std::byte{0xA7}) { return false; } }
    return true;
}

void seed_output(const OutputBuffer& output, bool bf16, std::size_t count, bool accumulate) {
    if (!accumulate) { return; }
    if (bf16) {
        std::vector<__nv_bfloat16> values(count);
        for (std::size_t i = 0; i < count; ++i) { values[i] = __float2bfloat16(float(int(i % 17) - 8) * 0.03125f); }
        check(cudaMemcpy(output.data, values.data(), output.bytes, cudaMemcpyHostToDevice), "seed BF16 output");
    } else {
        std::vector<float> values(count);
        for (std::size_t i = 0; i < count; ++i) { values[i] = float(int(i % 17) - 8) * 0.03125f; }
        check(cudaMemcpy(output.data, values.data(), output.bytes, cudaMemcpyHostToDevice), "seed FP32 output");
    }
}

VectorOutput make_output(const OutputBuffer& output, bool bf16, int rows, bool accumulate) {
    return {bf16 ? static_cast<__nv_bfloat16*>(output.data) : nullptr,
            bf16 ? nullptr : static_cast<float*>(output.data), rows,
            accumulate ? Epilogue::Accumulate : Epilogue::Store, nullptr, 0};
}

bool compare_outputs(const OutputBuffer& baseline, const OutputBuffer& candidate) {
    if (baseline.bytes != candidate.bytes || !canary_ok(baseline) || !canary_ok(candidate)) { return false; }
    std::vector<std::byte> a(baseline.bytes), b(candidate.bytes);
    check(cudaMemcpy(a.data(), baseline.data, baseline.bytes, cudaMemcpyDeviceToHost), "copy baseline output");
    check(cudaMemcpy(b.data(), candidate.data, candidate.bytes, cudaMemcpyDeviceToHost), "copy candidate output");
    return a == b;
}

template <bool WordRead>
void launch_gdn_probe(const ninfer::Weight& weight, const void* activation, int columns,
                      const VectorOutput& output, int ctas_per_sm, cudaStream_t stream) {
    if (weight.k != 5120 || weight.q4_fields.d_width != 12 || weight.q4_fields.dmin_width != 13) {
        throw std::runtime_error("private matched kernel requires Q4 GDN geometry");
    }
    const auto* base = static_cast<const std::uint8_t*>(activation);
    const VecStore store{output.bf16, output.f32, output.column_stride, output.epilogue,
                         output.gate, output.gate_column_stride};
    const VecArgs args{static_cast<const std::uint8_t*>(weight.qdata),
        std::int64_t(weight.k / 256) * 128, weight.n, weight.k,
        reinterpret_cast<const std::int8_t*>(base),
        reinterpret_cast<const half2*>(base + ninfer::ops::gguf::detail::vec_scales_offset(weight.k, columns)), store};
    const Metadata metadata{static_cast<const std::uint8_t*>(weight.scales), weight.q4_fields};
    const int groups = (args.rows + 7) / 8;
    const int grid = std::min(groups, ctas_per_sm * ninfer::ops::gguf::detail::device_facts().sm_count);
    ninfer::ops::gguf::detail::q4_gdn_compare::kernel<4, 5120, 12, 0, 13, 2, WordRead>
        <<<grid, 128, 0, stream>>>(args, metadata,
            static_cast<const std::uint8_t*>(weight.q4_d),
            static_cast<const std::uint8_t*>(weight.q4_dmin), weight.q4_field_row_begin);
    check(cudaGetLastError(), "private GDN kernel launch");
}

bool run_case(const ninfer::Weight& weight, const std::byte* raw, std::int64_t raw_row_bytes,
              int k, int columns, const void* activation, std::uint64_t begin_row, int rows,
              bool bf16, bool accumulate, bool graph, cudaStream_t stream) {
    const auto count = std::size_t(rows) * columns;
    auto baseline = alloc_output(bf16, count);
    auto candidate = alloc_output(bf16, count);
    auto byte6 = alloc_output(bf16, count);
    auto word6 = alloc_output(bf16, count);
    seed_output(baseline, bf16, count, accumulate);
    seed_output(candidate, bf16, count, accumulate);
    seed_output(byte6, bf16, count, accumulate);
    seed_output(word6, bf16, count, accumulate);
    const auto baseline_output = make_output(baseline, bf16, rows, accumulate);
    const auto candidate_output = make_output(candidate, bf16, rows, accumulate);
    const auto byte6_output = make_output(byte6, bf16, rows, accumulate);
    const auto word6_output = make_output(word6, bf16, rows, accumulate);
    const auto baseline_launch = [&] {
        ninfer::ops::gguf::vector_product(ninfer::ops::gguf::GgmlType::Q4_K,
            raw + std::int64_t(begin_row) * raw_row_bytes, raw_row_bytes, rows, k,
            activation, columns, baseline_output, stream);
    };
    const auto candidate_launch = [&] { ninfer::ops::gguf::vector_q4_fields(weight, activation, columns, candidate_output, stream); };
    const auto byte6_launch = [&] { launch_gdn_probe<false>(weight, activation, columns, byte6_output, 6, stream); };
    const auto word6_launch = [&] { launch_gdn_probe<true>(weight, activation, columns, word6_output, 6, stream); };
    if (graph) {
        for (int mode = 0; mode < 4; ++mode) {
            auto& output = mode == 0 ? baseline : mode == 1 ? candidate : mode == 2 ? byte6 : word6;
            seed_output(output, bf16, count, accumulate);
            cudaGraph_t graph_handle = nullptr;
            check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin graph capture");
            if (mode == 0) { baseline_launch(); }
            else if (mode == 1) { candidate_launch(); }
            else if (mode == 2) { byte6_launch(); }
            else { word6_launch(); }
            check(cudaStreamEndCapture(stream, &graph_handle), "end graph capture");
            cudaGraphExec_t executable = nullptr;
            check(cudaGraphInstantiate(&executable, graph_handle, nullptr, nullptr, 0), "instantiate graph");
            check(cudaGraphLaunch(executable, stream), "launch graph");
            check(cudaStreamSynchronize(stream), "graph synchronize");
            check(cudaGraphExecDestroy(executable), "destroy graph executable");
            check(cudaGraphDestroy(graph_handle), "destroy graph");
        }
    } else {
        baseline_launch();
        candidate_launch();
        byte6_launch();
        word6_launch();
        check(cudaStreamSynchronize(stream), "operator synchronize");
    }
    const bool public_exact = compare_outputs(baseline, candidate);
    const bool byte_exact = compare_outputs(baseline, byte6);
    const bool word6_exact = compare_outputs(baseline, word6);
    const bool exact = public_exact && byte_exact && word6_exact;
    check(cudaFree(word6.data), "free word cap-6 output");
    check(cudaFree(byte6.data), "free byte cap-6 output");
    check(cudaFree(candidate.data), "free candidate output");
    check(cudaFree(baseline.data), "free baseline output");
    std::cout << "{\"rows\":" << rows << ",\"row_begin\":" << begin_row
              << ",\"columns\":" << columns << ",\"dtype\":\"" << (bf16 ? "bf16" : "fp32")
              << "\",\"epilogue\":\"" << (accumulate ? "accumulate" : "store")
              << "\",\"graph\":" << (graph ? "true" : "false")
              << ",\"v6_public_word_bit_exact\":" << (public_exact ? "true" : "false")
              << ",\"byte_reconstructed_cap6_bit_exact\":" << (byte_exact ? "true" : "false")
              << ",\"word_cap6_bit_exact\":" << (word6_exact ? "true" : "false")
              << ",\"bit_exact\":" << (exact ? "true" : "false")
              << ",\"canaries_intact\":" << (exact ? "true" : "false") << "}\n";
    return exact;
}

void time_case(const ninfer::Weight& weight, const std::byte* raw, std::int64_t raw_row_bytes,
               int k, int columns, const void* activation, std::uint64_t begin_row, int rows,
               cudaStream_t stream) {
    const auto count = std::size_t(rows) * columns;
    std::array<OutputBuffer, 4> outputs;
    for (auto& output : outputs) { output = alloc_output(false, count); }
    const auto launch = [&](int mode) {
        const auto out = make_output(outputs[mode], false, rows, false);
        if (mode == 0) {
            ninfer::ops::gguf::vector_product(ninfer::ops::gguf::GgmlType::Q4_K,
                raw + std::int64_t(begin_row) * raw_row_bytes, raw_row_bytes, rows, k,
                activation, columns, out, stream);
        } else if (mode == 1) {
            ninfer::ops::gguf::vector_q4_fields(weight, activation, columns, out, stream);
        } else if (mode == 2) {
            launch_gdn_probe<false>(weight, activation, columns, out, 6, stream);
        } else {
            launch_gdn_probe<true>(weight, activation, columns, out, 6, stream);
        }
    };
    std::array<cudaGraphExec_t, 4> graphs{};
    for (int mode = 0; mode < 4; ++mode) {
        cudaGraph_t graph = nullptr;
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "timing graph capture begin");
        launch(mode);
        check(cudaStreamEndCapture(stream, &graph), "timing graph capture end");
        check(cudaGraphInstantiate(&graphs[mode], graph, nullptr, nullptr, 0), "timing graph instantiate");
        check(cudaGraphDestroy(graph), "timing graph destroy");
    }
    for (int round = 0; round < 3; ++round) {
        for (int position = 0; position < 4; ++position) {
            const int mode = (round & 1) ? 3 - position : position;
            check(cudaGraphLaunch(graphs[mode], stream), "timing graph warmup");
        }
    }
    check(cudaStreamSynchronize(stream), "timing graph warmup sync");
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start), "timing event create");
    check(cudaEventCreate(&stop), "timing event create");
    const char* names[4] = {"raw_vector", "v6_public_word_default_grid", "byte_reconstructed_cap6", "word_cap6"};
    for (int round = 0; round < 5; ++round) {
        for (int position = 0; position < 4; ++position) {
            const int mode = (position + round) % 4;
            check(cudaEventRecord(start, stream), "timing start record");
            for (int repeat = 0; repeat < 10; ++repeat) {
                check(cudaGraphLaunch(graphs[mode], stream), "timed graph replay");
            }
            check(cudaEventRecord(stop, stream), "timing stop record");
            check(cudaEventSynchronize(stop), "timing stop sync");
            float elapsed = 0.0f;
            check(cudaEventElapsedTime(&elapsed, start, stop), "timing elapsed");
            std::cout << "{\"kind\":\"timing\",\"variant\":\"" << names[mode]
                      << "\",\"rows\":" << rows << ",\"row_begin\":" << begin_row
                      << ",\"round\":" << round << ",\"milliseconds_per_launch\":" << elapsed / 10.0f << "}\n";
        }
    }
    check(cudaStreamSynchronize(stream), "timing replay sync");
    std::vector<std::byte> reference(outputs[0].bytes);
    check(cudaMemcpy(reference.data(), outputs[0].data, outputs[0].bytes, cudaMemcpyDeviceToHost),
          "timing reference readback");
    for (int mode = 0; mode < 4; ++mode) {
        if (!canary_ok(outputs[mode])) { throw std::runtime_error("timing output canary changed"); }
        std::vector<std::byte> actual(outputs[mode].bytes);
        check(cudaMemcpy(actual.data(), outputs[mode].data, outputs[mode].bytes, cudaMemcpyDeviceToHost),
              "timing exactness readback");
        if (actual != reference) { throw std::runtime_error("timing graph output mismatch"); }
        check(cudaGraphExecDestroy(graphs[mode]), "timing graph executable destroy");
        check(cudaFree(outputs[mode].data), "timing output free");
    }
    check(cudaEventDestroy(start), "timing event destroy");
    check(cudaEventDestroy(stop), "timing event destroy");
}

bool unpack_exact(const ninfer::Weight& weight, const std::byte* raw, int expected_begin, int first_row, int rows,
                  std::int64_t raw_row_bytes, cudaStream_t stream) {
    const auto bytes = std::size_t(rows) * std::size_t(raw_row_bytes);
    auto* output = device_alloc<std::byte>(bytes + 64);
    std::array<std::byte, 64> guard;
    guard.fill(std::byte{0xD3});
    check(cudaMemcpy(output + bytes, guard.data(), guard.size(), cudaMemcpyHostToDevice), "initialize unpack canary");
    ninfer::unpack_q4_fields(weight, first_row, rows, output, stream);
    check(cudaStreamSynchronize(stream), "unpack synchronize");
    std::vector<std::byte> decoded(bytes);
    check(cudaMemcpy(decoded.data(), output, bytes, cudaMemcpyDeviceToHost), "copy unpacked Q4 bytes");
    std::array<std::byte, 64> read_guard{};
    check(cudaMemcpy(read_guard.data(), output + bytes, read_guard.size(), cudaMemcpyDeviceToHost), "read unpack canary");
    const auto* expected = raw + std::int64_t(expected_begin + first_row) * raw_row_bytes;
    const bool exact = std::memcmp(decoded.data(), expected, bytes) == 0 &&
                       std::all_of(read_guard.begin(), read_guard.end(), [](std::byte b) { return b == std::byte{0xD3}; });
    check(cudaFree(output), "free unpack buffer");
    return exact;
}

void run(const std::string& model, std::uint64_t offset, int n, int k,
         const std::string& identity, const std::string& activation_path,
         std::uint64_t slice_begin, int slice_rows, std::uint64_t nested_begin, int nested_rows) {
    const std::int64_t raw_row_bytes = std::int64_t(k / 256) * 144;
    auto raw = read_model_bytes(model, offset, std::uint64_t(n) * raw_row_bytes);
    const std::array<std::uint64_t, 2> shape{std::uint64_t(n), std::uint64_t(k)};
    const auto stored = ninfer::weight_geometry(ninfer::QType::GGUF_Q4_K,
        ninfer::QuantLayout::GgufBlocks, shape);
    const auto geometry = ninfer::artifact::describe_q4_fields(stored, raw);
    if (!geometry) { throw std::runtime_error("actual artifact describe_q4_fields rejected tensor"); }
    auto encoded = ninfer::artifact::encode_q4_fields(*geometry, raw);
    if (encoded.size() != geometry->bytes) { throw std::runtime_error("encoded byte count differs from geometry"); }
    auto* device_encoded = device_alloc<std::byte>(encoded.size());
    check(cudaMemcpy(device_encoded, encoded.data(), encoded.size(), cudaMemcpyHostToDevice), "upload encoded weight");
    ninfer::WeightParent parent;
    parent.geometry = *geometry;
    parent.data = device_encoded;
    ninfer::WeightView root_view;
    root_view.shape = {std::uint64_t(n), std::uint64_t(k)};
    root_view.parts.push_back({&parent, 0, std::uint64_t(n) * k});
    const auto root_weight = ninfer::native_weight(root_view);
    const auto activation = load_activation(activation_path, k);
    auto* device_activation = device_alloc<std::byte>(activation.payload.size());
    auto* device_raw = device_alloc<std::byte>(raw.size());
    check(cudaMemcpy(device_activation, activation.payload.data(), activation.payload.size(), cudaMemcpyHostToDevice), "upload activation");
    check(cudaMemcpy(device_raw, raw.data(), raw.size(), cudaMemcpyHostToDevice), "upload original GGUF Q4 bytes");
    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
    bool all_exact = unpack_exact(root_weight, raw.data(), 0, 0, n, raw_row_bytes, stream);
    std::cout << "{\"kind\":\"full_raw_unpack\",\"rows\":" << n
              << ",\"bytes\":" << raw.size() << ",\"byte_exact\":" << (all_exact ? "true" : "false") << "}\n";
    const auto run_rows = [&](const ninfer::Weight& weight, int begin, int rows) {
        bool exact = true;
        for (const auto bf16 : {false, true}) {
            for (const auto accumulate : {false, true}) {
                for (const auto graph : {false, true}) {
                    exact = run_case(weight, device_raw, raw_row_bytes, k, activation.columns,
                                     device_activation, begin, rows, bf16, accumulate, graph, stream) && exact;
                }
            }
        }
        return exact;
    };
    all_exact = run_rows(root_weight, 0, n) && all_exact;
    if (k == 5120 && activation.columns == 4) {
        time_case(root_weight, device_raw, raw_row_bytes, k, activation.columns, device_activation, 0, n, stream);
    }
    if (slice_begin + std::uint64_t(slice_rows) > std::uint64_t(n)) { throw std::runtime_error("slice exceeds matrix rows"); }
    ninfer::WeightView slice_view;
    slice_view.shape = {std::uint64_t(slice_rows), std::uint64_t(k)};
    slice_view.parts.push_back({&parent, slice_begin * std::uint64_t(k),
        (slice_begin + std::uint64_t(slice_rows)) * std::uint64_t(k)});
    const auto slice_weight = ninfer::native_weight(slice_view);
    all_exact = unpack_exact(slice_weight, raw.data(), int(slice_begin), 0, slice_rows, raw_row_bytes, stream) && all_exact;
    all_exact = run_rows(slice_weight, int(slice_begin), slice_rows) && all_exact;
    if (k == 5120 && activation.columns == 4) {
        time_case(slice_weight, device_raw, raw_row_bytes, k, activation.columns,
                  device_activation, slice_begin, slice_rows, stream);
    }
    if (k == 5120) {
        ninfer::WeightView top_view;
        top_view.shape = {10240, std::uint64_t(k)};
        top_view.parts.push_back({&parent, 0, 10240ull * std::uint64_t(k)});
        const auto top_weight = ninfer::native_weight(top_view);
        all_exact = unpack_exact(top_weight, raw.data(), 0, 0, 10240, raw_row_bytes, stream) && all_exact;
        all_exact = run_rows(top_weight, 0, 10240) && all_exact;
        if (activation.columns == 4) {
            time_case(top_weight, device_raw, raw_row_bytes, k, activation.columns,
                      device_activation, 0, 10240, stream);
        }
    }
    if (nested_rows <= 0 || nested_begin + std::uint64_t(nested_rows) > std::uint64_t(slice_rows)) {
        throw std::runtime_error("nested row range exceeds its parent slice");
    }
    const auto nested_global = slice_begin + nested_begin;
    ninfer::WeightView nested_view;
    nested_view.shape = {std::uint64_t(nested_rows), std::uint64_t(k)};
    nested_view.parts.push_back({&parent, nested_global * std::uint64_t(k),
        (nested_global + std::uint64_t(nested_rows)) * std::uint64_t(k)});
    const auto nested_weight = ninfer::native_weight(nested_view);
    all_exact = unpack_exact(nested_weight, raw.data(), int(nested_global), 0, nested_rows, raw_row_bytes, stream) && all_exact;
    all_exact = run_rows(nested_weight, int(nested_global), nested_rows) && all_exact;
    check(cudaStreamDestroy(stream), "destroy stream");
    check(cudaFree(device_raw), "free original Q4 bytes");
    check(cudaFree(device_activation), "free activation");
    check(cudaFree(device_encoded), "free encoded weight");
    std::cout << "{\"matrix\":\"" << identity << "\",\"N\":" << n << ",\"K\":" << k
              << ",\"columns\":" << activation.columns << ",\"raw_bytes\":" << raw.size()
              << ",\"encoded_bytes\":" << encoded.size() << ",\"saved_bytes\":" << raw.size() - encoded.size()
              << ",\"slice_begin\":" << slice_begin << ",\"slice_rows\":" << slice_rows
              << ",\"nested_row_begin\":" << nested_global << ",\"nested_rows\":" << nested_rows
              << ",\"all_checks_exact\":" << (all_exact ? "true" : "false") << "}\n";
    if (!all_exact) { throw std::runtime_error("Q4 integrated operator or raw unpack oracle failed"); }
}

}

int main(int argc, char** argv) {
    if (argc != 11) {
        std::cerr << "model offset N K tensor activation slice_begin slice_rows nested_begin nested_rows\n";
        return 2;
    }
    try {
        run(argv[1], std::stoull(argv[2]), std::stoi(argv[3]), std::stoi(argv[4]),
            argv[5], argv[6], std::stoull(argv[7]), std::stoi(argv[8]), std::stoull(argv[9]), std::stoi(argv[10]));
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
