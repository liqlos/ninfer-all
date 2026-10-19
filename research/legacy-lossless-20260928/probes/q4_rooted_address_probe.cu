#include "artifact/q4_fields.h"
#include "core/q4_fields.h"
#include "core/weight_view.h"
#include "ops/linear/gguf/ggml_bridge.h"
#include "ops/linear/gguf/ggml_bridge_internal.cuh"
#include "ops/linear/gguf/q4_fields.cuh"
#include "q4_rooted_fields_private.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <array>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using ninfer::ops::gguf::Epilogue;
using ninfer::ops::gguf::VectorOutput;
using ninfer::ops::gguf::detail::VecArgs;
using ninfer::ops::gguf::detail::VecStore;
using ninfer::ops::gguf::detail::q4_fields::Metadata;

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

template <int T>
void launch_q4(const ninfer::Weight& weight, const void* activation, int columns,
               const VectorOutput& output, cudaStream_t stream) {
    const auto* base = static_cast<const std::uint8_t*>(activation);
    const VecStore store{output.bf16, output.f32, output.column_stride, output.epilogue,
                         output.gate, output.gate_column_stride};
    const VecArgs args{static_cast<const std::uint8_t*>(weight.qdata),
        std::int64_t(weight.k / 256) * 128, weight.n, weight.k,
        reinterpret_cast<const std::int8_t*>(base),
        reinterpret_cast<const half2*>(base + ninfer::ops::gguf::detail::vec_scales_offset(weight.k, columns)), store};
    const Metadata metadata{static_cast<const std::uint8_t*>(weight.scales), weight.q4_fields};
    ninfer::ops::gguf::detail::q4_fields::launch<T>(args, metadata,
        static_cast<const std::uint8_t*>(weight.q4_d), static_cast<const std::uint8_t*>(weight.q4_dmin),
        weight.q4_field_row_begin, stream);
}

void launch_q4_dispatch(const ninfer::Weight& weight, const void* activation, int columns,
                        const VectorOutput& output, cudaStream_t stream) {
    if (columns == 1) { launch_q4<1>(weight, activation, columns, output, stream); return; }
    if (columns == 2) { launch_q4<2>(weight, activation, columns, output, stream); return; }
    if (columns == 3) { launch_q4<3>(weight, activation, columns, output, stream); return; }
    if (columns == 4) { launch_q4<4>(weight, activation, columns, output, stream); return; }
    if (columns == 5) { launch_q4<5>(weight, activation, columns, output, stream); return; }
    if (columns == 6) { launch_q4<6>(weight, activation, columns, output, stream); return; }
    if (columns == 7) { launch_q4<7>(weight, activation, columns, output, stream); return; }
    if (columns == 8) { launch_q4<8>(weight, activation, columns, output, stream); return; }
    throw std::runtime_error("captured columns must be T1 through T8");
}

template <int T>
void launch_rooted_q4(const ninfer::Weight& weight, const void* activation, int columns,
                      const VectorOutput& output, cudaStream_t stream) {
    const auto* base = static_cast<const std::uint8_t*>(activation);
    const VecStore store{output.bf16, output.f32, output.column_stride, output.epilogue,
                         output.gate, output.gate_column_stride};
    const VecArgs args{static_cast<const std::uint8_t*>(weight.qdata),
        std::int64_t(weight.k / 256) * 128, weight.n, weight.k,
        reinterpret_cast<const std::int8_t*>(base),
        reinterpret_cast<const half2*>(base + ninfer::ops::gguf::detail::vec_scales_offset(weight.k, columns)), store};
    const Metadata metadata{static_cast<const std::uint8_t*>(weight.scales), weight.q4_fields};
    ninfer::ops::gguf::detail::q4_rooted_fields::launch<T>(args, metadata,
        static_cast<const std::uint8_t*>(weight.q4_d), static_cast<const std::uint8_t*>(weight.q4_dmin),
        weight.q4_field_row_begin, stream);
}

void launch_rooted_q4_dispatch(const ninfer::Weight& weight, const void* activation, int columns,
                               const VectorOutput& output, cudaStream_t stream) {
    if (columns == 1) { launch_rooted_q4<1>(weight, activation, columns, output, stream); return; }
    if (columns == 3) { launch_rooted_q4<3>(weight, activation, columns, output, stream); return; }
    if (columns == 4) { launch_rooted_q4<4>(weight, activation, columns, output, stream); return; }
    throw std::runtime_error("rooted Q4 probe expects captured T1, T3, or T4");
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

bool run_case(const ninfer::Weight& weight, const std::byte* raw, std::int64_t raw_row_bytes,
              int k, int columns, const void* activation, std::uint64_t begin_row, int rows,
              bool bf16, bool accumulate, bool graph, cudaStream_t stream) {
    const auto count = std::size_t(rows) * columns;
    auto baseline = alloc_output(bf16, count);
    auto candidate = alloc_output(bf16, count);
    auto rooted = alloc_output(bf16, count);
    seed_output(baseline, bf16, count, accumulate);
    seed_output(candidate, bf16, count, accumulate);
    seed_output(rooted, bf16, count, accumulate);
    const auto baseline_output = make_output(baseline, bf16, rows, accumulate);
    const auto candidate_output = make_output(candidate, bf16, rows, accumulate);
    const auto rooted_output = make_output(rooted, bf16, rows, accumulate);
    const auto baseline_launch = [&] {
        ninfer::ops::gguf::vector_product(ninfer::ops::gguf::GgmlType::Q4_K,
            raw + std::int64_t(begin_row) * raw_row_bytes, raw_row_bytes, rows, k,
            activation, columns, baseline_output, stream);
    };
    const auto candidate_launch = [&] { launch_q4_dispatch(weight, activation, columns, candidate_output, stream); };
    const auto rooted_launch = [&] { launch_rooted_q4_dispatch(weight, activation, columns, rooted_output, stream); };
    if (graph) {
        for (int mode = 0; mode < 3; ++mode) {
            auto& output = mode == 0 ? baseline : mode == 1 ? candidate : rooted;
            seed_output(output, bf16, count, accumulate);
            cudaGraph_t graph_handle = nullptr;
            check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin graph capture");
            if (mode == 0) { baseline_launch(); } else if (mode == 1) { candidate_launch(); } else { rooted_launch(); }
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
        rooted_launch();
        check(cudaStreamSynchronize(stream), "operator synchronize");
    }
    const bool current_exact = compare_outputs(baseline, candidate);
    const bool rooted_exact = compare_outputs(baseline, rooted);
    const bool exact = current_exact && rooted_exact;
    check(cudaFree(rooted.data), "free rooted output");
    check(cudaFree(candidate.data), "free candidate output");
    check(cudaFree(baseline.data), "free baseline output");
    std::cout << "{\"rows\":" << rows << ",\"row_begin\":" << begin_row
              << ",\"columns\":" << columns << ",\"dtype\":\"" << (bf16 ? "bf16" : "fp32")
              << "\",\"epilogue\":\"" << (accumulate ? "accumulate" : "store")
              << "\",\"graph\":" << (graph ? "true" : "false")
              << ",\"current_bit_exact\":" << (current_exact ? "true" : "false")
              << ",\"rooted_bit_exact\":" << (rooted_exact ? "true" : "false")
              << ",\"bit_exact\":" << (exact ? "true" : "false")
              << ",\"canaries_intact\":" << (exact ? "true" : "false") << "}\n";
    return exact;
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

template <int T>
void time_width(const ninfer::Weight& weight, const std::byte* raw, std::int64_t raw_row_bytes,
                int k, const void* activation, std::uint64_t begin_row, int rows,
                cudaStream_t stream) {
    const auto output = alloc_output(false, std::size_t(rows) * T);
    const auto store = make_output(output, false, rows, false);
    const auto baseline = [&] {
        ninfer::ops::gguf::vector_product(ninfer::ops::gguf::GgmlType::Q4_K,
            raw + std::int64_t(begin_row) * raw_row_bytes, raw_row_bytes, rows, k,
            activation, T, store, stream);
    };
    const auto current = [&] { launch_q4<T>(weight, activation, T, store, stream); };
    const auto rooted = [&] { launch_rooted_q4<T>(weight, activation, T, store, stream); };
    const std::array<const char*, 3> names{"baseline", "current", "rooted"};
    const std::array<std::function<void()>, 3> launches{baseline, current, rooted};
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start), "create timing start event");
    check(cudaEventCreate(&stop), "create timing stop event");
    for (int round = 0; round < 5; ++round) {
        for (int rotation = 0; rotation < 3; ++rotation) {
            for (int position = 0; position < 3; ++position) {
                const int mode = (position + rotation) % 3;
                check(cudaEventRecord(start, stream), "record timing start");
                for (int repetition = 0; repetition < 20; ++repetition) { launches[mode](); }
                check(cudaEventRecord(stop, stream), "record timing stop");
                check(cudaEventSynchronize(stop), "synchronize timing stop");
                float milliseconds = 0.0f;
                check(cudaEventElapsedTime(&milliseconds, start, stop), "read timing interval");
                std::cout << "{\"kind\":\"timing\",\"variant\":\"" << names[mode]
                          << "\",\"T\":" << T << ",\"rows\":" << rows
                          << ",\"row_begin\":" << begin_row << ",\"round\":" << round
                          << ",\"rotation\":" << rotation << ",\"launches\":20,\"milliseconds\":"
                          << milliseconds << ",\"milliseconds_per_launch\":" << milliseconds / 20.0f << "}\n";
            }
        }
    }
    check(cudaEventDestroy(stop), "destroy timing stop event");
    check(cudaEventDestroy(start), "destroy timing start event");
    check(cudaFree(output.data), "free timing output");
}

void time_dispatch(const ninfer::Weight& weight, const std::byte* raw, std::int64_t raw_row_bytes,
                   int k, int columns, const void* activation, std::uint64_t begin_row, int rows,
                   cudaStream_t stream) {
    if (columns == 1) { time_width<1>(weight, raw, raw_row_bytes, k, activation, begin_row, rows, stream); return; }
    if (columns == 3) { time_width<3>(weight, raw, raw_row_bytes, k, activation, begin_row, rows, stream); return; }
    if (columns == 4) { time_width<4>(weight, raw, raw_row_bytes, k, activation, begin_row, rows, stream); return; }
    throw std::runtime_error("timing expects captured T1, T3, or T4");
}

void run(const std::string& model, std::uint64_t offset, int n, int k,
         const std::string& identity, const std::string& activation_path,
         std::uint64_t slice_begin, int slice_rows) {
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
    const std::array<std::pair<std::uint64_t, int>, 2> regions{{{0, 10240}, {10240, 6144}}};
    for (const auto [begin, rows] : regions) {
        ninfer::WeightView view;
        view.shape = {std::uint64_t(rows), std::uint64_t(k)};
        view.parts.push_back({&parent, begin * std::uint64_t(k), (begin + rows) * std::uint64_t(k)});
        const auto weight = ninfer::native_weight(view);
        all_exact = unpack_exact(weight, raw.data(), int(begin), 0, rows, raw_row_bytes, stream) && all_exact;
        all_exact = run_rows(weight, int(begin), rows) && all_exact;
        time_dispatch(weight, device_raw, raw_row_bytes, k, activation.columns,
                      device_activation, begin, rows, stream);
    }
    time_dispatch(root_weight, device_raw, raw_row_bytes, k, activation.columns,
                  device_activation, 0, n, stream);
    check(cudaStreamDestroy(stream), "destroy stream");
    check(cudaFree(device_raw), "free original Q4 bytes");
    check(cudaFree(device_activation), "free activation");
    check(cudaFree(device_encoded), "free encoded weight");
    std::cout << "{\"matrix\":\"" << identity << "\",\"N\":" << n << ",\"K\":" << k
              << ",\"columns\":" << activation.columns << ",\"raw_bytes\":" << raw.size()
              << ",\"encoded_bytes\":" << encoded.size() << ",\"saved_bytes\":" << raw.size() - encoded.size()
              << ",\"slice_begin\":" << slice_begin << ",\"slice_rows\":" << slice_rows
              << ",\"all_checks_exact\":" << (all_exact ? "true" : "false") << "}\n";
    if (!all_exact) { throw std::runtime_error("Q4 integrated operator or raw unpack oracle failed"); }
}

}

int main(int argc, char** argv) {
    if (argc != 9) {
        std::cerr << "model offset N K tensor activation slice_begin slice_rows\n";
        return 2;
    }
    try {
        run(argv[1], std::stoull(argv[2]), std::stoi(argv[3]), std::stoi(argv[4]),
            argv[5], argv[6], std::stoull(argv[7]), std::stoi(argv[8]));
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
