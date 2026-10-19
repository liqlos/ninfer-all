#if defined(PAIRED_Q4)
#include "artifact/q4_fields.h"
#include "core/q4_fields.h"
#else
#include "artifact/q6_fields.h"
#include "core/q6_fields.h"
#endif
#include "core/weight_view.h"
#include "ops/linear/gguf/ggml_bridge.h"
#include "ops/linear/gguf/ggml_bridge_internal.cuh"
#include "ops/linear/gguf/q4_fields.cuh"
#include "ops/linear/gguf/q6_fields.cuh"
#include "q4q6_paired_fields.cuh"

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
#if defined(PAIRED_Q4)
constexpr auto kGgmlType = ninfer::ops::gguf::GgmlType::Q4_K;
#else
constexpr auto kGgmlType = ninfer::ops::gguf::GgmlType::Q6_K;
#endif
#if defined(PAIRED_Q4)
using ninfer::ops::gguf::detail::q4_fields::Metadata;
#endif

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
        read_u32(header.data() + 8) != (
#if defined(PAIRED_Q4)
        12
#else
        14
#endif
        ) || read_u32(header.data() + 12) != std::uint32_t(k) ||
        columns <= 0 || columns > 8 || read_u32(header.data() + 20) != 0) {
        throw std::runtime_error("activation snapshot header does not match paired vector geometry");
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

#if defined(PAIRED_Q4)
template <int T>
void launch_fields(const ninfer::Weight& weight, const void* activation, int columns,
                   const VectorOutput& output, cudaStream_t stream) {
    const auto* base = static_cast<const std::uint8_t*>(activation);
    const VecStore store{output.bf16, output.f32, output.column_stride, output.epilogue,
                         output.gate, output.gate_column_stride};
    const VecArgs args{static_cast<const std::uint8_t*>(weight.qdata),
        std::int64_t(weight.k / 256) * 128, weight.n, weight.k,
        reinterpret_cast<const std::int8_t*>(base),
        reinterpret_cast<const half2*>(base + ninfer::ops::gguf::detail::vec_scales_offset(weight.k, columns)), store};
    const ninfer::ops::gguf::detail::q4_fields::Metadata metadata{static_cast<const std::uint8_t*>(weight.scales), weight.q4_fields};
    ninfer::ops::gguf::detail::q4_fields::launch<T>(args, metadata,
        static_cast<const std::uint8_t*>(weight.q4_d), static_cast<const std::uint8_t*>(weight.q4_dmin),
        weight.q4_field_row_begin, stream);
}
#else
template <int T>
void launch_fields(const ninfer::Weight& weight, const void* activation, int columns,
                   const VectorOutput& output, cudaStream_t stream) {
    const auto* base = static_cast<const std::uint8_t*>(activation);
    const VecStore store{output.bf16, output.f32, output.column_stride, output.epilogue,
                         output.gate, output.gate_column_stride};
    const VecArgs args{static_cast<const std::uint8_t*>(weight.qdata),
        std::int64_t(weight.k / 1024) * 832, weight.n, weight.k,
        reinterpret_cast<const std::int8_t*>(base),
        reinterpret_cast<const half2*>(base + ninfer::ops::gguf::detail::vec_scales_offset(weight.k, columns)), store};
    ninfer::ops::gguf::detail::q6_fields::launch<T, false, true>(args,
        static_cast<const std::uint8_t*>(weight.scales), weight.q6_fields.width,
        weight.q6_fields.exponent_base, weight.q6_fields.exponent_bits,
        weight.q6_fields.sign_variable, weight.q6_fields.sign_value,
        std::uint32_t(weight.q6_field_row_begin), stream);
}
#endif

void launch_fields_dispatch(const ninfer::Weight& weight, const void* activation, int columns,
                            const VectorOutput& output, cudaStream_t stream) {
    if (columns == 4) { launch_fields<4>(weight, activation, columns, output, stream); return; }
    throw std::runtime_error("paired fields operator requires T4");
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
    seed_output(baseline, bf16, count, accumulate);
    seed_output(candidate, bf16, count, accumulate);
    const auto baseline_output = make_output(baseline, bf16, rows, accumulate);
    const auto candidate_output = make_output(candidate, bf16, rows, accumulate);
    const auto baseline_launch = [&] {
        ninfer::ops::gguf::vector_product(kGgmlType,
            raw + std::int64_t(begin_row) * raw_row_bytes, raw_row_bytes, rows, k,
            activation, columns, baseline_output, stream);
    };
    const auto candidate_launch = [&] { launch_fields_dispatch(weight, activation, columns, candidate_output, stream); };
    if (graph) {
        for (int mode = 0; mode < 2; ++mode) {
            auto& output = mode == 0 ? baseline : candidate;
            seed_output(output, bf16, count, accumulate);
            cudaGraph_t graph_handle = nullptr;
            check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin graph capture");
            if (mode == 0) { baseline_launch(); } else { candidate_launch(); }
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
        check(cudaStreamSynchronize(stream), "operator synchronize");
    }
    const bool exact = compare_outputs(baseline, candidate);
    check(cudaFree(candidate.data), "free candidate output");
    check(cudaFree(baseline.data), "free baseline output");
    std::cout << "{\"rows\":" << rows << ",\"row_begin\":" << begin_row
              << ",\"columns\":" << columns << ",\"dtype\":\"" << (bf16 ? "bf16" : "fp32")
              << "\",\"epilogue\":\"" << (accumulate ? "accumulate" : "store")
              << "\",\"graph\":" << (graph ? "true" : "false")
              << ",\"bit_exact\":" << (exact ? "true" : "false")
              << ",\"canaries_intact\":" << (exact ? "true" : "false") << "}\n";
    return exact;
}

void launch_fields_split(const ninfer::Weight& weight, const void* activation, int columns,
                         const OutputBuffer& first, const OutputBuffer& second, bool bf16,
                         cudaStream_t stream) {
    const auto* base = static_cast<const std::uint8_t*>(activation);
    const auto first_output = make_output(first, bf16, 10240, false);
    const auto second_output = make_output(second, bf16, 6144, false);
    const VecStore first_store{first_output.bf16, first_output.f32, first_output.column_stride,
                               first_output.epilogue, first_output.gate, first_output.gate_column_stride};
    const VecStore second_store{second_output.bf16, second_output.f32, second_output.column_stride,
                                second_output.epilogue, second_output.gate, second_output.gate_column_stride};
#if defined(PAIRED_Q4)
    constexpr std::int64_t row_bytes = 20 * 128;
    const auto* qdata = static_cast<const std::uint8_t*>(weight.qdata);
    const auto* scales = static_cast<const std::uint8_t*>(weight.scales);
    const auto* d = static_cast<const std::uint8_t*>(weight.q4_d);
    const auto* dmin = static_cast<const std::uint8_t*>(weight.q4_dmin);
    const ninfer::ops::gguf::detail::q4_fields::Metadata first_metadata{scales, weight.q4_fields};
    const ninfer::ops::gguf::detail::q4_fields::Metadata second_metadata{
        scales + std::int64_t(10240) * 20 * 12, weight.q4_fields};
    const VecArgs first_args{qdata, row_bytes, 10240, 5120,
        reinterpret_cast<const std::int8_t*>(base),
        reinterpret_cast<const half2*>(base + ninfer::ops::gguf::detail::vec_scales_offset(5120, columns)),
        first_store};
    const VecArgs second_args{qdata + std::int64_t(10240) * row_bytes, row_bytes, 6144, 5120,
        reinterpret_cast<const std::int8_t*>(base),
        reinterpret_cast<const half2*>(base + ninfer::ops::gguf::detail::vec_scales_offset(5120, columns)),
        second_store};
    ninfer::ops::gguf::detail::q4_fields::launch<4>(first_args, first_metadata, d, dmin, 0, stream);
    ninfer::ops::gguf::detail::q4_fields::launch<4>(second_args, second_metadata, d, dmin, 10240, stream);
#else
    constexpr std::int64_t row_bytes = 5 * 832;
    const auto* qdata = static_cast<const std::uint8_t*>(weight.qdata);
    const auto* d_bits = static_cast<const std::uint8_t*>(weight.scales);
    const VecArgs first_args{qdata, row_bytes, 10240, 5120,
        reinterpret_cast<const std::int8_t*>(base),
        reinterpret_cast<const half2*>(base + ninfer::ops::gguf::detail::vec_scales_offset(5120, columns)),
        first_store};
    const VecArgs second_args{qdata + std::int64_t(10240) * row_bytes, row_bytes, 6144, 5120,
        reinterpret_cast<const std::int8_t*>(base),
        reinterpret_cast<const half2*>(base + ninfer::ops::gguf::detail::vec_scales_offset(5120, columns)),
        second_store};
    ninfer::ops::gguf::detail::q6_regbalance::launch<4, false, true>(first_args, d_bits,
        weight.q6_fields.width, weight.q6_fields.exponent_base, weight.q6_fields.exponent_bits,
        weight.q6_fields.sign_variable, weight.q6_fields.sign_value, 0, stream);
    ninfer::ops::gguf::detail::q6_regbalance::launch<4, false, true>(second_args, d_bits,
        weight.q6_fields.width, weight.q6_fields.exponent_base, weight.q6_fields.exponent_bits,
        weight.q6_fields.sign_variable, weight.q6_fields.sign_value, 10240, stream);
#endif
}

void launch_paired(const ninfer::Weight& weight, const void* activation, int columns,
                   const OutputBuffer& first, const OutputBuffer& second, bool bf16,
                   cudaStream_t stream) {
    const auto* base = static_cast<const std::uint8_t*>(activation);
    const auto first_output = make_output(first, bf16, 10240, false);
    const auto second_output = make_output(second, bf16, 6144, false);
    const VecStore first_store{first_output.bf16, first_output.f32, first_output.column_stride,
                               first_output.epilogue, first_output.gate, first_output.gate_column_stride};
    const VecStore second_store{second_output.bf16, second_output.f32, second_output.column_stride,
                                second_output.epilogue, second_output.gate, second_output.gate_column_stride};
#if defined(PAIRED_Q4)
    constexpr int row_bytes = 20 * 128;
#else
    constexpr int row_bytes = 5 * 832;
#endif
    const VecArgs vector{static_cast<const std::uint8_t*>(weight.qdata), row_bytes, 16384, 5120,
        reinterpret_cast<const std::int8_t*>(base),
        reinterpret_cast<const half2*>(base + ninfer::ops::gguf::detail::vec_scales_offset(5120, columns)),
        {nullptr, nullptr, 16384, Epilogue::Store, nullptr, 0}};
    const ninfer::ops::gguf::detail::paired_split::Args args{vector, first_store, second_store, 10240};
#if defined(PAIRED_Q4)
    const ninfer::ops::gguf::detail::q4_fields::Metadata metadata{
        static_cast<const std::uint8_t*>(weight.scales), weight.q4_fields};
    ninfer::ops::gguf::detail::paired_split::launch_q4<4>(args, metadata,
        static_cast<const std::uint8_t*>(weight.q4_d), static_cast<const std::uint8_t*>(weight.q4_dmin),
        weight.q4_field_row_begin, stream);
#else
    ninfer::ops::gguf::detail::paired_split::launch_q6<4>(args,
        static_cast<const std::uint8_t*>(weight.scales), weight.q6_fields.width,
        weight.q6_fields.exponent_base, weight.q6_fields.exponent_bits,
        weight.q6_fields.sign_variable, weight.q6_fields.sign_value,
        weight.q6_field_row_begin, stream);
#endif
}

bool run_paired_case(const std::array<ninfer::Weight, 3>& weights,
                     const std::array<const std::byte*, 3>& raw_working,
                     std::int64_t raw_row_bytes, int columns, const void* activation,
                     bool bf16, cudaStream_t stream) {
    constexpr int top_rows = 10240;
    constexpr int bottom_rows = 6144;
    constexpr int k = 5120;
    const auto top_count = std::size_t(top_rows) * columns;
    const auto bottom_count = std::size_t(bottom_rows) * columns;
    auto top_raw = alloc_output(bf16, top_count);
    auto bottom_raw = alloc_output(bf16, bottom_count);
    auto top_fields = alloc_output(bf16, top_count);
    auto bottom_fields = alloc_output(bf16, bottom_count);
    auto top_candidate = alloc_output(bf16, top_count);
    auto bottom_candidate = alloc_output(bf16, bottom_count);
    const auto top_raw_output = make_output(top_raw, bf16, top_rows, false);
    const auto bottom_raw_output = make_output(bottom_raw, bf16, bottom_rows, false);
    const auto baseline_raw = [&](int index) {
        ninfer::ops::gguf::vector_product(kGgmlType, raw_working[index], raw_row_bytes, top_rows, k,
            activation, columns, top_raw_output, stream);
        ninfer::ops::gguf::vector_product(kGgmlType, raw_working[index] + std::int64_t(top_rows) * raw_row_bytes,
            raw_row_bytes, bottom_rows, k, activation, columns, bottom_raw_output, stream);
    };
    const auto baseline_fields = [&](int index) {
        launch_fields_split(weights[index], activation, columns, top_fields, bottom_fields, bf16, stream);
    };
    const auto candidate_launch = [&](int index) {
        launch_paired(weights[index], activation, columns, top_candidate, bottom_candidate, bf16, stream);
    };
    baseline_raw(0);
    baseline_fields(0);
    candidate_launch(0);
    check(cudaStreamSynchronize(stream), "paired direct synchronization");
    const bool raw_vs_fields = compare_outputs(top_raw, top_fields) && compare_outputs(bottom_raw, bottom_fields);
    const bool fields_vs_candidate = compare_outputs(top_fields, top_candidate) && compare_outputs(bottom_fields, bottom_candidate);
    const bool raw_vs_candidate = compare_outputs(top_raw, top_candidate) && compare_outputs(bottom_raw, bottom_candidate);
    const auto run_arm = [&](int arm, int index) {
        if (arm == 0) { baseline_raw(index); }
        else if (arm == 1) { baseline_fields(index); }
        else { candidate_launch(index); }
    };
    std::array<std::array<cudaGraphExec_t, 3>, 3> graphs{};
    for (int arm = 0; arm < 3; ++arm) {
        for (int index = 0; index < 3; ++index) {
            cudaGraph_t graph = nullptr;
            check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "paired graph capture begin");
            run_arm(arm, index);
            check(cudaStreamEndCapture(stream, &graph), "paired graph capture end");
            check(cudaGraphInstantiate(&graphs[arm][index], graph, nullptr, nullptr, 0), "paired graph instantiate");
            check(cudaGraphDestroy(graph), "paired graph destroy");
        }
    }
    bool graph_exact = raw_vs_fields && fields_vs_candidate && raw_vs_candidate;
    for (int index = 0; index < 3; ++index) {
        std::array<std::vector<std::byte>, 6> values{
            std::vector<std::byte>(top_raw.bytes), std::vector<std::byte>(bottom_raw.bytes),
            std::vector<std::byte>(top_fields.bytes), std::vector<std::byte>(bottom_fields.bytes),
            std::vector<std::byte>(top_candidate.bytes), std::vector<std::byte>(bottom_candidate.bytes)};
        for (int arm = 0; arm < 3; ++arm) {
            check(cudaGraphLaunch(graphs[arm][index], stream), "paired graph replay");
            check(cudaStreamSynchronize(stream), "paired graph synchronization");
            OutputBuffer* outputs[6]{&top_raw, &bottom_raw, &top_fields, &bottom_fields, &top_candidate, &bottom_candidate};
            const int first = arm * 2;
            check(cudaMemcpy(values[first].data(), outputs[first]->data, outputs[first]->bytes, cudaMemcpyDeviceToHost), "paired graph top readback");
            check(cudaMemcpy(values[first + 1].data(), outputs[first + 1]->data, outputs[first + 1]->bytes, cudaMemcpyDeviceToHost), "paired graph bottom readback");
            graph_exact = graph_exact && canary_ok(*outputs[first]) && canary_ok(*outputs[first + 1]);
        }
        graph_exact = graph_exact && values[0] == values[2] && values[0] == values[4] &&
                      values[1] == values[3] && values[1] == values[5];
    }
    for (int warmup = 0; warmup < 5; ++warmup) {
        for (int arm = 0; arm < 3; ++arm) {
            for (int index = 0; index < 3; ++index) {
                check(cudaGraphLaunch(graphs[arm][index], stream), "paired graph warmup");
            }
        }
    }
    check(cudaStreamSynchronize(stream), "paired graph warmup sync");
    cudaEvent_t start = nullptr, stop = nullptr;
    check(cudaEventCreate(&start), "paired benchmark start event");
    check(cudaEventCreate(&stop), "paired benchmark stop event");
    std::array<std::array<float, 5>, 3> samples{};
    for (int round = 0; round < 5; ++round) {
        for (int position = 0; position < 3; ++position) {
            const int arm = round % 2 == 0 ? position : 2 - position;
            check(cudaEventRecord(start, stream), "paired sample start");
            for (int repeat = 0; repeat < 12; ++repeat) {
                const int index = (repeat + round) % 3;
                check(cudaGraphLaunch(graphs[arm][index], stream), "paired rotating graph replay");
            }
            check(cudaEventRecord(stop, stream), "paired sample stop");
            check(cudaEventSynchronize(stop), "paired sample synchronize");
            float elapsed = 0.0f;
            check(cudaEventElapsedTime(&elapsed, start, stop), "paired sample elapsed");
            samples[arm][round] = elapsed / 12.0f;
        }
    }
    const bool guards_exact = canary_ok(top_raw) && canary_ok(bottom_raw) && canary_ok(top_fields) &&
                              canary_ok(bottom_fields) && canary_ok(top_candidate) && canary_ok(bottom_candidate);
    std::cout << "{\"kind\":\"paired_split\",\"dtype\":\"" << (bf16 ? "bf16" : "fp32")
              << "\",\"raw_vs_fields_direct_exact\":" << (raw_vs_fields ? "true" : "false")
              << ",\"fields_vs_candidate_direct_exact\":" << (fields_vs_candidate ? "true" : "false")
              << ",\"raw_vs_candidate_direct_exact\":" << (raw_vs_candidate ? "true" : "false")
              << ",\"all_arms_graph_exact\":" << (graph_exact ? "true" : "false")
              << ",\"all_output_canaries_exact\":" << (guards_exact ? "true" : "false")
              << ",\"raw_pair_ms\":[";
    for (int round = 0; round < 5; ++round) { if (round) { std::cout << ','; } std::cout << samples[0][round]; }
    std::cout << "],\"accepted_fields_pair_ms\":[";
    for (int round = 0; round < 5; ++round) { if (round) { std::cout << ','; } std::cout << samples[1][round]; }
    std::cout << "],\"one_paired_kernel_ms\":[";
    for (int round = 0; round < 5; ++round) { if (round) { std::cout << ','; } std::cout << samples[2][round]; }
    std::cout << "]}\n";
    for (int arm = 0; arm < 3; ++arm) {
        for (int index = 0; index < 3; ++index) { check(cudaGraphExecDestroy(graphs[arm][index]), "paired graph executable destroy"); }
    }
    check(cudaEventDestroy(start), "paired event destroy");
    check(cudaEventDestroy(stop), "paired event destroy");
    check(cudaFree(top_raw.data), "paired output free");
    check(cudaFree(bottom_raw.data), "paired output free");
    check(cudaFree(top_fields.data), "paired output free");
    check(cudaFree(bottom_fields.data), "paired output free");
    check(cudaFree(top_candidate.data), "paired output free");
    check(cudaFree(bottom_candidate.data), "paired output free");
    return raw_vs_fields && fields_vs_candidate && raw_vs_candidate && graph_exact && guards_exact;
}

bool unpack_exact(const ninfer::Weight& weight, const std::byte* raw, int expected_begin, int first_row, int rows,
                  std::int64_t raw_row_bytes, cudaStream_t stream) {
    const auto bytes = std::size_t(rows) * std::size_t(raw_row_bytes);
    auto* output = device_alloc<std::byte>(bytes + 64);
    std::array<std::byte, 64> guard;
    guard.fill(std::byte{0xD3});
    check(cudaMemcpy(output + bytes, guard.data(), guard.size(), cudaMemcpyHostToDevice), "initialize unpack canary");
#if defined(PAIRED_Q4)
    ninfer::unpack_q4_fields(weight, first_row, rows, output, stream);
#else
    ninfer::unpack_q6_fields(weight, first_row, rows, output, stream);
#endif
    check(cudaStreamSynchronize(stream), "unpack synchronize");
    std::vector<std::byte> decoded(bytes);
    check(cudaMemcpy(decoded.data(), output, bytes, cudaMemcpyDeviceToHost), "copy unpacked bytes");
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
         std::uint64_t slice_begin, int slice_rows, bool scan_only) {
    #if defined(PAIRED_Q4)
    const std::int64_t raw_row_bytes = std::int64_t(k / 256) * 144;
    constexpr auto qtype = ninfer::QType::GGUF_Q4_K;
#else
    const std::int64_t raw_row_bytes = std::int64_t(k / 256) * 210;
    constexpr auto qtype = ninfer::QType::GGUF_Q6_K;
#endif
    auto raw = read_model_bytes(model, offset, std::uint64_t(n) * raw_row_bytes);
    const std::array<std::uint64_t, 2> shape{std::uint64_t(n), std::uint64_t(k)};
    const auto stored = ninfer::weight_geometry(qtype, ninfer::QuantLayout::GgufBlocks, shape);
#if defined(PAIRED_Q4)
    const auto geometry = ninfer::artifact::describe_q4_fields(stored, raw);
    if (!geometry) { throw std::runtime_error("actual artifact describe_q4_fields rejected tensor"); }
    auto encoded = ninfer::artifact::encode_q4_fields(*geometry, raw);
#else
    const auto geometry = ninfer::artifact::describe_q6_fields(stored, raw);
    if (!geometry) { throw std::runtime_error("actual artifact describe_q6_fields rejected tensor"); }
    auto encoded = ninfer::artifact::encode_q6_fields(*geometry, raw);
#endif
    if (encoded.size() != geometry->bytes) { throw std::runtime_error("encoded byte count differs from geometry"); }
    if (scan_only) {
        std::cout << "{\"kind\":\"cpu_scan\",\"matrix\":\"" << identity
                  << "\",\"N\":" << n << ",\"K\":" << k
                  << ",\"raw_bytes\":" << raw.size() << ",\"encoded_bytes\":" << encoded.size()
                  << ",\"encoder_succeeded\":true,\"gpu_executed\":false}\n";
        return;
    }
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
    check(cudaMemcpy(device_raw, raw.data(), raw.size(), cudaMemcpyHostToDevice), "upload original GGUF bytes");
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
    if (n != 16384 || k != 5120 || activation.columns != 4) {
        throw std::runtime_error("paired split test requires the captured 16384x5120 T4 parent");
    }
    std::array<std::byte*, 3> pair_raw{};
    std::array<std::byte*, 3> pair_encoded{};
    std::array<const std::byte*, 3> pair_raw_const{};
    std::array<ninfer::WeightParent, 3> pair_parents{};
    std::array<ninfer::WeightView, 3> pair_views{};
    std::array<ninfer::Weight, 3> pair_weights{};
    pair_raw[0] = device_raw;
    pair_encoded[0] = device_encoded;
    pair_raw_const[0] = device_raw;
    pair_weights[0] = root_weight;
    for (int index = 1; index < 3; ++index) {
        pair_raw[index] = device_alloc<std::byte>(raw.size());
        check(cudaMemcpy(pair_raw[index], raw.data(), raw.size(), cudaMemcpyHostToDevice), "upload rotating raw weight");
        pair_encoded[index] = device_alloc<std::byte>(encoded.size());
        check(cudaMemcpy(pair_encoded[index], encoded.data(), encoded.size(), cudaMemcpyHostToDevice), "upload rotating encoded weight");
        pair_raw_const[index] = pair_raw[index];
        pair_parents[index].geometry = *geometry;
        pair_parents[index].data = pair_encoded[index];
        pair_views[index].shape = {std::uint64_t(n), std::uint64_t(k)};
        pair_views[index].parts.push_back({&pair_parents[index], 0, std::uint64_t(n) * k});
        pair_weights[index] = ninfer::native_weight(pair_views[index]);
    }
    const bool pair_fp32_exact = run_paired_case(pair_weights, pair_raw_const, raw_row_bytes,
        activation.columns, device_activation, false, stream);
    const bool pair_bf16_exact = run_paired_case(pair_weights, pair_raw_const, raw_row_bytes,
        activation.columns, device_activation, true, stream);
    all_exact = pair_fp32_exact && pair_bf16_exact && all_exact;
    for (int index = 1; index < 3; ++index) {
        check(cudaFree(pair_raw[index]), "free rotating raw weight");
        check(cudaFree(pair_encoded[index]), "free rotating encoded weight");
    }
    if (slice_begin + std::uint64_t(slice_rows) > std::uint64_t(n)) { throw std::runtime_error("slice exceeds matrix rows"); }
    ninfer::WeightView slice_view;
    slice_view.shape = {std::uint64_t(slice_rows), std::uint64_t(k)};
    slice_view.parts.push_back({&parent, slice_begin * std::uint64_t(k),
        (slice_begin + std::uint64_t(slice_rows)) * std::uint64_t(k)});
    const auto slice_weight = ninfer::native_weight(slice_view);
    all_exact = unpack_exact(slice_weight, raw.data(), int(slice_begin), 0, slice_rows, raw_row_bytes, stream) && all_exact;
    for (const int rows : {2048, 4096}) {
        if (rows <= n) {
            ninfer::WeightView view;
            const auto begin = std::uint64_t((n / 2 / rows) * rows);
            view.shape = {std::uint64_t(rows), std::uint64_t(k)};
            view.parts.push_back({&parent, begin * std::uint64_t(k), (begin + rows) * std::uint64_t(k)});
            const auto weight = ninfer::native_weight(view);
            all_exact = unpack_exact(weight, raw.data(), int(begin), 0, rows, raw_row_bytes, stream) && all_exact;
            all_exact = run_rows(weight, int(begin), rows) && all_exact;
        }
    }
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
    if (argc != 10) {
        std::cerr << "model offset N K tensor activation slice_begin slice_rows scan|bench\n";
        return 2;
    }
    try {
        run(argv[1], std::stoull(argv[2]), std::stoi(argv[3]), std::stoi(argv[4]),
            argv[5], argv[6], std::stoull(argv[7]), std::stoi(argv[8]), std::string(argv[9]) == "scan");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
