#include "probe_api.h"
#include "host_receipt.h"
#include "ops/linear/gguf/ggml_bridge_internal.cuh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "host_support.cuh"

namespace q4tc_native {
#include "native_producer.cuh"
}

namespace {

using ninfer::ops::gguf::detail::check;
constexpr int kK = 5120;
constexpr int kColumns = 4;
constexpr int kRepeats = 32;
constexpr std::size_t kGuard = 64;
constexpr std::size_t kRowBytes = 2880;
constexpr std::size_t kScaleOffset = kK * kColumns;
constexpr std::size_t kPackedBytes = kScaleOffset + (kK / 32) * kColumns * sizeof(half2);
constexpr std::size_t kFixtureBytes = 100270080;
constexpr const char* kFixtureSHA = "be10db58b96b5614830c01afe1e28f9d55c4581a6c481827fa7b9d885a6e2790";
bool capture_launches = false;
struct LaunchRecord {
    cudaKernel_t kernel;
    dim3 grid;
    dim3 block;
    std::size_t shared;
};
std::array<LaunchRecord, 8> launches{};
std::size_t launch_count = 0;

#include "host_common.cuh"
#include "host_guarded.cuh"
#include "host_samples.cuh"
#include "host_input.cuh"
#include "host_sha256.cuh"
#include "host_q4_oracle.cuh"

static_assert(sizeof(block_q8_1_mmq) == 144);
static_assert(kScaleOffset == ninfer::ops::gguf::detail::vec_scales_offset(kK, kColumns));
static_assert(sizeof(Q4TCArgs) == 72 && alignof(Q4TCArgs) == 8);

std::vector<std::uint8_t> read_fixture(const std::filesystem::path& root) {
    const auto path = root / "operators/q4-full-k5120.bin";
    require(std::filesystem::is_regular_file(path) && std::filesystem::file_size(path) == kFixtureBytes,
            "the pinned full Q4 K5120 fixture is required");
    std::vector<std::uint8_t> result(kFixtureBytes);
    std::ifstream input(path, std::ios::binary);
    input.read(reinterpret_cast<char*>(result.data()), std::streamsize(result.size()));
    require(input.gcount() == std::streamsize(result.size()), "incomplete full Q4 fixture read");
    require(sha256(result.data(), result.size()) == kFixtureSHA, "full Q4 fixture SHA256 differs");
    return result;
}

struct Activation {
    std::vector<__nv_bfloat16> input;
    Guarded source{kK * kColumns * sizeof(__nv_bfloat16)};
    Guarded packed{kPackedBytes};
    std::vector<std::uint8_t> host;
    explicit Activation(int rows) : input(make_input(kK, kColumns, 0x4a119U + rows)) {
        for (int i = 0; i < 32; ++i) {
            input[i] = __float2bfloat16(0.0f);
            input[kK + i] = __float2bfloat16(i & 1 ? -0.25f : 0.25f);
            input[2 * kK + i] = __float2bfloat16(0.03125f);
            input[3 * kK + i] = __float2bfloat16(float(i - 16) / 128.0f);
        }
        source.from_host(input.data(), source.bytes);
        prepare(nullptr);
        check(cudaDeviceSynchronize(), "initial native activation pack");
        host = packed.read();
        constexpr double epsilon = 8.0 * std::numeric_limits<float>::epsilon();
        for (int col = 0; col < kColumns; ++col) {
            for (int stripe = 0; stripe < kK / 32; ++stripe) {
                double maximum = 0, sum = 0, l1 = 0;
                for (int i = 0; i < 32; ++i) {
                    const double x = __bfloat162float(input[col * kK + stripe * 32 + i]);
                    maximum = std::max(maximum, std::abs(x));
                    sum += x;
                    l1 += std::abs(x);
                }
                const double expected_d = maximum / 127.0;
                const auto ds = load<half2>(host.data() + kScaleOffset + (col * (kK / 32) + stripe) * 4);
                const double d = __low2float(ds), stored_sum = __high2float(ds);
                const double d_round = std::abs(double(__half2float(__float2half_rn(float(expected_d)))) - expected_d);
                const double s_round = std::abs(double(__half2float(__float2half_rn(float(sum)))) - sum);
                require(std::isfinite(d) && d >= 0 && std::abs(d - expected_d) <= d_round + epsilon * expected_d,
                        "native Q8 scale differs from represented BF16 input");
                require(std::isfinite(stored_sum) && std::abs(stored_sum - sum) <= s_round + epsilon * l1,
                        "native Q8 sum differs from represented BF16 input");
                for (int i = 0; i < 32; ++i) {
                    const int q = load<std::int8_t>(host.data() + col * kK + stripe * 32 + i);
                    const double x = __bfloat162float(input[col * kK + stripe * 32 + i]);
                    require(q >= -127 && q <= 127 && (maximum != 0 || q == 0) &&
                            std::abs(d * q - x) <= 0.5 * expected_d + 127 * d_round + epsilon * maximum,
                            "native Q8 quant differs from represented BF16 input");
                }
            }
        }
        unchanged();
    }
    const std::int8_t* qs() const { return static_cast<const std::int8_t*>(packed.data); }
    const void* ds() const { return static_cast<const std::uint8_t*>(packed.data) + kScaleOffset; }
    void prepare(cudaStream_t stream) {
        q4tc_native::quantize_vector_kernel<<<dim3(kK / 256, kColumns), 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(source.data), nullptr,
            static_cast<std::int8_t*>(packed.data),
            reinterpret_cast<half2*>(static_cast<std::uint8_t*>(packed.data) + kScaleOffset), kK);
        check(cudaGetLastError(), "native BF16 to Q8 helperpack");
    }
    void unchanged() const {
        const auto represented = source.read();
        require(represented.size() == input.size() * sizeof(input[0]) &&
                std::memcmp(represented.data(), input.data(), represented.size()) == 0,
                "consumer changed represented input");
        require(packed.read() == host, "consumer changed common packed activation");
        source.guards();
        packed.guards();
    }
    std::vector<block_q8_1_mmq> oracle_input() const {
        std::vector<block_q8_1_mmq> result((kK / 128) * kColumns);
        for (int block = 0; block < kK / 128; ++block) {
            for (int col = 0; col < kColumns; ++col) {
                auto& out = result[block * kColumns + col];
                std::memcpy(out.qs, host.data() + col * kK + block * 128, 128);
                std::memcpy(out.ds4, host.data() + kScaleOffset + (col * (kK / 32) + block * 4) * 4, 16);
            }
        }
        return result;
    }
};

struct Case {
    int rows;
    int columns = kColumns;
    int stride;
    const Guarded& weight;
    const std::vector<std::uint8_t>& original;
    Activation activation;
    Guarded plain_a, plain_b, bf_a, bf_b, seed;
    std::vector<std::uint8_t> fp32, carry, stored;
    Case(int n, const Guarded& w, const std::vector<std::uint8_t>& raw)
        : rows(n), stride(n + 16), weight(w), original(raw), activation(n),
          plain_a(count() * 4), plain_b(count() * 4), bf_a(count() * 2),
          bf_b(count() * 2), seed(count() * 2) {}
    std::size_t count() const { return std::size_t(stride) * columns; }
    Guarded& output(Q4TCOutput mode, Q4TCRoute route) {
        return mode == Q4TCOutput::PlainFP32
            ? (route == Q4TCRoute::NativeW ? plain_a : plain_b)
            : (route == Q4TCRoute::NativeW ? bf_a : bf_b);
    }
    Q4TCArgs args(Q4TCOutput mode, Q4TCRoute route) {
        return {static_cast<const std::uint8_t*>(weight.data), kRowBytes, rows, kK, columns,
                activation.qs(), activation.ds(), output(mode, route).data, stride, mode};
    }
    void guards() const {
        plain_a.guards(); plain_b.guards(); bf_a.guards(); bf_b.guards(); seed.guards();
        weight.guards();
        activation.unchanged();
    }
};

#include "host_output.cuh"

std::string oracle(const Case& c, const std::vector<std::uint8_t>& actual) {
    const auto activation = c.activation.oracle_input();
    double maximum_absolute = 0, maximum_fraction = 0;
    int count = 0;
    for (int col = 0; col < kColumns; ++col) {
        for (int row : sample_indices(c.rows, 16)) {
            const double reference = q4k_q8_1_dot(c.original.data() + std::size_t(row) * kRowBytes,
                                                  activation.data(), col, kColumns);
            const double got = load<float>(actual.data() + (std::size_t(col) * c.stride + row) * 4);
            const double error = std::abs(got - reference), bound = 0.02 + 2e-4 * std::abs(reference);
            require(std::isfinite(reference) && std::isfinite(got) && error <= bound,
                    "Q4 result differs from independent original-byte FP64 oracle");
            maximum_absolute = std::max(maximum_absolute, error);
            maximum_fraction = std::max(maximum_fraction, error / bound);
            ++count;
        }
    }
    require(count == 64, "expected 64 independent oracle samples per arm");
    Json result;
    result.add("samples", count); result.add("maximum_absolute_error", maximum_absolute);
    result.add("maximum_bound_fraction", maximum_fraction);
    result.add("minimum_correction", "original half dmin, 6-bit minima and native stored half input sum");
    result.add("planar_to_oracle_adapter", "CPU literal copy only; never used for GPU preparation");
    return result.finish();
}

std::string semantic(const Case& c, Q4TCOutput mode, const std::vector<std::uint8_t>& actual, int calls) {
    std::size_t sensitive = 0, even = 0, odd = 0, adjacent = 0;
    for (int col = 0; col < c.columns; ++col) {
        for (int row = 0; row < c.rows; ++row) {
            const auto at = std::size_t(col) * c.stride + row;
            const float value = load<float>(c.fp32.data() + at * 4);
            require(std::isfinite(value), "nonfinite projection output");
            std::uint16_t expected = load<std::uint16_t>(c.carry.data() + at * 2);
            float before = value;
            for (int i = 0; i < calls; ++i) {
                before = mode == Q4TCOutput::AccumulateBF16 ? value + bf16_float(expected) : value;
                expected = rounded_bf16(before);
            }
            require(load<std::uint16_t>(actual.data() + at * 2) == expected &&
                    std::isfinite(bf16_float(expected)), "BF16 semantic boundary or evolving carry differs");
            const auto bits = load<std::uint32_t>(reinterpret_cast<const std::uint8_t*>(&before));
            const auto low = bits & 65535U;
            if (low == 32768) { ((bits >> 16) & 1 ? odd : even)++; }
            adjacent += low == 32767 || low == 32769;
            sensitive += expected != rounded_bf16(value);
        }
    }
    if (mode == Q4TCOutput::AccumulateBF16) { require(sensitive > 0, "carry was not exercised"); }
    output_padding(c, actual, 2);
    Json result;
    result.add("elements", c.rows * c.columns); result.add("calls", calls);
    result.add("carry_sensitive", sensitive); result.add("observed_final_halfway_even", even);
    result.add("observed_final_halfway_odd", odd); result.add("observed_final_adjacent_halfway", adjacent);
    result.add("host_rounding_boundary_selftests", 12);
    return result.finish();
}

std::string diagnostic(Case& c, Q4TCOutput mode, Q4TCRoute route) {
    launch_count = 0;
    capture_launches = true;
    const bool accepted = q4tc_run(c.args(mode, route), route, nullptr);
    capture_launches = false;
    require(accepted, "positive Q4 dispatch rejected");
    check(cudaDeviceSynchronize(), "diagnostic Q4 projection");
    require(launch_count == 1, "Q4 route must launch exactly one projection kernel");
    const auto actual = launches[0];
    const char* raw_name = nullptr;
    check(cudaFuncGetName(&raw_name, reinterpret_cast<const void*>(actual.kernel)), "diagnostic kernel name");
    require(raw_name, "missing diagnostic kernel name");
    const std::string name(raw_name);
    const char* expected = route == Q4TCRoute::Chain ? "q4_t4_chain_kernel" :
        "kernelIL9ggml_type12ELi4ELb0ELi5120ELb1ELb0ELi2EE";
    require(name.find(expected) != std::string::npos, "selected route differs from pinned current W/Chain");
    require(actual.grid.x > 0 && actual.grid.y == 1 && actual.grid.z == 1 &&
            actual.block.x == 128 && actual.block.y == 1 && actual.block.z == 1 && actual.shared == 0,
            "unexpected projection launch geometry");
    if (route == Q4TCRoute::Chain) {
        require(actual.grid.x == unsigned(c.rows / 16), "Chain CTA ownership differs");
    }
    Json record;
    record.add("kernel", name); record.add("grid_x", actual.grid.x);
    record.add("block_x", actual.block.x); record.add("dynamic_shared_bytes", actual.shared);
    record.add("outside_timing", true);
    return record.finish();
}

struct Graph {
    cudaStream_t stream{};
    cudaGraph_t graph{};
    cudaGraphExec_t executable{};
    Graph() { check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create graph stream"); }
    ~Graph() {
        if (executable) { cudaGraphExecDestroy(executable); }
        if (graph) { cudaGraphDestroy(graph); }
        if (stream) { cudaStreamDestroy(stream); }
    }
};

void evolving(Case& c) {
    std::vector<std::uint8_t> reference;
    for (auto route : {Q4TCRoute::NativeW, Q4TCRoute::Chain}) {
        auto& output = c.output(Q4TCOutput::AccumulateBF16, route);
        output.reset_from(c.seed);
        for (int i = 0; i < kRepeats; ++i) {
            require(q4tc_run(c.args(Q4TCOutput::AccumulateBF16, route), route, nullptr),
                    "evolving direct dispatch rejected");
        }
        check(cudaDeviceSynchronize(), "complete 32 evolving direct calls");
        const auto direct = output.read();
        semantic(c, Q4TCOutput::AccumulateBF16, direct, kRepeats);
        if (route == Q4TCRoute::NativeW) { reference = direct; }
        else { require(direct == reference, "32-call accumulated outputs are not bit exact"); }
        Graph graph;
        check(cudaStreamBeginCapture(graph.stream, cudaStreamCaptureModeThreadLocal), "begin graph capture");
        for (int i = 0; i < 4; ++i) {
            require(q4tc_run(c.args(Q4TCOutput::AccumulateBF16, route), route, graph.stream),
                    "graph capture dispatch rejected");
        }
        check(cudaStreamEndCapture(graph.stream, &graph.graph), "end graph capture");
        std::size_t node_count = 0;
        check(cudaGraphGetNodes(graph.graph, nullptr, &node_count), "count captured graph nodes");
        require(node_count == 4, "expected four projection graph nodes");
        check(cudaGraphInstantiate(&graph.executable, graph.graph, nullptr, nullptr, 0), "instantiate replay");
        output.reset_from(c.seed);
        for (int i = 0; i < 8; ++i) { check(cudaGraphLaunch(graph.executable, graph.stream), "replay graph"); }
        check(cudaStreamSynchronize(graph.stream), "complete evolving graph replay");
        const auto replay = output.read();
        require(replay == direct, "graph replay differs from direct evolving carry");
        semantic(c, Q4TCOutput::AccumulateBF16, replay, kRepeats);
    }
    c.guards();
}

void qualify(Case& c) {
    const auto initial = initial_output(c, true);
    c.carry = initial_output(c, false);
    c.seed.from_host(c.carry.data(), c.carry.size());
    c.plain_a.from_host(initial.data(), initial.size());
    c.plain_b.from_host(initial.data(), initial.size());
    Json record;
    record.add("event", "qualification"); record.add("N", c.rows);
    record.raw("native_plain_route", diagnostic(c, Q4TCOutput::PlainFP32, Q4TCRoute::NativeW));
    record.raw("chain_plain_route", diagnostic(c, Q4TCOutput::PlainFP32, Q4TCRoute::Chain));
    c.fp32 = c.plain_a.read();
    const auto alternate = c.plain_b.read();
    require(c.fp32 == alternate, "whole FP32 native/Chain comparison differs");
    output_padding(c, c.fp32, 4);
    record.raw("native_fp64_oracle", oracle(c, c.fp32));
    record.raw("chain_fp64_oracle", oracle(c, alternate));
    for (auto mode : {Q4TCOutput::StoreBF16, Q4TCOutput::AccumulateBF16}) {
        c.bf_a.reset_from(c.seed); c.bf_b.reset_from(c.seed);
        const bool store_mode = mode == Q4TCOutput::StoreBF16;
        record.raw(store_mode ? "native_store_route" : "native_accumulate_route",
                   diagnostic(c, mode, Q4TCRoute::NativeW));
        record.raw(store_mode ? "chain_store_route" : "chain_accumulate_route",
                   diagnostic(c, mode, Q4TCRoute::Chain));
        const auto a = c.bf_a.read(), b = c.bf_b.read();
        require(a == b, "whole BF16 native/Chain comparison differs");
        record.raw(store_mode ? "store_semantics" : "accumulate_semantics", semantic(c, mode, a, 1));
        semantic(c, mode, b, 1);
        if (store_mode) { c.stored = a; }
    }
    evolving(c);
    record.add("whole_modes_bit_exact", 3);
    record.add("direct_accumulate_calls", 32); record.add("graph_nodes", 4); record.add("graph_replays", 8);
    record.add("input_sha256", sha256(reinterpret_cast<const std::uint8_t*>(c.activation.input.data()),
                                    c.activation.source.bytes));
    record.add("packed_sha256", sha256(c.activation.host.data(), c.activation.host.size()));
    record.add("canaries_and_inputs_unchanged", true); record.add("status", "PASS");
    emit(record);
}

}

#include "host_launch_wrapper.cuh"

namespace {

void api_guards(Case& c) {
    int count = 0;
    std::ostringstream names;
    names << '[';
    const auto reject = [&](const char* name, Q4TCRoute route, auto edit) {
        auto args = c.args(Q4TCOutput::PlainFP32, route);
        edit(args);
        capture_launches = true;
        launch_count = 0;
        const bool accepted = q4tc_run(args, route, nullptr);
        capture_launches = false;
        require(!accepted && launch_count == 0, std::string("rejection failed: ") + name);
        names << (count++ ? "," : "") << quoted(name);
    };
    const auto old_a = c.plain_a.read(), old_b = c.plain_b.read();
    for (auto route : {Q4TCRoute::NativeW, Q4TCRoute::Chain}) {
        reject("zero_rows", route, [](auto& a) { a.rows = 0; });
        reject("negative_rows", route, [](auto& a) { a.rows = -1; });
        reject("unsupported_K", route, [](auto& a) { a.k = 5119; });
        reject("unsupported_T", route, [](auto& a) { a.columns = 3; });
        reject("row_bytes", route, [](auto& a) { --a.row_bytes; });
        reject("output_stride", route, [](auto& a) { a.output_column_stride = a.rows - 1; });
        reject("null_weight", route, [](auto& a) { a.weight = nullptr; });
        reject("null_qs", route, [](auto& a) { a.qs = nullptr; });
        reject("null_ds", route, [](auto& a) { a.ds = nullptr; });
        reject("null_output", route, [](auto& a) { a.output = nullptr; });
        reject("weight_alignment", route, [](auto& a) { ++a.weight; });
        reject("qs_alignment", route, [](auto& a) { ++a.qs; });
        reject("ds_alignment", route, [](auto& a) { a.ds = static_cast<const std::uint8_t*>(a.ds) + 1; });
        reject("fp32_alignment", route, [](auto& a) { a.output = static_cast<std::uint8_t*>(a.output) + 2; });
        reject("bf16_alignment", route, [](auto& a) {
            a.output_mode = Q4TCOutput::StoreBF16; a.output = static_cast<std::uint8_t*>(a.output) + 1;
        });
        reject("output_mode", route, [](auto& a) { a.output_mode = Q4TCOutput(99); });
    }
    reject("unsupported_chain_rows", Q4TCRoute::Chain, [](auto& a) { a.rows = 6143; });
    reject("N1024_no_chain", Q4TCRoute::Chain, [](auto& a) { a.rows = 1024; });
    reject("unknown_route", Q4TCRoute(99), [](auto&) {});
    require(!q4tc_candidate_shape(1024, kK, kColumns) &&
            q4tc_candidate_shape(34816, kK, kColumns) &&
            !q4tc_candidate_shape(34816, 5119, kColumns) &&
            !q4tc_candidate_shape(34816, kK, 1), "closed candidate shape contract differs");
    require(c.plain_a.read() == old_a && c.plain_b.read() == old_b, "rejected call changed output");
    Case small(1024, c.weight, c.original);
    const auto initial = initial_output(small, true);
    small.plain_a.from_host(initial.data(), initial.size());
    Json record;
    record.add("event", "api_guards");
    record.raw("N1024_native_route", diagnostic(small, Q4TCOutput::PlainFP32, Q4TCRoute::NativeW));
    const auto actual = small.plain_a.read();
    output_padding(small, actual, 4);
    record.raw("N1024_native_oracle", oracle(small, actual));
    for (int col = 0; col < small.columns; ++col) {
        for (int row = 0; row < small.rows; ++row) {
            require(std::isfinite(load<float>(actual.data() + (std::size_t(col) * small.stride + row) * 4)),
                    "N1024 NativeW left a nonfinite output");
        }
    }
    names << ']';
    c.guards(); small.guards();
    record.add("rejection_count", count); record.raw("rejections", names.str());
    record.add("rejects_without_launch", true); record.add("N1024_native_only", true);
    record.add("buffer_extents_and_disjointness", "caller-owned; API has no byte extent or alias validator");
    record.add("other_devices_or_default_OFF_exercised", false); record.add("status", "PASS");
    emit(record);
}

struct Events {
    cudaEvent_t first{}, last{};
    Events() {
        check(cudaEventCreate(&first), "create timing start");
        check(cudaEventCreate(&last), "create timing stop");
    }
    ~Events() { cudaEventDestroy(first); cudaEventDestroy(last); }
};

double sample(Case& c, bool prepared, Q4TCRoute route, Events& events) {
    require(!capture_launches, "diagnostic route capture leaked into timing");
    auto& output = c.output(Q4TCOutput::StoreBF16, route);
    output.reset_from(c.seed);
    const auto args = c.args(Q4TCOutput::StoreBF16, route);
    bool accepted = true;
    check(cudaEventRecord(events.first), "record start event");
    for (int i = 0; i < kRepeats; ++i) {
        if (prepared) { c.activation.prepare(nullptr); }
        accepted = q4tc_run(args, route, nullptr) && accepted;
    }
    check(cudaEventRecord(events.last), "record stop event");
    check(cudaEventSynchronize(events.last), "finish timed sample");
    require(accepted, "timed dispatch rejected");
    float milliseconds = 0;
    check(cudaEventElapsedTime(&milliseconds, events.first, events.last), "read elapsed events");
    require(std::isfinite(milliseconds) && milliseconds > 0, "invalid elapsed event duration");
    require(output.read() == c.stored, "timed Store output differs from qualification");
    output.guards();
    return double(milliseconds) / kRepeats;
}

bool screen(Case& c, bool prepared) {
    Events events;
    const auto route = [](char arm) { return arm == 'A' ? Q4TCRoute::NativeW : Q4TCRoute::Chain; };
    std::ostringstream warmups, blocks;
    warmups << '['; blocks << '[';
    int index = 0;
    for (char arm : std::string("ABAB")) {
        warmups << (index++ ? "," : "") << std::setprecision(12) << sample(c, prepared, route(arm), events);
    }
    warmups << ']';
    double total_a = 0, total_b = 0;
    int positive = 0, block_index = 0;
    for (const char* order : {"ABBA", "BAAB", "ABBA"}) {
        std::array<double, 4> samples{};
        double a = 0, b = 0;
        for (int i = 0; i < 4; ++i) {
            samples[i] = sample(c, prepared, route(order[i]), events);
            (order[i] == 'A' ? a : b) += samples[i] / 2;
        }
        const double gain = 100 * (1 - b / a);
        total_a += a; total_b += b; positive += gain > 0;
        Json block;
        block.add("order", order); block.add("native_ms", a); block.add("chain_ms", b);
        block.add("time_gain_percent", gain);
        std::ostringstream values;
        values << std::setprecision(12) << '[' << samples[0] << ',' << samples[1] << ','
               << samples[2] << ',' << samples[3] << ']';
        block.raw("samples_ms", values.str());
        blocks << (block_index++ ? "," : "") << block.finish();
        c.guards();
    }
    blocks << ']';
    const double gain = 100 * (1 - total_b / total_a);
    const bool primary = c.rows == 34816 && prepared;
    const bool accepted = primary ? gain >= 1.0 && positive >= 2 : gain >= -1.0;
    Json record;
    record.add("event", "screen"); record.add("N", c.rows);
    record.add("mode", "StoreBF16"); record.add("scope", prepared ? "prepared" : "projection_only");
    record.add("primary", primary); record.add("warmup_order", "ABAB");
    record.raw("warmups_ms", warmups.str()); record.raw("blocks", blocks.str());
    record.add("calls_per_sample", kRepeats); record.add("native_mean_ms", total_a / 3);
    record.add("chain_mean_ms", total_b / 3); record.add("mean_time_gain_percent", gain);
    record.add("positive_blocks", positive); record.add("criterion_pass", accepted);
    record.add("CI", false); record.add("fused_gate_up_qualified", false);
    record.add("end_to_end_claim", false);
    emit(record);
    return accepted;
}

}

int main(int argc, char** argv) {
    std::string launch_id;
    try {
        std::filesystem::path fixtures;
        bool correctness_only = false;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--correctness-only") { correctness_only = true; }
            else if ((arg == "--fixtures-root" || arg == "--launch-id") && i + 1 < argc) {
                const std::string value = argv[++i];
                if (arg == "--fixtures-root") { fixtures = value; }
                else { launch_id = value; }
            } else { throw std::runtime_error("usage: probe --fixtures-root DIR --launch-id ID [--correctness-only]"); }
        }
        require(!fixtures.empty() && !launch_id.empty(), "fixtures root and launch id are required");
        rounding_self_test();
        require(sha256(nullptr, 0) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                "SHA256 empty selftest failed");
        const std::array<std::uint8_t, 3> abc{{'a', 'b', 'c'}};
        require(sha256(abc.data(), abc.size()) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
                "SHA256 abc selftest failed");
        require(q4tc_candidate_enabled() && ninfer::ops::gguf::detail::device_facts().cc == 860,
                "enabled Chain object and SM86 required");
        int device = 0;
        cudaDeviceProp properties{};
        check(cudaGetDevice(&device), "query device");
        check(cudaGetDeviceProperties(&properties, device), "query device properties");
        Json start;
        start.add("event", "start"); start.add("launch_id", launch_id); start.add("device", properties.name);
        start.add("cc", 100 * properties.major + 10 * properties.minor);
        start.raw("host_receipt", kQ4TCHostReceipt);
        start.add("static_mechanism_review", "PARTIAL; original resource review is not numerical acceptance");
        emit(start);
        const auto original = read_fixture(fixtures);
        Guarded weight(original.size());
        weight.from_host(original.data(), original.size());
        Json fixture;
        fixture.add("event", "fixture_verified"); fixture.add("bytes", original.size());
        fixture.add("sha256", kFixtureSHA); fixture.add("shared_native_chain_weight_backings", 1);
        fixture.add("runtime_repack", false); emit(fixture);
        std::vector<std::unique_ptr<Case>> cases;
        for (int n : {6144, 17408, 34816}) {
            cases.push_back(std::make_unique<Case>(n, weight, original));
            qualify(*cases.back());
        }
        api_guards(*cases.front());
        require(weight.read() == original, "qualification changed immutable full weights");
        weight.guards();
        Json qualified;
        qualified.add("event", "qualification_complete"); qualified.add("status", "PASS");
        qualified.add("shape_count", 3); qualified.add("mode_case_count", 9);
        qualified.add("timing_started", false); qualified.add("planned_timing_endpoint_count", 6);
        qualified.add("N1024_native_guard", true); qualified.add("all_numerical_and_API_guards", true);
        emit(qualified);
        bool accepted = true;
        int endpoints = 0;
        if (!correctness_only) {
            for (auto& c : cases) {
                for (bool prepared : {false, true}) {
                    accepted = screen(*c, prepared) && accepted;
                    ++endpoints;
                }
            }
            require(weight.read() == original, "screen changed immutable full weights");
            weight.guards();
        }
        Json complete;
        complete.add("event", "complete"); complete.add("launch_id", launch_id);
        complete.add("status", correctness_only ? "CORRECTNESS_ONLY_PASS" : accepted ? "PASS" : "SCREEN_REJECTED");
        complete.add("shape_count", 3); complete.add("mode_case_count", 9);
        complete.add("timing_endpoint_count", endpoints); complete.add("numerical_PASS", true);
        complete.add("screen_accepted", !correctness_only && accepted);
        complete.add("warmup_order", "ABAB"); complete.add("calls_per_sample", 32);
        complete.raw("block_orders", "[\"ABBA\",\"BAAB\",\"ABBA\"]");
        complete.add("CI", false); complete.add("end_to_end_claim", false);
        complete.add("fused_gate_up_qualified", false); complete.add("runtime_integration", false);
        emit(complete);
        return correctness_only || accepted ? 0 : 2;
    } catch (const std::exception& error) {
        capture_launches = false;
        Json failure;
        failure.add("event", "failure"); failure.add("launch_id", launch_id);
        failure.add("status", "FAIL"); failure.add("error", error.what()); emit(failure);
        return 1;
    }
}
