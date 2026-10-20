#include "probe_api.h"
#include "host_receipt.h"
#include "mmq_two_input_native.cuh"
#include "ops/linear/gguf/ggml_bridge_internal.cuh"

#include <algorithm>
#include <array>
#include <climits>
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

namespace two_input_native {
#include "host_native_producer.cuh"
}

namespace {

using ninfer::ops::gguf::detail::check;
constexpr std::size_t kGuard = 64;
constexpr int kColumns = 1280;
constexpr int kRepeats = 8;
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
#include "host_fixture_type.cuh"
#include "host_samples.cuh"
#include "host_input.cuh"
#include "host_sha256.cuh"
#include "host_read_fixture.cuh"
#include "host_events.cuh"
#include "host_half_bound.cuh"

namespace q4_oracle {
constexpr int kK = 5120;
#include "host_q4_oracle.cuh"
}

namespace q6_oracle {
constexpr int kDownK = 17408;
constexpr int kDownRows = 5120;
using q4_oracle::half_value;
#include "host_q6_decoder.cuh"
#include "host_oracle_stats.cuh"
#include "host_down_oracle.cuh"
}

namespace q4_producer_oracle {
constexpr int kDownK = 5120;
#include "host_producer_oracle.cuh"
}

namespace q6_producer_oracle {
constexpr int kDownK = 17408;
#include "host_producer_oracle.cuh"
}

constexpr Fixture kQ6{"q6", "operators/q6-full-k17408.bin", 5120, 17408, 73113600,
    "a381c6be36f572170ccc9c7e4a3f8b25e5061e4eb633649399f3ec9759cb43e2"};
constexpr Fixture kQ4{"q4", "operators/q4-full-k5120.bin", 34816, 5120, 100270080,
    "be10db58b96b5614830c01afe1e28f9d55c4581a6c481827fa7b9d885a6e2790"};

struct Resident {
    Fixture spec;
    std::vector<std::uint8_t> original;
    Guarded weight;
    Resident(const std::filesystem::path& root, Fixture fixture)
        : spec(fixture), original(read_fixture(root, fixture)), weight(original.size()) {
        weight.from_host(original.data(), original.size());
        Json record;
        record.add("event", "fixture_verified"); record.add("family", spec.name);
        record.add("bytes", original.size()); record.add("sha256", spec.sha);
        record.add("synthetic_weight_fallback", false); emit(record);
    }
    void unchanged() const {
        require(weight.read() == original, "consumer changed real weight bytes");
        weight.guards();
    }
};

std::string string_pair(const std::array<std::string, 2>& value) {
    return "[" + quoted(value[0]) + "," + quoted(value[1]) + "]";
}

struct Activation {
    int k, layout;
    std::size_t panel_bytes, sample_stride;
    std::array<std::uint32_t, 2> seeds;
    std::array<std::vector<__nv_bfloat16>, 2> input;
    std::array<std::unique_ptr<Guarded>, 2> source;
    Guarded packed;
    std::vector<std::uint8_t> packed_initial;
    std::array<std::vector<block_q8_1_mmq>, 2> panels;
    std::array<std::string, 2> input_sha, packed_sha;
    std::array<int, 2> producer_groups{};
    Activation(int dimension, int format, std::uint32_t seed)
        : k(dimension), layout(format), panel_bytes(std::size_t(k / 128) * kColumns * 144),
          sample_stride(panel_bytes + 256), seeds{seed, seed + 1}, packed(2 * sample_stride) {
        for (int sample = 0; sample < 2; ++sample) {
            input[sample] = make_input(k, kColumns, seeds[sample]);
            source[sample] = std::make_unique<Guarded>(input[sample].size() * 2);
            source[sample]->from_host(input[sample].data(), source[sample]->bytes);
            input_sha[sample] = sha256(reinterpret_cast<const std::uint8_t*>(input[sample].data()),
                                       source[sample]->bytes);
        }
        require(input_sha[0] != input_sha[1], "the two BF16 inputs must differ");
        prepare();
        check(cudaDeviceSynchronize(), "complete initial native matrix packs");
        packed_initial = packed.read();
        for (int sample = 0; sample < 2; ++sample) {
            const auto* begin = packed_initial.data() + sample * sample_stride;
            panels[sample] = std::vector<block_q8_1_mmq>(panel_bytes / 144);
            std::memcpy(panels[sample].data(), begin, panel_bytes);
            packed_sha[sample] = sha256(begin, panel_bytes);
            producer_groups[sample] = layout == 0
                ? q6_producer_oracle::producer_oracle(layout, input[sample], panels[sample])
                : q4_producer_oracle::producer_oracle(layout, input[sample], panels[sample]);
            require(std::all_of(begin + panel_bytes, begin + sample_stride,
                               [](auto value) { return value == 0xa5; }),
                    "native pack changed sample padding");
        }
        require(packed_sha[0] != packed_sha[1], "the two native packed inputs must differ");
        unchanged();
    }
    void prepare() {
        for (int sample = 0; sample < 2; ++sample) {
            auto* output = reinterpret_cast<block_q8_1_mmq*>(
                static_cast<std::uint8_t*>(packed.data) + sample * sample_stride);
            const auto* values = static_cast<const __nv_bfloat16*>(source[sample]->data);
            if (layout == 0) {
                two_input_native::quantize_matrix_kernel<MMQ_Q8_1_DS_LAYOUT_D4>
                    <<<dim3(kColumns, k / 512, 1), 128>>>(values, nullptr, output, k, kColumns);
            } else {
                two_input_native::quantize_matrix_kernel<MMQ_Q8_1_DS_LAYOUT_DS4>
                    <<<dim3(kColumns, k / 512, 1), 128>>>(values, nullptr, output, k, kColumns);
            }
            check(cudaGetLastError(), "native BF16 matrix pack");
        }
    }
    void unchanged() const {
        for (int sample = 0; sample < 2; ++sample) {
            const auto actual = source[sample]->read();
            require(std::memcmp(actual.data(), input[sample].data(), actual.size()) == 0,
                    "consumer changed represented BF16 input");
            source[sample]->guards();
        }
        require(packed.read() == packed_initial, "consumer changed native packed input or padding");
        packed.guards();
    }
};

struct Case {
    const char* name;
    MmqTwoInputProfile profile;
    Resident& resident;
    int rows, k, layout, column_stride;
    std::size_t row_bytes, weight_offset, sample_stride;
    Activation activation;
    Guarded plain_a, plain_b, bf_a, bf_b, seed;
    std::vector<std::uint8_t> fp32, carry, stored;
    Case(const char* label, MmqTwoInputProfile kind, Resident& backing,
         int row_begin, std::uint32_t input_seed)
        : name(label), profile(kind), resident(backing),
          rows(kind == MmqTwoInputProfile::Q6Down ? 5120 : 17408),
          k(kind == MmqTwoInputProfile::Q6Down ? 17408 : 5120),
          layout(kind == MmqTwoInputProfile::Q6Down ? 0 : 1), column_stride(rows + 16),
          row_bytes(std::size_t(k / 256) * (layout == 0 ? 210 : 144)),
          weight_offset(std::size_t(row_begin) * row_bytes),
          sample_stride(std::size_t(kColumns) * column_stride + 64),
          activation(k, layout, input_seed), plain_a(2 * sample_stride * 4),
          plain_b(2 * sample_stride * 4), bf_a(2 * sample_stride * 2),
          bf_b(2 * sample_stride * 2), seed(2 * sample_stride * 2) {
        require(weight_offset + rows * row_bytes <= resident.original.size(), "weight view exceeds fixture");
        carry = initial(false);
        seed.from_host(carry.data(), carry.size());
    }
    std::size_t at(int sample, int col, int row) const {
        return sample * sample_stride + std::size_t(col) * column_stride + row;
    }
    std::size_t logical_elements() const { return std::size_t(2) * rows * kColumns; }
    std::vector<std::uint8_t> initial(bool plain) const {
        const unsigned width = plain ? 4 : 2;
        std::vector<std::uint8_t> result(2 * sample_stride * width, 0xa5);
        for (int sample = 0; sample < 2; ++sample) {
            for (int col = 0; col < kColumns; ++col) {
                for (int row = 0; row < rows; ++row) {
                    const auto offset = at(sample, col, row) * width;
                    if (plain) { store<std::uint32_t>(result.data() + offset, 0x7fc0a5a5U); }
                    else {
                        const float value = (1 + ((row + 3 * col + 7 * sample) % 17)) / 32.0f;
                        store<std::uint16_t>(result.data() + offset,
                            rounded_bf16((row + col + sample) & 1 ? -value : value));
                    }
                }
            }
        }
        return result;
    }
    Guarded& output(bool candidate, MmqTwoInputEpilogue mode) {
        return mode == MmqTwoInputEpilogue::PlainFp32
            ? (candidate ? plain_b : plain_a) : (candidate ? bf_b : bf_a);
    }
    MmqTwoInputArguments args(bool candidate, MmqTwoInputEpilogue mode) {
        auto& destination = output(candidate, mode);
        return {profile, mode, rows, k, kColumns, layout,
            static_cast<const std::uint8_t*>(resident.weight.data) + weight_offset,
            rows * row_bytes, std::int64_t(row_bytes), activation.packed.data,
            activation.packed.bytes, std::int64_t(activation.sample_stride),
            destination.data, destination.bytes, column_stride, std::int64_t(sample_stride)};
    }
    bool invoke(bool candidate, MmqTwoInputEpilogue mode, bool prepare = true) {
        if (prepare) { activation.prepare(); }
        const auto arguments = args(candidate, mode);
        return candidate ? mmq_two_input_candidate(arguments, nullptr)
                         : mmq_two_input_control(arguments, nullptr);
    }
    void padding(const std::vector<std::uint8_t>& output, unsigned width) const {
        require(output.size() == 2 * sample_stride * width, "output extent differs");
        for (int sample = 0; sample < 2; ++sample) {
            for (int col = 0; col < kColumns; ++col) {
                const auto begin = at(sample, col, rows) * width;
                const auto end = (sample * sample_stride + std::size_t(col + 1) * column_stride) * width;
                require(std::all_of(output.begin() + begin, output.begin() + end,
                                    [](auto value) { return value == 0xa5; }), "output column padding changed");
            }
            const auto begin = (sample * sample_stride + std::size_t(kColumns) * column_stride) * width;
            const auto end = (sample + 1) * sample_stride * width;
            require(std::all_of(output.begin() + begin, output.begin() + end,
                                [](auto value) { return value == 0xa5; }), "output sample padding changed");
        }
    }
    void unchanged() const {
        activation.unchanged(); resident.unchanged();
        plain_a.guards(); plain_b.guards(); bf_a.guards(); bf_b.guards(); seed.guards();
        require(seed.read() == carry, "carry seed changed");
    }
};

std::string oracle(const Case& c, const std::vector<std::uint8_t>& actual) {
    std::ostringstream out;
    out << '[';
    for (int sample = 0; sample < 2; ++sample) {
        int points = 0;
        double maximum_absolute = 0, maximum_fraction = 0;
        if (c.layout == 0) {
            std::vector<float> dense(std::size_t(c.rows) * kColumns);
            for (int col = 0; col < kColumns; ++col) {
                std::memcpy(dense.data() + std::size_t(col) * c.rows,
                            actual.data() + c.at(sample, col, 0) * 4, c.rows * 4);
            }
            const auto stats = q6_oracle::down_oracle(0, c.resident.original,
                                                     c.activation.panels[sample], dense);
            points = stats.samples;
            maximum_absolute = stats.max_abs;
            maximum_fraction = stats.max_bound_ratio;
        } else {
            for (int col : sample_indices(kColumns, 8)) {
                for (int row : sample_indices(c.rows, 8)) {
                    const auto* weight = c.resident.original.data() + c.weight_offset + row * c.row_bytes;
                    const double reference = q4_oracle::q4k_q8_1_dot(
                        weight, c.activation.panels[sample].data(), col, kColumns);
                    const double got = load<float>(actual.data() + c.at(sample, col, row) * 4);
                    const double bound = 0.02 + 2e-4 * std::abs(reference);
                    const double error = std::abs(got - reference);
                    require(std::isfinite(reference) && std::isfinite(got) && error <= bound,
                            "Q4 differs from independent original-byte FP64 oracle");
                    maximum_absolute = std::max(maximum_absolute, error);
                    maximum_fraction = std::max(maximum_fraction, error / bound);
                    ++points;
                }
            }
        }
        require(points == 64, "expected 64 independent FP64 points per panel and arm");
        Json stats;
        stats.add("samples", points); stats.add("maximum_absolute_error", maximum_absolute);
        stats.add("maximum_bound_fraction", maximum_fraction);
        out << (sample ? "," : "") << stats.finish();
    }
    out << ']';
    return out.str();
}

std::string semantics(const Case& c, MmqTwoInputEpilogue mode,
                      const std::vector<std::uint8_t>& actual, int calls, bool sampled) {
    std::size_t points = 0, sensitive = 0;
    const auto inspect = [&](int sample, int col, int row) {
        const auto at = c.at(sample, col, row);
        const float value = load<float>(c.fp32.data() + at * 4);
        require(std::isfinite(value), "nonfinite FP32 projection");
        auto expected = load<std::uint16_t>(c.carry.data() + at * 2);
        require(std::isfinite(bf16_float(expected)) && bf16_float(expected) != 0,
                "carry fixture must be finite and nonzero");
        for (int iteration = 0; iteration < calls; ++iteration) {
            expected = rounded_bf16(mode == MmqTwoInputEpilogue::AccumulateBf16
                                   ? value + bf16_float(expected) : value);
        }
        require(load<std::uint16_t>(actual.data() + at * 2) == expected,
                "BF16 output differs from independent Store/Accumulate rounding semantics");
        sensitive += expected != rounded_bf16(value);
        ++points;
    };
    for (int sample = 0; sample < 2; ++sample) {
        if (sampled) {
            for (int col : sample_indices(kColumns, 8)) {
                for (int row : sample_indices(c.rows, 8)) { inspect(sample, col, row); }
            }
        } else {
            for (int col = 0; col < kColumns; ++col) {
                for (int row = 0; row < c.rows; ++row) { inspect(sample, col, row); }
            }
        }
    }
    if (mode == MmqTwoInputEpilogue::AccumulateBf16) {
        require(sensitive > 0, "Accumulate did not exercise nonzero carry");
    }
    c.padding(actual, 2);
    Json result;
    result.add("exact", true); result.add("calls", calls);
    result.add("exact_elements", c.logical_elements());
    result.add(sampled ? "semantic_samples" : "semantic_elements", points);
    result.add("carry_sensitive", sensitive);
    return result.finish();
}

}

#include "host_launch_wrapper.cuh"

namespace {

std::string diagnostic(Case& c, bool candidate, MmqTwoInputEpilogue mode) {
    launch_count = 0;
    capture_launches = true;
    const bool accepted = c.invoke(candidate, mode);
    capture_launches = false;
    require(accepted, "positive native dispatch rejected");
    check(cudaDeviceSynchronize(), "complete diagnostic packs and projections");
    const std::size_t projection_count = candidate ? 1 : 2;
    require(launch_count == 2 + projection_count, "unexpected pack/projection launch count");
    std::ostringstream names, grids;
    names << '['; grids << '[';
    for (std::size_t i = 0; i < launch_count; ++i) {
        const auto& launch = launches[i];
        const char* raw_name = nullptr;
        check(cudaFuncGetName(&raw_name, reinterpret_cast<const void*>(launch.kernel)), "diagnostic name");
        require(raw_name != nullptr, "missing native kernel name");
        const std::string name(raw_name);
        if (i < 2) {
            require(name.find("quantize_matrix_kernel") != std::string::npos &&
                    launch.grid.x == kColumns && launch.grid.y == unsigned(c.k / 512) &&
                    launch.grid.z == 1 && launch.block.x == 128 && launch.block.y == 1 &&
                    launch.block.z == 1 && launch.shared == 0, "wrong native pack route");
        } else {
            const unsigned expected = (c.rows / 128) * 10 * (candidate ? 2 : 1);
            require(name.find("mul_mat_q") != std::string::npos &&
                    name.find("fixup") == std::string::npos && launch.grid.x == expected &&
                    launch.grid.y == 1 && launch.grid.z == 1 && launch.block.x == 32 &&
                    launch.block.y == 8 && launch.block.z == 1 && launch.shared == 57856,
                    "wrong native projection route or geometry");
            names << (i == 2 ? "" : ",") << quoted(name);
            grids << (i == 2 ? "" : ",") << launch.grid.x;
        }
    }
    names << ']'; grids << ']';
    Json result;
    result.add("projection_launches", projection_count); result.raw("grid_x", grids.str());
    result.raw("block", "[32,8,1]"); result.add("dynamic_shared_bytes", 57856);
    result.add("pack_launches", 2); result.raw("projection_kernel_names", names.str());
    result.raw("pack_grid", "[1280," + std::to_string(c.k / 512) + ",1]");
    result.raw("pack_block", "[128,1,1]"); result.add("outside_timing_events", true);
    return result.finish();
}

int api_guards(Case& c) {
    const auto old_a = c.plain_a.read(), old_b = c.plain_b.read();
    const auto old_bf_a = c.bf_a.read(), old_bf_b = c.bf_b.read();
    int count = 0;
    for (bool candidate : {false, true}) {
        for (auto mode : {MmqTwoInputEpilogue::PlainFp32, MmqTwoInputEpilogue::StoreBf16,
                          MmqTwoInputEpilogue::AccumulateBf16}) {
            const auto reject = [&](const char* label, auto edit) {
                auto a = c.args(candidate, mode);
                edit(a);
                capture_launches = true;
                launch_count = 0;
                const bool accepted = candidate ? mmq_two_input_candidate(a, nullptr)
                                                : mmq_two_input_control(a, nullptr);
                capture_launches = false;
                require(!accepted && launch_count == 0, std::string("API rejection failed: ") + label);
                ++count;
            };
            reject("profile", [](auto& a) { a.profile = MmqTwoInputProfile(99); });
            reject("epilogue", [](auto& a) { a.epilogue = MmqTwoInputEpilogue(99); });
            reject("rows", [](auto& a) { --a.rows; });
            reject("K", [](auto& a) { a.k -= 256; });
            reject("T2560", [](auto& a) { a.columns = 2560; });
            reject("layout", [](auto& a) { a.activation_layout = 1 - a.activation_layout; });
            reject("weight_row_bytes", [](auto& a) { ++a.weight_row_stride_bytes; });
            reject("weight_extent", [](auto& a) { --a.weight_capacity_bytes; });
            reject("null_weight", [](auto& a) { a.weight = nullptr; });
            reject("unaligned_weight", [](auto& a) {
                a.weight = static_cast<const std::uint8_t*>(a.weight) + 1;
            });
            reject("null_input", [](auto& a) { a.packed_inputs = nullptr; });
            reject("unaligned_input", [](auto& a) {
                a.packed_inputs = static_cast<const std::uint8_t*>(a.packed_inputs) + 4;
            });
            reject("input_extent", [&](auto& a) {
                a.packed_capacity_bytes = a.packed_sample_stride_bytes + c.activation.panel_bytes - 1;
            });
            reject("input_sample_overlap", [&](auto& a) {
                a.packed_sample_stride_bytes = c.activation.panel_bytes - 16;
            });
            reject("input_sample_alignment", [](auto& a) { a.packed_sample_stride_bytes += 4; });
            reject("input_sample_overflow", [](auto& a) {
                a.packed_sample_stride_bytes = (std::int64_t(INT_MAX) + 1) * 4;
            });
            reject("null_output", [](auto& a) { a.output = nullptr; });
            reject("unaligned_output", [](auto& a) {
                a.output = static_cast<std::uint8_t*>(a.output) + 2;
            });
            reject("output_extent", [&](auto& a) {
                const auto width = mode == MmqTwoInputEpilogue::PlainFp32 ? 4 : 2;
                a.output_capacity_bytes = (a.output_sample_stride_elements +
                    (kColumns - 1) * a.output_column_stride_elements + a.rows) * width - 1;
            });
            reject("column_overlap", [](auto& a) { a.output_column_stride_elements = a.rows - 1; });
            reject("column_overflow", [](auto& a) { a.output_column_stride_elements = INT_MAX; });
            reject("output_sample_overlap", [](auto& a) {
                a.output_sample_stride_elements = (kColumns - 1) * a.output_column_stride_elements + a.rows - 16;
            });
            reject("output_sample_negative", [](auto& a) { a.output_sample_stride_elements = -16; });
            reject("output_sample_alignment", [](auto& a) { ++a.output_sample_stride_elements; });
            reject("output_sample_overflow", [](auto& a) { a.output_sample_stride_elements = INT_MAX; });
            reject("output_pointer_overflow", [](auto& a) {
                a.output = reinterpret_cast<void*>(std::numeric_limits<std::uintptr_t>::max() - 15);
            });
            reject("weight_output_alias", [](auto& a) { a.output = const_cast<void*>(a.weight); });
            reject("input_output_alias", [](auto& a) { a.output = const_cast<void*>(a.packed_inputs); });
        }
    }
    check(cudaDeviceSynchronize(), "complete rejected API calls");
    require(c.plain_a.read() == old_a && c.plain_b.read() == old_b &&
            c.bf_a.read() == old_bf_a && c.bf_b.read() == old_bf_b,
            "rejected API call changed output");
    return count;
}

int qualify(Case& c) {
    const auto initial = c.initial(true);
    c.plain_a.from_host(initial.data(), initial.size());
    c.plain_b.from_host(initial.data(), initial.size());
    const auto route_a = diagnostic(c, false, MmqTwoInputEpilogue::PlainFp32);
    const auto route_b = diagnostic(c, true, MmqTwoInputEpilogue::PlainFp32);
    c.fp32 = c.plain_a.read();
    auto alternate = c.plain_b.read();
    require(c.fp32 == alternate, "full native FP32 outputs differ");
    c.padding(c.fp32, 4); c.padding(alternate, 4);
    for (int sample = 0; sample < 2; ++sample) {
        for (int col = 0; col < kColumns; ++col) {
            for (int row = 0; row < c.rows; ++row) {
                require(std::isfinite(load<float>(c.fp32.data() + c.at(sample, col, row) * 4)),
                        "unwritten or nonfinite FP32 output");
            }
        }
    }
    Json plain;
    plain.add("exact", true); plain.add("finite_elements", c.logical_elements());
    plain.raw("oracle_control", oracle(c, c.fp32)); plain.raw("oracle_candidate", oracle(c, alternate));
    std::array<std::string, 2> bf_semantics;
    std::array<std::string, 2> bf_routes;
    for (int index = 0; index < 2; ++index) {
        const auto mode = index ? MmqTwoInputEpilogue::AccumulateBf16 : MmqTwoInputEpilogue::StoreBf16;
        c.bf_a.reset_from(c.seed); c.bf_b.reset_from(c.seed);
        const auto first = diagnostic(c, false, mode), second = diagnostic(c, true, mode);
        const auto a = c.bf_a.read(), b = c.bf_b.read();
        require(a == b, "full native BF16 outputs differ");
        bf_semantics[index] = semantics(c, mode, a, 1, false);
        require(semantics(c, mode, b, 1, false) == bf_semantics[index], "BF16 semantic summaries differ");
        bf_routes[index] = "{\"control\":" + first + ",\"candidate\":" + second + "}";
        if (!index) { c.stored = a; }
    }
    c.bf_a.reset_from(c.seed); c.bf_b.reset_from(c.seed);
    for (bool candidate : {false, true}) {
        for (int repeat = 0; repeat < 32; ++repeat) {
            require(c.invoke(candidate, MmqTwoInputEpilogue::AccumulateBf16), "carry32 dispatch rejected");
        }
    }
    check(cudaDeviceSynchronize(), "complete evolving 32-call carry");
    const auto carry_a = c.bf_a.read(), carry_b = c.bf_b.read();
    require(carry_a == carry_b, "full 32-call evolving carry outputs differ");
    const auto carry_result = semantics(c, MmqTwoInputEpilogue::AccumulateBf16, carry_a, 32, true);
    require(semantics(c, MmqTwoInputEpilogue::AccumulateBf16, carry_b, 32, true) == carry_result,
            "32-call carry semantic summaries differ");
    const int rejected = api_guards(c);
    c.unchanged();
    Json record;
    record.add("event", "qualification"); record.add("case", c.name); record.add("rows", c.rows);
    record.add("k", c.k); record.add("columns", kColumns); record.add("samples", 2);
    record.add("layout", c.layout == 0 ? "D4" : "DS4");
    record.raw("seeds", "[" + std::to_string(c.activation.seeds[0]) + "," +
                        std::to_string(c.activation.seeds[1]) + "]");
    record.raw("input_sha256", string_pair(c.activation.input_sha));
    record.raw("packed_sha256", string_pair(c.activation.packed_sha));
    record.raw("producer_oracle_groups", "[" + std::to_string(c.activation.producer_groups[0]) + "," +
                                          std::to_string(c.activation.producer_groups[1]) + "]");
    record.raw("plain", plain.finish()); record.raw("store", bf_semantics[0]);
    record.raw("accumulate", bf_semantics[1]); record.raw("carry32", carry_result);
    record.raw("routes", "{\"control\":" + route_a + ",\"candidate\":" + route_b + "}");
    record.raw("store_routes", bf_routes[0]); record.raw("accumulate_routes", bf_routes[1]);
    record.add("api_negative_calls", rejected); record.add("canaries", true); record.add("padding", true);
    record.add("immutable_weights", true); record.add("immutable_inputs", true); record.add("immutable_packs", true);
    record.add("status", "PASS"); emit(record);
    return rejected;
}

double timed(Case& c, bool candidate, Events& events) {
    require(!capture_launches, "launch diagnostics must be outside timing");
    c.output(candidate, MmqTwoInputEpilogue::StoreBf16).reset_from(c.seed);
    check(cudaEventRecord(events.first), "record region start");
    bool accepted = true;
    for (int repeat = 0; repeat < kRepeats; ++repeat) {
        accepted &= c.invoke(candidate, MmqTwoInputEpilogue::StoreBf16);
    }
    check(cudaEventRecord(events.last), "record region stop");
    check(cudaEventSynchronize(events.last), "complete region sample");
    require(accepted, "timed native dispatch rejected");
    float elapsed = 0;
    check(cudaEventElapsedTime(&elapsed, events.first, events.last), "read region events");
    require(std::isfinite(elapsed) && elapsed > 0, "invalid region duration");
    return double(elapsed) / kRepeats;
}

struct ScreenResult {
    double mean;
    int positive;
};

ScreenResult screen(Case& c) {
    for (char arm : std::string("ABAB")) {
        for (int repeat = 0; repeat < kRepeats; ++repeat) {
            require(c.invoke(arm == 'B', MmqTwoInputEpilogue::StoreBf16), "warmup rejected");
        }
    }
    check(cudaDeviceSynchronize(), "complete untimed region warmup");
    Events events;
    std::ostringstream blocks;
    blocks << '[';
    double total_control = 0, total_candidate = 0;
    int positive = 0, block_index = 0;
    for (const char* order : {"ABBA", "BAAB", "ABBA"}) {
        std::array<double, 4> times{};
        double control = 0, candidate = 0;
        std::ostringstream milliseconds;
        milliseconds << std::setprecision(12) << '[';
        for (int index = 0; index < 4; ++index) {
            times[index] = timed(c, order[index] == 'B', events);
            (order[index] == 'B' ? candidate : control) += times[index] / 2;
            milliseconds << (index ? "," : "") << times[index];
        }
        milliseconds << ']';
        const double reduction = 100.0 * (1.0 - candidate / control);
        total_control += control;
        total_candidate += candidate;
        positive += reduction > 0;
        Json block;
        block.add("index", block_index); block.add("order", order);
        block.raw("milliseconds", milliseconds.str()); block.add("control_mean_ms", control);
        block.add("candidate_mean_ms", candidate); block.add("time_reduction_percent", reduction);
        blocks << (block_index++ ? "," : "") << block.finish();
    }
    blocks << ']';
    require(c.bf_a.read() == c.stored && c.bf_b.read() == c.stored,
            "timed region changed exact Store result");
    c.unchanged();
    Json record;
    record.add("event", "screen"); record.add("case", c.name);
    record.add("primary", c.profile == MmqTwoInputProfile::Q6Down);
    record.add("mode", "StoreBf16"); record.add("scope", "two_native_packs_plus_projection");
    record.add("calls_per_sample", kRepeats); record.raw("blocks", blocks.str());
    const double reduction = 100.0 * (1.0 - total_candidate / total_control);
    record.add("mean_time_reduction_percent", reduction);
    record.add("positive_blocks", positive); record.add("outputs_exact_after_timing", true);
    record.add("CUPTI", false); record.add("statistical_confirmation", false); emit(record);
    return {reduction, positive};
}

}

int main(int argc, char** argv) {
    try {
        std::filesystem::path fixtures;
        std::string launch_id;
        bool correctness_only = false;
        for (int i = 1; i < argc; ++i) {
            const std::string arg(argv[i]);
            if (arg == "--correctness-only") { correctness_only = true; }
            else if ((arg == "--fixtures-root" || arg == "--launch-id") && i + 1 < argc) {
                const std::string value(argv[++i]);
                if (arg == "--fixtures-root") { fixtures = value; } else { launch_id = value; }
            } else { throw std::runtime_error("usage: probe --fixtures-root DIR --launch-id ID [--correctness-only]"); }
        }
        require(!fixtures.empty() && !launch_id.empty(), "fixtures root and launch id required");
        rounding_self_test();
        require(sha256(nullptr, 0) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                "SHA256 empty known-answer test failed");
        const std::array<std::uint8_t, 3> abc{'a', 'b', 'c'};
        require(sha256(abc.data(), abc.size()) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
                "SHA256 known-answer test failed");
        int device = -1;
        cudaDeviceProp properties{};
        check(cudaGetDevice(&device), "current device");
        check(cudaGetDeviceProperties(&properties, device), "device properties");
        require(properties.major == 8 && properties.minor == 6 &&
                std::string(properties.name).find("RTX 3090") != std::string::npos,
                "this closed screen requires the RTX 3090");
        Json start;
        start.add("event", "start"); start.add("launch_id", launch_id); start.add("device", properties.name);
        start.add("cc", 100 * properties.major + 10 * properties.minor);
        start.add("correctness_only", correctness_only); start.raw("host_receipt", kMmqTwoInputHostReceipt); emit(start);
        Resident q6(fixtures, kQ6), q4(fixtures, kQ4);
        std::array<std::unique_ptr<Case>, 3> cases;
        cases[0] = std::make_unique<Case>("Q6Down", MmqTwoInputProfile::Q6Down, q6, 0, 303385U);
        cases[1] = std::make_unique<Case>("Q4Gate", MmqTwoInputProfile::Q4GateOrUp, q4, 0, 303401U);
        cases[2] = std::make_unique<Case>("Q4Up", MmqTwoInputProfile::Q4GateOrUp, q4, 17408, 303417U);
        int rejected = 0;
        std::ostringstream api_cases;
        api_cases << '[';
        for (std::size_t index = 0; index < cases.size(); ++index) {
            const int count = qualify(*cases[index]);
            rejected += count;
            Json entry;
            entry.add("case", cases[index]->name); entry.add("api_negative_calls", count);
            entry.add("both_arms", true); entry.add("all_three_epilogues", true);
            api_cases << (index ? "," : "") << entry.finish();
        }
        api_cases << ']';
        Json guards;
        guards.add("event", "api_guards"); guards.add("api_negative_calls", rejected);
        guards.raw("cases", api_cases.str()); guards.add("no_kernel_launches", true);
        guards.add("outputs_unchanged", true); guards.add("status", "PASS"); emit(guards);
        Json qualification;
        qualification.add("event", "qualification_complete"); qualification.add("cases", 3);
        qualification.add("all_correctness_before_timing", true); qualification.add("status", "PASS"); emit(qualification);
        ScreenResult primary{0, 0};
        if (!correctness_only) {
            for (std::size_t index = 0; index < cases.size(); ++index) {
                const auto result = screen(*cases[index]);
                if (index == 0) { primary = result; }
            }
        }
        const bool pass = !correctness_only && primary.mean >= 10.0 && primary.positive == 3;
        Json complete;
        complete.add("event", "complete"); complete.add("status", "PASS");
        complete.add("launch_id", launch_id); complete.add("correctness_only", correctness_only);
        complete.add("cases", 3); complete.add("timing_blocks", correctness_only ? 0 : 9);
        complete.add("primary_mean_time_reduction_percent", primary.mean);
        complete.add("primary_positive_blocks", primary.positive); complete.add("primary_screen_pass", pass);
        complete.add("decision", correctness_only ? "CORRECTNESS_ONLY" :
                     pass ? "PROCEED_TO_PAIRED_FFN_GATE" : "KILL_OPERATOR_SCREEN");
        complete.add("runtime_qualified", false); complete.add("full_ffn_qualified", false); emit(complete);
        return 0;
    } catch (const std::exception& error) {
        Json failure;
        failure.add("event", "failure"); failure.add("message", error.what()); emit(failure);
        return 1;
    }
}
