#include "probe_api.h"
#include "attention_v_shift_check.h"
#include "attention_v_shift_oracle.h"
#include "attention_v_shift_host_receipt.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace raw {
using namespace ninfer;
using namespace ninfer::test;
#include "attention_v_shift_host_helpers.cuh"
}

namespace {

using ninfer::DType;
using ninfer::Tensor;
constexpr int kRepeats = 32;
constexpr std::size_t kGuard = 256;
constexpr unsigned char kGuardByte = 0xa7;

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

std::string quoted(const std::string& value) {
    std::ostringstream out;
    out << '"';
    for (const unsigned char c : value) {
        if (c == '"' || c == '\\') { out << '\\' << char(c); }
        else if (c < 32) {
            out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
        } else { out << char(c); }
    }
    out << '"';
    return out.str();
}

struct Json {
    std::ostringstream stream;
    bool first = true;
    Json() { stream << std::setprecision(12) << '{'; }
    void key(const char* name) {
        if (!first) { stream << ','; }
        first = false;
        stream << quoted(name) << ':';
    }
    void raw(const char* name, const std::string& value) { key(name); stream << value; }
    void add(const char* name, const std::string& value) { key(name); stream << quoted(value); }
    void add(const char* name, const char* value) { add(name, std::string(value)); }
    void add(const char* name, bool value) { key(name); stream << (value ? "true" : "false"); }
    template <typename T, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
    void add(const char* name, T value) { key(name); stream << value; }
    std::string finish() { stream << '}'; return stream.str(); }
};

void emit(Json& value) { std::cout << value.finish() << '\n' << std::flush; }

struct LaunchRecord {
    cudaKernel_t kernel{};
    dim3 grid{}, block{};
    std::size_t shared{};
};

bool capture_launches = false;
std::size_t launch_count = 0;
LaunchRecord launch_record{};

struct Guarded {
    unsigned char* allocation = nullptr;
    void* data = nullptr;
    std::size_t bytes;
    explicit Guarded(std::size_t payload) : bytes(payload) {
        require(bytes > 0, "empty device allocation");
        check(cudaMalloc(reinterpret_cast<void**>(&allocation), bytes + 2 * kGuard), "allocate guarded");
        data = allocation + kGuard;
        check(cudaMemset(allocation, kGuardByte, bytes + 2 * kGuard), "initialize guarded");
    }
    ~Guarded() { if (allocation) { cudaFree(allocation); } }
    Guarded(const Guarded&) = delete;
    Guarded& operator=(const Guarded&) = delete;
    template <typename T>
    void upload(const std::vector<T>& values) {
        require(values.size() * sizeof(T) == bytes, "upload size mismatch");
        check(cudaMemcpy(data, values.data(), bytes, cudaMemcpyHostToDevice), "upload fixture");
    }
    template <typename T>
    std::vector<T> read() const {
        require(bytes % sizeof(T) == 0, "read element size mismatch");
        std::vector<T> values(bytes / sizeof(T));
        check(cudaMemcpy(values.data(), data, bytes, cudaMemcpyDeviceToHost), "read result");
        return values;
    }
    void guards(const std::string& label) const {
        std::array<unsigned char, 2 * kGuard> observed{};
        check(cudaMemcpy(observed.data(), allocation, kGuard, cudaMemcpyDeviceToHost), "read prefix guard");
        check(cudaMemcpy(observed.data() + kGuard, allocation + kGuard + bytes, kGuard,
                         cudaMemcpyDeviceToHost), "read suffix guard");
        require(std::all_of(observed.begin(), observed.end(),
                           [](unsigned char v) { return v == kGuardByte; }), label + " guard changed");
    }
    template <typename T>
    void unchanged(const std::vector<T>& expected, const std::string& label) const {
        require(expected.size() * sizeof(T) == bytes, "immutable payload size mismatch");
        std::vector<unsigned char> chunk(std::min<std::size_t>(bytes, 1U << 20));
        const auto* source = reinterpret_cast<const unsigned char*>(expected.data());
        for (std::size_t offset = 0; offset < bytes; offset += chunk.size()) {
            const auto count = std::min(chunk.size(), bytes - offset);
            check(cudaMemcpy(chunk.data(), static_cast<const unsigned char*>(data) + offset,
                             count, cudaMemcpyDeviceToHost), "read immutable payload");
            require(std::memcmp(chunk.data(), source + offset, count) == 0, label + " changed");
        }
        guards(label);
    }
};

Tensor tensor(void* data, DType type, std::array<std::int32_t, 4> shape, std::int64_t element_bytes) {
    Tensor result;
    result.data = data;
    result.dtype = type;
    for (int i = 0; i < 4; ++i) {
        result.ne[i] = shape[i];
        result.nb[i] = element_bytes;
        element_bytes *= shape[i];
    }
    return result;
}

struct Case {
    const char* name;
    int tokens, base, scale_pattern;
    std::uint32_t seed;
    bool long_case;
};

constexpr std::array<Case, 5> kCases{{
    {"tail1-valid1-low-boundary", 1, 192, 0, 1101u, false},
    {"tail33-valid63-high-boundary", 33, 158, 1, 1102u, false},
    {"tail127-valid1-maxfinite", 127, 130, 2, 1103u, false},
    {"tail129-valid63-stage-reuse", 129, 126, 3, 1104u, false},
    {"long-T1280-base31488", 1280, 31488, -1, 1105u, true},
}};

std::uint16_t edge_scale(int pattern, int page, int head) {
    constexpr std::array<std::uint16_t, 4> low{0x47ff, 0x4800, 0x4801, 0x3c00};
    constexpr std::array<std::uint16_t, 3> high{0x4bff, 0x4c00, 0x4c01};
    constexpr std::array<std::uint16_t, 5> maximum{0x7bff, 0x3c00, 0x4800, 0x4c01, 0x47ff};
    constexpr std::array<std::uint16_t, 4> reuse{0x4801, 0x4c00, 0x47ff, 0x4c01};
    const unsigned index = unsigned(page + head);
    if (pattern == 0) { return low[index % low.size()]; }
    if (pattern == 1) { return high[index % high.size()]; }
    if (pattern == 2) { return maximum[index % maximum.size()]; }
    require(pattern == 3, "unknown scale fixture");
    return reuse[index % reuse.size()];
}

struct Fixture {
    const Case& spec;
    int total, capacity, physical_pages;
    raw::MappingPattern mapping;
    std::vector<std::uint16_t> q, ks, vs;
    std::vector<std::int8_t> k, v;
    std::vector<std::int32_t> positions, table, rows{0};

    explicit Fixture(const Case& c)
        : spec(c), total(c.base + c.tokens),
          capacity(c.long_case ? 65536 : raw::align_up_page(total + 3)),
          physical_pages(raw::physical_page_count(capacity / ninfer::kPagedKVPageSize,
              c.long_case ? raw::MappingPattern::Identity : raw::MappingPattern::Fragmented)),
          mapping(c.long_case ? raw::MappingPattern::Identity : raw::MappingPattern::Fragmented),
          positions(c.tokens),
          table(raw::make_block_table(capacity / ninfer::kPagedKVPageSize, mapping)) {
        const auto& geometry = raw::kGeometries[0];
        require(total <= capacity && total >= 129, "invalid fixture page extent");
        auto queries = raw::make_bf16_values(std::size_t(c.tokens) * 24 * 256, c.seed, -0.25f, 0.25f);
        if (c.tokens == 1) { std::fill(queries.begin(), queries.end(), 0.0f); }
        q = raw::to_bf16_bits(queries);
        std::iota(positions.begin(), positions.end(), c.base);
        std::vector<std::int8_t> lk(raw::cache_elements(geometry, capacity), std::int8_t{77});
        std::vector<std::int8_t> lv(lk.size(), std::int8_t{-91});
        std::vector<std::uint16_t> lks(raw::scale_elements(geometry, capacity), 0x7bff);
        std::vector<std::uint16_t> lvs(lks.size(), 0x7bff);
        std::vector<float> key(256), value(256), unused_logical(256);
        std::mt19937 random(c.seed + 10u);
        std::uniform_real_distribution<float> distribution(-0.25f, 0.25f);
        for (int head = 0; head < 4; ++head) {
            for (int position = 0; position < total; ++position) {
                for (int d = 0; d < 256; ++d) {
                    key[d] = raw::bf16_to_f32(raw::f32_to_bf16(distribution(random)));
                    value[d] = raw::bf16_to_f32(raw::f32_to_bf16(distribution(random)));
                }
                const auto code = raw::cache_index(geometry, capacity, head, position, 0);
                const auto scale = raw::scale_index(geometry, capacity, head, position, 0);
                raw::encode_rotated_key_row(key, 0, lk, code, lks, scale, unused_logical, 0);
                for (int group = 0; group < 4; ++group) {
                    raw::encode_group(value, 64 * group, lv, code + 64 * group, lvs, scale + group);
                    if (c.scale_pattern >= 0) {
                        lvs[scale + group] = edge_scale(c.scale_pattern, position / 64, head);
                    }
                }
            }
        }
        k = raw::scatter_paged(lk, 256, geometry, capacity, table, physical_pages);
        v = raw::scatter_paged(lv, 256, geometry, capacity, table, physical_pages);
        ks = raw::scatter_paged(lks, 4, geometry, capacity, table, physical_pages);
        vs = raw::scatter_paged(lvs, 4, geometry, capacity, table, physical_pages);
    }

    std::size_t physical(int extent, int head, int position, int leading) const {
        return raw::paged_index(extent, raw::kGeometries[0], table[position / 64],
                                head, position, leading);
    }

    std::pair<std::vector<double>, std::vector<double>> represented_head(int head) const {
        std::vector<double> keys(std::size_t(total) * 256), values(keys.size());
        for (int position = 0; position < total; ++position) {
            std::array<double, 256> rotated{};
            for (int d = 0; d < 256; ++d) {
                const auto code = physical(256, head, position, d);
                const auto scale = physical(4, head, position, d / 64);
                rotated[d] = double(k[code]) * double(raw::f16_bits_to_f32(ks[scale]));
                values[std::size_t(position) * 256 + d] =
                    double(v[code]) * double(raw::f16_bits_to_f32(vs[scale]));
            }
            raw::normalized_hadamard_d256(rotated);
            std::copy(rotated.begin(), rotated.end(), keys.begin() + std::size_t(position) * 256);
        }
        return {std::move(keys), std::move(values)};
    }
};

struct DeviceFixture {
    const Fixture& host;
    Guarded q, k, v, ks, vs, positions, table, rows, off, on;
    Tensor tq, tp, tr, toff, ton;
    ninfer::PagedKVBatchLayerView cache;

    explicit DeviceFixture(const Fixture& f)
        : host(f), q(f.q.size() * 2), k(f.k.size()), v(f.v.size()),
          ks(f.ks.size() * 2), vs(f.vs.size() * 2), positions(f.positions.size() * 4),
          table(f.table.size() * 4), rows(4), off(f.q.size() * 2), on(f.q.size() * 2) {
        q.upload(f.q); k.upload(f.k); v.upload(f.v); ks.upload(f.ks); vs.upload(f.vs);
        positions.upload(f.positions); table.upload(f.table); rows.upload(f.rows);
        const std::vector<std::uint16_t> canary(f.q.size(), raw::kOutputCanary);
        off.upload(canary); on.upload(canary);
        tq = tensor(q.data, DType::BF16, {256, 24, f.spec.tokens, 1}, 2);
        tp = tensor(positions.data, DType::I32, {f.spec.tokens, 1, 1, 1}, 4);
        tr = tensor(rows.data, DType::I32, {1, 1, 1, 1}, 4);
        toff = tensor(off.data, DType::BF16, {256, 24, f.spec.tokens, 1}, 2);
        ton = tensor(on.data, DType::BF16, {256, 24, f.spec.tokens, 1}, 2);
        cache.k_pages = tensor(k.data, DType::I8, {256, 64, 4, f.physical_pages}, 1);
        cache.v_pages = tensor(v.data, DType::I8, {256, 64, 4, f.physical_pages}, 1);
        cache.k_scale_pages = tensor(ks.data, DType::FP16, {4, 64, 4, f.physical_pages}, 2);
        cache.v_scale_pages = tensor(vs.data, DType::FP16, {4, 64, 4, f.physical_pages}, 2);
        cache.block_tables = tensor(table.data, DType::I32, {int(f.table.size()), 1, 1, 1}, 4);
        cache.head_dim = 256;
        cache.num_kv_heads = 4;
        cache.storage = ninfer::KvCacheStorage::Int8Group64;
    }

    cudaError_t invoke(bool alternate) {
        return alternate
            ? attention_v_shift_on_wide_launch(tq, tp, tr, raw::kAttentionScale, cache, ton, nullptr)
            : attention_v_shift_off_wide_launch(tq, tp, tr, raw::kAttentionScale, cache, toff, nullptr);
    }

    void immutable() const {
        q.unchanged(host.q, "Q"); k.unchanged(host.k, "K"); v.unchanged(host.v, "V");
        ks.unchanged(host.ks, "K scales"); vs.unchanged(host.vs, "V scales");
        positions.unchanged(host.positions, "positions"); table.unchanged(host.table, "block table");
        rows.unchanged(host.rows, "table rows");
        off.guards("OFF output"); on.guards("ON output");
    }
};

std::string diagnostic_launch(DeviceFixture& f, bool alternate) {
    launch_count = 0;
    capture_launches = true;
    const auto status = f.invoke(alternate);
    capture_launches = false;
    check(status, "diagnostic attention launch");
    check(cudaDeviceSynchronize(), "complete diagnostic attention");
    require(launch_count == 1, "attention dispatch did not launch exactly one kernel");
    const auto actual = launch_record;
    const char* raw_name = nullptr;
    check(cudaFuncGetName(&raw_name, reinterpret_cast<const void*>(actual.kernel)), "diagnostic kernel name");
    require(raw_name, "kernel name unavailable");
    const std::string name(raw_name);
    require(name.find(alternate ? "causal_attention_prompt_i8_fast_v_shift_on_kernel"
                                : "causal_attention_prompt_i8_fast_v_shift_off_kernel") != std::string::npos,
            "unexpected attention route: " + name);
    require(actual.grid.x == unsigned((f.host.spec.tokens + 127) / 128) &&
            actual.grid.y == 24 && actual.grid.z == 1 &&
            actual.block.x == 256 && actual.block.y == 1 && actual.block.z == 1 &&
            actual.shared == 100352, "attention launch geometry changed");
    Json record;
    record.add("kernel", name);
    record.add("grid_x", actual.grid.x);
    record.add("grid_y", actual.grid.y);
    record.add("block_x", actual.block.x);
    record.add("dynamic_shared_bytes", actual.shared);
    record.add("outside_timing_events", true);
    return record.finish();
}

void paired_output(const std::vector<std::uint16_t>& off, const std::vector<std::uint16_t>& on) {
    require(off.size() == on.size() && !off.empty(), "output extents differ");
    std::size_t differences = 0, first = off.size(), nonfinite = 0, canaries = 0;
    for (std::size_t i = 0; i < off.size(); ++i) {
        if (off[i] != on[i]) { ++differences; first = std::min(first, i); }
        nonfinite += !std::isfinite(raw::bf16_to_f32(off[i]));
        nonfinite += !std::isfinite(raw::bf16_to_f32(on[i]));
        canaries += off[i] == raw::kOutputCanary;
        canaries += on[i] == raw::kOutputCanary;
    }
    require(differences == 0, "OFF/ON BF16 mismatches=" + std::to_string(differences) +
            " first_index=" + std::to_string(first));
    require(nonfinite == 0 && canaries == 0, "nonfinite or unwritten BF16 outputs");
}

struct OracleRow { int token, head; };

std::vector<OracleRow> oracle_rows(const Fixture& f) {
    if (f.spec.long_case) { return {{0, 0}, {127, 7}, {128, 12}, {1279, 23}}; }
    std::vector<OracleRow> result;
    for (int token = 0; token < f.spec.tokens; ++token) {
        for (int head = 0; head < 24; ++head) { result.push_back({token, head}); }
    }
    return result;
}

std::string oracle_check(const Fixture& f, const std::vector<std::uint16_t>& off,
                         const std::vector<std::uint16_t>& on) {
    const auto rows = oracle_rows(f);
    std::vector<double> reference(rows.size() * 256), a(reference.size()), b(reference.size());
    for (int kv_head = 0; kv_head < 4; ++kv_head) {
        const auto represented = f.represented_head(kv_head);
        for (std::size_t row = 0; row < rows.size(); ++row) {
            const auto selected = rows[row];
            if (selected.head / 6 != kv_head) { continue; }
            const int last_position = f.positions[selected.token];
            ninfer::test::naive_dense_softmax_attention(
                {256, 1, 1}, 1, last_position + 1, double(raw::kAttentionScale),
                [&](int d, int, int) {
                    return double(raw::bf16_to_f32(f.q[raw::q_index(
                        raw::kGeometries[0], selected.head, d, selected.token)]));
                },
                [&](int d, int, int position) { return represented.first[std::size_t(position) * 256 + d]; },
                [&](int d, int, int position) { return represented.second[std::size_t(position) * 256 + d]; },
                [&](int, int position) { return position <= last_position; },
                [&](int d, int, int, double value) {
                    const auto index = raw::q_index(raw::kGeometries[0], selected.head, d, selected.token);
                    const auto target = row * 256 + d;
                    reference[target] = value;
                    a[target] = double(raw::bf16_to_f32(off[index]));
                    b[target] = double(raw::bf16_to_f32(on[index]));
                });
        }
    }
    const auto sa = ninfer::test::compute_reduction_stats(a.data(), reference.data(), a.size());
    const auto sb = ninfer::test::compute_reduction_stats(b.data(), reference.data(), b.size());
    const bool pass_a = ninfer::test::reduction_passes(sa, a.size(), raw::kAttentionInt8Criterion);
    const bool pass_b = ninfer::test::reduction_passes(sb, b.size(), raw::kAttentionInt8Criterion);
    Json result;
    result.add("passed", pass_a && pass_b);
    result.add("query_head_rows", rows.size());
    result.add("output_elements", reference.size());
    result.add("full_output_oracle", !f.spec.long_case);
    result.add("represented_physical_bytes_decoded", true);
    result.add("key_inverse_hadamard_precision", "FP64");
    result.add("relative_l2_limit", raw::kAttentionInt8Criterion.relative_l2);
    result.add("gross_absolute", raw::kAttentionInt8Criterion.gross_absolute);
    result.add("gross_relative_to_max_reference", raw::kAttentionInt8Criterion.gross_relative_to_max_reference);
    result.add("off_relative_l2", sa.relative_l2);
    result.add("on_relative_l2", sb.relative_l2);
    result.add("off_max_absolute_error", sa.maximum_absolute_error);
    result.add("on_max_absolute_error", sb.maximum_absolute_error);
    result.add("gross_error_limit", ninfer::test::gross_error_limit(sa, raw::kAttentionInt8Criterion));
    const auto record = result.finish();
    if (!pass_a || !pass_b) {
        std::cout << "{\"event\":\"oracle_failure\",\"case\":" << quoted(f.spec.name)
                  << ",\"oracle\":" << record << "}\n" << std::flush;
        throw std::runtime_error("independent FP64 oracle criterion failed");
    }
    return record;
}

std::vector<std::uint16_t> qualify(DeviceFixture& device) {
    const auto route_a = diagnostic_launch(device, false);
    const auto route_b = diagnostic_launch(device, true);
    const auto off = device.off.read<std::uint16_t>();
    const auto on = device.on.read<std::uint16_t>();
    paired_output(off, on);
    const auto oracle = oracle_check(device.host, off, on);
    device.immutable();
    Json result;
    result.add("event", "paired_qualification");
    result.add("case", device.host.spec.name);
    result.add("tokens", device.host.spec.tokens);
    result.add("base", device.host.spec.base);
    result.add("context_capacity", device.host.capacity);
    result.add("visible_key_extent", device.host.total);
    result.add("last_page_valid_keys", (device.host.total - 1) % 64 + 1);
    result.add("mapping", raw::mapping_name(device.host.mapping));
    result.add("synthetic_represented_inputs", true);
    result.add("model_activation_capture", false);
    result.add("bf16_elements_compared", off.size());
    result.add("all_bf16_outputs_bit_exact", true);
    result.add("finite_and_canaries_passed", true);
    result.add("cache_and_inputs_byte_immutable", true);
    result.add("guards_passed", true);
    result.raw("off_route", route_a);
    result.raw("on_route", route_b);
    result.raw("oracle", oracle);
    emit(result);
    return off;
}

struct Events {
    cudaEvent_t first{}, last{};
    Events() {
        check(cudaEventCreate(&first), "create timing start");
        check(cudaEventCreate(&last), "create timing stop");
    }
    ~Events() { cudaEventDestroy(first); cudaEventDestroy(last); }
};

double timing_sample(DeviceFixture& f, bool alternate, Events& events) {
    require(!capture_launches, "route diagnostics enabled during timing");
    check(cudaEventRecord(events.first), "record timing start");
    for (int repeat = 0; repeat < kRepeats; ++repeat) {
        check(f.invoke(alternate), "timed attention launch");
    }
    check(cudaEventRecord(events.last), "record timing stop");
    check(cudaEventSynchronize(events.last), "complete timing sample");
    float elapsed = 0;
    check(cudaEventElapsedTime(&elapsed, events.first, events.last), "read timing");
    require(elapsed > 0 && std::isfinite(elapsed), "invalid timing duration");
    return double(elapsed) / kRepeats;
}

void repeated_result(DeviceFixture& f, const std::vector<std::uint16_t>& expected) {
    const auto a = f.off.read<std::uint16_t>();
    const auto b = f.on.read<std::uint16_t>();
    paired_output(a, b);
    require(a == expected, "repeated attention differs from qualified one-call output");
    f.immutable();
}

void screen(DeviceFixture& f, const std::vector<std::uint16_t>& expected) {
    require(f.host.spec.long_case && expected.size() == f.host.q.size(), "unqualified screen extent");
    Events events;
    std::array<double, 4> warm{};
    for (int i = 0; i < 4; ++i) { warm[i] = timing_sample(f, (i & 1) != 0, events); }
    repeated_result(f, expected);
    std::vector<double> a, b;
    std::ostringstream blocks;
    blocks << std::setprecision(12) << '[';
    for (int block = 0; block < 3; ++block) {
        const bool reverse = block == 1;
        const std::array<bool, 4> order{reverse, !reverse, !reverse, reverse};
        std::array<double, 4> elapsed{};
        for (int sample = 0; sample < 4; ++sample) {
            elapsed[sample] = timing_sample(f, order[sample], events);
            (order[sample] ? b : a).push_back(elapsed[sample]);
        }
        repeated_result(f, expected);
        Json entry;
        entry.add("block", block);
        entry.add("order", reverse ? "BAAB" : "ABBA");
        std::ostringstream times;
        times << std::setprecision(12) << '[' << elapsed[0] << ',' << elapsed[1] << ','
              << elapsed[2] << ',' << elapsed[3] << ']';
        entry.raw("ms_per_call", times.str());
        blocks << (block ? "," : "") << entry.finish();
    }
    blocks << ']';
    const double mean_a = std::accumulate(a.begin(), a.end(), 0.0) / a.size();
    const double mean_b = std::accumulate(b.begin(), b.end(), 0.0) / b.size();
    Json result;
    result.add("event", "exploratory_attention_screen");
    result.add("case", f.host.spec.name);
    result.add("timing_scope", "full_attention_only");
    result.add("calls_per_sample", kRepeats);
    result.add("warmup_samples", 4);
    result.add("warmup_order", "ABAB");
    std::ostringstream warm_json;
    warm_json << std::setprecision(12) << '[' << warm[0] << ',' << warm[1] << ',' << warm[2] << ',' << warm[3] << ']';
    result.raw("warmup_ms_per_call", warm_json.str());
    result.raw("blocks", blocks.str());
    result.add("off_mean_ms", mean_a);
    result.add("on_mean_ms", mean_b);
    result.add("time_reduction_percent", 100.0 * (mean_a - mean_b) / mean_a);
    result.add("same_cache_and_activation_all_arms", true);
    result.add("repacking_or_uploads_between_arms", false);
    result.add("diagnostics_or_copies_inside_events", false);
    result.add("post_block_exactness_and_immutability_passed", true);
    result.add("statistical_confirmation", false);
    result.add("full_request_result", false);
    emit(result);
}

}

extern "C" cudaError_t CUDARTAPI __real___cudaLaunchKernel(
    cudaKernel_t kernel, dim3 grid, dim3 block, void** args, std::size_t shared, cudaStream_t stream);

extern "C" cudaError_t CUDARTAPI __wrap___cudaLaunchKernel(
    cudaKernel_t kernel, dim3 grid, dim3 block, void** args, std::size_t shared, cudaStream_t stream) {
    if (capture_launches) {
        if (launch_count == 0) { launch_record = {kernel, grid, block, shared}; }
        ++launch_count;
    }
    return __real___cudaLaunchKernel(kernel, grid, block, args, shared, stream);
}

int main(int argc, char** argv) {
    try {
        require(argc == 1 || (argc == 2 && std::string(argv[1]) == "--correctness-only"),
                "usage: probe [--correctness-only]");
        const bool correctness_only = argc == 2;
        int device = 0, runtime = 0;
        check(cudaGetDevice(&device), "current device");
        check(cudaRuntimeGetVersion(&runtime), "runtime version");
        cudaDeviceProp properties{};
        check(cudaGetDeviceProperties(&properties, device), "device properties");
        require(properties.major == 8 && properties.minor == 6 &&
                std::string(properties.name) == "NVIDIA GeForce RTX 3090",
                "this screen requires the selected RTX 3090/SM86");
        require(properties.sharedMemPerBlockOptin >= 100368 && properties.multiProcessorCount == 82,
                "unexpected device resource envelope");
        Json start;
        start.add("event", "attention_v_shift_host_start");
        start.add("device", properties.name);
        start.add("runtime_version", runtime);
        start.add("correctness_only", correctness_only);
        start.raw("source_receipt", kAttentionVShiftHostReceiptJson);
        emit(start);
        for (const auto& c : kCases) {
            Fixture fixture(c);
            DeviceFixture gpu(fixture);
            const auto expected = qualify(gpu);
            if (c.long_case && !correctness_only) { screen(gpu, expected); }
        }
        Json done;
        done.add("event", "attention_v_shift_host_complete");
        done.add("qualified_cases", kCases.size());
        done.add("screen_completed", !correctness_only);
        done.add("statistical_confirmation", false);
        done.add("full_request_result", false);
        emit(done);
        return 0;
    } catch (const std::exception& error) {
        capture_launches = false;
        Json failure;
        failure.add("event", "attention_v_shift_host_failure");
        failure.add("error", error.what());
        emit(failure);
        return 1;
    }
}
