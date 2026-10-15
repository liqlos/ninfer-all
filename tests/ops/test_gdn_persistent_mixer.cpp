#include "core/arena.h"
#include "core/device.h"
#include "core/gdn_replay_records.h"
#include "core/weight.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/gated_rmsnorm.h"
#include "ninfer/ops/gdn_persistent_mixer.h"
#include "ops/linear/gguf/gguf_linear.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace ninfer;

constexpr int kHidden = 5120;
constexpr int kQkv = 10240;
constexpr int kValue = 6144;
constexpr int kQkHeads = 16;
constexpr int kValueHeads = 48;
constexpr int kDim = 128;
constexpr int kSlots = 4;
constexpr std::size_t kGuard = 256;
constexpr std::uint8_t kCanary = 0xa5;

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

class GuardedBuffer {
public:
    explicit GuardedBuffer(std::size_t payload_bytes) : payload_bytes_(payload_bytes) {
        check(cudaMalloc(&allocation_, payload_bytes_ + 2 * kGuard), "allocate guarded buffer");
        data_ = static_cast<std::byte*>(allocation_) + kGuard;
        check(cudaMemset(allocation_, kCanary, payload_bytes_ + 2 * kGuard), "initialize guards");
    }

    ~GuardedBuffer() {
        if (allocation_ != nullptr) { (void)cudaFree(allocation_); }
    }

    GuardedBuffer(const GuardedBuffer&) = delete;
    GuardedBuffer& operator=(const GuardedBuffer&) = delete;

    void upload(const void* source) {
        check(cudaMemcpy(data_, source, payload_bytes_, cudaMemcpyHostToDevice), "upload payload");
    }

    void upload_async(const void* source, cudaStream_t stream) {
        check(cudaMemcpyAsync(data_, source, payload_bytes_, cudaMemcpyHostToDevice, stream),
              "enqueue payload upload");
    }

    void download(void* destination) const {
        check(cudaMemcpy(destination, data_, payload_bytes_, cudaMemcpyDeviceToHost), "download payload");
    }

    void fill_payload(int value) {
        check(cudaMemset(data_, value, payload_bytes_), "fill payload");
    }

    bool guards_intact() const {
        std::array<std::uint8_t, 2 * kGuard> bytes{};
        check(cudaMemcpy(bytes.data(), allocation_, kGuard, cudaMemcpyDeviceToHost), "read leading guard");
        check(cudaMemcpy(bytes.data() + kGuard, static_cast<const std::byte*>(data_) + payload_bytes_,
                         kGuard, cudaMemcpyDeviceToHost), "read trailing guard");
        return std::all_of(bytes.begin(), bytes.end(), [](std::uint8_t value) { return value == kCanary; });
    }

    void* data() const { return data_; }
    std::size_t bytes() const { return payload_bytes_; }

private:
    void* allocation_ = nullptr;
    void* data_ = nullptr;
    std::size_t payload_bytes_ = 0;
};

std::vector<std::uint8_t> packed_weight(QType type, int rows, int columns, std::uint32_t seed) {
    const auto block = gguf_block_shape(type);
    const std::size_t row_bytes = static_cast<std::size_t>(columns / block.elements) * block.bytes;
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(rows) * row_bytes);
    std::mt19937 random(seed);
    std::uniform_int_distribution<int> byte(0, 255);
    for (auto& value : bytes) { value = static_cast<std::uint8_t>(byte(random)); }
    const std::uint16_t one = 0x3c00;
    for (std::size_t offset = 0; offset < bytes.size(); offset += block.bytes) {
        if (type == QType::GGUF_Q4_K) {
            std::memcpy(bytes.data() + offset, &one, sizeof(one));
            std::memcpy(bytes.data() + offset + 2, &one, sizeof(one));
        } else {
            std::memcpy(bytes.data() + offset + 208, &one, sizeof(one));
        }
    }
    return bytes;
}

Weight make_weight(QType type, const void* data, int rows, int columns) {
    Weight weight;
    weight.qtype = type;
    weight.layout = QuantLayout::GgufBlocks;
    weight.qdata = data;
    weight.payload = data;
    weight.n = weight.shape[0] = weight.padded_shape[0] = rows;
    weight.k = weight.shape[1] = weight.padded_shape[1] = columns;
    weight.ndim = 2;
    weight.group = weight.group_size = gguf_block_shape(type).elements;
    return weight;
}

std::vector<std::uint16_t> bf16_values(std::size_t count, std::uint32_t seed, float scale) {
    std::mt19937 random(seed);
    std::uniform_real_distribution<float> distribution(-scale, scale);
    std::vector<std::uint16_t> values(count);
    for (auto& value : values) {
        const std::uint32_t bits = std::bit_cast<std::uint32_t>(distribution(random));
        value = static_cast<std::uint16_t>((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
    }
    return values;
}

std::vector<float> fp32_values(std::size_t count, std::uint32_t seed, float scale) {
    std::mt19937 random(seed);
    std::uniform_real_distribution<float> distribution(-scale, scale);
    std::vector<float> values(count);
    for (auto& value : values) { value = distribution(random); }
    return values;
}

int fail(const std::string& message) {
    std::cerr << message << '\n';
    return 1;
}

struct Fixture {
    QType qkv_type;
    QType z_type;
    std::vector<std::uint8_t> qkv_host;
    std::vector<std::uint8_t> z_host;
    std::vector<std::uint8_t> output_host;
    GuardedBuffer qkv_data;
    GuardedBuffer z_data;
    GuardedBuffer output_data;
    GuardedBuffer gather_data;
    Weight qkv;
    Weight z;
    Weight output;
    ops::GgufProjectionWeights projection;

    Fixture(QType qkv_qtype, QType z_qtype)
        : qkv_type(qkv_qtype), z_type(z_qtype),
          qkv_host(packed_weight(qkv_type, qkv_type == z_type ? kQkv + kValue : kQkv,
                                 kHidden, 17)),
          z_host(qkv_type == z_type ? std::vector<std::uint8_t>{}
                                    : packed_weight(z_type, kValue, kHidden, 29)),
          output_host(packed_weight(QType::GGUF_Q4_K, kHidden, kValue, 41)),
          qkv_data(qkv_host.size()), z_data(std::max<std::size_t>(1, z_host.size())),
          output_data(output_host.size()),
          gather_data(static_cast<std::size_t>(kValue) * sizeof(std::int32_t)),
          qkv(make_weight(qkv_type, qkv_data.data(), qkv_type == z_type ? kQkv + kValue : kQkv,
                          kHidden)),
          z(qkv_type == z_type ? Weight{} : make_weight(z_type, z_data.data(), kValue, kHidden)),
          output(make_weight(QType::GGUF_Q4_K, output_data.data(), kHidden, kValue)) {
        qkv_data.upload(qkv_host.data());
        if (!z_host.empty()) { z_data.upload(z_host.data()); }
        output_data.upload(output_host.data());
        std::vector<std::int32_t> gather(kValue);
        for (int index = 0; index < kValue; ++index) { gather[index] = (index * 5) % kValue; }
        gather_data.upload(gather.data());
        output.input_columns = static_cast<const std::int32_t*>(gather_data.data());
        Weight qkv_part = qkv;
        qkv_part.n = qkv_part.shape[0] = qkv_part.padded_shape[0] = kQkv;
        Weight z_part = z;
        if (qkv_type == z_type) {
            const auto block = gguf_block_shape(qkv_type);
            const std::size_t row_bytes = static_cast<std::size_t>(kHidden / block.elements) * block.bytes;
            z_part = qkv;
            z_part.qdata = static_cast<const std::byte*>(qkv.qdata) + kQkv * row_bytes;
            z_part.payload = z_part.qdata;
            z_part.n = z_part.shape[0] = z_part.padded_shape[0] = kValue;
        }
        projection.parts = {{qkv_part, 0, 0}, {z_part, 1, 0}};
    }
};

int run_width(Fixture& fixture, int width, int valid, bool fp16_state = false) {
    const std::size_t hidden_bytes = static_cast<std::size_t>(kHidden) * width * sizeof(std::uint16_t);
    const std::size_t qkv_bytes = static_cast<std::size_t>(kQkv) * width * sizeof(std::uint16_t);
    const std::size_t value_bytes = static_cast<std::size_t>(kValue) * width * sizeof(std::uint16_t);
    const std::size_t qk_bytes = static_cast<std::size_t>(kDim) * kQkHeads * width * sizeof(std::uint16_t);
    const std::size_t recurrent_elements = static_cast<std::size_t>(kDim) * kDim * kValueHeads * kSlots;
    const std::size_t recurrent_bytes = recurrent_elements * (fp16_state ? sizeof(__half) : sizeof(float));
    const std::size_t conv_state_bytes = static_cast<std::size_t>(kQkv) * 3 * kSlots * sizeof(std::uint16_t);
    const std::size_t gate_bytes = static_cast<std::size_t>(2) * kValueHeads * width * sizeof(float);

    auto hidden_host = bf16_values(static_cast<std::size_t>(kHidden) * width, 53 + width, 0.12F);
    auto g_host = fp32_values(static_cast<std::size_t>(kValueHeads) * width, 67 + width, 0.8F);
    auto beta_host = fp32_values(static_cast<std::size_t>(kValueHeads) * width, 79 + width, 0.6F);
    for (auto& value : g_host) { value = -std::abs(value) - 0.02F; }
    for (auto& value : beta_host) { value = std::abs(value) + 0.05F; }
    auto convolution_host = bf16_values(static_cast<std::size_t>(kQkv) * 4, 83, 0.03F);
    auto norm_host = bf16_values(kDim, 89, 0.2F);
    auto conv_state_host = bf16_values(static_cast<std::size_t>(kQkv) * 3 * kSlots, 97, 0.04F);
    auto recurrent_host = fp32_values(recurrent_elements, 101, 0.015F);
    std::vector<__half> recurrent_half_host;
    if (fp16_state) {
        recurrent_half_host.resize(recurrent_elements);
        for (std::size_t i = 0; i < recurrent_elements; ++i) {
            recurrent_half_host[i] = __float2half_rn(recurrent_host[i]);
        }
    }
    std::vector<std::int32_t> valid_host{valid};
    std::vector<std::int32_t> source_host{1};

    GuardedBuffer hidden_data(hidden_bytes), g_data(g_host.size() * sizeof(float));
    GuardedBuffer beta_data(beta_host.size() * sizeof(float));
    GuardedBuffer convolution_data(convolution_host.size() * sizeof(std::uint16_t));
    GuardedBuffer norm_data(norm_host.size() * sizeof(std::uint16_t));
    GuardedBuffer conv_initial(conv_state_bytes), conv_candidate(conv_state_bytes);
    GuardedBuffer recurrent_initial(recurrent_bytes), recurrent_candidate(recurrent_bytes);
    GuardedBuffer valid_data(sizeof(std::int32_t)), source_data(sizeof(std::int32_t));
    GuardedBuffer baseline_conv(qkv_bytes), candidate_conv(qkv_bytes);
    GuardedBuffer projected_q(qk_bytes), projected_k(qk_bytes);
    GuardedBuffer projected_v(value_bytes), projected_z(value_bytes);
    GuardedBuffer baseline_key(qk_bytes), candidate_key(qk_bytes);
    GuardedBuffer baseline_value(value_bytes), candidate_value(value_bytes);
    GuardedBuffer baseline_gate(gate_bytes), candidate_gate(gate_bytes);
    GuardedBuffer baseline_out(value_bytes);
    GuardedBuffer normalized_data(value_bytes);
    GuardedBuffer baseline_residual(hidden_bytes), candidate_residual(hidden_bytes);
    GuardedBuffer workspace_data(64U * 1024U * 1024U);

    hidden_data.upload(hidden_host.data());
    g_data.upload(g_host.data());
    beta_data.upload(beta_host.data());
    convolution_data.upload(convolution_host.data());
    norm_data.upload(norm_host.data());
    conv_initial.upload(conv_state_host.data());
    conv_candidate.upload(conv_state_host.data());
    recurrent_initial.upload(fp16_state ? static_cast<const void*>(recurrent_half_host.data())
                                        : static_cast<const void*>(recurrent_host.data()));
    recurrent_candidate.upload(fp16_state ? static_cast<const void*>(recurrent_half_host.data())
                                          : static_cast<const void*>(recurrent_host.data()));
    valid_data.upload(valid_host.data());
    source_data.upload(source_host.data());
    baseline_conv.fill_payload(0xff);
    candidate_conv.fill_payload(0xff);
    baseline_key.fill_payload(0xff);
    candidate_key.fill_payload(0xff);
    baseline_value.fill_payload(0xff);
    candidate_value.fill_payload(0xff);
    baseline_gate.fill_payload(0xff);
    candidate_gate.fill_payload(0xff);
    baseline_out.fill_payload(0);

    auto residual_host = bf16_values(static_cast<std::size_t>(kHidden) * width, 103, 0.25F);
    baseline_residual.upload(residual_host.data());
    candidate_residual.upload(residual_host.data());

    Tensor hidden(hidden_data.data(), DType::BF16, {kHidden, width});
    Tensor g(g_data.data(), DType::FP32, {kValueHeads, width});
    Tensor beta(beta_data.data(), DType::FP32, {kValueHeads, width});
    Tensor convolution(convolution_data.data(), DType::BF16, {kQkv, 4});
    Tensor norm(norm_data.data(), DType::BF16, {kDim});
    Tensor conv_base(conv_initial.data(), DType::BF16, {kQkv, 3, kSlots});
    Tensor conv_fused(conv_candidate.data(), DType::BF16, {kQkv, 3, kSlots});
    const DType recurrent_dtype = fp16_state ? DType::FP16 : DType::FP32;
    Tensor recurrent_base(recurrent_initial.data(), recurrent_dtype, {kDim, kDim, kValueHeads, kSlots});
    Tensor recurrent_fused(recurrent_candidate.data(), recurrent_dtype, {kDim, kDim, kValueHeads, kSlots});
    Tensor valid_tensor(valid_data.data(), DType::I32, {1});
    Tensor source_tensor(source_data.data(), DType::I32, {1});
    Tensor baseline_conv_tensor(baseline_conv.data(), DType::BF16, {kQkv, width});
    Tensor candidate_conv_tensor(candidate_conv.data(), DType::BF16, {kQkv, width});
    Tensor baseline_q(projected_q.data(), DType::BF16, {kDim, kQkHeads, width});
    Tensor baseline_k(projected_k.data(), DType::BF16, {kDim, kQkHeads, width});
    Tensor candidate_key_tensor(candidate_key.data(), DType::BF16, {kDim, kQkHeads, width});
    Tensor baseline_v(projected_v.data(), DType::BF16, {kDim, kValueHeads, width});
    Tensor baseline_z(projected_z.data(), DType::BF16, {kDim, kValueHeads, width});
    Tensor baseline_key_record(baseline_key.data(), DType::BF16, {kDim, kQkHeads, width});
    Tensor baseline_value_record(baseline_value.data(), DType::BF16, {kDim, kValueHeads, width});
    Tensor candidate_value_tensor(candidate_value.data(), DType::BF16, {kDim, kValueHeads, width});
    Tensor candidate_gate_tensor(candidate_gate.data(), DType::FP32, {2, kValueHeads, width});
    Tensor baseline_gate_tensor(baseline_gate.data(), DType::FP32, {2, kValueHeads, width});
    Tensor baseline_out_tensor(baseline_out.data(), DType::BF16, {kDim, kValueHeads, width});
    Tensor normalized_heads(normalized_data.data(), DType::BF16, {kDim, kValueHeads, width});
    Tensor normalized(normalized_data.data(), DType::BF16, {kValue, width});
    Tensor baseline_residual_tensor(baseline_residual.data(), DType::BF16, {kHidden, width});
    Tensor candidate_residual_tensor(candidate_residual.data(), DType::BF16, {kHidden, width});

    cudaStream_t stream = nullptr;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
    WorkspaceArena workspace(DeviceSpan{workspace_data.data(), workspace_data.bytes()});
    ops::gdn_input_proj_conv_record(hidden, fixture.projection, convolution, conv_base,
                                    valid_tensor, source_tensor, baseline_conv_tensor,
                                    baseline_q, baseline_k, baseline_v, baseline_z,
                                    workspace, stream);
    ops::gated_delta_net_replay_record(
        baseline_q, baseline_k, baseline_v, g, beta, 1.0F / std::sqrt(128.0F),
        recurrent_base, valid_tensor, source_tensor, baseline_key_record,
        baseline_value_record, baseline_gate_tensor,
        baseline_out_tensor, stream);
    ops::gated_rmsnorm(baseline_out_tensor, norm, baseline_z, 1.0e-6F, normalized_heads, stream);
    ops::detail::gguf_linear_add(normalized, fixture.output, baseline_residual_tensor, workspace, stream);
    check(cudaStreamSynchronize(stream), "complete baseline sequence");

    ops::GdnPersistentState state;
    state.action = ops::GdnPersistentStateAction::Record;
    state.conv_states = conv_fused;
    state.recurrent_states = recurrent_fused;
    state.source_slots = source_tensor;
    state.valid_columns = valid_tensor;
    state.records = {candidate_conv_tensor, candidate_key_tensor, candidate_value_tensor,
                     candidate_gate_tensor};
    const bool executed = ops::gdn_persistent_mixer(hidden, g, beta, fixture.projection,
        convolution, norm, fixture.output, 1.0e-6F, state, candidate_residual_tensor,
        workspace, stream);
    if (!executed) {
        check(cudaStreamDestroy(stream), "destroy stream");
        return fail("eligible GDN persistent mixer declined width " + std::to_string(width));
    }
    check(cudaStreamSynchronize(stream), "complete fused sequence");

    const auto compare = [](const GuardedBuffer& lhs, const GuardedBuffer& rhs,
                            const std::string& label) {
        std::vector<std::byte> a(lhs.bytes()), b(rhs.bytes());
        lhs.download(a.data());
        rhs.download(b.data());
        return a == b ? 0 : fail(label + " differs from baseline bytes");
    };
    int failures = 0;
    failures += compare(baseline_conv, candidate_conv, "conv record");
    failures += compare(baseline_key, candidate_key, "key record");
    failures += compare(baseline_value, candidate_value, "value record");
    failures += compare(baseline_gate, candidate_gate, "gate record");
    failures += compare(baseline_residual, candidate_residual, "residual output");

    std::vector<std::uint16_t> conv_after(conv_state_host.size());
    std::vector<float> recurrent_after_fp32;
    std::vector<__half> recurrent_after_fp16;
    conv_candidate.download(conv_after.data());
    if (conv_after != conv_state_host) { failures += fail("Record mutated convolution state pool"); }
    if (fp16_state) {
        recurrent_after_fp16.resize(recurrent_elements);
        recurrent_candidate.download(recurrent_after_fp16.data());
        if (std::memcmp(recurrent_after_fp16.data(), recurrent_half_host.data(), recurrent_bytes) != 0) {
            failures += fail("Record mutated FP16 recurrent state pool");
        }
    } else {
        recurrent_after_fp32.resize(recurrent_elements);
        recurrent_candidate.download(recurrent_after_fp32.data());
        if (std::memcmp(recurrent_after_fp32.data(), recurrent_host.data(), recurrent_bytes) != 0) {
            failures += fail("Record mutated FP32 recurrent state pool");
        }
    }

    const std::array<const GuardedBuffer*, 28> guarded{
        &hidden_data, &g_data, &beta_data, &convolution_data, &norm_data,
        &conv_initial, &conv_candidate, &recurrent_initial, &recurrent_candidate,
        &valid_data, &source_data, &baseline_conv, &candidate_conv,
        &projected_q, &projected_k, &projected_v, &projected_z, &baseline_key,
        &candidate_key, &baseline_value, &candidate_value, &baseline_gate,
        &candidate_gate, &baseline_out, &normalized_data, &baseline_residual,
        &candidate_residual, &workspace_data};
    for (const auto* buffer : guarded) {
        if (!buffer->guards_intact()) { failures += fail("GDN persistent mixer crossed a buffer guard"); break; }
    }

    if (width == 4 && valid == 4 && failures == 0) {
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t executable = nullptr;
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "begin graph capture");
        const bool captured = ops::gdn_persistent_mixer(hidden, g, beta, fixture.projection,
            convolution, norm, fixture.output, 1.0e-6F, state, candidate_residual_tensor,
            workspace, stream);
        if (!captured) { throw std::runtime_error("persistent route declined during graph capture"); }
        check(cudaStreamEndCapture(stream, &graph), "end graph capture");
        check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), "instantiate graph");
        for (int replay = 0; replay < 2; ++replay) {
            candidate_residual.upload_async(residual_host.data(), stream);
            check(cudaGraphLaunch(executable, stream), "launch graph replay");
            check(cudaStreamSynchronize(stream), "synchronize graph replay");
            failures += compare(baseline_conv, candidate_conv, "graph conv record");
            failures += compare(baseline_key, candidate_key, "graph key record");
            failures += compare(baseline_value, candidate_value, "graph value record");
            failures += compare(baseline_gate, candidate_gate, "graph gate record");
            failures += compare(baseline_residual, candidate_residual, "graph residual");
            conv_candidate.download(conv_after.data());
            if (conv_after != conv_state_host) {
                failures += fail("graph Record mutated convolution state pool");
            }
            if (fp16_state) {
                recurrent_candidate.download(recurrent_after_fp16.data());
                if (std::memcmp(recurrent_after_fp16.data(), recurrent_half_host.data(), recurrent_bytes) != 0) {
                    failures += fail("graph Record mutated FP16 recurrent state pool");
                }
            } else {
                recurrent_candidate.download(recurrent_after_fp32.data());
                if (std::memcmp(recurrent_after_fp32.data(), recurrent_host.data(), recurrent_bytes) != 0) {
                    failures += fail("graph Record mutated FP32 recurrent state pool");
                }
            }
        }
        check(cudaGraphExecDestroy(executable), "destroy graph executable");
        check(cudaGraphDestroy(graph), "destroy graph");
    }

    GuardedBuffer destination_data(sizeof(std::int32_t));
    const std::int32_t destination_slot = 2;
    destination_data.upload(&destination_slot);
    state.destination_slots = Tensor(destination_data.data(), DType::I32, {1});
    const std::size_t workspace_before = workspace.used();
    if (ops::gdn_persistent_mixer(hidden, g, beta, fixture.projection, convolution, norm,
                                  fixture.output, 1.0e-6F, state, candidate_residual_tensor,
                                  workspace, stream)) {
        failures += fail("Record route accepted a destination selector");
    }
    check(cudaStreamSynchronize(stream), "complete fallback check");
    if (workspace.used() != workspace_before) { failures += fail("fallback consumed workspace"); }
    failures += compare(baseline_conv, candidate_conv, "fallback conv record");
    failures += compare(baseline_key, candidate_key, "fallback key record");
    failures += compare(baseline_value, candidate_value, "fallback value record");
    failures += compare(baseline_gate, candidate_gate, "fallback gate record");
    failures += compare(baseline_residual, candidate_residual, "fallback residual");
    if (!workspace_data.guards_intact() || !destination_data.guards_intact()) {
        failures += fail("GDN persistent mixer crossed a fallback buffer guard");
    }

    check(cudaStreamDestroy(stream), "destroy stream");
    return failures;
}

int run() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) { return 77; }
    int failures = 0;
    for (const auto pair : {std::pair{QType::GGUF_Q4_K, QType::GGUF_Q4_K},
                            std::pair{QType::GGUF_Q6_K, QType::GGUF_Q4_K}}) {
        Fixture fixture(pair.first, pair.second);
        for (int width = 2; width <= 4; ++width) {
            for (int valid = 1; valid <= width; ++valid) {
                failures += run_width(fixture, width, valid);
            }
        }
        failures += run_width(fixture, 4, 4, true);
        if (!fixture.qkv_data.guards_intact() || !fixture.z_data.guards_intact() ||
            !fixture.output_data.guards_intact() || !fixture.gather_data.guards_intact()) {
            failures += fail("GDN persistent mixer crossed a weight buffer guard");
        }
    }
    std::cout << (failures == 0 ? "OK" : "FAIL") << " gdn_persistent_mixer\n";
    return failures == 0 ? 0 : 1;
}

}

int main() {
    try { return run(); }
    catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
