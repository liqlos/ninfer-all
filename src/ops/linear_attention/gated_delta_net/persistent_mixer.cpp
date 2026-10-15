#include "ninfer/ops/gdn_persistent_mixer.h"

#include "ops/linear/gguf/ggml_bridge.h"
#include "ops/linear/gguf/ggml_gdn_persistent.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

namespace gguf = ninfer::ops::gguf;
namespace bridge = ninfer::ops::gguf::detail;

constexpr int kHidden = 5120;
constexpr int kQkv = 10240;
constexpr int kZ = 6144;
constexpr int kQkHeads = 16;
constexpr int kValueHeads = 48;
constexpr int kDim = 128;
constexpr int kMaxColumns = 4;
constexpr int kMaxParts = 8;

bool aligned_to(const void* p, std::uintptr_t alignment) {
    return p != nullptr && (reinterpret_cast<std::uintptr_t>(p) & (alignment - 1)) == 0;
}

void require_tensor(const Tensor& t, DType dtype, const std::array<std::int32_t, 4>& shape,
                    std::uintptr_t alignment, const char* name) {
    if (t.dtype != dtype || t.data == nullptr || !t.is_contiguous() || !aligned_to(t.data, alignment)) {
        throw std::invalid_argument(std::string("gdn_persistent_mixer: invalid ") + name);
    }
    for (int i = 0; i < 4; ++i) {
        if (t.ne[i] != shape[static_cast<std::size_t>(i)]) {
            throw std::invalid_argument(std::string("gdn_persistent_mixer: invalid ") + name);
        }
    }
}

std::size_t align_bytes(std::size_t n) { return (n + 255) & ~std::size_t(255); }

bool build_projection(const GgufProjectionWeights& projection, bridge::GdnPersistentLaunchArgs& args) {
    if (projection.parts.empty() || projection.parts.size() > kMaxParts ||
        projection.hadamard_signs.data != nullptr) {
        return false;
    }
    std::array<std::array<bool, kQkv>, 2> covered{};
    args.part_count = static_cast<int>(projection.parts.size());
    for (std::size_t i = 0; i < projection.parts.size(); ++i) {
        const GgufProjectionPart& part = projection.parts[i];
        if ((part.weight.qtype != QType::GGUF_Q4_K && part.weight.qtype != QType::GGUF_Q6_K) ||
            part.weight.k != kHidden || part.weight.n <= 0 || !aligned_to(part.weight.qdata, 4) ||
            part.weight.input_columns != nullptr || part.output < 0 || part.output > 1 ||
            part.row < 0 || part.row + part.weight.n > (part.output == 0 ? kQkv : kZ)) {
            return false;
        }
        const auto type = part.weight.qtype == QType::GGUF_Q4_K ? gguf::GgmlType::Q4_K
                                                               : gguf::GgmlType::Q6_K;
        const auto block = gguf::block_shape(type);
        const std::int64_t row_bytes = std::int64_t(part.weight.k / block.elements) * block.bytes;
        for (int row = part.row; row < part.row + part.weight.n; ++row) {
            if (covered[static_cast<std::size_t>(part.output)][static_cast<std::size_t>(row)]) {
                return false;
            }
            covered[static_cast<std::size_t>(part.output)][static_cast<std::size_t>(row)] = true;
        }
        args.parts[i] = {static_cast<const std::uint8_t*>(part.weight.qdata), row_bytes,
                         static_cast<int>(part.weight.qtype), part.weight.n, part.row, part.output};
    }
    for (int row = 0; row < kQkv; ++row) {
        if (!covered[0][static_cast<std::size_t>(row)]) { return false; }
    }
    for (int row = 0; row < kZ; ++row) {
        if (!covered[1][static_cast<std::size_t>(row)]) { return false; }
    }
    return true;
}

}

std::size_t gdn_persistent_mixer_workspace_bytes(std::int32_t columns) {
    if (columns < 2 || columns > kMaxColumns) { return 0; }
    const std::size_t t = static_cast<std::size_t>(columns);
    const std::size_t q8 = gguf::vector_activation_bytes(6144, columns);
#if defined(NINFER_GDN_PERSISTENT_PROJECTION_CONV_FUSION)
    return align_bytes(q8) + align_bytes(sizeof(__nv_bfloat16) * t * kQkv) +
           align_bytes(sizeof(__nv_bfloat16) * t * kZ) +
           align_bytes(sizeof(__nv_bfloat16) * t * 6144) * 2;
#else
    return align_bytes(q8) + align_bytes(sizeof(__nv_bfloat16) * t * kQkv) * 2 +
           align_bytes(sizeof(__nv_bfloat16) * t * kZ) +
           align_bytes(sizeof(__nv_bfloat16) * t * 6144) * 2;
#endif
}

bool gdn_persistent_mixer(const Tensor& hidden, const Tensor& g, const Tensor& beta,
                          const GgufProjectionWeights& projection, const Tensor& convolution,
                          const Tensor& norm_weight, const Weight& output_projection,
                          float rms_norm_eps, GdnPersistentState& state, Tensor& residual,
                          WorkspaceArena& workspace, cudaStream_t stream) {
    const int columns = hidden.ne[1];
    if (columns < 2 || columns > kMaxColumns || hidden.ne[2] != 1 || hidden.ne[3] != 1 ||
        state.action != GdnPersistentStateAction::Record ||
        output_projection.qtype != QType::GGUF_Q4_K || output_projection.n != kHidden ||
        output_projection.k != 6144 || output_projection.qdata == nullptr ||
        output_projection.input_columns == nullptr || !aligned_to(output_projection.input_columns, 4) ||
        !aligned_to(output_projection.qdata, 4)) {
        return false;
    }
    if (!std::isfinite(rms_norm_eps) || rms_norm_eps <= 0.0f ||
        state.destination_slots.data != nullptr) {
        return false;
    }
    bridge::GdnPersistentLaunchArgs args{};
    args.columns = columns;
    args.action = static_cast<int>(state.action);
    args.state_fp16 = state.recurrent_states.dtype == DType::FP16;
    args.eps = rms_norm_eps;
    if (!build_projection(projection, args)) { return false; }
    if (state.conv_states.dtype != DType::BF16 || state.conv_states.ne[0] != kQkv ||
        state.conv_states.ne[1] != 3 || state.conv_states.ne[2] <= 0 || state.conv_states.ne[3] != 1 ||
        !state.conv_states.is_contiguous() || !aligned_to(state.conv_states.data, 2) ||
        (state.recurrent_states.dtype != DType::FP32 && state.recurrent_states.dtype != DType::FP16) ||
        state.recurrent_states.ne[0] != kDim || state.recurrent_states.ne[1] != kDim ||
        state.recurrent_states.ne[2] != kValueHeads || state.recurrent_states.ne[3] <= 0 ||
        state.conv_states.ne[2] != state.recurrent_states.ne[3] ||
        !state.recurrent_states.is_contiguous() ||
        !aligned_to(state.recurrent_states.data, state.recurrent_states.dtype == DType::FP32 ? 4 : 2)) {
        return false;
    }
    if (state.source_slots.dtype != DType::I32 || state.source_slots.ne[0] != 1 ||
        state.source_slots.ne[1] != 1 || state.source_slots.ne[2] != 1 || state.source_slots.ne[3] != 1 ||
        !state.source_slots.is_contiguous() || !aligned_to(state.source_slots.data, 4) ||
        state.valid_columns.dtype != DType::I32 || state.valid_columns.ne[0] != 1 ||
        state.valid_columns.ne[1] != 1 || state.valid_columns.ne[2] != 1 || state.valid_columns.ne[3] != 1 ||
        !state.valid_columns.is_contiguous() || !aligned_to(state.valid_columns.data, 4)) {
        return false;
    }
    if (state.records.conv.dtype != DType::BF16 || state.records.conv.ne[0] != kQkv ||
        state.records.conv.ne[1] != columns || state.records.conv.ne[2] != 1 || state.records.conv.ne[3] != 1 ||
        !state.records.conv.is_contiguous() || !aligned_to(state.records.conv.data, 2) ||
        state.records.key.dtype != DType::BF16 || state.records.key.ne[0] != kDim ||
        state.records.key.ne[1] != kQkHeads || state.records.key.ne[2] != columns || state.records.key.ne[3] != 1 ||
        !state.records.key.is_contiguous() || !aligned_to(state.records.key.data, 4) ||
        state.records.value.dtype != DType::BF16 || state.records.value.ne[0] != kDim ||
        state.records.value.ne[1] != kValueHeads || state.records.value.ne[2] != columns ||
        state.records.value.ne[3] != 1 || !state.records.value.is_contiguous() || !aligned_to(state.records.value.data, 4) ||
        state.records.gate.dtype != DType::FP32 || state.records.gate.ne[0] != 2 ||
        state.records.gate.ne[1] != kValueHeads || state.records.gate.ne[2] != columns ||
        state.records.gate.ne[3] != 1 || !state.records.gate.is_contiguous() || !aligned_to(state.records.gate.data, alignof(uint2))) {
        return false;
    }
    require_tensor(hidden, DType::BF16, {kHidden, columns, 1, 1}, 2, "hidden");
    require_tensor(g, DType::FP32, {kValueHeads, columns, 1, 1}, 4, "g");
    require_tensor(beta, DType::FP32, {kValueHeads, columns, 1, 1}, 4, "beta");
    require_tensor(convolution, DType::BF16, {kQkv, 4, 1, 1}, 2, "convolution");
    require_tensor(norm_weight, DType::BF16, {kDim, 1, 1, 1}, 2, "norm weight");
    require_tensor(residual, DType::BF16, {kHidden, columns, 1, 1}, 2, "residual");

    auto scope = workspace.scope();
    const std::size_t required = gdn_persistent_mixer_workspace_bytes(columns);
    if (required == 0 || workspace.capacity() - workspace.used() < required) { return false; }
    void* scratch = workspace.alloc_bytes(required).data;
    args.hidden = hidden.data;
    args.g = static_cast<const float*>(g.data);
    args.beta = static_cast<const float*>(beta.data);
    args.convolution = convolution.data;
    args.norm_weight = norm_weight.data;
    args.output_columns = output_projection.input_columns;
    args.output_weight = static_cast<const std::uint8_t*>(output_projection.qdata);
    const auto block = gguf::block_shape(gguf::GgmlType::Q4_K);
    args.output_row_bytes = std::int64_t(output_projection.k / block.elements) * block.bytes;
    args.residual = residual.data;
    args.residual_out = residual.data;
    args.conv_states = state.conv_states.data;
    args.recurrent_states = state.recurrent_states.data;
    args.source_slots = static_cast<const std::int32_t*>(state.source_slots.data);
    args.valid_columns = static_cast<const std::int32_t*>(state.valid_columns.data);
    args.conv_record = state.records.conv.data;
    args.key_record = state.records.key.data;
    args.value_record = state.records.value.data;
    args.gate_record = state.records.gate.data;
    args.workspace = scratch;
    return bridge::gdn_persistent_mixer_launch(args, stream);
}

}
