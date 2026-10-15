#include "models/qwen3_5/execution/ffn.h"
#include "models/qwen3_5/execution/rotation.h"

#include "core/layout.h"
#include "ninfer/ops/hadamard_transform.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/silu_mul.h"
#include "ops/linear/gguf/gguf_linear.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::models::qwen3_5::execution {
namespace {

bool rotated_dense(const DenseParameters& p) {
    return rotated(p.gate_up.hadamard_signs) || rotated(p.down.hadamard_signs);
}

// A Hadamard-rotated Dense FFN projects gate/up into one plane from the rotated input; when down
// is rotated, silu_mul_hadamard writes its input already in the rotated basis.
std::size_t rotated_dense_workspace_bytes(const DenseParameters& p, std::int32_t first,
                                          std::int32_t last) {
    const auto& gu   = p.gate_up.weight;
    const auto& down = p.down.weight;
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {gu.n, last});
    {
        auto scope = layout.scope();
        (void)layout.alloc_bytes(
            rotated_workspace_bytes(p.gate_up.hadamard_signs, gu.k, last,
                                    ops::linear_workspace_capacity_bytes(
                                        gu.qtype, gu.n, gu.k, p.gate_up.policy, first, last)));
    }
    (void)layout.alloc(DType::BF16, {gu.n / 2, last});
    (void)layout.alloc_bytes(ops::linear_add_workspace_capacity_bytes(down.qtype, down.n, down.k,
                                                                      p.down.policy, first, last));
    return layout.peak_bytes(1);
}

void rotated_dense_ffn(const Tensor& hidden, const DenseParameters& p, Tensor& residual,
                       WorkspaceArena& workspace, cudaStream_t stream, InputBasis basis) {
    const auto columns      = hidden.ne[1];
    const auto& gu          = p.gate_up.weight;
    const auto intermediate = gu.n / 2;
    Tensor plane            = workspace.alloc(DType::BF16, {gu.n, columns});
    {
        auto call      = workspace.scope();
        const Tensor x = rotated_input(hidden, p.gate_up.hadamard_signs, workspace, stream, basis);
        ops::linear(x, gu, plane, p.gate_up.policy, workspace, stream);
    }
    Tensor activation = workspace.alloc(DType::BF16, {intermediate, columns});
    if (rotated(p.down.hadamard_signs)) {
        ops::silu_mul_hadamard(plane, p.down.hadamard_signs, activation, stream);
    } else {
        ops::silu_mul(plane.slice(0, 0, intermediate), plane.slice(0, intermediate, intermediate),
                      activation, stream);
    }
    ops::linear_add(activation, p.down.weight, residual, p.down.policy, workspace, stream);
}

} // namespace

std::size_t ffn_workspace_bytes(const FfnParameters& parameters, std::int32_t first,
                                std::int32_t last, bool mtp, bool verify) {
    if (first <= 0 || last < first) { throw std::invalid_argument("FFN: invalid column interval"); }
    if (const auto* moe = std::get_if<ops::SparseMoeWeights>(&parameters)) {
        return ops::sparse_moe_workspace_capacity_bytes(moe->routed_gate_up.qtype,
                                                        moe->routed_down.qtype, first, last);
    }
    const auto& p = std::get<DenseParameters>(parameters);
    if (rotated_dense(p)) { return rotated_dense_workspace_bytes(p, first, last); }
    const auto& gu   = p.gate_up.weight;
    const auto& down = p.down.weight;
    if (is_gguf(gu.qtype)) {
        const auto fused = !p.up
            ? ops::detail::gguf_swiglu_linear_add_workspace_capacity_bytes(
                  gu, nullptr, down, first, last)
            : 0;
        WorkspaceLayoutBuilder layout;
        const std::int32_t rows = p.up ? gu.n : gu.n / 2;
        (void)layout.alloc(DType::BF16, {rows, last});
        {
            auto scope = layout.scope();
            (void)layout.alloc_bytes(
                p.up ? ops::linear_swiglu_pair_workspace_capacity_bytes(
                           gu.qtype, p.up->weight.qtype, gu.n, gu.k, p.gate_up.policy, first, last)
                     : ops::linear_swiglu_workspace_capacity_bytes(gu.qtype, gu.n, gu.k,
                                                                   p.gate_up.policy, first, last));
        }
        {
            auto scope = layout.scope();
            (void)layout.alloc_bytes(ops::linear_add_workspace_capacity_bytes(
                down.qtype, down.n, down.k, p.down.policy, first, last));
        }
        return std::max(fused, layout.peak_bytes(1));
    }
    WorkspaceLayoutBuilder layout;
    if (mtp || gu.qtype == QType::Q6_G64_FP16) {
        (void)layout.alloc(DType::BF16, {gu.n, last});
        {
            auto scope = layout.scope();
            (void)layout.alloc_bytes(ops::linear_workspace_capacity_bytes(
                gu.qtype, gu.n, gu.k, p.gate_up.policy, first, last));
        }
        (void)layout.alloc(DType::BF16, {gu.n / 2, last});
        if (mtp) {
            (void)layout.alloc(DType::BF16, {down.n, last});
            (void)layout.alloc_bytes(ops::linear_workspace_capacity_bytes(
                down.qtype, down.n, down.k, p.down.policy, first, last));
        } else {
            auto scope = layout.scope();
            (void)layout.alloc_bytes(ops::linear_add_workspace_capacity_bytes(
                down.qtype, down.n, down.k, p.down.policy, first, last));
        }
    } else {
        (void)layout.alloc(DType::BF16, {gu.n / 2, last});
        {
            auto scope = layout.scope();
            (void)layout.alloc_bytes(ops::linear_swiglu_workspace_capacity_bytes(
                gu.qtype, gu.n, gu.k, verify ? p.verify_gate_up_policy : p.gate_up.policy, first,
                last));
        }
        {
            auto scope = layout.scope();
            (void)layout.alloc_bytes(ops::linear_add_workspace_capacity_bytes(
                down.qtype, down.n, down.k, p.down.policy, first, last));
        }
    }
    return layout.peak_bytes(1);
}

const Tensor* ffn_input_signs(const FfnParameters& parameters) {
    const auto* dense = std::get_if<DenseParameters>(&parameters);
    return dense != nullptr && rotated(dense->gate_up.hadamard_signs)
               ? &dense->gate_up.hadamard_signs
               : nullptr;
}

bool ffn_prequantized_q8_input_eligible(const FfnParameters& parameters,
                                        std::int32_t columns, InputBasis basis) {
#if defined(NINFER_GGUF_FFN_RMSNORM_Q8_FUSION)
    if (columns < 1 || columns > 8 || basis != InputBasis::Primal) { return false; }
    const auto* p = std::get_if<DenseParameters>(&parameters);
    if (p == nullptr || p->up || rotated_dense(*p)) { return false; }
    const auto& gate = p->gate_up.weight;
    const auto& down = p->down.weight;
    return gate.qtype == QType::GGUF_Q4_K && gate.layout == QuantLayout::GgufBlocks &&
           gate.qdata != nullptr && gate.n == 34816 && gate.k == 5120 &&
           gate.input_columns == nullptr && down.qtype == QType::GGUF_Q6_K &&
           down.layout == QuantLayout::GgufBlocks && down.qdata != nullptr && down.n == 5120 &&
           down.k == 17408 && down.input_columns == nullptr;
#else
    (void)parameters;
    (void)columns;
    (void)basis;
    return false;
#endif
}

void ffn(const Tensor& hidden, const FfnParameters& parameters, Tensor& residual,
         const ops::SparseMoeHints& hints, WorkspaceArena& workspace, cudaStream_t stream, bool mtp,
         bool verify, InputBasis basis, const void* prequantized_q8_input) {
    auto scope         = workspace.scope();
    const auto columns = hidden.ne[1];
    if (prequantized_q8_input != nullptr &&
        (mtp || !ffn_prequantized_q8_input_eligible(parameters, columns, basis))) {
        throw std::invalid_argument("FFN: unsupported prequantized q8 input");
    }
    if (const auto* moe = std::get_if<ops::SparseMoeWeights>(&parameters)) {
        const auto storage =
            workspace.alloc_bytes(ffn_workspace_bytes(parameters, columns, columns));
        WorkspaceArena scratch(storage);
        ops::sparse_moe(hidden, *moe, ops::SparseMoeEpilogue::AddResidual, residual, hints, scratch,
                        stream);
        return;
    }
    const auto& p = std::get<DenseParameters>(parameters);
    if (prequantized_q8_input != nullptr) {
        if (ops::detail::gguf_swiglu_linear_add_prequantized(
                prequantized_q8_input, p.gate_up.weight.k, columns, p.gate_up.weight, nullptr,
                p.down.weight, residual, workspace, stream)) {
            return;
        }
        Tensor activation = workspace.alloc(DType::BF16, {p.gate_up.weight.n / 2, columns});
        if (!ops::detail::gguf_swiglu_prequantized(prequantized_q8_input, p.gate_up.weight.k,
                                                  columns, p.gate_up.weight, activation, stream)) {
            throw std::logic_error("FFN: prepared q8 SwiGLU route became unavailable");
        }
        ops::linear_add(activation, p.down.weight, residual, p.down.policy, workspace, stream);
        return;
    }
    if (rotated_dense(p)) {
        rotated_dense_ffn(hidden, p, residual, workspace, stream, basis);
        return;
    }
    const auto& gu   = p.gate_up.weight;
    const auto& down = p.down.weight;
    if (is_gguf(gu.qtype)) {
        if (ops::detail::gguf_swiglu_linear_add(hidden, gu, p.up ? &p.up->weight : nullptr, down,
                                                residual, workspace, stream)) {
            return;
        }
        Tensor activation = workspace.alloc(DType::BF16, {p.up ? gu.n : gu.n / 2, columns});
        {
            auto call = workspace.scope();
            if (p.up) {
                ops::linear_swiglu(hidden, gu, p.up->weight, activation, p.gate_up.policy,
                                   workspace, stream);
            } else {
                ops::linear_swiglu(hidden, gu, activation, p.gate_up.policy, workspace, stream);
            }
        }
        ops::linear_add(activation, down, residual, p.down.policy, workspace, stream);
        return;
    }
    if (mtp || gu.qtype == QType::Q6_G64_FP16) {
        Tensor gate_up = workspace.alloc(DType::BF16, {gu.n, columns});
        {
            auto call = workspace.scope();
            ops::linear(hidden, gu, gate_up, p.gate_up.policy, workspace, stream);
        }
        Tensor activation = workspace.alloc(DType::BF16, {gu.n / 2, columns});
        ops::silu_mul(gate_up.slice(0, 0, gu.n / 2), gate_up.slice(0, gu.n / 2, gu.n / 2),
                      activation, stream);
        if (mtp) {
            Tensor delta = workspace.alloc(DType::BF16, {down.n, columns});
            ops::linear(activation, down, delta, p.down.policy, workspace, stream);
            ops::residual_add(delta, residual, stream);
            return;
        }
        ops::linear_add(activation, down, residual, p.down.policy, workspace, stream);
        return;
    }
    Tensor activation = workspace.alloc(DType::BF16, {gu.n / 2, columns});
    {
        auto call = workspace.scope();
        ops::linear_swiglu(hidden, gu, activation,
                           verify ? p.verify_gate_up_policy : p.gate_up.policy, workspace, stream);
    }
    ops::linear_add(activation, down, residual, p.down.policy, workspace, stream);
}

} // namespace ninfer::models::qwen3_5::execution
