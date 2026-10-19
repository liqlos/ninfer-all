#include "ops/linear/gguf/gguf_linear.h"
#include "ops/linear/gguf/ggml_bridge_dual_mmq.h"

#include "core/layout.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::ops::detail {
namespace {

// Rows the dequantize-and-GEMM route converts per pass for a type with no integer kernel.
constexpr std::size_t kDequantizedScratchBytes = std::size_t{16} << 20;
// Up to this many columns such a type loops the vector kernel instead: those are decode and verify
// batches, captured in graphs, where a cuBLAS call is best avoided.
constexpr std::int32_t kDequantizedVectorColumns = 64;
// The widest stream-k fixup plane any launch can ask for: one 128 x 128 FP32 tile per SM.
constexpr std::size_t kFixupTileBytes = 128 * 128 * sizeof(float);

std::int64_t row_bytes(const Weight& w) {
    const auto block = gguf_block_shape(w.qtype);
    return std::int64_t(w.k / block.elements) * block.bytes;
}

bool matrix_kernel(const Weight& w) { return gguf::has_matrix_kernel(gguf_type(w.qtype)); }

gguf::Epilogue bridge_epilogue(GgufEpilogue epilogue) {
    switch (epilogue) {
    case GgufEpilogue::Store: return gguf::Epilogue::Store;
    case GgufEpilogue::Accumulate: return gguf::Epilogue::Accumulate;
    case GgufEpilogue::GateProduct: return gguf::Epilogue::GateProduct;
    }
    throw std::invalid_argument("gguf: invalid epilogue");
}

// The destination of columns [first, ...) of a product.
gguf::VectorOutput output(const GgufProduct& p, std::int32_t first) {
    gguf::VectorOutput out;
    out.epilogue = bridge_epilogue(p.epilogue);
    if (p.f32 != nullptr) {
        out.f32           = p.f32 + std::int64_t(first) * p.f32_rows + p.row;
        out.column_stride = p.f32_rows;
    } else {
        out.bf16 = static_cast<__nv_bfloat16*>(p.out->data) + std::int64_t(first) * p.out->ne[0] +
                   p.row;
        out.column_stride = p.out->ne[0];
    }
    if (p.epilogue == GgufEpilogue::GateProduct) {
        out.gate               = p.gate + std::int64_t(first) * p.weight->n;
        out.gate_column_stride = p.weight->n;
    }
    return out;
}

void require_activation(const Tensor& x, const char* label) {
    if (x.dtype != DType::BF16 || x.ne[0] <= 0 || x.ne[1] <= 0 || x.ne[2] != 1 || x.ne[3] != 1 ||
        !x.is_contiguous() || x.data == nullptr) {
        throw std::invalid_argument(std::string(label) + ": x must be contiguous BF16 [K,T]");
    }
}

void require_product(const Tensor& x, const GgufProduct& p) {
    require_gguf(*p.weight, "gguf product");
    if (p.weight->k != x.ne[0]) { throw std::invalid_argument("gguf product: K differs from x"); }
    if (p.f32 != nullptr) {
        if (p.row < 0 || p.row + p.weight->n > p.f32_rows) {
            throw std::invalid_argument("gguf product: rows exceed the FP32 plane");
        }
        return;
    }
    if (p.out == nullptr || p.out->dtype != DType::BF16 || !p.out->is_contiguous() ||
        p.out->ne[1] != x.ne[1] || p.row < 0 || p.row + p.weight->n > p.out->ne[0]) {
        throw std::invalid_argument("gguf product: rows exceed the BF16 destination");
    }
    if (p.epilogue == GgufEpilogue::GateProduct && p.gate == nullptr) {
        throw std::invalid_argument("gguf product: the gate product needs its gate plane");
    }
}

std::size_t fixup_bound() {
    return std::size_t(gguf::multiprocessor_count()) * kFixupTileBytes;
}

#if defined(NINFER_GGUF_SWIGLU_Q8_FUSION) || defined(NINFER_GGUF_PERSISTENT_FFN)
bool fused_swiglu_q8_weights_admit(const Weight& gate, const Weight* up, const Weight& down,
                                   std::int32_t min_tokens, std::int32_t max_tokens) {
    return up == nullptr &&
           (gate.qtype == QType::GGUF_Q4_K || gate.qtype == QType::GGUF_Q6_K) &&
           is_gguf(down.qtype) && gate.layout == QuantLayout::GgufBlocks &&
           down.layout == QuantLayout::GgufBlocks && gate.qdata != nullptr && down.qdata != nullptr &&
           down.input_columns == nullptr && gate.n > 0 && gate.n % 512 == 0 && gate.k > 0 &&
           gate.k % 256 == 0 && down.k == gate.n / 2 && down.n > 0 && min_tokens > 0 &&
           max_tokens >= min_tokens && max_tokens <= 8;
}
#endif

} // namespace

gguf::GgmlType gguf_type(QType format) {
    switch (format) {
    case QType::GGUF_Q8_0: return gguf::GgmlType::Q8_0;
    case QType::GGUF_Q2_K: return gguf::GgmlType::Q2_K;
    case QType::GGUF_Q3_K: return gguf::GgmlType::Q3_K;
    case QType::GGUF_Q4_K: return gguf::GgmlType::Q4_K;
    case QType::GGUF_Q5_K: return gguf::GgmlType::Q5_K;
    case QType::GGUF_Q6_K: return gguf::GgmlType::Q6_K;
    case QType::GGUF_IQ2_XXS: return gguf::GgmlType::IQ2_XXS;
    case QType::GGUF_IQ2_XS: return gguf::GgmlType::IQ2_XS;
    case QType::GGUF_IQ2_S: return gguf::GgmlType::IQ2_S;
    case QType::GGUF_IQ3_XXS: return gguf::GgmlType::IQ3_XXS;
    case QType::GGUF_IQ3_S: return gguf::GgmlType::IQ3_S;
    case QType::GGUF_IQ1_S: return gguf::GgmlType::IQ1_S;
    case QType::GGUF_IQ1_M: return gguf::GgmlType::IQ1_M;
    case QType::GGUF_IQ4_NL: return gguf::GgmlType::IQ4_NL;
    case QType::GGUF_IQ4_XS: return gguf::GgmlType::IQ4_XS;
    default: break;
    }
    throw std::invalid_argument("gguf: not a GGUF block format");
}

void require_gguf(const Weight& w, const char* label) {
    const auto block = gguf_block_shape(w.qtype);
    if (!is_gguf(w.qtype) || w.layout != QuantLayout::GgufBlocks || w.qdata == nullptr ||
        w.n <= 0 || w.k <= 0 || w.k % block.elements != 0 || w.k % 512 != 0) {
        throw std::invalid_argument(std::string(label) +
                                    ": expected a resident GGUF matrix over 512-column groups");
    }
}

std::size_t gguf_project_workspace_bytes(std::span<const GgufShape> shapes,
                                         std::int32_t min_tokens, std::int32_t max_tokens) {
    if (shapes.empty() || min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("gguf workspace: invalid products or token interval");
    }
    const std::int32_t k = shapes.front().k;
    std::vector<int> layouts;
    std::size_t rows = 0;
    bool dequantized = false;
#if defined(NINFER_GGUF_DUAL_MMQ_FUSION)
    std::size_t dual_q4_rows = 0;
#endif
    for (const GgufShape& shape : shapes) {
        const auto block = gguf_block_shape(shape.qtype);
        if (!is_gguf(shape.qtype) || shape.rows <= 0 || shape.k != k || k <= 0 ||
            k % block.elements != 0 || k % 512 != 0) {
            throw std::invalid_argument("gguf workspace: unsupported product geometry");
        }
        const auto type = gguf_type(shape.qtype);
        if (gguf::has_matrix_kernel(type)) {
            const int layout = gguf::matrix_activation_layout(type);
            if (std::find(layouts.begin(), layouts.end(), layout) == layouts.end()) {
                layouts.push_back(layout);
            }
        } else {
            dequantized = true;
        }
        rows = std::max<std::size_t>(rows, shape.rows);
#if defined(NINFER_GGUF_DUAL_MMQ_FUSION)
        if (shape.qtype == QType::GGUF_Q4_K && shape.rows == 6144 && shape.k == 5120) {
            ++dual_q4_rows;
        }
#endif
    }
    // Every product may carry its own input gather, so each may quantize its own activation.
    const std::size_t activations = shapes.size();
    std::size_t widest            = 0;
    if (min_tokens <= gguf::kMaxVectorColumns) {
        const std::int32_t t = std::min(max_tokens, gguf::kMaxVectorColumns);
        WorkspaceLayoutBuilder layout;
        for (std::size_t i = 0; i < activations; ++i) {
            (void)layout.alloc_bytes(gguf::vector_activation_bytes(k, t));
        }
        widest = layout.peak_bytes(1);
    }
    if (max_tokens > gguf::kMaxVectorColumns) {
        const std::int32_t t = max_tokens;
        WorkspaceLayoutBuilder layout;
        (void)layout.alloc_bytes(rows * t * sizeof(float));
        (void)layout.alloc_bytes(fixup_bound());
#if defined(NINFER_GGUF_DUAL_MMQ_FUSION)
        if (min_tokens <= 1024 && dual_q4_rows >= 2) {
            (void)layout.alloc_bytes(rows * std::min(t, 1024) * sizeof(float));
            (void)layout.alloc_bytes(fixup_bound());
        }
#endif
        if (dequantized) {
            (void)layout.alloc_bytes(gguf::vector_activation_bytes(k, gguf::kMaxVectorColumns));
            (void)layout.alloc_bytes(kDequantizedScratchBytes);
        }
        for (std::size_t i = 0; i < std::max<std::size_t>(layouts.size(), 1) * activations; ++i) {
            (void)layout.alloc_bytes(gguf::matrix_activation_bytes(k, t));
        }
        if (dequantized) { (void)layout.alloc_bytes(std::size_t(k) * t * 2); }
        widest = std::max(widest, layout.peak_bytes(1));
    }
    return widest;
}

bool gguf_project(const Tensor& x, std::span<const GgufProduct> products,
                  WorkspaceArena& workspace, cudaStream_t stream,
                  bool* used_mmq_store, bool* used_dual_layout_mmq_pack,
                  bool* used_mmq_gate_epilogue) {
    if (used_mmq_store != nullptr) { *used_mmq_store = false; }
    if (used_dual_layout_mmq_pack != nullptr) { *used_dual_layout_mmq_pack = false; }
    if (used_mmq_gate_epilogue != nullptr) { *used_mmq_gate_epilogue = false; }
    require_activation(x, "gguf product");
    if (products.empty()) { return false; }
    for (const auto& p : products) { require_product(x, p); }
    const std::int32_t k = x.ne[0];
    const std::int32_t t = x.ne[1];
    const auto* xb       = static_cast<const __nv_bfloat16*>(x.data);
    auto scope           = workspace.scope();

#if defined(NINFER_GGUF_DUAL_MMQ_FUSION)
    if (k == 5120 && t > gguf::kMaxVectorColumns && t <= 1024) {
        std::size_t eligible_q4 = 0;
        bool all_native = true;
        for (const GgufProduct& product : products) {
            const Weight& weight = *product.weight;
            all_native = all_native && matrix_kernel(weight) && weight.input_columns == nullptr;
            if (weight.qtype == QType::GGUF_Q4_K && weight.n == 6144 && weight.k == k) {
                ++eligible_q4;
            }
        }
        if (all_native && eligible_q4 >= 2) {
            std::vector<gguf::MatrixActivationPlane> matrix_activations;
            for (const GgufProduct& product : products) {
                const auto type = gguf_type(product.weight->qtype);
                const int layout = gguf::matrix_activation_layout(type);
                const bool present = std::any_of(
                    matrix_activations.begin(), matrix_activations.end(), [&](const auto& plane) {
                        return gguf::matrix_activation_layout(plane.type) == layout;
                    });
                if (!present) {
                    void* data = workspace.alloc_bytes(gguf::matrix_activation_bytes(k, t)).data;
                    gguf::quantize_matrix_activation(type, xb, k, t, nullptr, data, stream);
                    matrix_activations.push_back({type, data});
                }
            }
            return gguf_project_matrix_q8_1_activations(k, t, matrix_activations, products,
                                                        workspace, stream, used_mmq_store,
                                                        used_mmq_gate_epilogue);
        }
    }
#endif

    if (t <= gguf::kMaxVectorColumns) {
        struct Activation {
            const std::int32_t* columns;
            void* data;
        };
        std::vector<Activation> activations;
        for (const auto& p : products) {
            const Weight& w = *p.weight;
            auto found      = std::find_if(activations.begin(), activations.end(),
                                           [&](const Activation& a) { return a.columns == w.input_columns; });
            if (found == activations.end()) {
                void* data = workspace.alloc_bytes(gguf::vector_activation_bytes(k, t)).data;
                gguf::quantize_vector_activation(xb, k, t, w.input_columns, data, stream);
                activations.push_back({w.input_columns, data});
                found = activations.end() - 1;
            }
#if defined(NINFER_GGUF_Q6_HEAD_ALIGNED_PLANES)
            if (w.q6_head_aligned_planes != nullptr && w.qtype == QType::GGUF_Q6_K &&
                w.n == 248320 && k == 5120 && (t == 1 || t == 4) &&
                w.input_columns == nullptr && p.row == 0 &&
                p.epilogue == GgufEpilogue::Store && p.f32 == nullptr &&
                gguf::q6_head_aligned_vector_product(w.q6_head_aligned_planes, w.n, k,
                                                     found->data, t, output(p, 0), stream)) {
                continue;
            }
#endif
            gguf::vector_product(gguf_type(w.qtype), w.qdata, row_bytes(w), w.n, k, found->data,
                                 t, output(p, 0), stream);
        }
        return false;
    }

    struct Activation {
        int layout;
        const std::int32_t* columns;
        void* data;
    };
    std::vector<Activation> activations;
    std::int32_t plane_rows = 0;
    bool dequantized        = false;
    for (const auto& p : products) {
        plane_rows = std::max(plane_rows, p.weight->n);
        dequantized |= !matrix_kernel(*p.weight);
    }
    auto* plane = static_cast<float*>(
        workspace.alloc_bytes(std::size_t(plane_rows) * t * sizeof(float)).data);
    void* fixup = workspace.alloc_bytes(fixup_bound()).data;
    void* vector_activation = nullptr;
    void* scratch           = nullptr;
    if (dequantized) {
        vector_activation =
            workspace.alloc_bytes(gguf::vector_activation_bytes(k, gguf::kMaxVectorColumns)).data;
        scratch = workspace.alloc_bytes(kDequantizedScratchBytes).data;
    }

    for (std::size_t product_index = 0; product_index < products.size(); ++product_index) {
        const auto& p = products[product_index];
        const Weight& w  = *p.weight;
        const auto type  = gguf_type(w.qtype);
        const bool direct = p.f32 != nullptr && p.epilogue == GgufEpilogue::Store;
        float* out        = direct ? p.f32 + p.row : plane;
        const std::int64_t stride = direct ? p.f32_rows : w.n;
        if (gguf::has_matrix_kernel(type)) {
            const int layout = gguf::matrix_activation_layout(type);
            auto found = std::find_if(activations.begin(), activations.end(), [&](const Activation& a) {
                return a.layout == layout && a.columns == w.input_columns;
            });
            if (found == activations.end()) {
#if defined(NINFER_GGUF_DUAL_LAYOUT_MMQ_PACK)
                const GgufProduct* partner = nullptr;
                const bool q4_candidate = w.qtype == QType::GGUF_Q4_K;
                const bool q6_candidate = w.qtype == QType::GGUF_Q6_K;
                if (t > gguf::kMaxVectorColumns && (q4_candidate || q6_candidate)) {
                    const QType partner_type = q4_candidate ? QType::GGUF_Q6_K : QType::GGUF_Q4_K;
                    const auto partner_layout = gguf::matrix_activation_layout(gguf_type(partner_type));
                    const bool partner_cached = std::any_of(
                        activations.begin(), activations.end(), [&](const Activation& a) {
                            return a.layout == partner_layout && a.columns == w.input_columns;
                        });
                    if (!partner_cached) {
                        for (std::size_t candidate = product_index + 1; candidate < products.size();
                             ++candidate) {
                            const Weight& candidate_weight = *products[candidate].weight;
                            if (candidate_weight.qtype == partner_type &&
                                candidate_weight.input_columns == w.input_columns &&
                                matrix_kernel(candidate_weight)) {
                                partner = &products[candidate];
                                break;
                            }
                        }
                    }
                }
                if (partner != nullptr) {
                    void* current_data =
                        workspace.alloc_bytes(gguf::matrix_activation_bytes(k, t)).data;
                    void* partner_data =
                        workspace.alloc_bytes(gguf::matrix_activation_bytes(k, t)).data;
                    if (gguf::quantize_matrix_activation_pair(type, gguf_type(partner->weight->qtype),
                                                              xb, k, t, w.input_columns, current_data,
                                                              partner_data, stream)) {
                        activations.push_back({layout, w.input_columns, current_data});
                        const int partner_layout = gguf::matrix_activation_layout(
                            gguf_type(partner->weight->qtype));
                        activations.push_back({partner_layout, w.input_columns, partner_data});
                        if (used_dual_layout_mmq_pack != nullptr) {
                            *used_dual_layout_mmq_pack = true;
                        }
                        found = std::find_if(activations.begin(), activations.end(),
                                             [&](const Activation& a) {
                                                 return a.layout == layout &&
                                                        a.columns == w.input_columns;
                                             });
                    }
                }
#endif
                if (found == activations.end()) {
                    void* data = workspace.alloc_bytes(gguf::matrix_activation_bytes(k, t)).data;
                    gguf::quantize_matrix_activation(type, xb, k, t, w.input_columns, data, stream);
                    activations.push_back({layout, w.input_columns, data});
                    found = activations.end() - 1;
                }
            }
#if defined(NINFER_GGUF_MMQ_STORE_FUSION)
            if (p.f32 == nullptr &&
                (p.epilogue == GgufEpilogue::Store || p.epilogue == GgufEpilogue::Accumulate) &&
                w.input_columns == nullptr &&
                (w.qtype == QType::GGUF_Q4_K || w.qtype == QType::GGUF_Q6_K)) {
                gguf::MatrixBf16StorePlan plan;
                if (gguf::matrix_bf16_store_plan(type, w.n, t, plan)) {
                    const gguf::VectorOutput destination = output(p, 0);
                    if (gguf::matrix_product_bf16_store(type, w.qdata, row_bytes(w), w.n, k,
                                                        found->data, t, destination.bf16,
                                                        destination.column_stride, stream,
                                                        p.epilogue == GgufEpilogue::Accumulate)) {
                        if (used_mmq_store != nullptr) { *used_mmq_store = true; }
                        continue;
                    }
                }
            }
#endif
#if defined(NINFER_GGUF_MMQ_GATE_EPILOGUE_FUSION)
            if (p.f32 == nullptr && p.epilogue == GgufEpilogue::GateProduct &&
                p.out != nullptr && p.gate != nullptr && w.qtype == QType::GGUF_Q4_K &&
                w.input_columns == nullptr) {
                gguf::MatrixBf16StorePlan plan;
                if (gguf::matrix_bf16_gate_store_plan(type, w.n, t, plan)) {
                    const gguf::VectorOutput destination = output(p, 0);
                    const auto gate_begin = reinterpret_cast<std::uintptr_t>(destination.gate);
                    const auto gate_end = gate_begin + std::size_t(w.n) * t * sizeof(float);
                    const auto out_begin = reinterpret_cast<std::uintptr_t>(destination.bf16);
                    const auto out_end = out_begin +
                        (std::size_t(t - 1) * destination.column_stride + w.n) *
                            sizeof(__nv_bfloat16);
                    if ((gate_end <= out_begin || out_end <= gate_begin) &&
                        gguf::matrix_product_bf16_gate_store(
                            type, w.qdata, row_bytes(w), w.n, k, found->data, t,
                            destination.bf16, destination.column_stride, destination.gate,
                            destination.gate_column_stride, stream)) {
                        if (used_mmq_gate_epilogue != nullptr) {
                            *used_mmq_gate_epilogue = true;
                        }
                        continue;
                    }
                }
            }
#endif
            gguf::matrix_product(type, w.qdata, row_bytes(w), w.n, k, found->data, t, out, stride,
                                 fixup, stream);
        } else if (t <= kDequantizedVectorColumns) {
            for (std::int32_t first = 0; first < t; first += gguf::kMaxVectorColumns) {
                const std::int32_t columns = std::min(gguf::kMaxVectorColumns, t - first);
                gguf::quantize_vector_activation(xb + std::int64_t(first) * k, k, columns,
                                                 w.input_columns, vector_activation, stream);
                gguf::vector_product(type, w.qdata, row_bytes(w), w.n, k, vector_activation,
                                     columns, output(p, first), stream);
            }
            continue;
        } else {
            const __nv_bfloat16* source = xb;
            if (w.input_columns != nullptr) {
                auto* gathered = static_cast<__nv_bfloat16*>(
                    workspace.alloc_bytes(std::size_t(k) * t * 2).data);
                gguf::gather_columns(xb, k, t, w.input_columns, gathered, stream);
                source = gathered;
            }
            gguf::dequantized_product(type, w.qdata, row_bytes(w), w.n, k, source, t, out, stride,
                                      scratch, kDequantizedScratchBytes, stream);
        }
        if (!direct) { gguf::store_plane(plane, w.n, w.n, t, output(p, 0), stream); }
    }
    return false;
}

void gguf_project_q8_1_activation(int k, int columns, const void* activation,
                                  std::span<const GgufProduct> products,
                                  cudaStream_t stream) {
    if (k != 5120 || columns <= 0 || columns > gguf::kMaxVectorColumns || activation == nullptr ||
        (reinterpret_cast<std::uintptr_t>(activation) & (alignof(half2) - 1)) != 0 ||
        products.empty()) {
        throw std::invalid_argument("gguf prepared vector activation: unsupported geometry");
    }
    for (const GgufProduct& p : products) {
        require_gguf(*p.weight, "gguf prepared vector product");
        require_product(Tensor(nullptr, DType::BF16, {k, columns}), p);
        if (p.weight->input_columns != nullptr) {
            throw std::invalid_argument("gguf prepared vector activation: gather is unsupported");
        }
        gguf::vector_product(gguf_type(p.weight->qtype), p.weight->qdata,
                             row_bytes(*p.weight), p.weight->n, k, activation, columns,
                             output(p, 0), stream);
    }
}

void gguf_project_matrix_q8_1_activation(int k, int columns, QType activation_format,
                                         const void* activation,
                                         std::span<const GgufProduct> products,
                                         WorkspaceArena& workspace, cudaStream_t stream) {
    const gguf::MatrixActivationPlane plane{gguf_type(activation_format), activation};
    gguf_project_matrix_q8_1_activations(k, columns, {&plane, 1}, products, workspace, stream);
}

bool gguf_project_matrix_q8_1_activations(
    int k, int columns, std::span<const gguf::MatrixActivationPlane> activations,
    std::span<const GgufProduct> products, WorkspaceArena& workspace, cudaStream_t stream,
    bool* used_mmq_store, bool* used_mmq_gate_epilogue) {
    if (used_mmq_store != nullptr) { *used_mmq_store = false; }
    if (used_mmq_gate_epilogue != nullptr) { *used_mmq_gate_epilogue = false; }
    int max_columns = 1024;
#if defined(NINFER_GGUF_MMQ_GATE_EPILOGUE_FUSION)
    if (std::all_of(products.begin(), products.end(), [](const GgufProduct& product) {
            return product.weight != nullptr && product.weight->qtype == QType::GGUF_Q4_K &&
                   product.f32 == nullptr && product.epilogue == GgufEpilogue::GateProduct;
        })) {
        max_columns = gguf::kMaxGateMatrixColumns;
    }
#endif
    if (k != 5120 || columns <= gguf::kMaxVectorColumns || columns > max_columns || products.empty() ||
        activations.empty()) {
        throw std::invalid_argument("gguf prepared matrix activation: unsupported geometry");
    }
    for (const auto& activation : activations) {
        if (activation.data == nullptr ||
            (reinterpret_cast<std::uintptr_t>(activation.data) &
             (alignof(std::uint32_t) - 1)) != 0 ||
            !gguf::has_matrix_kernel(activation.type)) {
            throw std::invalid_argument("gguf prepared matrix activation: unsupported plane");
        }
    }
    const Tensor input(nullptr, DType::BF16, {k, columns});
    std::int32_t plane_rows = 0;
    for (const GgufProduct& p : products) {
        require_gguf(*p.weight, "gguf prepared matrix product");
        require_product(input, p);
        const auto type = gguf_type(p.weight->qtype);
        const int product_layout = gguf::matrix_activation_layout(type);
        const bool found = std::any_of(activations.begin(), activations.end(), [&](const auto& a) {
            return gguf::matrix_activation_layout(a.type) == product_layout;
        });
        if (!matrix_kernel(*p.weight) || p.weight->input_columns != nullptr || !found) {
            throw std::invalid_argument("gguf prepared matrix activation: incompatible product");
        }
        plane_rows = std::max(plane_rows, p.weight->n);
    }
    auto scope = workspace.scope();
    auto* plane = static_cast<float*>(
        workspace.alloc_bytes(std::size_t(plane_rows) * columns * sizeof(float)).data);
    void* fixup = workspace.alloc_bytes(fixup_bound()).data;
#if defined(NINFER_GGUF_DUAL_MMQ_FUSION)
    const auto dual_pair_admitted = [&](std::size_t a, std::size_t b) {
        const GgufProduct& first = products[a];
        const GgufProduct& second = products[b];
        const Weight& wa = *first.weight;
        const Weight& wb = *second.weight;
        if (columns <= gguf::kMaxVectorColumns || columns > 1024 || k != 5120 ||
            wa.qtype != QType::GGUF_Q4_K || wb.qtype != QType::GGUF_Q4_K ||
            wa.n != 6144 || wb.n != 6144 || wa.k != k || wb.k != k ||
            wa.input_columns != nullptr || wb.input_columns != nullptr ||
            first.epilogue != GgufEpilogue::Store || second.epilogue != GgufEpilogue::Store ||
            first.f32 != nullptr || second.f32 != nullptr || first.out == nullptr ||
            second.out == nullptr || row_bytes(wa) != row_bytes(wb) ||
            (reinterpret_cast<std::uintptr_t>(wa.qdata) & (alignof(std::uint32_t) - 1)) != 0 ||
            (reinterpret_cast<std::uintptr_t>(wb.qdata) & (alignof(std::uint32_t) - 1)) != 0) {
            return false;
        }
        const auto begin = [](const GgufProduct& p) {
            return reinterpret_cast<std::uintptr_t>(p.out->data) +
                   std::size_t(p.row) * sizeof(__nv_bfloat16);
        };
        const auto end = [&](const GgufProduct& p) {
            return begin(p) +
                   (std::size_t(columns - 1) * p.out->ne[0] + p.weight->n) *
                       sizeof(__nv_bfloat16);
        };
        const bool outputs_overlap = begin(first) < end(second) && begin(second) < end(first);
        return !outputs_overlap && gguf::dual_q4_k_matrix_supported(6144, columns);
    };
    std::size_t pair_a = products.size();
    std::size_t pair_b = products.size();
    for (std::size_t a = 0; a < products.size() && pair_a == products.size(); ++a) {
        for (std::size_t b = a + 1; b < products.size(); ++b) {
            if (dual_pair_admitted(a, b)) {
                pair_a = a;
                pair_b = b;
                break;
            }
        }
    }
    float* dual_plane = nullptr;
    void* dual_fixup = nullptr;
    if (pair_a != products.size()) {
        dual_plane = static_cast<float*>(
            workspace.alloc_bytes(std::size_t(plane_rows) * columns * sizeof(float)).data);
        const auto fixup_bytes = gguf::dual_q4_k_matrix_fixup_bytes(6144, columns);
        if (fixup_bytes != 0) { dual_fixup = workspace.alloc_bytes(fixup_bytes).data; }
    }
    bool pair_done = false;
#endif
    for (std::size_t i = 0; i < products.size(); ++i) {
#if defined(NINFER_GGUF_DUAL_MMQ_FUSION)
        if (pair_a != products.size() && !pair_done && i == pair_a) {
            const GgufProduct& a = products[pair_a];
            const GgufProduct& b = products[pair_b];
            const Weight& wa = *a.weight;
            const Weight& wb = *b.weight;
            const auto activation = std::find_if(activations.begin(), activations.end(),
                [&](const auto& plane) {
                    return gguf::matrix_activation_layout(plane.type) ==
                           gguf::matrix_activation_layout(gguf::GgmlType::Q4_K);
                });
            if (activation == activations.end()) {
                throw std::invalid_argument("gguf prepared matrix activation: Q4_K plane missing");
            }
            const bool fused = gguf::dual_q4_k_matrix_product(
                wa.qdata, row_bytes(wa), wb.qdata, row_bytes(wb), wa.n, k,
                activation->data, columns, plane, wa.n, dual_plane, wb.n,
                fixup, dual_fixup, stream);
            if (fused) {
                gguf::store_plane(plane, wa.n, wa.n, columns, output(a, 0), stream);
                gguf::store_plane(dual_plane, wb.n, wb.n, columns, output(b, 0), stream);
                pair_done = true;
                continue;
            }
        }
        if (pair_done && i == pair_b) { continue; }
#endif
        const GgufProduct& p = products[i];
        const Weight& w = *p.weight;
        const auto type = gguf_type(w.qtype);
        const int product_layout = gguf::matrix_activation_layout(type);
        const auto activation = std::find_if(activations.begin(), activations.end(), [&](const auto& a) {
            return gguf::matrix_activation_layout(a.type) == product_layout;
        });
#if defined(NINFER_GGUF_MMQ_STORE_FUSION)
        if (p.f32 == nullptr &&
            (p.epilogue == GgufEpilogue::Store || p.epilogue == GgufEpilogue::Accumulate) &&
            (w.qtype == QType::GGUF_Q4_K || w.qtype == QType::GGUF_Q6_K)) {
            gguf::MatrixBf16StorePlan plan;
            if (gguf::matrix_bf16_store_plan(type, w.n, columns, plan)) {
                const gguf::VectorOutput destination = output(p, 0);
                if (gguf::matrix_product_bf16_store(type, w.qdata, row_bytes(w), w.n, k,
                                                    activation->data, columns, destination.bf16,
                                                    destination.column_stride, stream,
                                                    p.epilogue == GgufEpilogue::Accumulate)) {
                    if (used_mmq_store != nullptr) { *used_mmq_store = true; }
                    continue;
                }
            }
        }
#endif
#if defined(NINFER_GGUF_MMQ_GATE_EPILOGUE_FUSION)
        if (p.f32 == nullptr && p.epilogue == GgufEpilogue::GateProduct &&
            p.out != nullptr && p.gate != nullptr && w.qtype == QType::GGUF_Q4_K) {
            gguf::MatrixBf16StorePlan plan;
            if (gguf::matrix_bf16_gate_store_plan(type, w.n, columns, plan)) {
                const gguf::VectorOutput destination = output(p, 0);
                const auto gate_begin = reinterpret_cast<std::uintptr_t>(destination.gate);
                const auto gate_end = gate_begin + std::size_t(w.n) * columns * sizeof(float);
                const auto out_begin = reinterpret_cast<std::uintptr_t>(destination.bf16);
                const auto out_end = out_begin +
                    (std::size_t(columns - 1) * destination.column_stride + w.n) *
                        sizeof(__nv_bfloat16);
                if ((gate_end <= out_begin || out_end <= gate_begin) &&
                    gguf::matrix_product_bf16_gate_store(
                        type, w.qdata, row_bytes(w), w.n, k, activation->data, columns,
                        destination.bf16, destination.column_stride, destination.gate,
                        destination.gate_column_stride, stream)) {
                    if (used_mmq_gate_epilogue != nullptr) {
                        *used_mmq_gate_epilogue = true;
                    }
                    continue;
                }
            }
        }
#endif
        const bool direct = p.f32 != nullptr && p.epilogue == GgufEpilogue::Store;
        float* out = direct ? p.f32 + p.row : plane;
        const std::int64_t stride = direct ? p.f32_rows : w.n;
        gguf::matrix_product(type, w.qdata, row_bytes(w), w.n, k, activation->data, columns, out,
                             stride, fixup, stream);
        if (!direct) { gguf::store_plane(plane, w.n, w.n, columns, output(p, 0), stream); }
    }
#if defined(NINFER_GGUF_DUAL_MMQ_FUSION)
    return pair_done;
#else
    return false;
#endif
}

void gguf_linear(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& workspace,
                 cudaStream_t stream) {
    const GgufProduct product{&w, &out, 0, GgufEpilogue::Store};
    gguf_project(x, {&product, 1}, workspace, stream);
}

void gguf_linear_add(const Tensor& x, const Weight& w, Tensor& residual,
                     WorkspaceArena& workspace, cudaStream_t stream) {
    const GgufProduct product{&w, &residual, 0, GgufEpilogue::Accumulate};
    gguf_project(x, {&product, 1}, workspace, stream);
}

std::size_t gguf_swiglu_linear_add_workspace_capacity_bytes(const Weight& gate, const Weight* up,
                                                            const Weight& down,
                                                            std::int32_t min_tokens,
                                                            std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("gguf fused swiglu q8: invalid token interval");
    }
#if defined(NINFER_GGUF_SWIGLU_Q8_FUSION) || defined(NINFER_GGUF_PERSISTENT_FFN)
    if (!fused_swiglu_q8_weights_admit(gate, up, down, min_tokens, max_tokens)) { return 0; }
    bool route_admitted = false;
    for (std::int32_t tokens = min_tokens; tokens <= max_tokens; ++tokens) {
        route_admitted |= gguf_swiglu_linear_add_route(gate, up, down, tokens) !=
                          GgufSwiGluRoute::Fallback;
    }
    if (!route_admitted) { return 0; }
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc_bytes(gguf::vector_activation_bytes(gate.k, max_tokens));
    (void)layout.alloc_bytes(gguf::vector_activation_bytes(gate.n / 2, max_tokens));
    return layout.peak_bytes(1);
#else
    (void)gate;
    (void)down;
    return 0;
#endif
}

GgufSwiGluRoute gguf_swiglu_linear_add_route(const Weight& gate, const Weight* up,
                                            const Weight& down, std::int32_t tokens) {
#if defined(NINFER_GGUF_SWIGLU_Q8_FUSION) || defined(NINFER_GGUF_PERSISTENT_FFN)
    if (fused_swiglu_q8_weights_admit(gate, up, down, tokens, tokens)) {
#if defined(NINFER_GGUF_PERSISTENT_FFN)
        if (gate.qtype == QType::GGUF_Q4_K && gate.n == 34816 && gate.k == 5120 &&
            gate.input_columns == nullptr && down.qtype == QType::GGUF_Q6_K &&
            down.n == 5120 && down.k == 17408 &&
            gguf::persistent_swiglu_q8_add_available(tokens, nullptr)) {
            return GgufSwiGluRoute::PersistentFfn;
        }
#endif
#if defined(NINFER_GGUF_SWIGLU_Q8_FUSION)
        return GgufSwiGluRoute::ProducerQ8;
#endif
    }
#else
    (void)gate; (void)up; (void)down; (void)tokens;
#endif
    return GgufSwiGluRoute::Fallback;
}

bool gguf_swiglu_linear_add(const Tensor& x, const Weight& gate, const Weight* up,
                            const Weight& down, Tensor& residual, WorkspaceArena& workspace,
                            cudaStream_t stream) {
#if defined(NINFER_GGUF_SWIGLU_Q8_FUSION) || defined(NINFER_GGUF_PERSISTENT_FFN)
    if (!fused_swiglu_q8_weights_admit(gate, up, down, x.ne[1], x.ne[1]) ||
        down.n <= 0 || down.n != residual.ne[0] || x.ne[1] < 1 || x.ne[1] > 8 ||
        residual.ne[1] != x.ne[1] || residual.dtype != DType::BF16 || !residual.is_contiguous() ||
        x.ne[0] != gate.k || x.dtype != DType::BF16 || !x.is_contiguous()) {
        return false;
    }
    auto scope = workspace.scope();
    const std::int32_t rows = gate.n / 2;
    const auto route = gguf_swiglu_linear_add_route(gate, up, down, x.ne[1]);
    if (route == GgufSwiGluRoute::Fallback) { return false; }
    if (route == GgufSwiGluRoute::PersistentFfn) {
        void* input = workspace.alloc_bytes(gguf::vector_activation_bytes(gate.k, x.ne[1])).data;
        void* output = workspace.alloc_bytes(gguf::vector_activation_bytes(rows, x.ne[1])).data;
        gguf::quantize_vector_activation(static_cast<const __nv_bfloat16*>(x.data), gate.k, x.ne[1],
                                         gate.input_columns, input, stream);
        return gguf::persistent_swiglu_q8_add(
            gguf_type(gate.qtype), gate.qdata, row_bytes(gate), rows, gate.k, input,
            gguf_type(down.qtype), down.qdata, row_bytes(down), down.n, down.k, output,
            static_cast<__nv_bfloat16*>(residual.data), residual.ne[0], x.ne[1], stream);
    }
    void* input = workspace.alloc_bytes(gguf::vector_activation_bytes(gate.k, x.ne[1])).data;
    void* output = workspace.alloc_bytes(gguf::vector_activation_bytes(rows, x.ne[1])).data;
    gguf::quantize_vector_activation(static_cast<const __nv_bfloat16*>(x.data), gate.k, x.ne[1],
                                     gate.input_columns, input, stream);
#if defined(NINFER_GGUF_SWIGLU_Q8_FUSION)
    gguf::vector_swiglu_q8(gguf_type(gate.qtype), gate.qdata, row_bytes(gate), rows, gate.k,
                           input, x.ne[1], output, stream, NINFER_GGUF_SWIGLU_Q8_WARPS);
#else
    return false;
#endif
    gguf::VectorOutput destination;
    destination.bf16 = static_cast<__nv_bfloat16*>(residual.data);
    destination.column_stride = residual.ne[0];
    destination.epilogue = gguf::Epilogue::Accumulate;
    gguf::vector_product(gguf_type(down.qtype), down.qdata, row_bytes(down), down.n, down.k,
                         output, x.ne[1], destination, stream);
    return true;
#else
    (void)x;
    (void)gate;
    (void)up;
    (void)down;
    (void)residual;
    (void)workspace;
    (void)stream;
    return false;
#endif
}

bool gguf_swiglu_linear_add_prequantized(const void* activation, int k, int columns,
                                         const Weight& gate, const Weight* up,
                                         const Weight& down, Tensor& residual,
                                         WorkspaceArena& workspace, cudaStream_t stream) {
#if defined(NINFER_GGUF_SWIGLU_Q8_FUSION) || defined(NINFER_GGUF_PERSISTENT_FFN)
    if (activation == nullptr || k != gate.k || gate.input_columns != nullptr || columns < 1 ||
        columns > 8 || !fused_swiglu_q8_weights_admit(gate, up, down, columns, columns) ||
        down.n <= 0 || down.n != residual.ne[0] || residual.ne[1] != columns ||
        residual.ne[2] != 1 || residual.ne[3] != 1 || residual.dtype != DType::BF16 ||
        !residual.is_contiguous() || residual.data == nullptr ||
        (reinterpret_cast<std::uintptr_t>(activation) & (alignof(int4) - 1)) != 0) {
        return false;
    }
    auto scope = workspace.scope();
    const std::int32_t rows = gate.n / 2;
    void* output = workspace.alloc_bytes(gguf::vector_activation_bytes(rows, columns)).data;
    const auto route = gguf_swiglu_linear_add_route(gate, up, down, columns);
    if (route == GgufSwiGluRoute::PersistentFfn) {
        return gguf::persistent_swiglu_q8_add(
            gguf_type(gate.qtype), gate.qdata, row_bytes(gate), rows, gate.k, activation,
            gguf_type(down.qtype), down.qdata, row_bytes(down), down.n, down.k, output,
            static_cast<__nv_bfloat16*>(residual.data), residual.ne[0], columns, stream);
    }
    if (route != GgufSwiGluRoute::ProducerQ8) { return false; }
#if defined(NINFER_GGUF_SWIGLU_Q8_FUSION)
    gguf::vector_swiglu_q8(gguf_type(gate.qtype), gate.qdata, row_bytes(gate), rows, gate.k,
                           activation, columns, output, stream, NINFER_GGUF_SWIGLU_Q8_WARPS);
#else
    return false;
#endif
    gguf::VectorOutput destination;
    destination.bf16 = static_cast<__nv_bfloat16*>(residual.data);
    destination.column_stride = residual.ne[0];
    destination.epilogue = gguf::Epilogue::Accumulate;
    gguf::vector_product(gguf_type(down.qtype), down.qdata, row_bytes(down), down.n, down.k,
                         output, columns, destination, stream);
    return true;
#else
    (void)activation;
    (void)k;
    (void)columns;
    (void)gate;
    (void)up;
    (void)down;
    (void)residual;
    (void)workspace;
    (void)stream;
    return false;
#endif
}

bool gguf_swiglu_prequantized(const void* activation, int k, int columns,
                              const Weight& gate, Tensor& output, cudaStream_t stream) {
    if (activation == nullptr || gate.qtype != QType::GGUF_Q4_K || gate.k != k || k != 5120 ||
        gate.n != 34816 || gate.layout != QuantLayout::GgufBlocks || gate.qdata == nullptr ||
        gate.input_columns != nullptr || columns < 1 || columns > gguf::kMaxVectorColumns ||
        output.dtype != DType::BF16 || !output.is_contiguous() || output.ne[0] != gate.n / 2 ||
        output.ne[1] != columns || output.ne[2] != 1 || output.ne[3] != 1 || output.data == nullptr ||
        (reinterpret_cast<std::uintptr_t>(activation) & (alignof(int4) - 1)) != 0) {
        return false;
    }
    gguf::vector_swiglu(gguf_type(gate.qtype), gate.qdata, row_bytes(gate), gate.n / 2, k,
                        activation, columns, static_cast<__nv_bfloat16*>(output.data),
                        output.ne[0], stream);
    return true;
}

std::size_t gguf_swiglu_workspace_bytes(const GgufShape& gate, const GgufShape* up,
                                        std::int32_t min_tokens, std::int32_t max_tokens) {
    if (up == nullptr) {
        const GgufShape halves[] = {{gate.qtype, gate.rows / 2, gate.k},
                                    {gate.qtype, gate.rows / 2, gate.k}};
        return gguf_project_workspace_bytes(halves, min_tokens, max_tokens) +
               std::size_t(gate.rows / 2) * max_tokens * sizeof(float) + 256;
    }
    const GgufShape parents[] = {gate, *up};
    return gguf_project_workspace_bytes(parents, min_tokens, max_tokens) +
           std::size_t(gate.rows) * max_tokens * sizeof(float) + 256;
}

void gguf_swiglu(const Tensor& x, const Weight& gate, const Weight* up, Tensor& out,
                 WorkspaceArena& workspace, cudaStream_t stream) {
    require_activation(x, "gguf swiglu");
    require_gguf(gate, "gguf swiglu gate");
    const std::int32_t t    = x.ne[1];
    const std::int32_t rows = out.ne[0];
    if (out.dtype != DType::BF16 || !out.is_contiguous() || out.ne[1] != t) {
        throw std::invalid_argument("gguf swiglu: out must be contiguous BF16 [N,T]");
    }
    auto scope = workspace.scope();
    Weight gate_rows = gate;
    Weight up_rows;
    if (up == nullptr) {
        if (gate.n != 2 * rows) { throw std::invalid_argument("gguf swiglu: parent is not [gate; up]"); }
        if (t <= gguf::kMaxVectorColumns) {
            void* activation =
                workspace.alloc_bytes(gguf::vector_activation_bytes(gate.k, t)).data;
            gguf::quantize_vector_activation(static_cast<const __nv_bfloat16*>(x.data), gate.k, t,
                                             gate.input_columns, activation, stream);
            gguf::vector_swiglu(gguf_type(gate.qtype), gate.qdata, row_bytes(gate), rows, gate.k,
                                activation, t, static_cast<__nv_bfloat16*>(out.data), rows,
                                stream);
            return;
        }
        gate_rows.n = rows;
        up_rows     = gate;
        up_rows.n   = rows;
        up_rows.qdata = static_cast<const std::byte*>(gate.qdata) + std::int64_t(rows) * row_bytes(gate);
    } else {
        require_gguf(*up, "gguf swiglu up");
        if (gate.n != rows || up->n != rows || up->k != gate.k) {
            throw std::invalid_argument("gguf swiglu: gate and up must both be [N,K]");
        }
        up_rows = *up;
    }
    auto* gate_plane =
        static_cast<float*>(workspace.alloc_bytes(std::size_t(rows) * t * sizeof(float)).data);
    GgufProduct products[2];
    products[0].weight   = &gate_rows;
    products[0].f32      = gate_plane;
    products[0].f32_rows = rows;
    products[1].weight   = &up_rows;
    products[1].out      = &out;
    products[1].epilogue = GgufEpilogue::GateProduct;
    products[1].gate     = gate_plane;
    gguf_project(x, products, workspace, stream);
}

void gguf_embedding(const Weight& table, const Tensor& ids, Tensor& out, cudaStream_t stream) {
    require_gguf(table, "gguf embedding");
    if (ids.dtype != DType::I32 || !ids.is_contiguous() || out.dtype != DType::BF16 ||
        !out.is_contiguous() || out.ne[0] != table.k || out.numel() != std::int64_t(table.k) * ids.numel()) {
        throw std::invalid_argument("gguf embedding: expected I32 ids and BF16 [K, ids]");
    }
    gguf::dequantize_rows(gguf_type(table.qtype), table.qdata, row_bytes(table), table.k,
                          static_cast<const std::int32_t*>(ids.data),
                          static_cast<int>(ids.numel()), static_cast<__nv_bfloat16*>(out.data),
                          table.k, stream);
}

} // namespace ninfer::ops::detail
