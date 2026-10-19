#include "ops/linear/gguf/gguf_linear.h"

#include "core/layout.h"
#include "core/q4_fields.h"
#include "ops/linear/gguf/ggml_bridge_q6_mmq.h"
#include "ops/linear/gguf/ggml_bridge_q4_mmq.h"
#include "ops/linear/gguf/ggml_bridge_q4_paired.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <cstdlib>
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
constexpr std::size_t kQ4FieldsRawScratchBytes = std::size_t{48} << 20;

std::int64_t row_bytes(const Weight& w) {
    const auto block = gguf_block_shape(w.qtype);
    return std::int64_t(w.k / block.elements) * block.bytes;
}

bool matrix_kernel(const Weight& w) { return gguf::has_matrix_kernel(gguf_type(w.qtype)); }

bool q4_matrix_fields_supported(const Weight& w) {
    const auto& fields = w.q4_fields;
    return (w.k == 5120 && fields.d_width == 12 && fields.d_exponent_base == 0 &&
            fields.d_exponent_bits == 2 && fields.dmin_width == 13 &&
            fields.dmin_exponent_base == 2 && fields.dmin_exponent_bits == 3) ||
           (w.k == 17408 && fields.d_width == 13 && fields.d_exponent_base == 0 &&
            fields.d_exponent_bits == 3 && fields.dmin_width == 13 &&
            fields.dmin_exponent_base == 2 && fields.dmin_exponent_bits == 3);
}

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


bool q4_paired_products_supported(const Tensor& x, std::span<const GgufProduct> products) {
    if (x.ne[0] != 5120 || x.ne[1] != 4 || products.size() != 2) { return false; }
    const auto& first = products[0];
    const auto& second = products[1];
    const auto valid_product = [&](const GgufProduct& p, int rows) {
        return p.weight && p.out && p.weight->layout == QuantLayout::GgufQ4Fields &&
               p.weight->qtype == QType::GGUF_Q4_K && p.weight->n == rows && p.weight->k == 5120 &&
               p.row == 0 && p.epilogue == GgufEpilogue::Store && p.gate == nullptr && p.f32 == nullptr &&
               p.out->dtype == DType::BF16 && p.out->is_contiguous() && p.out->ne[0] == rows &&
               p.out->ne[1] == 4 && p.out->ne[2] == 1 && p.out->ne[3] == 1 && p.out->data != nullptr;
    };
    if (!valid_product(first, 10240) || !valid_product(second, 6144)) { return false; }
    const Weight& a = *first.weight;
    const Weight& b = *second.weight;
    if (a.input_columns != b.input_columns || a.payload == nullptr || a.payload != b.payload ||
        a.payload_bytes != b.payload_bytes || a.q4_field_row_begin != 0 ||
        b.q4_field_row_begin != 10240 || a.q4_fields.d_width != 12 ||
        a.q4_fields.d_exponent_base != 0 || a.q4_fields.d_exponent_bits != 2 ||
        a.q4_fields.dmin_width != 13 || a.q4_fields.dmin_exponent_base != 2 ||
        a.q4_fields.dmin_exponent_bits != 3 || a.q4_fields.d_width != b.q4_fields.d_width ||
        a.q4_fields.dmin_width != b.q4_fields.dmin_width ||
        a.q4_fields.d_exponent_base != b.q4_fields.d_exponent_base ||
        a.q4_fields.d_exponent_bits != b.q4_fields.d_exponent_bits ||
        a.q4_fields.dmin_exponent_base != b.q4_fields.dmin_exponent_base ||
        a.q4_fields.dmin_exponent_bits != b.q4_fields.dmin_exponent_bits ||
        !a.qdata || !b.qdata || !a.scales || !b.scales || !a.q4_d || !a.q4_dmin ||
        a.q4_d != b.q4_d || a.q4_dmin != b.q4_dmin ||
        (reinterpret_cast<std::uintptr_t>(a.q4_d) & 3u) != 0 ||
        (reinterpret_cast<std::uintptr_t>(a.q4_dmin) & 3u) != 0) { return false; }
    const auto qdata_a = reinterpret_cast<std::uintptr_t>(a.qdata);
    const auto qdata_b = reinterpret_cast<std::uintptr_t>(b.qdata);
    const auto scales_a = reinterpret_cast<std::uintptr_t>(a.scales);
    const auto scales_b = reinterpret_cast<std::uintptr_t>(b.scales);
    if (qdata_a > std::numeric_limits<std::uintptr_t>::max() - std::uintptr_t(10240) * 2560 ||
        scales_a > std::numeric_limits<std::uintptr_t>::max() - std::uintptr_t(10240) * 20 * 12 ||
        qdata_b != qdata_a + std::uintptr_t(10240) * 2560 ||
        scales_b != scales_a + std::uintptr_t(10240) * 20 * 12) { return false; }
    const auto in_parent = [&](const void* pointer, std::uint64_t bytes) {
        const auto parent = reinterpret_cast<std::uintptr_t>(a.payload);
        const auto address = reinterpret_cast<std::uintptr_t>(pointer);
        if (!pointer || address < parent ||
            a.payload_bytes > std::numeric_limits<std::uintptr_t>::max() - parent) { return false; }
        const auto end = parent + static_cast<std::uintptr_t>(a.payload_bytes);
        return address <= end && bytes <= end - address;
    };
    if (!in_parent(a.qdata, std::uint64_t(16384) * 2560) ||
        !in_parent(a.scales, std::uint64_t(16384) * 20 * 12) ||
        !in_parent(a.q4_d, (std::uint64_t(16384) * 20 * 12 + 7) / 8) ||
        !in_parent(a.q4_dmin, (std::uint64_t(16384) * 20 * 13 + 7) / 8)) { return false; }
    const auto first_begin = reinterpret_cast<std::uintptr_t>(first.out->data);
    const auto second_begin = reinterpret_cast<std::uintptr_t>(second.out->data);
    constexpr std::uintptr_t first_bytes = std::uintptr_t(10240) * 4 * sizeof(__nv_bfloat16);
    constexpr std::uintptr_t second_bytes = std::uintptr_t(6144) * 4 * sizeof(__nv_bfloat16);
    if (first_begin > std::numeric_limits<std::uintptr_t>::max() - first_bytes ||
        second_begin > std::numeric_limits<std::uintptr_t>::max() - second_bytes ||
        (first_begin < second_begin + second_bytes && second_begin < first_begin + first_bytes)) {
        return false;
    }
    return true;
}

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
    if (!is_gguf(w.qtype) || (w.layout != QuantLayout::GgufBlocks && w.layout != QuantLayout::GgufQ6Fields &&
        w.layout != QuantLayout::GgufQ4Fields) || w.qdata == nullptr ||
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
    bool q4_fields = false;
    const auto* lossless_mode = std::getenv("NINFER_LOSSLESS_MODE");
    for (const GgufShape& shape : shapes) {
        const auto block = gguf_block_shape(shape.qtype);
        if (!is_gguf(shape.qtype) || shape.rows <= 0 || shape.k != k || k <= 0 ||
            k % block.elements != 0 || k % 512 != 0) {
            throw std::invalid_argument("gguf workspace: unsupported product geometry");
        }
        const auto type = gguf_type(shape.qtype);
        q4_fields |= lossless_mode && shape.qtype == QType::GGUF_Q4_K &&
                     std::string(lossless_mode) == "q4q6-fields";
        if (gguf::has_matrix_kernel(type)) {
            const int layout = gguf::matrix_activation_layout(type);
            if (std::find(layouts.begin(), layouts.end(), layout) == layouts.end()) {
                layouts.push_back(layout);
            }
        } else {
            dequantized = true;
        }
        rows = std::max<std::size_t>(rows, shape.rows);
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
        if (q4_fields) { (void)layout.alloc_bytes(kQ4FieldsRawScratchBytes); }
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

void gguf_project(const Tensor& x, std::span<const GgufProduct> products,
                  WorkspaceArena& workspace, cudaStream_t stream) {
    require_activation(x, "gguf product");
    if (products.empty()) { return; }
    for (const auto& p : products) { require_product(x, p); }
    const std::int32_t k = x.ne[0];
    const std::int32_t t = x.ne[1];
    const auto* xb       = static_cast<const __nv_bfloat16*>(x.data);
    auto scope           = workspace.scope();

    if (t <= gguf::kMaxVectorColumns) {
        if (q4_paired_products_supported(x, products)) {
            const Weight& first = *products[0].weight;
            void* activation = workspace.alloc_bytes(gguf::vector_activation_bytes(k, t)).data;
            gguf::quantize_vector_activation(xb, k, t, first.input_columns, activation, stream);
            gguf::vector_q4_fields_paired(first, *products[1].weight, activation, t,
                                          output(products[0], 0), output(products[1], 0), stream);
            return;
        }
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
            if (w.layout == QuantLayout::GgufQ6Fields) {
                gguf::vector_q6_fields(w, found->data, t, output(p, 0), stream);
            } else if (w.layout == QuantLayout::GgufQ4Fields) {
                gguf::vector_q4_fields(w, found->data, t, output(p, 0), stream);
            } else {
                gguf::vector_product(gguf_type(w.qtype), w.qdata, row_bytes(w), w.n, k, found->data,
                                     t, output(p, 0), stream);
            }
        }
        return;
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
    void* q4_raw = nullptr;
    if (std::any_of(products.begin(), products.end(), [](const auto& product) {
        return product.weight->layout == QuantLayout::GgufQ4Fields &&
               !q4_matrix_fields_supported(*product.weight);
    })) { q4_raw = workspace.alloc_bytes(kQ4FieldsRawScratchBytes).data; }
    void* vector_activation = nullptr;
    void* scratch           = nullptr;
    if (dequantized) {
        vector_activation =
            workspace.alloc_bytes(gguf::vector_activation_bytes(k, gguf::kMaxVectorColumns)).data;
        scratch = workspace.alloc_bytes(kDequantizedScratchBytes).data;
    }

    for (const auto& p : products) {
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
                void* data = workspace.alloc_bytes(gguf::matrix_activation_bytes(k, t)).data;
                gguf::quantize_matrix_activation(type, xb, k, t, w.input_columns, data, stream);
                activations.push_back({layout, w.input_columns, data});
                found = activations.end() - 1;
            }
            const void* matrix_weight = w.qdata;
            if (w.layout == QuantLayout::GgufQ6Fields) {
                gguf::matrix_q6_fields(w, found->data, t, out, stride, fixup, stream);
                if (!direct) { gguf::store_plane(plane, w.n, w.n, t, output(p, 0), stream); }
                continue;
            } else if (w.layout == QuantLayout::GgufQ4Fields) {
                if (q4_matrix_fields_supported(w)) {
                    gguf::matrix_q4_fields(w, found->data, t, out, stride, fixup, stream);
                    if (!direct) { gguf::store_plane(plane, w.n, w.n, t, output(p, 0), stream); }
                    continue;
                }
                if (std::uint64_t(w.n) * row_bytes(w) > kQ4FieldsRawScratchBytes) {
                    throw std::invalid_argument("selected Q4 fields exceed bounded raw MMQ scratch");
                }
                unpack_q4_fields(w, 0, w.n, q4_raw, stream);
                matrix_weight = q4_raw;
            }
            gguf::matrix_product(type, matrix_weight, row_bytes(w), w.n, k, found->data, t, out, stride,
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
        if (t <= gguf::kMaxVectorColumns && gate.layout != QuantLayout::GgufQ6Fields &&
            gate.layout != QuantLayout::GgufQ4Fields) {
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
        const auto stride = gate.layout == QuantLayout::GgufQ6Fields ? std::int64_t(gate.k / 256) * 208 :
                            gate.layout == QuantLayout::GgufQ4Fields ? std::int64_t(gate.k / 256) * 128 : row_bytes(gate);
        up_rows.qdata = static_cast<const std::byte*>(gate.qdata) + std::int64_t(rows) * stride;
        up_rows.q6_field_row_begin += rows;
        if (gate.layout == QuantLayout::GgufQ4Fields) {
            up_rows.scales = static_cast<const std::byte*>(gate.scales) + std::int64_t(rows) * (gate.k / 256) * 12;
            up_rows.q4_field_row_begin += rows;
        }
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
    if (table.layout == QuantLayout::GgufQ6Fields || table.layout == QuantLayout::GgufQ4Fields) {
        throw std::invalid_argument("lossless fields embedding is not enabled for this experiment");
    }
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
