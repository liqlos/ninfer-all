#pragma once

#include "probe_api.h"
#include <climits>
#include <limits>

namespace q6wg {

inline bool range_valid(const void* pointer, std::uint64_t bytes, unsigned alignment) {
    const auto address = reinterpret_cast<std::uintptr_t>(pointer);
    return pointer && bytes && (address & (alignment - 1)) == 0 &&
           bytes <= std::numeric_limits<std::uintptr_t>::max() - address;
}

inline bool overlaps(const void* a, std::uint64_t an, const void* b, std::uint64_t bn) {
    const auto x = reinterpret_cast<std::uintptr_t>(a);
    const auto y = reinterpret_cast<std::uintptr_t>(b);
    return an && bn && (x <= y ? y - x < an : x - y < bn);
}

inline bool validate_view(const Q6WGView& view) {
    if (!view.parent_rows || !view.parent_B || view.parent_B > INT_MAX / 256 ||
        !view.rows || view.rows > INT_MAX || view.row_begin > view.parent_rows ||
        view.rows > view.parent_rows - view.row_begin) {
        return false;
    }
    const std::uint64_t blocks = std::uint64_t(view.parent_rows) * view.parent_B;
    if (view.storage == Q6WGStorage::Native210) {
        return view.bytes == blocks * 210 && view.final_slab_blocks == 0 &&
               range_valid(view.base, view.bytes, 2);
    }
    if (view.storage == Q6WGStorage::WordGroupSlab16) {
        return view.bytes == ((blocks + 15) / 16) * 3360 &&
               view.final_slab_blocks == (blocks - 1) % 16 + 1 &&
               range_valid(view.base, view.bytes, 16);
    }
    return false;
}

inline bool output_extent(const Q6WGOutput& out, int rows, int columns, std::size_t& bytes) {
    if (rows <= 0 || columns <= 0 || out.column_stride < rows ||
        out.column_stride > INT_MAX ||
        (out.mode != Q6WGOutputMode::PlainFP32 && out.mode != Q6WGOutputMode::StoreBF16 &&
         out.mode != Q6WGOutputMode::AccumulateBF16)) {
        return false;
    }
    const std::uint64_t elements = std::uint64_t(columns - 1) * out.column_stride + rows;
    if (elements > INT_MAX) { return false; }
    const unsigned element_bytes = out.mode == Q6WGOutputMode::PlainFP32 ? 4 : 2;
    bytes = static_cast<std::size_t>(elements * element_bytes);
    return out.buffer.bytes >= bytes && range_valid(out.buffer.data, bytes, element_bytes);
}

inline bool output_disjoint(const Q6WGView& view, Q6WGReadBuffer activation,
                            const Q6WGOutput& output, std::size_t output_bytes) {
    return !overlaps(output.buffer.data, output_bytes, view.base, view.bytes) &&
           !overlaps(output.buffer.data, output_bytes, activation.data, activation.bytes);
}

}
