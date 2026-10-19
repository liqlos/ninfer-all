#include "core/q4_fields.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer {
namespace {

__device__ unsigned half_bits(const unsigned char* plane, unsigned block,
                               unsigned width, unsigned exponent_base) {
    const auto bit = block * width;
    const auto at = bit >> 3;
    const auto shift = bit & 7;
    auto code = unsigned(plane[at]) | (unsigned(plane[at + 1]) << 8);
    if (shift + width > 16) { code |= unsigned(plane[at + 2]) << 16; }
    code = (code >> shift) & ((1u << width) - 1u);
    return (code & 1023u) | (((code >> 10) + exponent_base) << 10);
}

__global__ void unpack_kernel(const unsigned char* codes, const unsigned char* metadata,
                              const unsigned char* d, const unsigned char* dmin,
                              unsigned first_block, unsigned blocks, unsigned origin,
                              Q4FieldEncoding fields, unsigned char* out) {
    const auto count = blocks * 144u;
    for (unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
         index < count; index += blockDim.x * gridDim.x) {
        const auto block = first_block + index / 144u;
        const auto byte = index % 144u;
        if (byte >= 16u) { out[index] = codes[block * 128u + byte - 16u]; }
        else if (byte >= 4u) { out[index] = metadata[block * 12u + byte - 4u]; }
        else {
            const bool minimum = byte >= 2u;
            const auto bits = half_bits(minimum ? dmin : d, origin + block,
                minimum ? fields.dmin_width : fields.d_width,
                minimum ? fields.dmin_exponent_base : fields.d_exponent_base);
            out[index] = byte & 1u ? bits >> 8 : bits & 255u;
        }
    }
}

}

void unpack_q4_fields(const Weight& weight, int first_row, int rows, void* raw,
                      cudaStream_t stream) {
    if (weight.layout != QuantLayout::GgufQ4Fields || weight.qtype != QType::GGUF_Q4_K ||
        first_row < 0 || rows <= 0 || first_row > weight.n || rows > weight.n - first_row ||
        weight.k <= 0 || weight.k % 256 || !weight.qdata || !weight.scales ||
        !weight.q4_d || !weight.q4_dmin || !raw ||
        weight.q4_fields.d_width < 10 || weight.q4_fields.d_width > 15 ||
        weight.q4_fields.dmin_width < 10 || weight.q4_fields.dmin_width > 15) {
        throw std::invalid_argument("invalid Q4 field unpack");
    }
    const auto blocks_per_row = unsigned(weight.k / 256);
    const auto blocks64 = std::uint64_t(rows) * blocks_per_row;
    const auto last64 = (weight.q4_field_row_begin + std::uint64_t(first_row) + rows) * blocks_per_row;
    const auto width = std::max(weight.q4_fields.d_width, weight.q4_fields.dmin_width);
    if (blocks64 * 144u > std::numeric_limits<unsigned>::max() ||
        last64 * width > std::numeric_limits<unsigned>::max()) {
        throw std::invalid_argument("Q4 unpack exceeds device index range");
    }
    const auto blocks = unsigned(blocks64);
    const auto grid = std::min((blocks * 144u + 255u) / 256u, 65535u);
    unpack_kernel<<<grid, 256, 0, stream>>>(static_cast<const unsigned char*>(weight.qdata),
        static_cast<const unsigned char*>(weight.scales), static_cast<const unsigned char*>(weight.q4_d),
        static_cast<const unsigned char*>(weight.q4_dmin), unsigned(first_row) * blocks_per_row,
        blocks, unsigned(weight.q4_field_row_begin) * blocks_per_row, weight.q4_fields,
        static_cast<unsigned char*>(raw));
    const auto status = cudaGetLastError();
    if (status != cudaSuccess) { throw std::runtime_error(std::string("Q4 unpack: ") + cudaGetErrorString(status)); }
}

}
