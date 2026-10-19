#include "core/q6_fields.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace ninfer {
namespace {

__global__ void unpack_kernel(const unsigned char* plane, const unsigned char* d_bits,
                              std::uint32_t first_block, std::uint32_t blocks,
                              std::uint32_t d_block_origin, Q6FieldEncoding fields,
                              unsigned char* out) {
    const std::uint32_t count = blocks * 210;
    for (std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < count; index += blockDim.x * gridDim.x) {
        const auto local_block = index / 210;
        const auto block = first_block + local_block;
        const auto byte = index % 210;
        const auto tile = (block / 4) * 832;
        const auto slot = block % 4;
        if (byte < 128) { out[index] = plane[tile + slot * 128 + byte]; }
        else if (byte < 192) { out[index] = plane[tile + 512 + slot * 64 + byte - 128]; }
        else if (byte < 208) { out[index] = plane[tile + 768 + slot * 16 + byte - 192]; }
        else {
            const auto at = (d_block_origin + block) * fields.width;
            const auto start = at >> 3;
            const auto shift = at & 7;
            auto code = unsigned(d_bits[start]) | (unsigned(d_bits[start + 1]) << 8);
            if (shift + fields.width > 16) { code |= unsigned(d_bits[start + 2]) << 16; }
            code = (code >> shift) & ((1u << fields.width) - 1u);
            const auto exponent = fields.exponent_base + ((code >> 10) & ((1u << fields.exponent_bits) - 1u));
            const auto sign = fields.sign_variable ? (code >> (10 + fields.exponent_bits)) & 1u : fields.sign_value;
            const auto bits = (code & 0x3ffu) | (exponent << 10) | (sign << 15);
            out[index] = byte == 208 ? bits & 255u : bits >> 8;
        }
    }
}

}

void unpack_q6_fields(const Weight& weight, int first_row, int rows, void* raw,
                      cudaStream_t stream) {
    if (weight.layout != QuantLayout::GgufQ6Fields || weight.qtype != QType::GGUF_Q6_K ||
        first_row < 0 || rows <= 0 || first_row > weight.n || rows > weight.n - first_row ||
        weight.k <= 0 || weight.k % 1024 || !weight.qdata || !weight.scales || !raw) {
        throw std::invalid_argument("invalid Q6 field unpack");
    }
    const auto blocks_per_row = unsigned(weight.k / 256);
    const auto blocks = unsigned(rows) * blocks_per_row;
    const auto count = blocks * 210;
    const auto grid = std::min((count + 255) / 256, 65535u);
    unpack_kernel<<<grid, 256, 0, stream>>>(static_cast<const unsigned char*>(weight.qdata),
        static_cast<const unsigned char*>(weight.scales), unsigned(first_row) * blocks_per_row,
        blocks, unsigned(weight.q6_field_row_begin) * blocks_per_row, weight.q6_fields,
        static_cast<unsigned char*>(raw));
    const auto status = cudaGetLastError();
    if (status != cudaSuccess) { throw std::runtime_error(std::string("Q6 unpack: ") + cudaGetErrorString(status)); }
}

}
