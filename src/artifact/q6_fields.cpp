#include "artifact/q6_fields.h"

#include "artifact/schema.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string_view>

namespace ninfer::artifact {
namespace {

std::uint16_t half_bits(const std::byte* data) {
    return std::to_integer<unsigned>(data[0]) | (std::to_integer<unsigned>(data[1]) << 8);
}

std::uint32_t read_bits(std::span<const std::byte> data, std::uint64_t at, unsigned width) {
    std::uint32_t value = 0;
    for (unsigned bit = 0; bit < width; ++bit) {
        const auto position = at + bit;
        value |= ((std::to_integer<unsigned>(data[position / 8]) >> (position % 8)) & 1u) << bit;
    }
    return value;
}

}

bool q6_fields_mode_enabled() {
    const auto* mode = std::getenv("NINFER_LOSSLESS_MODE");
    if (!mode || std::string_view(mode) == "off") { return false; }
    if (std::string_view(mode) != "q6-fields" && std::string_view(mode) != "q4q6-fields") {
        throw ArtifactError("unknown experimental lossless mode");
    }
    return true;
}

std::optional<WeightGeometry> describe_q6_fields(const WeightGeometry& stored,
                                               std::span<const std::byte> source) {
    if (stored.format != QType::GGUF_Q6_K || stored.layout != QuantLayout::GgufBlocks ||
        stored.shape.size() != 2 || stored.shape[1] % 1024 || source.size() != stored.bytes) {
        return {};
    }
    const auto blocks = stored.elements / 256;
    std::uint8_t minimum = 31, maximum = 0;
    std::uint64_t negative = 0;
    for (std::uint64_t block = 0; block < blocks; ++block) {
        const auto bits = half_bits(source.data() + block * 210 + 208);
        const auto exponent = std::uint8_t((bits >> 10) & 31);
        minimum = std::min(minimum, exponent);
        maximum = std::max(maximum, exponent);
        negative += bits >> 15;
    }
    Q6FieldEncoding fields;
    fields.exponent_base = minimum;
    while ((1u << fields.exponent_bits) < unsigned(maximum - minimum + 1)) { ++fields.exponent_bits; }
    fields.sign_variable = negative != 0 && negative != blocks;
    fields.sign_value = negative == blocks;
    fields.width = 10 + fields.exponent_bits + fields.sign_variable;
    if (blocks > std::numeric_limits<std::uint32_t>::max() / fields.width) { return {}; }
    auto geometry = stored;
    geometry.layout = QuantLayout::GgufQ6Fields;
    geometry.code_offset = 32;
    geometry.code_bytes_per_row = stored.shape[1] / 256 * 208;
    geometry.code_bytes = stored.shape[0] * geometry.code_bytes_per_row;
    geometry.scale_offset = geometry.code_offset + geometry.code_bytes;
    geometry.scale_bytes_per_row = 0;
    geometry.scale_bytes = (blocks * fields.width + 7) / 8;
    geometry.bytes = geometry.scale_offset + geometry.scale_bytes;
    geometry.q6_fields = fields;
    if (geometry.bytes >= stored.bytes) { return {}; }
    return geometry;
}

std::vector<std::byte> encode_q6_fields(const WeightGeometry& g,
                                      std::span<const std::byte> source) {
    if (g.layout != QuantLayout::GgufQ6Fields || g.format != QType::GGUF_Q6_K ||
        source.size() != g.elements / 256 * 210 || g.shape.size() != 2) {
        throw ArtifactError("invalid Q6 field encoding geometry");
    }
    const auto& fields = g.q6_fields;
    const auto flags = unsigned(fields.exponent_base) | (unsigned(fields.exponent_bits) << 5) |
                       (unsigned(fields.sign_variable) << 10) | (unsigned(fields.sign_value) << 11);
    const std::array<std::uint32_t, 8> header{0x31503651u, 1, 14, std::uint32_t(g.shape[0]),
                                            std::uint32_t(g.shape[1]), 832, fields.width, flags};
    std::vector<std::byte> encoded(g.bytes, std::byte{0});
    std::memcpy(encoded.data(), header.data(), sizeof(header));
    const auto blocks = g.elements / 256;
    for (std::uint64_t block = 0; block < blocks; ++block) {
        const auto* raw = source.data() + block * 210;
        auto* tile = encoded.data() + 32 + (block / 4) * 832;
        const auto slot = block % 4;
        std::memcpy(tile + slot * 128, raw, 128);
        std::memcpy(tile + 512 + slot * 64, raw + 128, 64);
        std::memcpy(tile + 768 + slot * 16, raw + 192, 16);
        const auto bits = half_bits(raw + 208);
        const auto exponent = ((bits >> 10) & 31u) - fields.exponent_base;
        std::uint32_t code = (bits & 0x3ffu) | (exponent << 10);
        if (fields.sign_variable) { code |= (bits >> 15) << (10 + fields.exponent_bits); }
        const auto at = block * fields.width;
        for (unsigned bit = 0; bit < fields.width; ++bit) {
            const auto position = at + bit;
            encoded[g.scale_offset + position / 8] |= std::byte(((code >> bit) & 1u) << (position % 8));
        }
    }
    const auto packed = std::span<const std::byte>(encoded).subspan(g.scale_offset);
    for (std::uint64_t block = 0; block < blocks; ++block) {
        const auto* raw = source.data() + block * 210;
        const auto* tile = encoded.data() + 32 + (block / 4) * 832;
        const auto slot = block % 4;
        const auto code = read_bits(packed, block * fields.width, fields.width);
        const auto exponent = fields.exponent_base + ((code >> 10) & ((1u << fields.exponent_bits) - 1));
        const auto sign = fields.sign_variable ? (code >> (10 + fields.exponent_bits)) & 1u : fields.sign_value;
        const auto bits = std::uint16_t((code & 0x3ffu) | (exponent << 10) | (sign << 15));
        if (std::memcmp(tile + slot * 128, raw, 128) ||
            std::memcmp(tile + 512 + slot * 64, raw + 128, 64) ||
            std::memcmp(tile + 768 + slot * 16, raw + 192, 16) || bits != half_bits(raw + 208)) {
            throw ArtifactError("Q6 field full CPU byte reconstruction failed");
        }
    }
    return encoded;
}

}
