#include "artifact/q4_fields.h"

#include "artifact/schema.h"

#include <array>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string_view>

namespace ninfer::artifact {
namespace {

constexpr std::uint32_t kMagic = 0x31463451u;
constexpr std::uint32_t kVersion = 2;
constexpr std::uint32_t kHeaderBytes = 64;

std::uint64_t bits_bytes(std::uint64_t count, std::uint32_t width) {
    if (count > (std::numeric_limits<std::uint64_t>::max() - 7) / width) {
        throw ArtifactError("Q4 field stream size overflows u64");
    }
    return (count * width + 7) / 8;
}

std::uint64_t stream_bytes(std::uint64_t count, std::uint32_t width) {
    const auto bytes = bits_bytes(count, width);
    if (width != 13) { return bytes; }
    if (bytes > std::numeric_limits<std::uint64_t>::max() - 4) {
        throw ArtifactError("Q4 field stream padding overflows u64");
    }
    return bytes + 4;
}

bool gdn_geometry(const WeightGeometry& g) {
    return g.shape.size() == 2 && g.shape[0] == 16384 && g.shape[1] == 5120 &&
           g.q4_fields.d_width == 12 && g.q4_fields.d_exponent_base == 0 &&
           g.q4_fields.d_exponent_bits == 2 && g.q4_fields.dmin_width == 13 &&
           g.q4_fields.dmin_exponent_base == 2 && g.q4_fields.dmin_exponent_bits == 3;
}

bool down_geometry(const WeightGeometry& g) {
    return g.shape.size() == 2 && g.shape[0] == 5120 && g.shape[1] == 17408 &&
           g.q4_fields.d_width == 13 && g.q4_fields.d_exponent_base == 0 &&
           g.q4_fields.d_exponent_bits == 3 && g.q4_fields.dmin_width == 13 &&
           g.q4_fields.dmin_exponent_base == 2 && g.q4_fields.dmin_exponent_bits == 3;
}

void write_u32_le(std::span<std::byte> data, std::uint64_t offset, std::uint32_t value) {
    if (offset + 4 > data.size()) { throw ArtifactError("Q4 field header write exceeds payload"); }
    for (unsigned i = 0; i < 4; ++i) { data[offset + i] = std::byte((value >> (8 * i)) & 255u); }
}

std::uint32_t read_u32_le(std::span<const std::byte> data, std::uint64_t offset) {
    if (offset + 4 > data.size()) { throw ArtifactError("Q4 field header read exceeds payload"); }
    std::uint32_t value = 0;
    for (unsigned i = 0; i < 4; ++i) { value |= std::to_integer<std::uint32_t>(data[offset + i]) << (8 * i); }
    return value;
}

std::uint16_t read_half(std::span<const std::byte> data, std::uint64_t offset) {
    return std::uint16_t(std::to_integer<unsigned>(data[offset]) |
                         (std::to_integer<unsigned>(data[offset + 1]) << 8));
}

void write_bits(std::span<std::byte> data, std::uint64_t bit_offset,
                std::uint32_t value, std::uint32_t width) {
    for (std::uint32_t bit = 0; bit < width; ++bit) {
        const auto position = bit_offset + bit;
        if (position / 8 >= data.size()) { throw ArtifactError("Q4 field bit write exceeds payload"); }
        if ((value >> bit) & 1u) {
            data[position / 8] |= std::byte(1u << (position % 8));
        }
    }
}

std::uint32_t read_bits(std::span<const std::byte> data, std::uint64_t bit_offset,
                        std::uint32_t width) {
    std::uint32_t value = 0;
    for (std::uint32_t bit = 0; bit < width; ++bit) {
        const auto position = bit_offset + bit;
        if (position / 8 >= data.size()) { throw ArtifactError("Q4 field bit read exceeds payload"); }
        value |= ((std::to_integer<unsigned>(data[position / 8]) >> (position % 8)) & 1u) << bit;
    }
    return value;
}

void validate_geometry(const WeightGeometry& g) {
    if (g.format != QType::GGUF_Q4_K || g.layout != QuantLayout::GgufQ4Fields ||
        g.shape.size() != 2 || (!gdn_geometry(g) && !down_geometry(g)) ||
        g.elements != g.shape[0] * g.shape[1]) {
        throw ArtifactError("invalid Q4 field geometry");
    }
    const std::uint64_t rows = g.shape[0];
    const std::uint64_t columns = g.shape[1];
    const std::uint64_t blocks_per_row = columns / 256;
    const std::uint64_t blocks = rows * blocks_per_row;
    const std::uint64_t code_row = blocks_per_row * 128;
    const std::uint64_t metadata_row = blocks_per_row * 12;
    const std::uint64_t code_bytes = rows * code_row;
    const std::uint64_t metadata_bytes = rows * metadata_row;
    const std::uint64_t d_bytes = stream_bytes(blocks, g.q4_fields.d_width);
    const std::uint64_t dmin_bytes = stream_bytes(blocks, g.q4_fields.dmin_width);
    const std::uint64_t code_offset = kHeaderBytes;
    const std::uint64_t scale_offset = code_offset + code_bytes;
    const std::uint64_t d_offset = scale_offset + metadata_bytes;
    const std::uint64_t dmin_offset = d_offset + d_bytes;
    if (g.code_offset != code_offset || g.code_bytes_per_row != code_row || g.code_bytes != code_bytes ||
        g.scale_offset != scale_offset || g.scale_bytes_per_row != metadata_row || g.scale_bytes != metadata_bytes ||
        g.q4_d_offset != d_offset || g.q4_dmin_offset != dmin_offset || g.bytes != dmin_offset + dmin_bytes) {
        throw ArtifactError("Q4 field geometry offsets or byte counts do not match the encoding");
    }
}

std::array<std::uint32_t, 16> expected_header(const WeightGeometry& g) {
    validate_geometry(g);
    const auto d_params = std::uint32_t(g.q4_fields.d_exponent_base) |
                          (std::uint32_t(g.q4_fields.d_exponent_bits) << 8);
    const auto dmin_params = std::uint32_t(g.q4_fields.dmin_exponent_base) |
                             (std::uint32_t(g.q4_fields.dmin_exponent_bits) << 8);
    return {kMagic, kVersion, kHeaderBytes, std::uint32_t(g.shape[0]), std::uint32_t(g.shape[1]),
            std::uint32_t(g.code_offset), std::uint32_t(g.code_bytes_per_row),
            std::uint32_t(g.scale_offset), std::uint32_t(g.scale_bytes_per_row),
            std::uint32_t(g.q4_d_offset), g.q4_fields.d_width, d_params,
            std::uint32_t(g.q4_dmin_offset), g.q4_fields.dmin_width, dmin_params,
            std::uint32_t(g.bytes)};
}

void validate_header(const WeightGeometry& g, std::span<const std::byte> encoded) {
    if (encoded.size() != g.bytes || encoded.size() < kHeaderBytes) {
        throw ArtifactError("Q4 field payload size does not match geometry");
    }
    const auto expected = expected_header(g);
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (read_u32_le(encoded, i * 4) != expected[i]) {
            throw ArtifactError("Q4 field header version, shape, or plane metadata mismatch");
        }
    }
}

std::vector<std::byte> recover_raw(const WeightGeometry& g, std::span<const std::byte> encoded) {
    validate_header(g, encoded);
    const std::uint64_t rows = g.shape[0];
    const std::uint64_t columns = g.shape[1];
    const std::uint64_t blocks_per_row = columns / 256;
    const std::uint64_t blocks = rows * blocks_per_row;
    const auto d_stream = encoded.subspan(g.q4_d_offset, stream_bytes(blocks, g.q4_fields.d_width));
    const auto dmin_stream = encoded.subspan(g.q4_dmin_offset, stream_bytes(blocks, g.q4_fields.dmin_width));
    std::vector<std::byte> raw(blocks * 144);
    for (std::uint64_t row = 0; row < rows; ++row) {
        for (std::uint64_t kblock = 0; kblock < blocks_per_row; ++kblock) {
            const auto block = row * blocks_per_row + kblock;
            const auto dst_offset = block * 144;
            const auto code_offset = g.code_offset + row * g.code_bytes_per_row + kblock * 128;
            const auto metadata_offset = g.scale_offset + row * g.scale_bytes_per_row + kblock * 12;
            std::memcpy(raw.data() + dst_offset + 16, encoded.data() + code_offset, 128);
            std::memcpy(raw.data() + dst_offset + 4, encoded.data() + metadata_offset, 12);
            const auto d_code = read_bits(d_stream, block * g.q4_fields.d_width, g.q4_fields.d_width);
            const auto dmin_code = read_bits(dmin_stream, block * g.q4_fields.dmin_width, g.q4_fields.dmin_width);
            const auto d = std::uint16_t((d_code & 0x3ffu) | ((g.q4_fields.d_exponent_base + (d_code >> 10)) << 10));
            const auto dmin = std::uint16_t((dmin_code & 0x3ffu) | ((g.q4_fields.dmin_exponent_base + (dmin_code >> 10)) << 10));
            raw[dst_offset] = std::byte(d & 255u);
            raw[dst_offset + 1] = std::byte(d >> 8);
            raw[dst_offset + 2] = std::byte(dmin & 255u);
            raw[dst_offset + 3] = std::byte(dmin >> 8);
        }
    }
    return raw;
}

void require_header_rejection(const WeightGeometry& g, std::span<std::byte> encoded) {
    auto* bytes = encoded.data();
    std::array<std::byte, kHeaderBytes> saved{};
    std::memcpy(saved.data(), bytes, saved.size());
    const auto rejected = [&](std::span<const std::byte> candidate) {
        try { validate_header(g, candidate); }
        catch (const ArtifactError&) { return true; }
        return false;
    };
    write_u32_le(encoded, 4, kVersion + 1);
    if (!rejected(encoded)) { throw ArtifactError("Q4 field invalid-version fixture was accepted"); }
    std::memcpy(bytes, saved.data(), saved.size());
    write_u32_le(encoded, 12, std::uint32_t(g.shape[0] - 1));
    if (!rejected(encoded)) { throw ArtifactError("Q4 field invalid-shape fixture was accepted"); }
    std::memcpy(bytes, saved.data(), saved.size());
    if (!rejected(encoded.first(encoded.size() - 1))) { throw ArtifactError("Q4 field invalid-size fixture was accepted"); }
}

}

bool q4_fields_mode_enabled() {
    const auto* mode = std::getenv("NINFER_LOSSLESS_MODE");
    if (!mode || std::string_view(mode) == "off") { return false; }
    if (std::string_view(mode) != "q4q6-fields") { throw ArtifactError("unknown experimental lossless mode"); }
    return true;
}

std::optional<WeightGeometry> describe_q4_fields(const WeightGeometry& stored,
                                                 std::span<const std::byte> source) {
    if (stored.format != QType::GGUF_Q4_K || stored.layout != QuantLayout::GgufBlocks ||
        stored.shape.size() != 2 || source.size() != stored.bytes) {
        return {};
    }
    const bool is_gdn = stored.shape[0] == 16384 && stored.shape[1] == 5120;
    const bool is_down = stored.shape[0] == 5120 && stored.shape[1] == 17408;
    if (!is_gdn && !is_down) { return {}; }
    const auto rows = stored.shape[0];
    const auto columns = stored.shape[1];
    const std::uint64_t blocks_per_row = columns / 256;
    const std::uint64_t blocks = rows * blocks_per_row;
    if (stored.elements != rows * columns || stored.code_bytes_per_row != blocks_per_row * 144 ||
        stored.code_bytes != rows * stored.code_bytes_per_row || source.size() != blocks * 144) {
        return {};
    }
    for (std::uint64_t block = 0; block < blocks; ++block) {
        const auto d = read_half(source, block * 144);
        const auto dmin = read_half(source, block * 144 + 2);
        const auto d_exp = (d >> 10) & 31u;
        const auto dmin_exp = (dmin >> 10) & 31u;
        if ((d & 0x8000u) || (dmin & 0x8000u) || d_exp > (is_gdn ? 3u : 7u) ||
            dmin_exp < 2 || dmin_exp > 9) { return {}; }
    }
    auto geometry = stored;
    geometry.layout = QuantLayout::GgufQ4Fields;
    geometry.q4_fields = is_gdn ? Q4FieldEncoding{12, 13, 0, 2, 2, 3}
                                : Q4FieldEncoding{13, 13, 0, 3, 2, 3};
    geometry.code_offset = kHeaderBytes;
    geometry.code_bytes_per_row = blocks_per_row * 128;
    geometry.code_bytes = rows * geometry.code_bytes_per_row;
    geometry.scale_offset = geometry.code_offset + geometry.code_bytes;
    geometry.scale_bytes_per_row = blocks_per_row * 12;
    geometry.scale_bytes = rows * geometry.scale_bytes_per_row;
    geometry.q4_d_offset = geometry.scale_offset + geometry.scale_bytes;
    geometry.q4_dmin_offset = geometry.q4_d_offset + stream_bytes(blocks, geometry.q4_fields.d_width);
    geometry.bytes = geometry.q4_dmin_offset + stream_bytes(blocks, geometry.q4_fields.dmin_width);
    if (geometry.bytes >= stored.bytes) { return {}; }
    validate_geometry(geometry);
    return geometry;
}

std::vector<std::byte> encode_q4_fields(const WeightGeometry& geometry,
                                        std::span<const std::byte> source) {
    validate_geometry(geometry);
    const std::uint64_t rows = geometry.shape[0];
    const std::uint64_t columns = geometry.shape[1];
    const std::uint64_t blocks_per_row = columns / 256;
    const std::uint64_t blocks = rows * blocks_per_row;
    if (source.size() != blocks * 144) { throw ArtifactError("Q4 field source size does not match geometry"); }
    std::vector<std::byte> encoded(geometry.bytes, std::byte{0});
    const auto header = expected_header(geometry);
    for (std::size_t i = 0; i < header.size(); ++i) { write_u32_le(encoded, i * 4, header[i]); }
    for (std::uint64_t row = 0; row < rows; ++row) {
        for (std::uint64_t kblock = 0; kblock < blocks_per_row; ++kblock) {
            const auto block = row * blocks_per_row + kblock;
            const auto* raw = source.data() + block * 144;
            const auto code_offset = geometry.code_offset + row * geometry.code_bytes_per_row + kblock * 128;
            const auto metadata_offset = geometry.scale_offset + row * geometry.scale_bytes_per_row + kblock * 12;
            std::memcpy(encoded.data() + code_offset, raw + 16, 128);
            std::memcpy(encoded.data() + metadata_offset, raw + 4, 12);
            const auto d = read_half(source, block * 144);
            const auto dmin = read_half(source, block * 144 + 2);
            const auto d_exp = (d >> 10) & 31u;
            const auto dmin_exp = (dmin >> 10) & 31u;
            const auto d_exp_limit = 1u << geometry.q4_fields.d_exponent_bits;
            const auto dmin_exp_limit = 1u << geometry.q4_fields.dmin_exponent_bits;
            if ((d & 0x8000u) || (dmin & 0x8000u) ||
                d_exp < geometry.q4_fields.d_exponent_base ||
                d_exp - geometry.q4_fields.d_exponent_base >= d_exp_limit ||
                dmin_exp < geometry.q4_fields.dmin_exponent_base ||
                dmin_exp - geometry.q4_fields.dmin_exponent_base >= dmin_exp_limit) {
                throw ArtifactError("Q4 field source does not fit the fixed exponent windows");
            }
            const auto d_code = std::uint32_t(d & 0x3ffu) |
                ((d_exp - geometry.q4_fields.d_exponent_base) << 10);
            const auto dmin_code = std::uint32_t(dmin & 0x3ffu) |
                ((dmin_exp - geometry.q4_fields.dmin_exponent_base) << 10);
            write_bits(encoded, (geometry.q4_d_offset * 8) + block * geometry.q4_fields.d_width,
                       d_code, geometry.q4_fields.d_width);
            write_bits(encoded, (geometry.q4_dmin_offset * 8) + block * geometry.q4_fields.dmin_width,
                       dmin_code, geometry.q4_fields.dmin_width);
        }
    }
    validate_header(geometry, encoded);
    const auto recovered = recover_raw(geometry, encoded);
    if (!std::equal(recovered.begin(), recovered.end(), source.begin(), source.end())) {
        throw ArtifactError("Q4 field full CPU byte reconstruction failed");
    }
    require_header_rejection(geometry, encoded);
    validate_header(geometry, encoded);
    return encoded;
}

}
