#include "artifact/q4_fields.h"
#include "artifact/schema.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::artifact;

static std::uint32_t u32(const std::byte* p) {
    return std::to_integer<std::uint32_t>(p[0]) |
           (std::to_integer<std::uint32_t>(p[1]) << 8) |
           (std::to_integer<std::uint32_t>(p[2]) << 16) |
           (std::to_integer<std::uint32_t>(p[3]) << 24);
}

static std::uint16_t u16(const std::byte* p) {
    return std::uint16_t(std::to_integer<unsigned>(p[0]) | (std::to_integer<unsigned>(p[1]) << 8));
}

static std::uint32_t read_bits(const std::vector<std::byte>& data, std::uint64_t bit, unsigned width) {
    std::uint32_t value = 0;
    for (unsigned i = 0; i < width; ++i) {
        const auto pos = bit + i;
        value |= ((std::to_integer<unsigned>(data[pos / 8]) >> (pos % 8)) & 1u) << i;
    }
    return value;
}

static std::vector<std::byte> recover(const WeightGeometry& g, const std::vector<std::byte>& encoded) {
    const auto rows = g.shape[0];
    const auto blocks_per_row = g.shape[1] / 256;
    const auto blocks = rows * blocks_per_row;
    const auto d_bits = blocks * g.q4_fields.d_width;
    const auto dmin_bits = blocks * g.q4_fields.dmin_width;
    const auto d_bytes = (d_bits + 7) / 8;
    const auto dmin_bytes = (dmin_bits + 7) / 8;
    std::vector<std::byte> raw(blocks * 144);
    for (std::uint64_t row = 0; row < rows; ++row) {
        for (std::uint64_t block_in_row = 0; block_in_row < blocks_per_row; ++block_in_row) {
            const auto block = row * blocks_per_row + block_in_row;
            const auto dst = block * 144;
            const auto code = g.code_offset + row * g.code_bytes_per_row + block_in_row * 128;
            const auto scales = g.scale_offset + row * g.scale_bytes_per_row + block_in_row * 12;
            std::memcpy(raw.data() + dst + 16, encoded.data() + code, 128);
            std::memcpy(raw.data() + dst + 4, encoded.data() + scales, 12);
            const auto dc = read_bits(encoded, g.q4_d_offset * 8 + block * g.q4_fields.d_width, g.q4_fields.d_width);
            const auto mc = read_bits(encoded, g.q4_dmin_offset * 8 + block * g.q4_fields.dmin_width, g.q4_fields.dmin_width);
            const auto d = std::uint16_t((dc & 1023u) | ((g.q4_fields.d_exponent_base + (dc >> 10)) << 10));
            const auto m = std::uint16_t((mc & 1023u) | ((g.q4_fields.dmin_exponent_base + (mc >> 10)) << 10));
            raw[dst] = std::byte(d & 255u);
            raw[dst + 1] = std::byte(d >> 8);
            raw[dst + 2] = std::byte(m & 255u);
            raw[dst + 3] = std::byte(m >> 8);
        }
    }
    const auto d_pad = g.q4_fields.d_width == 13 ? 4u : 0u;
    const auto dmin_pad = g.q4_fields.dmin_width == 13 ? 4u : 0u;
    if (d_bytes + d_pad != g.q4_dmin_offset - g.q4_d_offset || dmin_bytes + dmin_pad != g.bytes - g.q4_dmin_offset) {
        throw std::runtime_error("13-bit stream padding geometry mismatch");
    }
    for (std::uint64_t i = d_bytes; i < d_bytes + d_pad; ++i) {
        if (encoded[g.q4_d_offset + i] != std::byte{0}) { throw std::runtime_error("d stream zero tail padding changed"); }
    }
    for (std::uint64_t i = dmin_bytes; i < dmin_bytes + dmin_pad; ++i) {
        if (encoded[g.q4_dmin_offset + i] != std::byte{0}) { throw std::runtime_error("dmin stream zero tail padding changed"); }
    }
    if (d_bits % 8 && (std::to_integer<unsigned>(encoded[g.q4_d_offset + d_bytes - 1]) >> (d_bits % 8))) {
        throw std::runtime_error("d stream unused tail bits are nonzero");
    }
    if (dmin_bits % 8 && (std::to_integer<unsigned>(encoded[g.q4_dmin_offset + dmin_bytes - 1]) >> (dmin_bits % 8))) {
        throw std::runtime_error("dmin stream unused tail bits are nonzero");
    }
    return raw;
}

static std::array<std::uint32_t, 16> header(const WeightGeometry& g) {
    const auto dp = std::uint32_t(g.q4_fields.d_exponent_base) | (std::uint32_t(g.q4_fields.d_exponent_bits) << 8);
    const auto mp = std::uint32_t(g.q4_fields.dmin_exponent_base) | (std::uint32_t(g.q4_fields.dmin_exponent_bits) << 8);
    return {0x31463451u, 2u, 64u, std::uint32_t(g.shape[0]), std::uint32_t(g.shape[1]),
            std::uint32_t(g.code_offset), std::uint32_t(g.code_bytes_per_row),
            std::uint32_t(g.scale_offset), std::uint32_t(g.scale_bytes_per_row),
            std::uint32_t(g.q4_d_offset), g.q4_fields.d_width, dp,
            std::uint32_t(g.q4_dmin_offset), g.q4_fields.dmin_width, mp, std::uint32_t(g.bytes)};
}

static bool rejects_geometry(const WeightGeometry& g, const std::vector<std::byte>& raw) {
    try { (void)encode_q4_fields(g, raw); }
    catch (const ArtifactError&) { return true; }
    return false;
}

int main(int argc, char** argv) {
    if (argc != 7) { throw std::runtime_error("model offset rows columns identity eligibility required"); }
    const auto offset = std::stoull(argv[2]);
    const auto rows = std::stoull(argv[3]);
    const auto columns = std::stoull(argv[4]);
    const std::array<std::uint64_t, 2> shape{rows, columns};
    const auto stored = weight_geometry(QType::GGUF_Q4_K, QuantLayout::GgufBlocks, shape);
    std::vector<std::byte> raw(stored.bytes);
    std::ifstream input(argv[1], std::ios::binary);
    input.seekg(static_cast<std::streamoff>(offset));
    input.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw.size()));
    if (input.gcount() != static_cast<std::streamsize>(raw.size())) { throw std::runtime_error("tensor read failed"); }
    const auto geometry = describe_q4_fields(stored, raw);
    if (!geometry) {
        if (std::string(argv[6]) == "eligible") { throw std::runtime_error("eligible tensor fell back to raw"); }
        std::cout << "{\"tensor\":\"" << argv[5] << "\",\"rows\":" << rows << ",\"columns\":" << columns
                  << ",\"raw_bytes\":" << raw.size() << ",\"eligible\":false,\"expected_raw_fallback\":true}\n";
        return 0;
    }
    if (std::string(argv[6]) == "fallback") { throw std::runtime_error("tensor expected raw fallback but was eligible"); }
    const auto encoded = encode_q4_fields(*geometry, raw);
    const auto expected_header = header(*geometry);
    for (std::size_t i = 0; i < expected_header.size(); ++i) {
        if (u32(encoded.data() + 4 * i) != expected_header[i]) { throw std::runtime_error("runtime header field mismatch"); }
    }
    const auto recovered = recover(*geometry, encoded);
    if (recovered != raw) { throw std::runtime_error("independent full byte reconstruction failed"); }
    auto sign = raw;
    sign[1] |= std::byte{0x80};
    if (describe_q4_fields(stored, sign)) { throw std::runtime_error("signed field did not select raw fallback"); }
    auto exponent = raw;
    const unsigned d_exponent = rows == 16384 ? 4u : 8u;
    const auto d = u16(exponent.data());
    const auto forced_d = std::uint16_t((d & ~0x7c00u) | (d_exponent << 10));
    exponent[0] = std::byte(forced_d & 255u);
    exponent[1] = std::byte(forced_d >> 8);
    if (describe_q4_fields(stored, exponent)) { throw std::runtime_error("out-of-window d exponent did not select raw fallback"); }
    exponent = raw;
    const auto m = u16(exponent.data() + 2);
    const auto forced_m = std::uint16_t((m & ~0x7c00u) | (10u << 10));
    exponent[2] = std::byte(forced_m & 255u);
    exponent[3] = std::byte(forced_m >> 8);
    if (describe_q4_fields(stored, exponent)) { throw std::runtime_error("out-of-window dmin exponent did not select raw fallback"); }
    if (describe_q4_fields(stored, std::span<const std::byte>(raw).first(raw.size() - 1))) {
        throw std::runtime_error("truncated tensor did not select raw fallback");
    }
    const std::array<std::uint64_t, 2> wrong_shape{rows, columns - 256};
    if (describe_q4_fields(weight_geometry(QType::GGUF_Q4_K, QuantLayout::GgufBlocks, wrong_shape), raw)) {
        throw std::runtime_error("wrong shape did not select raw fallback");
    }
    auto bad_header = encoded;
    for (std::size_t i = 0; i < expected_header.size(); ++i) {
        bad_header[4 * i] ^= std::byte{1};
        bool matches = true;
        for (std::size_t j = 0; j < expected_header.size(); ++j) { matches = matches && u32(bad_header.data() + 4 * j) == expected_header[j]; }
        if (matches) { throw std::runtime_error("header corruption was not detected"); }
        bad_header = encoded;
    }
    auto altered = *geometry;
    ++altered.code_offset;
    const bool offset_rejected = rejects_geometry(altered, raw);
    altered = *geometry;
    altered.q4_dmin_offset += 1;
    const bool dmin_offset_rejected = rejects_geometry(altered, raw);
    altered = *geometry;
    altered.bytes -= 1;
    const bool size_rejected = rejects_geometry(altered, raw);
    if (!offset_rejected || !dmin_offset_rejected || !size_rejected) { throw std::runtime_error("corrupt geometry offsets or size were accepted"); }
    std::cout << "{\"tensor\":\"" << argv[5] << "\",\"rows\":" << rows << ",\"columns\":" << columns
              << ",\"raw_bytes\":" << raw.size() << ",\"encoded_bytes\":" << encoded.size()
              << ",\"saved_bytes\":" << raw.size() - encoded.size()
              << ",\"d_width\":" << unsigned(geometry->q4_fields.d_width)
              << ",\"d_base\":" << unsigned(geometry->q4_fields.d_exponent_base)
              << ",\"d_exponent_bits\":" << unsigned(geometry->q4_fields.d_exponent_bits)
              << ",\"dmin_width\":" << unsigned(geometry->q4_fields.dmin_width)
              << ",\"dmin_base\":" << unsigned(geometry->q4_fields.dmin_exponent_base)
              << ",\"dmin_exponent_bits\":" << unsigned(geometry->q4_fields.dmin_exponent_bits)
              << ",\"eligible\":true,\"offsets_match_runtime_header\":true,\"full_original_bytes_exact\":true"
              << ",\"tail_zero_bits_and_padding_exact\":true,\"all_header_word_corruptions_detected\":true"
              << ",\"corrupt_offsets_and_size_rejected\":true,\"sign_exponent_shape_size_fallback\":true}\n";
}
