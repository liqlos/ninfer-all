#include "artifact/q4_fields.h"
#include "artifact/schema.h"

#include <array>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace ninfer;
using namespace ninfer::artifact;

int main(int argc, char** argv) {
    if (argc != 5) { throw std::runtime_error("model offset rows columns required"); }
    const auto offset = std::stoull(argv[2]);
    const auto rows = std::stoull(argv[3]);
    const auto columns = std::stoull(argv[4]);
    const std::array<std::uint64_t, 2> shape{rows, columns};
    const auto stored = weight_geometry(QType::GGUF_Q4_K, QuantLayout::GgufBlocks, shape);
    std::vector<std::byte> raw(stored.bytes);
    std::ifstream input(argv[1], std::ios::binary);
    input.seekg(static_cast<std::streamoff>(offset));
    input.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw.size()));
    if (input.gcount() != static_cast<std::streamsize>(raw.size())) { throw std::runtime_error("full tensor read failed"); }
    if (setenv("NINFER_LOSSLESS_MODE", "off", 1) || q4_fields_mode_enabled()) {
        throw std::runtime_error("off mode check failed");
    }
    setenv("NINFER_LOSSLESS_MODE", "q4q6-fields", 1);
    if (!q4_fields_mode_enabled()) { throw std::runtime_error("combined mode check failed"); }
    setenv("NINFER_LOSSLESS_MODE", "unsupported", 1);
    bool unknown_rejected = false;
    try { (void)q4_fields_mode_enabled(); }
    catch (const ArtifactError&) { unknown_rejected = true; }
    if (!unknown_rejected) { throw std::runtime_error("unknown mode was accepted"); }
    setenv("NINFER_LOSSLESS_MODE", "q4q6-fields", 1);
    const auto geometry = describe_q4_fields(stored, raw);
    if (!geometry) { throw std::runtime_error("eligible tensor fell back to raw"); }
    const auto encoded = encode_q4_fields(*geometry, raw);
    if (encoded.size() != geometry->bytes || geometry->q4_d_offset <= geometry->scale_offset ||
        geometry->q4_dmin_offset <= geometry->q4_d_offset || geometry->bytes >= raw.size()) {
        throw std::runtime_error("encoded geometry or byte accounting mismatch");
    }
    auto sign_fixture = raw;
    sign_fixture[1] |= std::byte{0x80};
    if (describe_q4_fields(stored, sign_fixture)) { throw std::runtime_error("signful source did not fall back"); }
    auto exponent_fixture = raw;
    exponent_fixture[1] = (exponent_fixture[1] & std::byte{0x83}) | std::byte{0x10};
    if (describe_q4_fields(stored, exponent_fixture)) { throw std::runtime_error("out-of-window exponent did not fall back"); }
    if (describe_q4_fields(stored, std::span<const std::byte>(raw).first(raw.size() - 1))) {
        throw std::runtime_error("truncated source did not fall back");
    }
    const std::array<std::uint64_t, 2> wrong_shape{rows, columns - 256};
    const auto wrong_geometry = weight_geometry(QType::GGUF_Q4_K, QuantLayout::GgufBlocks, wrong_shape);
    if (describe_q4_fields(wrong_geometry, raw)) { throw std::runtime_error("wrong shape did not fall back"); }
    std::cout << "{\"rows\":" << rows << ",\"columns\":" << columns
              << ",\"raw_bytes\":" << raw.size() << ",\"encoded_bytes\":" << encoded.size()
              << ",\"saved_bytes\":" << raw.size() - encoded.size()
              << ",\"d_width\":" << unsigned(geometry->q4_fields.d_width)
              << ",\"d_exponent_base\":" << unsigned(geometry->q4_fields.d_exponent_base)
              << ",\"d_exponent_bits\":" << unsigned(geometry->q4_fields.d_exponent_bits)
              << ",\"dmin_width\":" << unsigned(geometry->q4_fields.dmin_width)
              << ",\"dmin_exponent_base\":" << unsigned(geometry->q4_fields.dmin_exponent_base)
              << ",\"dmin_exponent_bits\":" << unsigned(geometry->q4_fields.dmin_exponent_bits)
              << ",\"code_offset\":" << geometry->code_offset
              << ",\"code_bytes_per_row\":" << geometry->code_bytes_per_row
              << ",\"scale_offset\":" << geometry->scale_offset
              << ",\"scale_bytes_per_row\":" << geometry->scale_bytes_per_row
              << ",\"q4_d_offset\":" << geometry->q4_d_offset
              << ",\"q4_dmin_offset\":" << geometry->q4_dmin_offset
              << ",\"full_original_bytes_exact\":true,\"header_invalid_version_shape_size_rejected\":true"
              << ",\"ineligible_sign_exponent_shape_size_fallback\":true}\n";
}
