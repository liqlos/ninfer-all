#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace q6wg_codec {

inline std::size_t packed_bytes(std::size_t blocks) {
    return ((blocks + 15) / 16) * 3360;
}

inline unsigned original_code(const std::uint8_t* block, unsigned k) {
    const unsigned half = k / 128, part = (k % 128) / 32, lane = k % 32;
    const unsigned low = block[64 * half + 32 * (part & 1) + lane];
    const unsigned high = block[128 + 32 * half + lane];
    return ((low >> (4 * (part / 2))) & 15) | (((high >> (2 * part)) & 3) << 4);
}

inline std::vector<std::uint8_t> pack(const std::vector<std::uint8_t>& original) {
    if (original.empty() || original.size() % 210) {
        throw std::invalid_argument("word-group CPU pack needs complete native blocks");
    }
    const std::size_t blocks = original.size() / 210;
    std::vector<std::uint8_t> result(packed_bytes(blocks), 0);
    for (std::size_t b = 0; b < blocks; ++b) {
        const auto* source = original.data() + b * 210;
        auto* slab = result.data() + (b / 16) * 3360;
        const unsigned slot = b % 16;
        for (unsigned g = 0; g < 16; ++g) {
            std::array<unsigned, 16> q{};
            for (unsigned k = 0; k < 16; ++k) { q[k] = original_code(source, 16 * g + k); }
            for (unsigned t = 0; t < 4; ++t) {
                slab[slot * 128 + g * 8 + t] = (q[t] & 15) | ((q[t + 4] & 15) << 4);
                slab[slot * 128 + g * 8 + 4 + t] = (q[t + 8] & 15) | ((q[t + 12] & 15) << 4);
                slab[2048 + slot * 64 + g * 4 + t] =
                    (q[t] >> 4) | ((q[t + 4] >> 4) << 2) |
                    ((q[t + 8] >> 4) << 4) | ((q[t + 12] >> 4) << 6);
            }
        }
        std::memcpy(slab + 3072 + slot * 16, source + 192, 16);
        std::memcpy(slab + 3328 + slot * 2, source + 208, 2);
    }
    return result;
}

inline std::array<std::uint8_t, 210> inverse_block(const std::uint8_t* packed, std::size_t b) {
    std::array<std::uint8_t, 210> result{};
    const auto* slab = packed + (b / 16) * 3360;
    const unsigned slot = b % 16;
    for (unsigned k = 0; k < 256; ++k) {
        const unsigned g = k / 16, within = k % 16, quartet = within / 4, t = within % 4;
        const unsigned low = slab[slot * 128 + g * 8 + (quartet / 2) * 4 + t];
        const unsigned high = slab[2048 + slot * 64 + g * 4 + t];
        const unsigned value = ((low >> (4 * (quartet & 1))) & 15) |
                               (((high >> (2 * quartet)) & 3) << 4);
        const unsigned half = k / 128, part = (k % 128) / 32, lane = k % 32;
        result[half * 64 + (part & 1) * 32 + lane] |= (value & 15) << (4 * (part / 2));
        result[128 + half * 32 + lane] |= (value >> 4) << (2 * part);
    }
    std::memcpy(result.data() + 192, slab + 3072 + slot * 16, 16);
    std::memcpy(result.data() + 208, slab + 3328 + slot * 2, 2);
    return result;
}

inline void verify_inverse(const std::vector<std::uint8_t>& original,
                           const std::vector<std::uint8_t>& packed) {
    const std::size_t blocks = original.size() / 210;
    if (original.empty() || original.size() % 210 || packed.size() != packed_bytes(blocks)) {
        throw std::invalid_argument("word-group inverse extent mismatch");
    }
    for (std::size_t b = 0; b < blocks; ++b) {
        const auto restored = inverse_block(packed.data(), b);
        if (std::memcmp(restored.data(), original.data() + b * 210, 210) != 0) {
            throw std::runtime_error("literal Q6 full-parent inverse differs from original bytes");
        }
    }
    for (std::size_t b = blocks; b < ((blocks + 15) / 16) * 16; ++b) {
        const auto restored = inverse_block(packed.data(), b);
        for (const auto value : restored) {
            if (value != 0) { throw std::runtime_error("unused slab slot is not zero"); }
        }
    }
}

inline void self_test() {
    for (unsigned bit = 0; bit < 210 * 8; ++bit) {
        std::vector<std::uint8_t> original(210, 0);
        original[bit / 8] = 1u << (bit % 8);
        verify_inverse(original, pack(original));
    }
    for (const std::size_t blocks : {1, 3, 15, 16, 17, 31, 33}) {
        std::vector<std::uint8_t> original(blocks * 210);
        for (std::size_t i = 0; i < original.size(); ++i) { original[i] = (i * 73 + i / 17 + 19) & 255; }
        verify_inverse(original, pack(original));
    }
}

}
