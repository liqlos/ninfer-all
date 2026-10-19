#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>

namespace ninfer::artifact {

inline void pack_q6_head_aligned_planes(std::span<const std::byte> source,
                                        std::size_t rows,
                                        std::span<std::byte> destination) {
    constexpr std::size_t source_row_bytes = 4200;
    constexpr std::size_t destination_row_bytes = 4224;
    constexpr std::size_t blocks_per_row = 20;
    if (rows == 0 || source.size() != rows * source_row_bytes ||
        destination.size() != rows * destination_row_bytes) {
        throw std::invalid_argument("Q6 head aligned plane transform has invalid extent");
    }
    for (std::size_t row = 0; row < rows; ++row) {
        const auto* row_source = source.data() + row * source_row_bytes;
        auto* row_destination = destination.data() + row * destination_row_bytes;
        for (std::size_t block = 0; block < blocks_per_row; ++block) {
            const auto* block_source = row_source + block * 210;
            std::memcpy(row_destination + block * 128, block_source, 128);
            std::memcpy(row_destination + 2560 + block * 64, block_source + 128, 64);
            std::memcpy(row_destination + 3840 + block * 16, block_source + 192, 16);
            std::memcpy(row_destination + 4160 + block * 2, block_source + 208, 2);
        }
    }
}

}
