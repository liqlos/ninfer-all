#pragma once

#include "../mtp_proposal_census/observer.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace ninfer::lab::head_census {

inline constexpr int kK = 5120;
inline constexpr int kPacked = kK + kK / 32 * 4;
inline constexpr int kRoles = 7;
inline constexpr std::uint64_t kMagic = 0x315355534e454348ULL;

struct Sample {
    std::uint32_t epoch;
    std::uint32_t round;
    std::int32_t role;
    std::int32_t frontier;
    std::int32_t anchor;
    std::int32_t extent;
    std::uint64_t weight_address;
    std::uint8_t packed[kPacked];
    std::uint16_t logits[mtp_census::kPhysical];
};

struct Payload {
    std::uint32_t counts_epoch;
    std::uint32_t reserved;
    std::int32_t counts[mtp_census::kDomain];
    std::uint32_t count_padding;
    Sample samples[mtp_census::kRecords][kRoles];
};

struct FileHeader {
    std::uint64_t magic;
    std::uint32_t schema;
    std::uint32_t epoch;
    std::uint32_t records;
    std::uint32_t k;
    std::uint32_t physical;
    std::uint32_t logical;
    std::uint32_t packed_bytes;
    std::uint32_t sample_bytes;
    std::uint64_t payload_bytes;
};

static_assert(sizeof(Sample) == 502432);
static_assert(offsetof(Sample, packed) == 32);
static_assert(offsetof(Sample, logits) == 5792);
static_assert(sizeof(FileHeader) == 48);
static_assert(std::is_trivially_copyable_v<Payload>);
static_assert(sizeof(Payload) == 57264704);

cudaError_t snapshot(const mtp_census::State*, Payload*, int, const void*, const void*,
                     const void*, mtp_census::View, cudaStream_t);

}
