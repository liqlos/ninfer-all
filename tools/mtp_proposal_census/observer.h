#pragma once

#include "ninfer/ops/sampling.h"

#include <cstdint>
#include <type_traits>

namespace ninfer::lab::mtp_census {

inline constexpr int kDomain = 248077;
inline constexpr int kPhysical = 248320;
inline constexpr int kDrafts = 3;
inline constexpr int kWidth = 4;
inline constexpr int kRecords = 16;
inline constexpr int kCandidates = 4;

enum Error : std::uint32_t {
    None = 0,
    Sampling = 1,
    Extent = 2,
    Join = 3,
    Token = 4,
    NonFinite = 5,
    RawWinner = 6,
    CountJoin = 7,
    Acceptance = 8,
    CountUpdate = 9,
    ProposalOrder = 10,
};

struct Score {
    std::int32_t id;
    std::uint32_t raw_bits;
    std::uint32_t adjusted_bits;
    std::int32_t base_count;
    std::int32_t prefix_count;
};

struct Top2 {
    Score raw[2];
    Score adjusted[2];
};

struct Proposal {
    Top2 top;
    std::int32_t actual;
    std::int32_t candidate_count;
    Score candidates[kCandidates];
};

struct Target {
    Top2 top;
    std::int32_t actual_raw;
    std::int32_t candidate_count;
    Score candidates[kCandidates];
};

struct Pending {
    std::uint32_t epoch;
    std::uint32_t round;
    std::uint32_t mask;
    std::int32_t valid;
    std::int32_t frontier;
    std::int32_t anchor;
    std::int32_t extent;
    Proposal q[kDrafts];
};

struct Record {
    std::uint32_t epoch;
    std::uint32_t round;
    std::uint32_t producer_round;
    std::uint32_t q_epoch;
    std::uint32_t causal_columns;
    std::int32_t q_frontier;
    std::int32_t q_anchor;
    std::int32_t q_extent;
    std::int32_t frontier;
    std::int32_t anchor;
    std::int32_t extent;
    std::int32_t drafts[kDrafts];
    std::int32_t accepted;
    std::int32_t licensed_count;
    std::int32_t licensed[kWidth];
    std::int32_t post_frontier;
    std::int32_t post_anchor;
    Proposal q[kDrafts];
    Target p[kWidth];
};

struct State {
    std::uint32_t epoch;
    std::uint32_t round;
    std::uint32_t completed;
    std::uint32_t error;
    std::int32_t active;
    std::int32_t slot;
    std::int32_t first_frontier;
    std::int32_t first_anchor;
    std::int32_t first_budget;
    const std::int32_t* counts;
    ops::SamplingConfig sampling;
    Pending pending;
    Record records[kRecords];
};

struct View {
    const std::int32_t* drafts;
    const std::int32_t* extent;
    const std::int32_t* valid_columns;
    const std::int32_t* budget;
    const std::int32_t* frontier;
    const std::int32_t* anchor;
    const std::int32_t* accepted;
    const std::int32_t* licensed_count;
    const std::int32_t* licensed;
    const std::int32_t* next_extent;
    const std::int32_t* next_drafts;
    const ops::SamplingConfig* sampling;
};

static_assert(std::is_trivially_copyable_v<State>);
static_assert(sizeof(State) < 32768);
static_assert(sizeof(Score) == 20);

cudaError_t reset(State*, bool, const std::int32_t*, cudaStream_t);
cudaError_t before_accept(State*, const void*, const std::int32_t*, View, cudaStream_t);
cudaError_t after_accept(State*, View, cudaStream_t);
cudaError_t proposal(State*, const void*, const std::int32_t*, int, View, cudaStream_t);

}
