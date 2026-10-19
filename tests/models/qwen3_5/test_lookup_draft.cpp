// Context-lookup drafting is exact by construction -- the verify step rejects a wrong guess -- so
// what these checks defend is not correctness of output but the properties the round economics
// depend on: that a miss reports zero rather than a stale or partial draft, that the *most recent*
// occurrence is the one continued, and that a proposal never runs off the end of the ledger.

#include "models/qwen3_5/program/speculative/lookup_draft.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <exception>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::qwen3_5;

std::uint32_t longest_suffix_oracle(std::span<const TokenId> ledger, std::uint32_t min_ngram,
                                    std::uint32_t max_drafts, TokenId* out) {
    if (min_ngram == 0 || max_drafts == 0 || out == nullptr || ledger.size() < 2) { return 0; }
    const std::size_t size = ledger.size();
    const std::size_t max_suffix =
        std::min<std::size_t>(size, kMtpLookupMaximumSuffix);
    if (min_ngram > max_suffix) { return 0; }
    const std::size_t oldest_end =
        size - 1 > kMtpLookupMaximumHistory ? size - 1 - kMtpLookupMaximumHistory : 0;
    std::size_t best_end = 0;
    std::size_t best_length = 0;
    for (std::size_t end = size - 1; end-- > oldest_end;) {
        for (std::size_t length = max_suffix; length >= min_ngram; --length) {
            if (length <= end + 1) {
                bool match = true;
                for (std::size_t i = 0; i < length; ++i) {
                    if (ledger[size - length + i] != ledger[end - length + 1 + i]) {
                        match = false;
                        break;
                    }
                }
                if (match) {
                    if (length > best_length) {
                        best_length = length;
                        best_end = end;
                    }
                    break;
                }
            }
            if (length == min_ngram) { break; }
        }
    }
    if (best_length == 0) { return 0; }
    const std::size_t source = best_end + 1;
    const auto count = static_cast<std::uint32_t>(
        std::min<std::size_t>(size - source, max_drafts));
    for (std::uint32_t i = 0; i < count; ++i) { out[i] = ledger[source + i]; }
    return count;
}

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

void continues_the_earlier_occurrence() {
    const std::vector<TokenId> ledger{1, 2, 3, 4, 1, 2};
    TokenId out[8]{};
    // The earlier "1 2" was followed by "3 4", then by the tail itself.
    require(lookup_draft(ledger, 2, 4, out) == 4, "expected four drafts");
    require(out[0] == 3 && out[1] == 4 && out[2] == 1 && out[3] == 2, "wrong continuation");
}

void a_miss_reports_zero() {
    const std::vector<TokenId> ledger{1, 2, 3, 4, 5, 6};
    TokenId out[8]{};
    // No earlier "5 6": the round must see zero and fall back to a plain step.
    require(lookup_draft(ledger, 2, 4, out) == 0, "a miss must report zero");
}

void the_most_recent_occurrence_wins() {
    // Two earlier "9 9" runs. The nearer one was followed by 7, the older by 5; in repetitive text
    // the nearer copy is the better guess, so the scan runs backwards.
    const std::vector<TokenId> ledger{9, 9, 5, 9, 9, 7, 0, 9, 9};
    TokenId out[8]{};
    require(lookup_draft(ledger, 2, 2, out) == 2, "expected two drafts");
    require(out[0] == 7 && out[1] == 0, "took an older occurrence over a nearer one");
}

void a_proposal_never_runs_past_the_ledger() {
    const std::vector<TokenId> ledger{4, 5, 6, 4, 5};
    TokenId out[8]{};
    // Only three tokens followed the earlier "4 5", however many were asked for.
    require(lookup_draft(ledger, 2, 8, out) == 3, "clamp to what actually followed");
    require(out[0] == 6 && out[1] == 4 && out[2] == 5, "wrong clamped continuation");
}

void degenerate_arguments_are_refused() {
    const std::vector<TokenId> ledger{1, 2, 3};
    TokenId out[4]{};
    require(lookup_draft(ledger, 0, 4, out) == 0, "a zero n-gram matches everything; refuse it");
    require(lookup_draft(ledger, 2, 0, out) == 0, "no room for drafts");
    require(lookup_draft(ledger, 2, 4, nullptr) == 0, "no destination");
    const std::vector<TokenId> shorter{1, 2};
    require(lookup_draft(shorter, 3, 2, out) == 0, "ledger shorter than the n-gram");
    const std::vector<TokenId> exact{1, 2};
    require(lookup_draft(exact, 2, 2, out) == 0, "nothing precedes the tail to match against");
}

void longest_suffix_prefers_more_context_and_recent_ties() {
    const std::vector<TokenId> ledger{8, 1, 2, 3, 4, 1, 2, 9, 1, 2, 3, 4};
    TokenId out[4]{};
    require(lookup_draft_longest_suffix(ledger, 2, 4, out) == 4,
            "longest suffix should find the longer contextual match");
    require(out[0] == 1 && out[1] == 2 && out[2] == 9 && out[3] == 1,
            "longest suffix selected the short-context continuation");

    const std::vector<TokenId> tied{1, 2, 7, 1, 2, 8, 1, 2};
    require(lookup_draft_longest_suffix(tied, 2, 1, out) == 1 && out[0] == 8,
            "equal-length suffix matches should prefer the most recent occurrence");
}

void longest_suffix_clips_minimum_and_bounds() {
    const std::vector<TokenId> ledger{4, 5, 6, 7, 4, 5, 6};
    TokenId out[8]{};
    require(lookup_draft_longest_suffix(ledger, 3, 8, out) == 4 && out[0] == 7,
            "proposal must stop at the end of the ledger");
    require(lookup_draft_longest_suffix(ledger, 4, 8, out) == 0,
            "a match shorter than the configured minimum must be rejected");
    require(lookup_draft_longest_suffix(ledger, kMtpLookupMaximumSuffix + 1, 8, out) == 0,
            "suffixes above the bounded maximum must be rejected");

    std::vector<TokenId> bounded(kMtpLookupMaximumHistory + 4, 0);
    bounded[0] = 1;
    bounded[1] = 2;
    bounded[bounded.size() - 2] = 1;
    bounded[bounded.size() - 1] = 2;
    require(lookup_draft_longest_suffix(bounded, 2, 8, out) == 0,
            "matches outside the bounded recent-history window must be ignored");
}

void longest_suffix_matches_exhaustive_oracle() {
    for (std::size_t length = 0; length <= 7; ++length) {
        std::size_t cases = 1;
        for (std::size_t i = 0; i < length; ++i) { cases *= 3; }
        for (std::size_t encoded = 0; encoded < cases; ++encoded) {
            std::size_t value = encoded;
            std::vector<TokenId> ledger(length);
            for (auto& token : ledger) {
                token = static_cast<TokenId>(value % 3);
                value /= 3;
            }
            for (std::uint32_t minimum = 0; minimum <= 5; ++minimum) {
                for (std::uint32_t max_drafts = 0; max_drafts <= 4; ++max_drafts) {
                    std::array<TokenId, 6> actual{};
                    std::array<TokenId, 6> expected{};
                    actual.fill(-17);
                    expected.fill(-17);
                    const auto actual_count = lookup_draft_longest_suffix(
                        ledger, minimum, max_drafts, actual.data() + 1);
                    const auto expected_count = longest_suffix_oracle(
                        ledger, minimum, max_drafts, expected.data() + 1);
                    require(actual_count == expected_count,
                            "longest-suffix count differs from exhaustive oracle");
                    require(actual == expected,
                            "longest-suffix output or guard differs from exhaustive oracle");
                }
            }
        }
    }
}

} // namespace

int main() {
    try {
        continues_the_earlier_occurrence();
        a_miss_reports_zero();
        the_most_recent_occurrence_wins();
        a_proposal_never_runs_past_the_ledger();
        degenerate_arguments_are_refused();
        longest_suffix_prefers_more_context_and_recent_ties();
        longest_suffix_clips_minimum_and_bounds();
        longest_suffix_matches_exhaustive_oracle();
        std::printf("lookup draft checks passed\n");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
