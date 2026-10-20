#include "core/device.h"
#include "models/qwen3_5/program/context.h"
#include "models/qwen3_5/program/program_impl.h"
#include "ninfer/engine.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifndef NINFER_LAB_MTP_COPY_CENSUS
#define NINFER_LAB_MTP_COPY_CENSUS 0
#endif

#if NINFER_LAB_MTP_COPY_CENSUS != 0 && NINFER_LAB_MTP_COPY_CENSUS != 1
#error NINFER_LAB_MTP_COPY_CENSUS_must_be_zero_or_one
#endif

#if NINFER_LAB_MTP_COPY_CENSUS
#include "nlohmann/json.hpp"
#endif

namespace qe = ninfer::models::qwen3_5::execution;
namespace qd = ninfer::models::qwen3_5::detail;

#define CENSUS_RUN "_ZN6ninfer6models7qwen3_59execution16mtp_decode_batchERNS2_15MtpBatchContextEijjNS2_27MtpCausalAttentionEnvelopesEPNS_21DecodeGraphExecutableE"
#define CENSUS_INSTALL "_ZN6ninfer6models7qwen3_56detail11ProgramImpl16install_samplingERNS2_13SequenceStateERNS2_14RequestControlERKNS_3ops14SamplingConfigE"
#define CENSUS_WAIT "_ZN6ninfer16GenerationHandle4waitEPNS_10OutputSinkERKNS_16CancellationViewE"

extern "C" void real_run(qe::MtpBatchContext&, std::int32_t, std::uint32_t, std::uint32_t,
    qe::MtpCausalAttentionEnvelopes, ninfer::DecodeGraphExecutable*) asm("__real_" CENSUS_RUN);
extern "C" void wrap_run(qe::MtpBatchContext&, std::int32_t, std::uint32_t, std::uint32_t,
    qe::MtpCausalAttentionEnvelopes, ninfer::DecodeGraphExecutable*) asm("__wrap_" CENSUS_RUN);
extern "C" void real_install(qd::ProgramImpl*, qd::SequenceState&, qd::RequestControl&,
    const ninfer::ops::SamplingConfig&) asm("__real_" CENSUS_INSTALL);
extern "C" void wrap_install(qd::ProgramImpl*, qd::SequenceState&, qd::RequestControl&,
    const ninfer::ops::SamplingConfig&) asm("__wrap_" CENSUS_INSTALL);
extern "C" ninfer::GenerationResult real_wait(ninfer::GenerationHandle*, ninfer::OutputSink*,
    const ninfer::CancellationView&) asm("__real_" CENSUS_WAIT);
extern "C" ninfer::GenerationResult wrap_wait(ninfer::GenerationHandle*, ninfer::OutputSink*,
    const ninfer::CancellationView&) asm("__wrap_" CENSUS_WAIT);

#if NINFER_LAB_MTP_COPY_CENSUS
namespace {

using Json = nlohmann::json;
using Tokens = std::vector<ninfer::TokenId>;

struct Record {
    std::uint64_t epoch = 0;
    std::uint64_t wait_id = 0;
    bool active = false;
    bool install_before_wait = false;
    bool monotonic_prefix = true;
    Json sampling;
    Json rows = Json::array();
    std::vector<std::string> errors;
    Tokens first_ledger;
    Tokens previous_ledger;

    void check(bool condition, const char* reason) {
        if (!condition && std::find(errors.begin(), errors.end(), reason) == errors.end()) {
            errors.emplace_back(reason);
        }
    }
};

struct Collector {
    std::mutex mutex;
    std::uint64_t next_epoch = 0;
    std::uint64_t next_wait = 0;
    std::uint64_t active_wait = 0;
    std::shared_ptr<Record> current;
    qd::ProgramImpl* program = nullptr;
    qd::SequenceState* sequence = nullptr;
    std::thread::id worker;
    bool lifecycle_error = false;
};

Collector collector;

bool supported(const qd::ProgramImpl& p, const qd::SequenceState& sequence,
               const qd::RequestControl& request, const ninfer::ops::SamplingConfig& c) {
    return p.max_concurrency == 1 && sequence.lane == 0 && p.lookup_ngram == 0 &&
           p.speculative_backend == ninfer::SpeculativeBackend::Mtp && p.draft_window == 3 &&
           p.mtp_policy == ninfer::MtpDraftPolicy::Fixed && !p.mtp_controller &&
           p.proposal_head == ninfer::ProposalHead::Full && !p.parameters.proposal &&
           p.parameters.mtp && p.parameters.model.options().speculative == ninfer::SpeculativeBackend::Mtp &&
           p.device.size() == 1 && p.device.props.major == 8 && p.device.props.minor == 6 &&
           p.parameters.model.config().text.hidden_size == 5120 &&
           p.parameters.model.config().text.vocab_size == 248320 &&
           p.parameters.model.resources().public_token_count == 248077 &&
           !request.grammar && c.token_mask == nullptr && c.token_mask_stride == 0 &&
           c.temperature == 0.0F && c.top_k == 1 && c.presence_penalty == 1.5F &&
           c.frequency_penalty == 0.0F;
}

void emit(const Json& value) {
    const std::string line = "MTP_COPY_CENSUS " + value.dump() + "\n";
    std::fwrite(line.data(), 1, line.size(), stdout);
    std::fflush(stdout);
    if (value.value("kind", "") == "request") {
        const Json binding = {{"epoch", value["epoch"]}, {"wait_id", value["binding"]["wait_id"]},
                              {"active", value["active"]}};
        const std::string marker = "MTP_COPY_CENSUS_BIND " + binding.dump() + "\n";
        std::fwrite(marker.data(), 1, marker.size(), stderr);
        std::fflush(stderr);
    }
}

void observer_failure(const char* phase) noexcept {
    try {
        std::lock_guard lock(collector.mutex);
        collector.lifecycle_error = true;
        if (collector.current) { collector.current->check(false, phase); }
        else { emit({{"schema", 1}, {"kind", "observer_error"}, {"phase", phase}}); }
    } catch (...) {
        std::fputs("MTP_COPY_CENSUS {\"schema\":1,\"kind\":\"observer_error\"}\n", stdout);
        std::fflush(stdout);
    }
}

void install(qd::ProgramImpl& program, qd::SequenceState& sequence,
             const qd::RequestControl& request, const ninfer::ops::SamplingConfig& config) {
    auto record = std::make_shared<Record>();
    record->active = supported(program, sequence, request, config);
    record->sampling = {{"temperature", config.temperature}, {"top_k", config.top_k},
        {"top_p", config.top_p}, {"min_p", config.min_p},
        {"presence_penalty", config.presence_penalty}, {"frequency_penalty", config.frequency_penalty}};
    std::lock_guard lock(collector.mutex);
    record->epoch = ++collector.next_epoch;
    record->wait_id = collector.active_wait;
    record->install_before_wait = collector.active_wait == 0;
    if (collector.current) {
        collector.lifecycle_error = true;
        record->check(false, "install_before_previous_wait_completed");
    }
    record->check(!collector.lifecycle_error, "collector_lifecycle_error");
    collector.current = std::move(record);
    collector.program = &program;
    collector.sequence = &sequence;
    collector.worker = std::this_thread::get_id();
}

std::uint64_t begin_wait() {
    std::lock_guard lock(collector.mutex);
    const std::uint64_t id = ++collector.next_wait;
    if (collector.active_wait != 0) {
        collector.lifecycle_error = true;
        if (collector.current) { collector.current->check(false, "overlapping_public_waits"); }
        return id;
    }
    collector.active_wait = id;
    if (collector.current) {
        collector.current->check(collector.current->wait_id == 0, "epoch_already_bound");
        collector.current->wait_id = id;
    }
    return id;
}

void before_run(qe::MtpBatchContext& state, std::int32_t batch,
                std::uint32_t verify, std::uint32_t draft, ninfer::DecodeGraphExecutable* executable) {
    if (executable == nullptr) { return; }
    std::lock_guard lock(collector.mutex);
    if (!collector.current) {
        collector.lifecycle_error = true;
        emit({{"schema", 1}, {"kind", "observer_error"}, {"phase", "run_without_install"}});
        return;
    }
    Record& r = *collector.current;
    if (!r.active) { return; }
    const bool bound = collector.program != nullptr && collector.sequence != nullptr &&
        collector.worker == std::this_thread::get_id();
    r.check(bound, "run_worker_binding");
    if (!bound) { return; }
    r.check(&collector.program->device == &state.execution.device &&
            &collector.program->parameters == &state.execution.parameters, "run_program_binding");
    r.check(batch == 1 && verify == 3 && draft == 3 && state.execution.stages == nullptr &&
            state.execution.proposal_head == ninfer::ProposalHead::Full && executable != nullptr, "run_scope");
    if (&collector.program->device != &state.execution.device ||
        &collector.program->parameters != &state.execution.parameters ||
        batch != 1 || verify != 3 || draft != 3 || executable == nullptr) {
        return;
    }
    const qd::SequenceState& s = *collector.sequence;
    const auto& ingress = state.host_ingress;
    const auto& sampling = ingress.sampling[0];
    r.check(sampling.temperature == 0.0F && sampling.top_k == 1 &&
        sampling.presence_penalty == 1.5F && sampling.frequency_penalty == 0.0F &&
        sampling.top_p == r.sampling["top_p"].get<float>() &&
        sampling.min_p == r.sampling["min_p"].get<float>() &&
        sampling.token_mask == nullptr && sampling.token_mask_stride == 0, "host_sampling_contract");
    const bool frontier = s.ledger_frontier == s.execution_frontier + 1 &&
        s.ledger.size() == s.ledger_frontier && s.mtp_kv_valid == s.execution_frontier &&
        s.mtp_draft_count <= draft;
    r.check(frontier, "committed_frontier_contract");
    if (!frontier) { return; }
    const auto anchor = s.ledger[s.execution_frontier];
    const std::int32_t extent = ingress.current_extents[0];
    const std::int32_t budget = ingress.remaining_budgets[0];
    r.check(ingress.base_frontiers[0] == static_cast<std::int32_t>(s.execution_frontier) &&
        ingress.anchors[0] == anchor && budget > 0 && extent >= 0 &&
        extent <= static_cast<std::int32_t>(std::min(verify, s.mtp_draft_count)) &&
        extent < budget && ingress.target_valid_columns[0] == extent + 1, "host_ingress_contract");
    if (extent < 0 || extent > static_cast<std::int32_t>(s.mtp_draft_count)) { return; }
    for (std::uint32_t i = 0; i < verify; ++i) {
        r.check(ingress.current_drafts[i] == (i < static_cast<std::uint32_t>(extent) ?
            s.mtp_drafts[i] : anchor), "host_draft_binding");
    }
    Tokens delta;
    if (r.rows.empty()) {
        r.first_ledger = s.ledger;
    } else {
        const bool prefix = s.ledger.size() > r.previous_ledger.size() &&
            std::equal(r.previous_ledger.begin(), r.previous_ledger.end(), s.ledger.begin());
        r.monotonic_prefix = r.monotonic_prefix && prefix;
        r.check(prefix, "nonmonotonic_committed_ledger");
        if (!prefix) { return; }
        delta.assign(s.ledger.begin() + static_cast<std::ptrdiff_t>(r.previous_ledger.size()), s.ledger.end());
    }
    Tokens drafts(s.mtp_drafts.begin(), s.mtp_drafts.begin() + s.mtp_draft_count);
    r.rows.push_back({{"round", r.rows.size() + 1}, {"E", s.execution_frontier},
        {"S", s.ledger_frontier}, {"anchor", anchor}, {"mtp_kv_valid", s.mtp_kv_valid},
        {"verify_window", verify}, {"draft_window", draft}, {"target_width", verify + 1},
        {"remaining_budget", budget}, {"extent", extent}, {"mtp_draft_count", s.mtp_draft_count},
        {"drafts", std::move(drafts)}, {"committed_delta", std::move(delta)}});
    r.previous_ledger = s.ledger;
}

const char* finish_name(ninfer::FinishReason reason) {
    switch (reason) {
        case ninfer::FinishReason::None: return "none";
        case ninfer::FinishReason::OutputLimit: return "output_limit";
        case ninfer::FinishReason::ContextCapacity: return "context_capacity";
        case ninfer::FinishReason::StopToken: return "stop_token";
        case ninfer::FinishReason::StopString: return "stop_string";
        case ninfer::FinishReason::Cancelled: return "cancelled";
    }
    return "unknown";
}

void finish_wait(std::uint64_t id, const ninfer::GenerationResult* result) {
    std::shared_ptr<Record> record;
    {
        std::lock_guard lock(collector.mutex);
        if (collector.active_wait == id && collector.current && collector.current->wait_id == id) {
            record = std::move(collector.current);
        } else {
            record = std::make_shared<Record>();
            record->wait_id = id;
            record->check(false, "wait_without_unique_install_epoch");
            collector.lifecycle_error = true;
        }
        record->check(!collector.lifecycle_error, "collector_lifecycle_error");
        if (collector.active_wait == id) {
            collector.active_wait = 0;
            collector.program = nullptr;
            collector.sequence = nullptr;
            collector.worker = {};
        }
    }
    Record& r = *record;
    Json native = nullptr;
    bool prefixes = false;
    std::size_t terminal_tokens = 0;
    if (result != nullptr) {
        const auto& stats = result->speculative;
        native = {{"prompt_tokens", result->prompt.prompt_tokens}, {"has_media", result->prompt.has_media},
            {"generated_token_ids", result->generated_token_ids}, {"finish_reason", finish_name(result->finish_reason)},
            {"rounds", stats.rounds}, {"drafted_tokens", stats.drafted_tokens}, {"accepted_tokens", stats.accepted_tokens},
            {"speculative", {{"enabled", stats.enabled}, {"backend", stats.backend == ninfer::SpeculativeBackend::Mtp ? "mtp" : "other"},
                {"draft_window", stats.draft_window}, {"rounds", stats.rounds}, {"drafted_tokens", stats.drafted_tokens},
                {"accepted_tokens", stats.accepted_tokens}, {"fallback_steps", stats.fallback_steps},
                {"accepted_per_position", stats.accepted_per_position}, {"adaptive", stats.adaptive},
                {"window_transitions", stats.window_transitions}, {"rounds_per_window", stats.rounds_per_window}}}};
        if (r.active) {
            const std::size_t prompt = result->prompt.prompt_tokens;
            const auto& generated = result->generated_token_ids;
            prefixes = r.first_ledger.size() > prompt && r.previous_ledger.size() >= prompt &&
                r.previous_ledger.size() - prompt <= generated.size() &&
                std::equal(r.previous_ledger.begin() + static_cast<std::ptrdiff_t>(prompt),
                           r.previous_ledger.end(), generated.begin());
            r.check(prefixes && r.monotonic_prefix, "native_result_prefix_mismatch");
            r.check(!result->prompt.has_media && !r.rows.empty() && r.rows.size() == stats.rounds + stats.fallback_steps &&
                stats.backend == ninfer::SpeculativeBackend::Mtp && stats.enabled && stats.draft_window == 3 &&
                !stats.adaptive, "native_result_scope");
            if (prefixes && !r.rows.empty()) {
                terminal_tokens = generated.size() - (r.previous_ledger.size() - prompt);
                r.check(terminal_tokens > 0 && terminal_tokens <= r.rows.back()["extent"].get<std::size_t>() + 1,
                        "terminal_committed_span");
                if (result->finish_reason == ninfer::FinishReason::OutputLimit) {
                    for (const auto& row : r.rows) {
                        r.check(row["remaining_budget"].get<std::size_t>() ==
                            generated.size() - (row["S"].get<std::size_t>() - prompt), "native_result_budget");
                    }
                }
            }
        }
    } else {
        r.check(false, "native_wait_threw");
    }
    emit({{"schema", 1}, {"kind", "request"}, {"diagnostic_only", true},
        {"epoch", r.epoch}, {"active", r.active}, {"error", r.errors}, {"sampling", r.sampling},
        {"first_ledger", r.first_ledger}, {"rows", r.rows}, {"result", std::move(native)},
        {"binding", {{"wait_id", r.wait_id}, {"install_before_wait", r.install_before_wait},
            {"captured_on_program_thread", true}, {"sequence_pointer_cleared", true}}},
        {"validation", {{"monotonic_prefix", r.monotonic_prefix}, {"native_result_prefixes", prefixes},
            {"terminal_committed_tokens", terminal_tokens}, {"terminal_source", "generation_result"},
            {"terminal_ledger_snapshot", false}}}});
}

}
#endif

extern "C" void wrap_install(qd::ProgramImpl* self, qd::SequenceState& sequence,
    qd::RequestControl& request, const ninfer::ops::SamplingConfig& config) {
#if NINFER_LAB_MTP_COPY_CENSUS
    try { install(*self, sequence, request, config); }
    catch (...) { observer_failure("install_observer_exception"); }
#endif
    real_install(self, sequence, request, config);
}

extern "C" void wrap_run(qe::MtpBatchContext& state, std::int32_t batch,
    std::uint32_t verify, std::uint32_t draft, qe::MtpCausalAttentionEnvelopes envelopes,
    ninfer::DecodeGraphExecutable* executable) {
#if NINFER_LAB_MTP_COPY_CENSUS
    try { before_run(state, batch, verify, draft, executable); }
    catch (...) { observer_failure("run_observer_exception"); }
#endif
    real_run(state, batch, verify, draft, envelopes, executable);
}

extern "C" ninfer::GenerationResult wrap_wait(ninfer::GenerationHandle* self,
    ninfer::OutputSink* sink, const ninfer::CancellationView& cancellation) {
#if NINFER_LAB_MTP_COPY_CENSUS
    std::uint64_t id = 0;
    try { id = begin_wait(); }
    catch (...) { observer_failure("wait_entry_observer_exception"); }
    ninfer::GenerationResult result;
    try { result = real_wait(self, sink, cancellation); }
    catch (...) {
        try { finish_wait(id, nullptr); }
        catch (...) { observer_failure("wait_failure_observer_exception"); }
        throw;
    }
    try { finish_wait(id, &result); }
    catch (...) { observer_failure("wait_return_observer_exception"); }
    return result;
#else
    return real_wait(self, sink, cancellation);
#endif
}
