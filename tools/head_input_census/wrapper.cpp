#include "observer.h"
#include "core/device.h"
#include "models/qwen3_5/program/context.h"
#include "models/qwen3_5/program/program_impl.h"
#include "ninfer/ops/speculative_round.h"
#include "ops/linear/gguf/ggml_bridge.h"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifndef NINFER_LAB_HEAD_INPUT_CENSUS
#define NINFER_LAB_HEAD_INPUT_CENSUS 0
#endif

#if NINFER_LAB_HEAD_INPUT_CENSUS != 0 && NINFER_LAB_HEAD_INPUT_CENSUS != 1
#error NINFER_LAB_HEAD_INPUT_CENSUS_must_be_zero_or_one
#endif

#if NINFER_LAB_HEAD_INPUT_CENSUS
#include "nlohmann/json.hpp"
#endif

namespace qe = ninfer::models::qwen3_5::execution;
namespace qd = ninfer::models::qwen3_5::detail;
namespace qm = ninfer::models::qwen3_5;
namespace census = ninfer::lab::mtp_census;
namespace head = ninfer::lab::head_census;
namespace gguf = ninfer::ops::gguf;

#ifndef HEAD_VECTOR_SYMBOL
#error HEAD_VECTOR_SYMBOL_requires_native_preflight
#endif

static_assert(sizeof(gguf::VectorOutput) == 48);
static_assert(offsetof(gguf::VectorOutput, epilogue) == 24);
static_assert(offsetof(gguf::VectorOutput, gate) == 32);

extern "C" void real_vector(gguf::GgmlType, const void*, std::int64_t, int, int,
    const void*, int, const gguf::VectorOutput&, cudaStream_t) asm("__real_" HEAD_VECTOR_SYMBOL);
extern "C" void wrap_vector(gguf::GgmlType, const void*, std::int64_t, int, int,
    const void*, int, const gguf::VectorOutput&, cudaStream_t) asm("__wrap_" HEAD_VECTOR_SYMBOL);

#define CENSUS_CAPTURE "_ZN6ninfer6models7qwen3_59execution24capture_mtp_decode_batchERNS2_15MtpBatchContextEijjNS2_27MtpCausalAttentionEnvelopesERNS_21DecodeGraphDefinitionE"
#define CENSUS_RUN "_ZN6ninfer6models7qwen3_59execution16mtp_decode_batchERNS2_15MtpBatchContextEijjNS2_27MtpCausalAttentionEnvelopesEPNS_21DecodeGraphExecutableE"
#define CENSUS_PROPOSE "_ZN6ninfer6models7qwen3_59execution11TextContext17mtp_propose_batchERKNS_6TensorERS4_S7_"
#define CENSUS_ACCEPT "_ZN6ninfer3ops32speculative_accept_greedy_draftsERKNS_6TensorES3_S3_S3_RS1_S4_S4_S4_S4_iPKNS0_14SamplingConfigERNS_11DeviceArenaEP11CUstream_st"
#define CENSUS_INSTALL "_ZN6ninfer6models7qwen3_56detail11ProgramImpl16install_samplingERNS2_13SequenceStateERNS2_14RequestControlERKNS_3ops14SamplingConfigE"
#define CENSUS_SYNC "_ZNK6ninfer13DeviceContext11synchronizeEv"

extern "C" void real_capture(qe::MtpBatchContext&, std::int32_t, std::uint32_t, std::uint32_t,
    qe::MtpCausalAttentionEnvelopes, ninfer::DecodeGraphDefinition&) asm("__real_" CENSUS_CAPTURE);
extern "C" void wrap_capture(qe::MtpBatchContext&, std::int32_t, std::uint32_t, std::uint32_t,
    qe::MtpCausalAttentionEnvelopes, ninfer::DecodeGraphDefinition&) asm("__wrap_" CENSUS_CAPTURE);
extern "C" void real_run(qe::MtpBatchContext&, std::int32_t, std::uint32_t, std::uint32_t,
    qe::MtpCausalAttentionEnvelopes, ninfer::DecodeGraphExecutable*) asm("__real_" CENSUS_RUN);
extern "C" void wrap_run(qe::MtpBatchContext&, std::int32_t, std::uint32_t, std::uint32_t,
    qe::MtpCausalAttentionEnvelopes, ninfer::DecodeGraphExecutable*) asm("__wrap_" CENSUS_RUN);
extern "C" void real_propose(qe::TextContext*, const ninfer::Tensor&, ninfer::Tensor&,
    ninfer::Tensor&) asm("__real_" CENSUS_PROPOSE);
extern "C" void wrap_propose(qe::TextContext*, const ninfer::Tensor&, ninfer::Tensor&,
    ninfer::Tensor&) asm("__wrap_" CENSUS_PROPOSE);
extern "C" void real_accept(const ninfer::Tensor&, const ninfer::Tensor&, const ninfer::Tensor&,
    const ninfer::Tensor&, ninfer::Tensor&, ninfer::Tensor&, ninfer::Tensor&, ninfer::Tensor&,
    ninfer::Tensor&, std::int32_t, const ninfer::ops::SamplingConfig*, ninfer::WorkspaceArena&,
    cudaStream_t) asm("__real_" CENSUS_ACCEPT);
extern "C" void wrap_accept(const ninfer::Tensor&, const ninfer::Tensor&, const ninfer::Tensor&,
    const ninfer::Tensor&, ninfer::Tensor&, ninfer::Tensor&, ninfer::Tensor&, ninfer::Tensor&,
    ninfer::Tensor&, std::int32_t, const ninfer::ops::SamplingConfig*, ninfer::WorkspaceArena&,
    cudaStream_t) asm("__wrap_" CENSUS_ACCEPT);
extern "C" void real_install(qd::ProgramImpl*, qd::SequenceState&, qd::RequestControl&,
    const ninfer::ops::SamplingConfig&) asm("__real_" CENSUS_INSTALL);
extern "C" void wrap_install(qd::ProgramImpl*, qd::SequenceState&, qd::RequestControl&,
    const ninfer::ops::SamplingConfig&) asm("__wrap_" CENSUS_INSTALL);
extern "C" void real_sync(const ninfer::DeviceContext*) asm("__real_" CENSUS_SYNC);
extern "C" void wrap_sync(const ninfer::DeviceContext*) asm("__wrap_" CENSUS_SYNC);

#if NINFER_LAB_HEAD_INPUT_CENSUS
namespace {

using ninfer::DType;
using ninfer::Tensor;
using Json = nlohmann::json;

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

bool shape(const Tensor& t, DType type, std::initializer_list<int> dimensions) {
    if (t.data == nullptr || t.dtype != type || !t.is_contiguous()) { return false; }
    const auto alignment = type == DType::BF16 ? 2U : 4U;
    if (reinterpret_cast<std::uintptr_t>(t.data) % alignment != 0) { return false; }
    int i = 0;
    for (int n : dimensions) { if (t.ne[i++] != n) { return false; } }
    for (; i < 4; ++i) { if (t.ne[i] != 1) { return false; } }
    return true;
}

bool same(const Tensor& a, const Tensor& b) {
    return a.data == b.data && a.dtype == b.dtype &&
           std::memcmp(a.ne, b.ne, sizeof(a.ne)) == 0 && a.is_contiguous() && b.is_contiguous();
}

bool closed(qe::MtpBatchContext& state, int batch, unsigned verify, unsigned draft) {
    const auto& e = state.execution;
    const auto& f = state.frame;
    const auto& model = e.parameters.model;
    if (batch != 1 || verify != census::kDrafts || draft != census::kDrafts ||
        e.device.size() != 1 || e.device.props.major != 8 || e.device.props.minor != 6 ||
        e.stages != nullptr || e.proposal_head != ninfer::ProposalHead::Full ||
        !e.parameters.mtp || e.parameters.proposal || model.options().speculative !=
            ninfer::SpeculativeBackend::Mtp ||
        model.config().text.hidden_size != 5120 || model.config().text.vocab_size != census::kPhysical ||
        model.resources().public_token_count != census::kDomain) { return false; }
    if (f.ingress.data == nullptr || f.egress.data == nullptr ||
        f.ingress.bytes < sizeof(qm::MtpDecodeIngress) ||
        f.egress.bytes < sizeof(qm::MtpDecodeEgress)) { return false; }
    const auto* in = static_cast<const unsigned char*>(f.ingress.data);
    const auto* out = static_cast<const unsigned char*>(f.egress.data);
    if (f.sampling != reinterpret_cast<const ninfer::ops::SamplingConfig*>(
            in + offsetof(qm::MtpDecodeIngress, sampling))) { return false; }
    const struct Field { const Tensor* tensor; const void* address; int first; int second; } fields[] = {
        {&f.anchors, in + offsetof(qm::MtpDecodeIngress, anchors), 1, 1},
        {&f.base_frontiers, in + offsetof(qm::MtpDecodeIngress, base_frontiers), 1, 1},
        {&f.remaining_budgets, in + offsetof(qm::MtpDecodeIngress, remaining_budgets), 1, 1},
        {&f.current_extents, in + offsetof(qm::MtpDecodeIngress, current_extents), 1, 1},
        {&f.target_valid_columns, in + offsetof(qm::MtpDecodeIngress, target_valid_columns), 1, 1},
        {&f.current_drafts, in + offsetof(qm::MtpDecodeIngress, current_drafts), 3, 1},
        {&f.licensed_tokens, out + offsetof(qm::MtpDecodeEgress, licensed_tokens), 4, 1},
        {&f.licensed_counts, out + offsetof(qm::MtpDecodeEgress, licensed_counts), 1, 1},
        {&f.accepted_drafts, out + offsetof(qm::MtpDecodeEgress, accepted_drafts), 1, 1},
        {&f.next_drafts, out + offsetof(qm::MtpDecodeEgress, next_drafts), 1, 3},
        {&f.next_extents, out + offsetof(qm::MtpDecodeEgress, next_extents), 1, 1},
    };
    for (const auto& field : fields) {
        if (field.tensor->data != field.address ||
            !shape(*field.tensor, DType::I32, {field.first, field.second})) { return false; }
    }
    return shape(f.target_logits, DType::BF16, {census::kPhysical, 4, 1}) &&
           shape(f.target_argmax, DType::I32, {4, 1}) &&
           shape(f.proposal_logits, DType::BF16, {census::kPhysical, 1});
}

census::View view(const qm::MtpDecodeState& f) {
    const auto ptr = [](const Tensor& t) { return static_cast<const std::int32_t*>(t.data); };
    return {ptr(f.current_drafts), ptr(f.current_extents), ptr(f.target_valid_columns),
            ptr(f.remaining_budgets), ptr(f.base_frontiers), ptr(f.anchors),
            ptr(f.accepted_drafts), ptr(f.licensed_counts), ptr(f.licensed_tokens),
            ptr(f.next_extents), ptr(f.next_drafts), f.sampling};
}

Json score_json(const census::Score& s) {
    return {{"id", s.id}, {"raw_bits", s.raw_bits}, {"adjusted_bits", s.adjusted_bits},
            {"base_count", s.base_count}, {"prefix_count", s.prefix_count}};
}

Json top_json(const census::Top2& t) {
    return {{"raw", Json::array({score_json(t.raw[0]), score_json(t.raw[1])})},
            {"adjusted", Json::array({score_json(t.adjusted[0]), score_json(t.adjusted[1])})}};
}

template <class Column>
Json column_json(const Column& c) {
    Json out = top_json(c.top);
    out["candidates"] = Json::array();
    for (int j = 0; j < c.candidate_count && j < census::kCandidates; ++j) {
        out["candidates"].push_back(score_json(c.candidates[j]));
    }
    return out;
}

Json record_json(const census::Record& r) {
    Json out = {{"epoch", r.epoch}, {"round", r.round}, {"producer_round", r.producer_round},
        {"q_epoch", r.q_epoch}, {"q_frontier", r.q_frontier}, {"q_anchor", r.q_anchor},
        {"q_extent", r.q_extent}, {"frontier", r.frontier}, {"anchor", r.anchor},
        {"extent", r.extent}, {"accepted", r.accepted}, {"licensed_count", r.licensed_count},
        {"post_frontier", r.post_frontier}, {"post_anchor", r.post_anchor},
        {"causal_columns", r.causal_columns}, {"drafts", Json::array()},
        {"licensed", Json::array()}, {"q", Json::array()}, {"p", Json::array()}};
    for (int j = 0; j < r.extent; ++j) {
        out["drafts"].push_back(r.drafts[j]);
        Json q = column_json(r.q[j]);
        q["actual"] = r.q[j].actual;
        out["q"].push_back(std::move(q));
    }
    for (int j = 0; j < r.licensed_count; ++j) { out["licensed"].push_back(r.licensed[j]); }
    for (int col = 0; col <= r.extent; ++col) {
        Json p = column_json(r.p[col]);
        p["actual_raw"] = r.p[col].actual_raw;
        p["causal"] = col <= r.accepted;
        p["q_score_at_target_winner"] = nullptr;
        if (col < r.extent) {
            for (int j = 0; j < r.q[col].candidate_count; ++j) {
                if (r.q[col].candidates[j].id == r.p[col].top.adjusted[0].id) {
                    p["q_score_at_target_winner"] = score_json(r.q[col].candidates[j]);
                }
            }
        }
        out["p"].push_back(std::move(p));
    }
    return out;
}

struct Host {
    census::State* device = nullptr;
    census::State* mirror = nullptr;
    head::Payload* payload_device = nullptr;
    head::Payload* payload_host = nullptr;
    const ninfer::DeviceContext* owner = nullptr;
    qd::ProgramImpl* program = nullptr;
    qd::SequenceState* sequence = nullptr;
    std::thread::id worker;
    cudaStream_t stream = nullptr;
    const void* weight = nullptr;
    std::filesystem::path directory;
    Json ledger_rows = Json::array();
    std::vector<std::int32_t> first_ledger;
    std::vector<std::int32_t> previous_ledger;
    unsigned host_round = 0;
    bool active_request = false;
    bool capture_this_run = false;
    std::uint32_t reported_epoch = 0;
    unsigned captures = 0;
    bool pending = false;
    bool ready = false;

    void bind(qd::ProgramImpl& p, qd::SequenceState& s, bool active) {
        program = &p;
        sequence = &s;
        worker = std::this_thread::get_id();
        host_round = 0;
        active_request = active;
        ledger_rows = Json::array();
        first_ledger.clear();
        previous_ledger.clear();
    }

    void before_run(qe::MtpBatchContext& state) {
        capture_this_run = false;
        if (!active_request || host_round >= census::kRecords + 1) { return; }
        require(program != nullptr && sequence != nullptr &&
                worker == std::this_thread::get_id() && &program->device == owner &&
                &program->parameters == &state.execution.parameters,
                "head census native Program/Sequence binding");
        const auto& s = *sequence;
        const auto& in = state.host_ingress;
        require(s.ledger_frontier == s.execution_frontier + 1 &&
                s.ledger.size() == s.ledger_frontier && s.mtp_kv_valid == s.execution_frontier &&
                s.mtp_draft_count <= census::kDrafts, "head census committed frontier");
        const auto anchor = s.ledger[s.execution_frontier];
        const int extent = in.current_extents[0];
        require(in.base_frontiers[0] == static_cast<int>(s.execution_frontier) &&
                in.anchors[0] == anchor && extent >= 0 &&
                extent <= static_cast<int>(s.mtp_draft_count) &&
                extent < in.remaining_budgets[0] && in.target_valid_columns[0] == extent + 1,
                "head census host ingress binding");
        for (int i = 0; i < census::kDrafts; ++i) {
            require(in.current_drafts[i] == (i < extent ? s.mtp_drafts[i] : anchor),
                    "head census host draft binding");
        }
        std::vector<std::int32_t> delta;
        if (host_round == 0) { first_ledger = s.ledger; }
        else {
            require(s.ledger.size() > previous_ledger.size() &&
                    std::equal(previous_ledger.begin(), previous_ledger.end(), s.ledger.begin()),
                    "head census nonmonotonic native ledger");
            delta.assign(s.ledger.begin() + static_cast<std::ptrdiff_t>(previous_ledger.size()),
                         s.ledger.end());
        }
        ledger_rows.push_back({{"round", host_round + 1}, {"frontier", s.execution_frontier},
            {"ledger_frontier", s.ledger_frontier}, {"anchor", anchor}, {"extent", extent},
            {"budget", in.remaining_budgets[0]}, {"committed_delta", std::move(delta)}});
        previous_ledger = s.ledger;
        ++host_round;
        capture_this_run = true;
    }

    void copy_payload() {
        if (!capture_this_run) { return; }
        const unsigned r = host_round;
        if (r <= census::kRecords) {
            CUDA_CHECK(cudaMemcpyAsync(&payload_host->samples[r - 1][0],
                &payload_device->samples[r - 1][0], 3 * sizeof(head::Sample),
                cudaMemcpyDeviceToHost, stream));
        }
        if (r >= 2) {
            CUDA_CHECK(cudaMemcpyAsync(&payload_host->samples[r - 2][3],
                &payload_device->samples[r - 2][3], 4 * sizeof(head::Sample),
                cudaMemcpyDeviceToHost, stream));
        }
        if (r == 2) {
            CUDA_CHECK(cudaMemcpyAsync(payload_host, payload_device,
                offsetof(head::Payload, samples), cudaMemcpyDeviceToHost, stream));
        }
    }

    Json save(const census::State& s) {
        if (s.completed == 0) { return nullptr; }
        require(s.error == census::None && s.completed <= census::kRecords &&
                payload_host->counts_epoch == s.epoch && ledger_rows.size() >= s.completed + 1,
                "head census incomplete payload provenance");
        std::uint64_t address = 0;
        for (unsigned record = 0; record < s.completed; ++record) {
            const auto& r = s.records[record];
            require(r.round == record + 2 && r.producer_round == record + 1,
                    "head census nonconsecutive linked window");
            const auto& row = ledger_rows[record + 1];
            require(row["frontier"].get<int>() == r.frontier &&
                    row["anchor"].get<int>() == r.anchor && row["extent"].get<int>() == r.extent,
                    "head census ledger/device join");
            for (int role = 0; role < head::kRoles; ++role) {
                const auto& sample = payload_host->samples[record][role];
                const bool target = role >= 3;
                require(sample.epoch == s.epoch && sample.round == record + (target ? 2 : 1) &&
                        sample.role == role && sample.frontier == r.frontier &&
                        sample.anchor == r.anchor &&
                        sample.extent == (target ? r.extent : r.q_extent) &&
                        sample.weight_address != 0, "head census sample binding");
                if (address == 0) { address = sample.weight_address; }
                require(sample.weight_address == address, "head census head weights changed");
            }
        }
        const std::uint64_t bytes = sizeof(payload_host->counts) +
            s.completed * head::kRoles * sizeof(head::Sample);
        const head::FileHeader header{head::kMagic, 1, s.epoch, s.completed, head::kK,
            census::kPhysical, census::kDomain, head::kPacked, sizeof(head::Sample), bytes};
        const auto path = directory / ("head-input-epoch-" + std::to_string(s.epoch) + ".bin");
        FILE* file = std::fopen(path.string().c_str(), "wbx");
        require(file != nullptr, "head census binary path must be new and writable");
        bool ok = std::fwrite(&header, sizeof(header), 1, file) == 1 &&
            std::fwrite(payload_host->counts, sizeof(payload_host->counts), 1, file) == 1 &&
            std::fwrite(payload_host->samples, head::kRoles * sizeof(head::Sample),
                         s.completed, file) == s.completed;
        ok = std::fclose(file) == 0 && ok;
        require(ok, "head census binary write failed");
        return {{"path", path.string()}, {"bytes", sizeof(header) + bytes},
            {"schema", 1}, {"initial_counts", "logical_vocab_before_first_saved_P_acceptance"},
            {"sample_roles", {"Q0", "Q1", "Q2", "P0", "P1", "P2", "P3"}},
            {"first_ledger", first_ledger}, {"ledger_rows", ledger_rows},
            {"head_weight_address", address}, {"weight_content_identity_verified", false}};
    }

    void emit(const char* reason) {
        if (!ready || pending || mirror->epoch == 0 || mirror->epoch == reported_epoch) { return; }
        const auto& s = *mirror;
        Json j = {{"kind", "ninfer_lab_head_input_census"}, {"schema", 1}, {"diagnostic_only", true},
            {"reason", reason}, {"epoch", s.epoch}, {"active", s.active}, {"error", s.error},
            {"device_observed_rounds", s.round}, {"completed_linked_records", s.completed},
            {"first_frontier", s.first_frontier}, {"first_anchor", s.first_anchor},
            {"first_budget", s.first_budget}, {"physical_rows", census::kPhysical},
            {"valid_rows", census::kDomain}, {"bootstrap_omitted", true},
            {"native_q_policy", "raw_argmax"}, {"q_adjusted_is_counterfactual", true},
            {"target_policy", "processed_greedy"},
            {"sampling", {{"temperature", s.sampling.temperature}, {"top_k", s.sampling.top_k},
                {"presence_penalty", s.sampling.presence_penalty},
                {"frequency_penalty", s.sampling.frequency_penalty}}},
            {"records", Json::array()}};
        j["payload"] = s.error == census::None ? save(s) : Json(nullptr);
        for (unsigned i = 0; i < s.completed && i < census::kRecords; ++i) {
            j["records"].push_back(record_json(s.records[i]));
        }
        const std::string encoded = j.dump();
        std::fprintf(stderr, "%s\n", encoded.c_str());
        std::fflush(stderr);
        reported_epoch = s.epoch;
    }

    void ensure(ninfer::DeviceContext& context) {
        if (device != nullptr) {
            require(owner == &context && stream == context.stream, "census supports one Program only");
            return;
        }
        cudaStreamCaptureStatus status{};
        CUDA_CHECK(cudaStreamIsCapturing(context.stream, &status));
        require(status == cudaStreamCaptureStatusNone, "census allocation during capture");
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device), sizeof(census::State)));
        try {
            CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&mirror), sizeof(census::State),
                                     cudaHostAllocDefault));
            CUDA_CHECK(cudaMemsetAsync(device, 0, sizeof(census::State), context.stream));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&payload_device), sizeof(head::Payload)));
            CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&payload_host), sizeof(head::Payload),
                                     cudaHostAllocDefault));
            const char* path = std::getenv("NINFER_HEAD_INPUT_CENSUS_DIR");
            require(path != nullptr && std::filesystem::path(path).is_absolute() &&
                    std::filesystem::is_directory(path),
                    "head census requires an existing absolute output directory");
            directory = path;
        } catch (...) {
            if (payload_host != nullptr) { cudaFreeHost(payload_host); payload_host = nullptr; }
            if (payload_device != nullptr) { cudaFree(payload_device); payload_device = nullptr; }
            if (mirror != nullptr) { cudaFreeHost(mirror); mirror = nullptr; }
            cudaFree(device);
            device = nullptr;
            throw;
        }
        std::memset(mirror, 0, sizeof(census::State));
        owner = &context;
        stream = context.stream;
    }

    ~Host() {
        try { emit("process_exit_partial"); } catch (...) {}
        if (mirror != nullptr && !pending) { cudaFreeHost(mirror); }
        if (device != nullptr && !pending) { cudaFree(device); }
        if (payload_host != nullptr && !pending) { cudaFreeHost(payload_host); }
        if (payload_device != nullptr && !pending) { cudaFree(payload_device); }
    }
};

Host host;

struct Scope;
thread_local Scope* current = nullptr;

struct Scope {
    qe::MtpBatchContext& state;
    census::View operands;
    int accepted_calls = 0;
    int proposal_calls = 0;
    int head_calls = 0;
    int proposal_role = -1;

    explicit Scope(qe::MtpBatchContext& s) : state(s), operands(view(s.frame)) {
        require(current == nullptr, "nested census MTP transaction");
        current = this;
    }
    ~Scope() { current = nullptr; }
    void complete(bool body) const {
        require(accepted_calls == (body ? 1 : 0) && proposal_calls == (body ? 3 : 0) &&
                head_calls == (body ? 4 : 0),
                "census MTP body was only partially intercepted");
    }
};

bool request_supported(const qd::ProgramImpl& p, const qd::SequenceState& sequence,
                       const qd::RequestControl& request, const ninfer::ops::SamplingConfig& c) {
    return p.max_concurrency == 1 && sequence.lane == 0 && p.lookup_ngram == 0 &&
           p.speculative_backend == ninfer::SpeculativeBackend::Mtp && p.draft_window == 3 &&
           p.mtp_policy == ninfer::MtpDraftPolicy::Fixed &&
           p.proposal_head == ninfer::ProposalHead::Full && !p.parameters.proposal &&
           p.device.size() == 1 && p.device.props.major == 8 && p.device.props.minor == 6 &&
           p.parameters.model.config().text.vocab_size == census::kPhysical &&
           p.parameters.model.resources().public_token_count == census::kDomain &&
           shape(p.token_counts, DType::I32, {census::kDomain, 1}) &&
           !request.grammar && c.token_mask == nullptr && c.token_mask_stride == 0 &&
           c.temperature == 0.0F && c.top_k == 1 &&
           c.presence_penalty == 1.5F && c.frequency_penalty == 0.0F;
}

}
#endif

extern "C" void wrap_capture(qe::MtpBatchContext& state, std::int32_t batch,
    std::uint32_t verify, std::uint32_t draft, qe::MtpCausalAttentionEnvelopes envelopes,
    ninfer::DecodeGraphDefinition& definition) {
#if NINFER_LAB_HEAD_INPUT_CENSUS
    if (closed(state, batch, verify, draft)) {
        host.ensure(state.execution.device);
        Scope scope(state);
        real_capture(state, batch, verify, draft, envelopes, definition);
        scope.complete(true);
        ++host.captures;
        return;
    }
#endif
    real_capture(state, batch, verify, draft, envelopes, definition);
}

extern "C" void wrap_run(qe::MtpBatchContext& state, std::int32_t batch,
    std::uint32_t verify, std::uint32_t draft, qe::MtpCausalAttentionEnvelopes envelopes,
    ninfer::DecodeGraphExecutable* executable) {
#if NINFER_LAB_HEAD_INPUT_CENSUS
    if (closed(state, batch, verify, draft)) {
        host.ensure(state.execution.device);
        require(!host.pending, "census egress was not synchronized");
        require(executable == nullptr || host.captures != 0, "census graph was not instrumented");
        host.before_run(state);
        Scope scope(state);
        real_run(state, batch, verify, draft, envelopes, executable);
        scope.complete(executable == nullptr);
        host.copy_payload();
        CUDA_CHECK(cudaMemcpyAsync(host.mirror, host.device, sizeof(census::State),
                                   cudaMemcpyDeviceToHost, state.execution.device.stream));
        host.pending = true;
        return;
    }
#endif
    real_run(state, batch, verify, draft, envelopes, executable);
}

extern "C" void wrap_propose(qe::TextContext* self, const ninfer::Tensor& hidden,
                            ninfer::Tensor& logits, ninfer::Tensor& tokens) {
#if NINFER_LAB_HEAD_INPUT_CENSUS
    if (current != nullptr) {
        Scope& scope = *current;
        const auto& frame = scope.state.frame;
        const int step = scope.proposal_calls;
        const auto* expected = static_cast<const std::int32_t*>(frame.next_drafts.data) + step;
        require(scope.accepted_calls == 1 && step < 3 &&
                shape(hidden, DType::BF16, {5120, 1}) &&
                same(logits, frame.proposal_logits) && shape(tokens, DType::I32, {1}) &&
                tokens.data == expected, "census unexpected proposal operands");
        require(scope.proposal_role == -1, "head census nested proposal");
        scope.proposal_role = step;
        real_propose(self, hidden, logits, tokens);
        scope.proposal_role = -1;
        CUDA_CHECK(census::proposal(host.device, logits.data,
            static_cast<const std::int32_t*>(tokens.data), step, scope.operands, host.stream));
        ++scope.proposal_calls;
        return;
    }
#endif
    real_propose(self, hidden, logits, tokens);
}

extern "C" void wrap_accept(const ninfer::Tensor& target, const ninfer::Tensor& logits,
    const ninfer::Tensor& drafts, const ninfer::Tensor& extents, ninfer::Tensor& lengths,
    ninfer::Tensor& anchors, ninfer::Tensor& licensed, ninfer::Tensor& counts,
    ninfer::Tensor& accepted, std::int32_t domain, const ninfer::ops::SamplingConfig* sampling,
    ninfer::WorkspaceArena& workspace, cudaStream_t stream) {
#if NINFER_LAB_HEAD_INPUT_CENSUS
    if (current != nullptr) {
        Scope& scope = *current;
        const auto& f = scope.state.frame;
        require(scope.accepted_calls == 0 && scope.proposal_calls == 0 &&
                domain == census::kDomain && stream == host.stream && sampling == f.sampling &&
                same(target, f.target_argmax) && same(logits, f.target_logits) &&
                same(drafts, f.current_drafts) && same(extents, f.current_extents) &&
                same(lengths, f.base_frontiers) && same(anchors, f.anchors) &&
                same(licensed, f.licensed_tokens) && same(counts, f.licensed_counts) &&
                same(accepted, f.accepted_drafts), "census unexpected acceptance operands");
        CUDA_CHECK(census::before_accept(host.device, logits.data,
            static_cast<const std::int32_t*>(target.data), scope.operands, stream));
        real_accept(target, logits, drafts, extents, lengths, anchors, licensed, counts,
                    accepted, domain, sampling, workspace, stream);
        CUDA_CHECK(census::after_accept(host.device, scope.operands, stream));
        ++scope.accepted_calls;
        return;
    }
#endif
    real_accept(target, logits, drafts, extents, lengths, anchors, licensed, counts,
                accepted, domain, sampling, workspace, stream);
}

extern "C" void wrap_install(qd::ProgramImpl* self, qd::SequenceState& sequence,
    qd::RequestControl& request, const ninfer::ops::SamplingConfig& config) {
#if NINFER_LAB_HEAD_INPUT_CENSUS
    const bool supported = request_supported(*self, sequence, request, config);
    if (supported) { host.ensure(self->device); }
    if (host.device != nullptr) {
        require(host.owner == &self->device && !host.pending, "census request reset before egress");
        host.emit("request_boundary_partial");
    }
#endif
    real_install(self, sequence, request, config);
#if NINFER_LAB_HEAD_INPUT_CENSUS
    if (host.device != nullptr) {
        CUDA_CHECK(census::reset(host.device, supported, request.sampling_host.token_counts,
                                 self->device.stream));
        host.bind(*self, sequence, supported);
        host.ready = false;
    }
    if (!supported) {
        std::fputs("{\"kind\":\"ninfer_lab_head_input_census_declined\",\"reason\":\"unsupported_request\"}\n",
                   stderr);
    }
#endif
}

extern "C" void wrap_vector(gguf::GgmlType type, const void* weight, std::int64_t row_bytes,
    int rows, int k, const void* activation, int columns, const gguf::VectorOutput& output,
    cudaStream_t stream) {
#if NINFER_LAB_HEAD_INPUT_CENSUS
    if (current != nullptr) {
        Scope& scope = *current;
        const auto& frame = scope.state.frame;
        const bool target = output.bf16 == frame.target_logits.data;
        const bool proposal = output.bf16 == frame.proposal_logits.data;
        if (target || proposal) {
            const int role = target ? 3 : scope.proposal_role;
            require(type == gguf::GgmlType::Q6_K && row_bytes == 4200 &&
                    rows == census::kPhysical && k == head::kK &&
                    columns == (target ? 4 : 1) && activation != nullptr &&
                    (reinterpret_cast<std::uintptr_t>(activation) & 15U) == 0 &&
                    weight != nullptr && output.f32 == nullptr &&
                    output.column_stride == census::kPhysical &&
                    output.epilogue == gguf::Epilogue::Store && output.gate == nullptr &&
                    stream == host.stream &&
                    (target ? scope.accepted_calls == 0 && scope.head_calls == 0 :
                              scope.accepted_calls == 1 && role >= 0 && role < 3 &&
                              scope.head_calls == role + 1), "head census vector contract");
            if (host.weight == nullptr) { host.weight = weight; }
            require(host.weight == weight, "head census multiple weight planes");
            real_vector(type, weight, row_bytes, rows, k, activation, columns, output, stream);
            CUDA_CHECK(head::snapshot(host.device, host.payload_device, role, activation,
                                       output.bf16, weight, scope.operands, stream));
            ++scope.head_calls;
            return;
        }
        require(!(type == gguf::GgmlType::Q6_K && rows == census::kPhysical && k == head::kK),
                "head census unrecognized full-vocabulary output");
    }
#endif
    real_vector(type, weight, row_bytes, rows, k, activation, columns, output, stream);
}

extern "C" void wrap_sync(const ninfer::DeviceContext* self) {
    real_sync(self);
#if NINFER_LAB_HEAD_INPUT_CENSUS
    if (host.owner == self && host.pending) {
        host.pending = false;
        host.ready = true;
        if (host.mirror->error != census::None || host.mirror->completed == census::kRecords) {
            host.emit(host.mirror->error == census::None ? "complete" : "device_contract_failure");
        }
    }
#endif
}
