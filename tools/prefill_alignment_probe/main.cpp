#include "core/device.h"
#include "models/load_options.h"
#include "models/qwen3_5/execution/linear.h"
#include "models/qwen3_5/frontend/digest.h"
#include "models/qwen3_5/frontend/frontend.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"
#include "models/qwen3_5/load.h"
#include "models/qwen3_5/program/planning/startup.h"
#include "models/qwen3_5/program/program_impl.h"
#include "ops/common/device_route.h"
#include "runtime/engine/device_profile.h"
#include "runtime/engine/model_instance.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

using Json = nlohmann::json;
namespace qwen = ninfer::models::qwen3_5;
namespace rt = ninfer::runtime;
namespace detail = ninfer::models::qwen3_5::detail;
using ninfer::DType;
using ninfer::Tensor;
using ninfer::TokenId;

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

std::string digest(std::span<const std::uint8_t> bytes) {
    return qwen::frontend::sha256_hex(qwen::frontend::sha256(bytes));
}

template <class T>
std::string digest_values(std::span<const T> values) {
    return digest({reinterpret_cast<const std::uint8_t*>(values.data()), values.size_bytes()});
}

std::string dtype_name(DType dtype) {
    switch (dtype) {
    case DType::BF16: return "bf16";
    case DType::FP32: return "fp32";
    case DType::I32: return "i32";
    case DType::U8: return "u8";
    case DType::I64: return "i64";
    case DType::I8: return "i8";
    case DType::FP16: return "fp16";
    case DType::FP8_E4M3FN: return "fp8_e4m3fn_codes";
    }
    throw std::runtime_error("unknown tensor dtype");
}

float floating_value(std::span<const std::uint8_t> bytes, DType dtype, std::size_t index) {
    if (dtype == DType::FP32) {
        float value;
        std::memcpy(&value, bytes.data() + index * sizeof(value), sizeof(value));
        return value;
    }
    std::uint16_t bits;
    std::memcpy(&bits, bytes.data() + index * sizeof(bits), sizeof(bits));
    if (dtype == DType::BF16) {
        return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16U);
    }
    const int exponent = (bits >> 10U) & 31U;
    const int fraction = bits & 1023U;
    const float magnitude = exponent == 31
                                ? (fraction == 0 ? std::numeric_limits<float>::infinity()
                                                 : std::numeric_limits<float>::quiet_NaN())
                            : exponent == 0
                                ? std::ldexp(static_cast<float>(fraction), -24)
                                : std::ldexp(static_cast<float>(fraction + 1024), exponent - 25);
    return (bits & 0x8000U) != 0 ? -magnitude : magnitude;
}

Json summarize(std::span<const std::uint8_t> bytes, DType dtype, Json shape) {
    Json result{{"sha256", digest(bytes)},
                {"bytes", bytes.size()},
                {"dtype", dtype_name(dtype)},
                {"shape", std::move(shape)}};
    std::uint64_t set_bits = 0;
    std::uint64_t zero_bytes = 0;
    for (const std::uint8_t value : bytes) {
        set_bits += std::popcount(value);
        zero_bytes += value == 0;
    }
    result["set_bits"] = set_bits;
    result["zero_bytes"] = zero_bytes;
    if (dtype != DType::FP32 && dtype != DType::BF16 && dtype != DType::FP16) {
        return result;
    }
    const std::size_t count = bytes.size() / ninfer::dtype_size(dtype);
    std::size_t nonfinite = 0;
    double sum = 0;
    double sum_squared = 0;
    double max_abs = 0;
    float minimum = std::numeric_limits<float>::infinity();
    float maximum = -std::numeric_limits<float>::infinity();
    for (std::size_t index = 0; index < count; ++index) {
        const float value = floating_value(bytes, dtype, index);
        if (!std::isfinite(value)) {
            ++nonfinite;
            continue;
        }
        sum += value;
        sum_squared += static_cast<double>(value) * value;
        max_abs = std::max(max_abs, std::abs(static_cast<double>(value)));
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
    }
    result["numeric"] = {
        {"elements", count}, {"nonfinite", nonfinite}, {"sum", sum},
        {"sum_squared", sum_squared}, {"max_abs", max_abs},
        {"minimum", std::isfinite(minimum) ? Json(minimum) : Json(nullptr)},
        {"maximum", std::isfinite(maximum) ? Json(maximum) : Json(nullptr)}};
    Json samples = Json::array();
    for (std::size_t sample = 0; sample < std::min<std::size_t>(count, 8); ++sample) {
        const std::size_t index = count <= 8 ? sample : sample * (count - 1) / 7;
        const float value = floating_value(bytes, dtype, index);
        samples.push_back({{"index", index},
                           {"value", std::isfinite(value) ? Json(value) : Json(nullptr)},
                           {"fp32_bits", std::bit_cast<std::uint32_t>(value)}});
    }
    result["samples"] = std::move(samples);
    return result;
}

Json host_tensor(const void* data, std::size_t bytes, DType dtype, Json shape) {
    require(data != nullptr || bytes == 0, "null host tensor");
    return summarize({static_cast<const std::uint8_t*>(data), bytes}, dtype, std::move(shape));
}

Json device_tensor(const Tensor& tensor, ninfer::DeviceContext& device) {
    require(tensor.data != nullptr && tensor.is_contiguous(), "expected a contiguous live tensor");
    std::vector<std::uint8_t> bytes(tensor.bytes());
    CUDA_CHECK(cudaMemcpyAsync(bytes.data(), tensor.data, bytes.size(), cudaMemcpyDeviceToHost,
                               device.stream));
    device.synchronize();
    return summarize(bytes, tensor.dtype,
                     Json::array({tensor.ne[0], tensor.ne[1], tensor.ne[2], tensor.ne[3]}));
}

ninfer::KvCacheStorage parse_kv(const std::string& value) {
    if (value == "bf16") { return ninfer::KvCacheStorage::BFloat16; }
    if (value == "int8") { return ninfer::KvCacheStorage::Int8Group64; }
    if (value == "fp8") { return ninfer::KvCacheStorage::Fp8E4M3Row256; }
    if (value == "rk8v4") { return ninfer::KvCacheStorage::RotatedInt8KeyInt4ValueGroup64; }
    throw std::invalid_argument("kv_dtype must be bf16, int8, fp8 or rk8v4");
}

ninfer::EngineOptions engine_options(const Json& input) {
    ninfer::EngineOptions options;
    options.artifact_path = input.at("artifact").get<std::string>();
    options.max_context = input.at("max_context").get<std::uint32_t>();
    options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(options.max_context);
    options.prefill_chunk = input.at("prefill_chunk").get<std::uint32_t>();
    options.kv_cache = parse_kv(input.at("kv_dtype").get<std::string>());
    const std::string backend = input.at("backend").get<std::string>();
    require(backend == "none" || backend == "mtp", "only none and mtp are covered by this probe");
    options.speculative.backend = backend == "mtp" ? ninfer::SpeculativeBackend::Mtp
                                                   : ninfer::SpeculativeBackend::None;
    options.speculative.draft_tokens = input.at("draft_tokens").get<std::uint32_t>();
    const std::string proposal = input.at("proposal_head").get<std::string>();
    require(proposal == "full" || proposal == "optimized", "invalid proposal_head");
    options.speculative.proposal_head = proposal == "full" ? ninfer::ProposalHead::Full
                                                           : ninfer::ProposalHead::Optimized;
    options.speculative.lookup_ngram = input.value("lookup_ngram", std::uint32_t{0});
    options.speculative.mtp_policy = ninfer::MtpDraftPolicy::Fixed;
    options.max_concurrency = 1;
    options.max_pending_requests = 1;
    require(!input.at("context_cache").get<bool>(), "this gate requires root-only context_cache=false");
    require(input.at("host_kv_mib").get<std::uint64_t>() == 0 &&
                input.at("host_state_slots").get<std::uint32_t>() == 0,
            "this gate requires host_kv_mib=0 and host_state_slots=0");
    options.context_cache = ninfer::ContextCacheOptions{
        .enabled = false, .host_state_slots = 0, .host_kv_capacity_bytes = 0};
    options.enable_vision = input.at("enable_vision").get<bool>();
    const std::string residency = input.at("vision_residency").get<std::string>();
    require(residency == "resident" || residency == "overlay", "invalid vision_residency");
    require(options.enable_vision || residency == "resident",
            "vision_residency=overlay requires enable_vision=true");
    options.vision_residency = residency == "overlay" ? ninfer::VisionResidency::Overlay
                                                     : ninfer::VisionResidency::Resident;
    options.use_cuda_graph = input.at("use_cuda_graph").get<bool>();
    options.fast_prefill_kernel = input.at("fast_prefill_kernel").get<bool>();
    options.device_profile = input.at("device_profile").get<std::string>();
    options.device_profile_path = input.value("device_profile_path", std::string{});
    options.rope_yarn = input.value("rope_yarn", false);
    options.lm_head_q4 = input.value("lm_head_q4", false);
    options.lm_head_q6 = input.value("lm_head_q6", false);
    options.embedding_q4 = input.value("embedding_q4", false);
    options.embedding_q6 = input.value("embedding_q6", false);
    options.mtp_experts_q4 = input.value("mtp_experts_q4", false);
    options.mlp_a8_decode = input.value("mlp_a8_decode", false);
    options.prefill_a8 = input.value("prefill_a8", true);
    options.prefill_cublas = input.value("prefill_cublas", false);
    options.prefill_cublas_projections = input.value("prefill_cublas_projections", true);
    options.gdn_state_fp16 = input.value("gdn_state_fp16", false);
    require(options.prefill_chunk == 1024, "this experiment requests prefill_chunk=1024");
    require(options.speculative.lookup_ngram == 0, "lookup drafting is outside this state gate");
    return rt::normalize_engine_options(std::move(options));
}

Json resolved_profile(const ninfer::EngineOptions& options) {
    return {
        {"max_context", options.max_context}, {"requested_prefill_chunk", options.prefill_chunk},
        {"kv_storage", static_cast<int>(options.kv_cache)},
        {"backend", static_cast<int>(options.speculative.backend)},
        {"draft_tokens", options.speculative.draft_tokens},
        {"proposal_head", static_cast<int>(options.speculative.proposal_head)},
        {"mtp_policy", static_cast<int>(options.speculative.mtp_policy)},
        {"max_concurrency", options.max_concurrency}, {"context_cache", options.context_cache.enabled},
        {"host_kv_mib", options.context_cache.host_kv_capacity_bytes / (1024ULL * 1024ULL)},
        {"host_state_slots", options.context_cache.host_state_slots},
        {"device_state_slots", options.context_cache.device_state_slots.value()},
        {"max_private_continuations", options.context_cache.max_private_continuations.value()},
        {"max_shared_prefixes", options.context_cache.max_shared_prefixes.value()},
        {"max_long_anchors_per_continuation",
         options.context_cache.max_long_anchors_per_continuation.value()},
        {"enable_vision", options.enable_vision},
        {"vision_residency", options.vision_residency == ninfer::VisionResidency::Overlay
                                 ? "overlay" : "resident"},
        {"use_cuda_graph", options.use_cuda_graph},
        {"fast_prefill_kernel", options.fast_prefill_kernel}, {"rope_yarn", options.rope_yarn},
        {"lm_head_q4", options.lm_head_q4}, {"lm_head_q6", options.lm_head_q6},
        {"embedding_q4", options.embedding_q4}, {"embedding_q6", options.embedding_q6},
        {"mtp_experts_q4", options.mtp_experts_q4}, {"mlp_a8_decode", options.mlp_a8_decode},
        {"prefill_a8", options.prefill_a8}, {"prefill_cublas", options.prefill_cublas},
        {"prefill_cublas_projections", options.prefill_cublas_projections},
        {"gdn_state_fp16", options.gdn_state_fp16}, {"device_profile", options.device_profile}};
}

Json install_routes(const ninfer::EngineOptions& options, ninfer::DeviceContext& device) {
    if (options.device_profile == "off") {
        ninfer::ops::install_device_route_profile(device.device, nullptr);
        return {{"mode", "off"}};
    }
    require(options.device_profile == "auto", "calibration is not permitted by this probe");
    const auto path = options.device_profile_path.empty() ? rt::default_device_profile_path()
                                                          : options.device_profile_path;
    const auto hardware = rt::context_cost_hardware_class(device.props.name, device.props.major,
                                                          device.props.minor);
    auto profile = rt::find_device_route_profile(hardware, device.props.multiProcessorCount, path);
    require(profile.has_value(), "no existing device route profile; refusing calibration");
    const std::string serialized = rt::serialize_device_route_profiles({*profile});
    const Json result{{"mode", "auto"}, {"hardware_class", hardware},
                      {"multiprocessors", device.props.multiProcessorCount},
                      {"origin", profile->origin},
                      {"sha256", qwen::frontend::sha256_hex(qwen::frontend::sha256(serialized))}};
    ninfer::ops::install_device_route_profile(
        device.device, std::make_shared<const ninfer::ops::DeviceRouteProfile>(std::move(*profile)));
    return result;
}

rt::ResolvedExecutionOptions request_options(const Json& input) {
    const Json& sampling = input.at("sampling");
    rt::ResolvedExecutionOptions result;
    result.sampling = {
        .temperature = sampling.at("temperature").get<float>(),
        .top_k = sampling.at("top_k").get<std::int32_t>(),
        .top_p = sampling.at("top_p").get<float>(),
        .min_p = sampling.at("min_p").get<float>(),
        .presence_penalty = sampling.at("presence_penalty").get<float>(),
        .frequency_penalty = sampling.at("frequency_penalty").get<float>(),
        .seed = sampling.at("seed").get<std::uint64_t>()};
    const std::size_t forced = input.at("forced_tokens").size();
    require(forced < std::numeric_limits<std::uint32_t>::max() - 2U, "too many forced tokens");
    result.requested_output_tokens = static_cast<std::uint32_t>(forced + 2U);
    result.allow_prefix_reuse = false;
    result.first_token_top_logprobs = 1;
    return result;
}

qwen::SequenceHandle start(detail::ProgramImpl& program, qwen::PreparedPromptData&& prompt,
                           const rt::ResolvedExecutionOptions& request) {
    const auto base = program.plan_request(prompt, request);
    auto candidate = program.inspect_admission(prompt, base, rt::LaneId{0}, nullptr, nullptr,
                                               std::nullopt, false);
    require(candidate.has_value(), "root admission did not produce a candidate");
    auto sealed = program.seal_materialization(*candidate, prompt, {}, {}, {}, {}, {}, {});
    require(sealed.has_value(), "root admission could not be sealed");
    require(program.reserve_materialization(std::move(*sealed), std::move(prompt), {}) ==
                rt::ContextTransactionReserveStatus::Reserved,
            "root materialization was not reserved");
    for (;;) {
        auto progress = program.progress_context_transaction({});
        if (std::holds_alternative<rt::ContextTransactionInProgress>(progress)) {
            std::this_thread::yield();
            continue;
        }
        const auto* completed = std::get_if<qwen::MaterializationResult>(&progress);
        require(completed != nullptr && completed->status == rt::ContextTransactionStatus::Published &&
                    completed->published.has_value(),
                "root materialization did not publish");
        const auto sequence = completed->published->sequence;
        program.finalize_context_transaction();
        return sequence;
    }
}

Json canonical_kv(qwen::PagedKVCache& cache, detail::KVAddressSpaceStore& addresses,
                   detail::KVAddressSpaceHandle address, std::uint32_t frontier,
                   ninfer::DeviceContext& device) {
    require(addresses.active(address), "KV address is not active");
    require(addresses.committed_frontier(address) == frontier, "KV committed frontier differs");
    const auto& pages = cache.page_pool();
    const auto& geometry = pages.geometry();
    require(geometry.device_plane_order == ninfer::PagedKVPlaneOrder::PageMajor,
            "probe only supports the current page-major KV layout");
    require(pages.rank_count() == 1, "probe requires a single GPU");
    const auto view = cache.execution_view(addresses.execution_row(address));
    const Tensor table = view.layer_view(0).block_table;
    const std::size_t logical_pages = (frontier + geometry.page_tokens - 1) / geometry.page_tokens;
    require(logical_pages <= static_cast<std::size_t>(table.ne[0]), "KV table is too short");
    std::vector<std::int32_t> mapping(logical_pages);
    CUDA_CHECK(cudaMemcpyAsync(mapping.data(), table.data, mapping.size() * sizeof(mapping[0]),
                               cudaMemcpyDeviceToHost, device.stream));
    device.synchronize();
    const std::set<std::int32_t> distinct(mapping.begin(), mapping.end());
    require(distinct.size() == mapping.size(), "root KV mapping aliases live pages");
    for (const std::int32_t physical : mapping) {
        require(physical >= 0 && static_cast<std::uint32_t>(physical) < pages.capacity_pages(),
                "KV mapping is outside its physical pool");
    }
    Json result{{"frontier", frontier}, {"logical_pages", logical_pages},
                {"layers", cache.layers()}, {"page_tokens", geometry.page_tokens},
                {"planes", Json::array()}};
    for (std::size_t plane_index = 0; plane_index < pages.plane_count(); ++plane_index) {
        const Tensor& plane = pages.plane(plane_index);
        const auto& geometry_plane = geometry.planes.at(plane_index);
        require(plane.ne[0] == geometry_plane.leading_extent &&
                    plane.ne[1] == static_cast<std::int32_t>(geometry.page_tokens) &&
                    plane.ne[2] == geometry_plane.head_extent && plane.is_contiguous(),
                "unexpected KV plane geometry");
        const std::size_t token_bytes =
            static_cast<std::size_t>(plane.ne[0]) * ninfer::dtype_size(plane.dtype);
        std::vector<std::uint8_t> canonical(
            static_cast<std::size_t>(frontier) * plane.ne[2] * token_bytes);
        std::size_t destination = 0;
        for (std::size_t logical = 0; logical < logical_pages; ++logical) {
            const std::size_t live = std::min<std::size_t>(
                geometry.page_tokens, frontier - logical * geometry.page_tokens);
            for (int head = 0; head < plane.ne[2]; ++head) {
                const auto* source = static_cast<const std::uint8_t*>(plane.data) +
                                     static_cast<std::size_t>(mapping[logical]) * plane.nb[3] +
                                     static_cast<std::size_t>(head) * plane.nb[2];
                const std::size_t bytes = live * token_bytes;
                CUDA_CHECK(cudaMemcpyAsync(canonical.data() + destination, source, bytes,
                                           cudaMemcpyDeviceToHost, device.stream));
                destination += bytes;
            }
        }
        device.synchronize();
        require(destination == canonical.size(), "KV canonicalization is incomplete");
        Json component = summarize(canonical, plane.dtype,
                                   Json::array({frontier, plane.ne[2], plane.ne[0]}));
        component["plane_index"] = plane_index;
        component["order"] = "logical_page,head,live_token,stored_dimension";
        result["planes"].push_back(std::move(component));
    }
    return result;
}

Json snapshot(detail::ProgramImpl& program, const std::vector<TokenId>& expected_ledger) {
    program.device.synchronize();
    require(program.active_continuations[0] < program.continuation_states.size(),
            "no active sequence");
    const auto& sequence = program.continuation_states.at(program.active_continuations[0]);
    require(sequence.ledger == expected_ledger, "physical ledger differs from the forced input");
    require(sequence.ledger_frontier == sequence.execution_frontier + 1U &&
                sequence.ledger_frontier == sequence.ledger.size() &&
                sequence.text_kv_valid == sequence.execution_frontier && sequence.tail_hidden_valid,
            "sequence frontiers or continuation hidden are incomplete");
    require(!sequence.state.fork_pending && sequence.state.read == sequence.state.write,
            "state fork is not settled");
    require(program.state_store->role(sequence.state.write) == detail::StateImageRole::ActiveMutable,
            "snapshot state is not the active mutable image");
    require(sequence.kv.has_value(), "sequence has no KV bundle");
    require(program.state_images->shard_count() == 1 &&
                program.state_images->dflash_local() == nullptr,
            "snapshot would omit an unsupported state shard");
    const std::int32_t slot = program.state_store->physical_slot(sequence.state.write);
    auto& linear = program.state_images->linear();
    Json result{
        {"execution_frontier", sequence.execution_frontier},
        {"ledger_frontier", sequence.ledger_frontier},
        {"text_kv_valid", sequence.text_kv_valid}, {"mtp_kv_valid", sequence.mtp_kv_valid},
        {"rope_delta", sequence.rope_delta}, {"tail_hidden_valid", sequence.tail_hidden_valid},
        {"state_fork_pending", sequence.state.fork_pending},
        {"physical_prefix_sha256", digest_values<TokenId>(
             std::span<const TokenId>(sequence.ledger).first(sequence.execution_frontier))},
        {"ledger_sha256", digest_values<TokenId>(sequence.ledger)},
        {"unexecuted_anchor", sequence.ledger.back()},
        {"gdn_layers", linear.layer_count()}, {"components", Json::object()}};
    for (std::uint32_t layer = 0; layer < linear.layer_count(); ++layer) {
        result["components"]["conv." + std::to_string(layer)] =
            device_tensor(linear.conv_slot(layer, slot), program.device);
        result["components"]["gdn." + std::to_string(layer)] =
            device_tensor(linear.recurrent_slot(layer, slot), program.device);
    }
    require(linear.layer_count() == program.parameters.model.config().text.linear_attention_layers,
            "snapshot does not cover every GDN layer");
    result["components"]["continuation_hidden"] =
        device_tensor(program.state_images->continuation_hidden_slot(slot), program.device);
    result["components"]["tail_hidden"] = device_tensor(sequence.tail_hidden, program.device);
    require(result["components"]["continuation_hidden"]["sha256"] ==
                result["components"]["tail_hidden"]["sha256"],
            "tail view and state image disagree");
    result["main_kv"] = canonical_kv(program.decoder->text_kv, *program.text_kv_addresses,
                                     sequence.kv->text, sequence.text_kv_valid, program.device);
    if (program.speculative_backend == ninfer::SpeculativeBackend::Mtp) {
        require(sequence.mtp_kv_valid == sequence.execution_frontier &&
                    sequence.kv->backend.has_value() && program.decoder->mtp_cache() != nullptr,
                "MTP committed state is incomplete");
        result["mtp_kv"] = canonical_kv(*program.decoder->mtp_cache(),
                                        *program.backend_kv_addresses, *sequence.kv->backend,
                                        sequence.mtp_kv_valid, program.device);
        result["mtp_draft_count"] = sequence.mtp_draft_count;
        result["mtp_drafts"] = std::vector<TokenId>(
            sequence.mtp_drafts.begin(), sequence.mtp_drafts.begin() + sequence.mtp_draft_count);
    }
    const auto& sampling = program.requests[0].sampling_host;
    require(sampling.token_mask == nullptr, "snapshot would omit a structured sampler mask");
    result["sampler"] = {{"temperature", sampling.temperature}, {"top_k", sampling.top_k},
                          {"top_p", sampling.top_p}, {"min_p", sampling.min_p},
                          {"presence_penalty", sampling.presence_penalty},
                          {"frequency_penalty", sampling.frequency_penalty}, {"seed", sampling.seed},
                          {"token_counts_active", sampling.token_counts != nullptr}};
    if (sampling.token_counts != nullptr) {
        result["components"]["token_counts"] =
            device_tensor(program.token_counts.slice(1, 0, 1), program.device);
    }
    program.work.reset();
    Tensor logits = program.io.logits.slice(1, 0, 1);
    qwen::execution::project(sequence.tail_hidden, program.parameters.text.output_head,
                              logits, program.work, program.device.stream);
    program.device.synchronize();
    const auto public_tokens =
        static_cast<std::int32_t>(program.parameters.model.resources().public_token_count);
    result["components"]["target_logits"] =
        device_tensor(logits.slice(0, 0, public_tokens), program.device);
    program.work.reset();
    return result;
}

Json run(const Json& input, std::optional<TokenId> expected_anchor) {
    const char* align = std::getenv("NINFER_PREFILL_ALIGN");
    require(align != nullptr && (std::string_view(align) == "0" || std::string_view(align) == "1"),
            "set NINFER_PREFILL_ALIGN explicitly to 0 or 1 in a fresh process");
    auto options = engine_options(input);
    const auto request = request_options(input);
    const auto prompt_tokens = input.at("token_ids").get<std::vector<TokenId>>();
    const auto forced = input.at("forced_tokens").get<std::vector<TokenId>>();
    require(!prompt_tokens.empty() && !forced.empty(), "prompt and forced tokens must be nonempty");
    require(prompt_tokens.size() + forced.size() + 2 <= options.max_context,
            "context must contain prompt, anchor and forced continuation");
    ninfer::DeviceContext device(options.device);
    const Json routes = install_routes(options, device);
    auto model = qwen::load_model(options.artifact_path, ninfer::models::load_options(options), device);
    const qwen::execution::Parameters parameters(*model);
    auto frontend = qwen::make_frontend(
        model->resources(), {.architecture = model->config().text.architecture,
                              .vision_enabled = options.enable_vision,
                              .max_context = options.max_context});
    auto prepared = frontend.prepare_tokens(prompt_tokens, false);
    auto forced_validation = frontend.prepare_tokens(forced, false);
    require(static_cast<bool>(forced_validation), "forced-token validation failed");
    auto prompt = qwen::PreparedPromptAccess::take(std::move(prepared));
    const std::string positions_sha256 = digest_values<std::int32_t>(prompt.positions);
    const std::size_t overlay_window_bytes =
        qwen::prepare_vision_overlay(parameters, device, options);
    auto planner = detail::make_sequence_planner_impl(parameters, device, options);
    const auto page_groups = planner->curve.minimum_main_page_groups;
    auto plan = detail::finalize_sequence_plan_impl(std::move(planner), page_groups);
    const std::uint32_t expected_chunk = std::string_view(align) == "0" ? 1024U : 1280U;
    require(plan->prefill_chunk == expected_chunk,
            "actual planner chunk " + std::to_string(plan->prefill_chunk) +
                " differs from expected " + std::to_string(expected_chunk));
    require(plan->capacity == options.max_context &&
                plan->max_concurrency == options.max_concurrency &&
                plan->kv_storage == options.kv_cache &&
                plan->speculative_backend == options.speculative.backend &&
                plan->proposal_head == options.speculative.proposal_head &&
                plan->fast_prefill_kernel == options.fast_prefill_kernel &&
                plan->use_cuda_graph == options.use_cuda_graph,
            "sequence plan changed the bound execution options");
    detail::ProgramImpl program(parameters, *plan, device, {});
    Json report{{"schema", "ninfer.prefill-alignment-state.v1"},
                {"status", "running"}, {"align", align},
                {"input_contract", "raw_token_ids"},
                {"artifact", options.artifact_path.string()},
                {"artifact_id", model->info().artifact_id},
                {"profile", resolved_profile(options)}, {"routes", routes},
                {"overlay_window_bytes", overlay_window_bytes},
                {"sampling", input.at("sampling")},
                {"prompt_tokens", prompt_tokens.size()},
                {"token_ids_sha256", digest_values<TokenId>(prompt_tokens)},
                {"positions_sha256", positions_sha256},
                {"forced_tokens", forced},
                {"forced_tokens_sha256", digest_values<TokenId>(forced)},
                {"actual_prefill_chunk", program.prefill_chunk},
                {"actual_context_capacity", plan->capacity},
                {"actual_kv_capacity", plan->kv_capacity},
                {"actual_main_page_groups", plan->main_page_groups},
                {"actual_chunks", Json::array()},
                {"snapshots", Json::object()},
                {"state_dump_written", false},
                {"snapshot_scope", "committed_target_state_and_committed_mtp_kv"},
                {"excluded_storage", {"workspace", "arena_padding", "invalid_kv_page_tail",
                                       "speculative_kv_beyond_committed_frontier"}},
                {"numeric_diagnostics_are_pairwise_error", false}};
    const qwen::SequenceHandle sequence = start(program, std::move(prompt), request);
    std::optional<qwen::PendingBatch> pending;
    std::uint32_t processed = 0;
    while (!pending) {
        auto progress = program.advance_prefill(sequence, nullptr);
        require(!progress.capture.has_value(), "root probe unexpectedly requested a checkpoint split");
        require(progress.summary.reused_prompt_tokens == 0 &&
                    progress.summary.prefix_reuse_path == ninfer::PrefixReusePath::Root,
                "probe unexpectedly reused a prefix");
        require(progress.processed_prompt_tokens != 0, "prefill made no progress");
        report["actual_chunks"].push_back(progress.processed_prompt_tokens);
        processed += progress.processed_prompt_tokens;
        require(processed <= prompt_tokens.size(), "prefill exceeded its physical token prefix");
        if (progress.complete) {
            require(progress.pending.has_value(), "completed prefill has no pending token");
            pending.emplace(std::move(*progress.pending));
        } else {
            require(!progress.pending.has_value(), "incomplete prefill has a pending token");
        }
    }
    require(processed == prompt_tokens.size(), "prefill did not consume the complete input");
    require(program.first_token_logits_host.has_value(), "native first-token logits were not captured");
    const auto public_tokens = model->resources().public_token_count;
    report["native_first_token_logits"] =
        host_tensor(program.first_token_logits_host->data(),
                    static_cast<std::size_t>(public_tokens) * sizeof(std::uint16_t), DType::BF16,
                    Json::array({public_tokens, 1, 1, 1}));
    const TokenId anchor = program.host_tokens[0];
    report["prefill_anchor"] = anchor;
    const std::array<rt::CommitDecision, 1> decision{{{.accepted_tokens = 1}}};
    auto committed = program.commit(std::move(*pending), decision, rt::CommitObservation::AllRows,
                                      nullptr);
    require(committed.row_count == 1 &&
                committed.rows[0].disposition == rt::CommitDisposition::Active &&
                !committed.captures[0].has_value(),
            "prefill commit did not establish a clean active sequence");
    std::vector<TokenId> expected_ledger = prompt_tokens;
    expected_ledger.push_back(anchor);
    report["snapshots"]["prefill"] = snapshot(program, expected_ledger);
    report["native_logits_reprojection_equal"] =
        report["native_first_token_logits"]["sha256"] ==
        report["snapshots"]["prefill"]["components"]["target_logits"]["sha256"];
    if (!report["native_logits_reprojection_equal"].get<bool>()) {
        report["status"] = "native_logits_reprojection_mismatch";
        require(program.abort(sequence).status == rt::ConsumeStatus::Consumed, "abort failed");
        return report;
    }
    if (expected_anchor && anchor != *expected_anchor) {
        report["status"] = "anchor_mismatch";
        report["expected_anchor"] = *expected_anchor;
        require(program.abort(sequence).status == rt::ConsumeStatus::Consumed, "abort failed");
        return report;
    }
    const std::array<qwen::SequenceHandle, 1> members{sequence};
    const std::array<std::optional<std::uint32_t>, 1> no_splits{std::nullopt};
    for (const TokenId token : forced) {
        const std::array<TokenId, 1> row{token};
        (void)program.append_forced_tokens(members, row, 1, no_splits, nullptr);
        expected_ledger.push_back(token);
    }
    report["forced_execution_width"] = 1;
    report["snapshots"]["continuation"] = snapshot(program, expected_ledger);
    require(program.abort(sequence).status == rt::ConsumeStatus::Consumed, "cleanup failed");
    report["cleanup"] = "consumed";
    report["status"] = "complete";
    return report;
}

}

int main(int argc, char** argv) {
    try {
        require(argc == 2 || argc == 3, "usage: prefill_alignment_probe INPUT_JSON [EXPECTED_ANCHOR]");
        std::ifstream file(argv[1]);
        require(file.good(), "cannot open input JSON");
        Json input;
        file >> input;
        std::optional<TokenId> anchor;
        if (argc == 3) {
            const long long value = std::stoll(argv[2]);
            require(value >= 0 && value <= std::numeric_limits<TokenId>::max(),
                    "expected anchor is outside TokenId");
            anchor = static_cast<TokenId>(value);
        }
        const Json result = run(input, anchor);
        std::cout << result.dump() << '\n';
        return result.at("status") == "complete" ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "prefill_alignment_probe: " << error.what() << '\n';
        std::cout << Json({{"schema", "ninfer.prefill-alignment-state.v1"},
                           {"status", "error"}, {"error", error.what()}}).dump() << '\n';
        return 1;
    }
}
