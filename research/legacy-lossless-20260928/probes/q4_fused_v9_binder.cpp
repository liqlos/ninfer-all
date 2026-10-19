#include "artifact/binder.h"

#include "artifact/framing.h"
#include "artifact/transcode.h"
#include "artifact/q6_fields.h"
#include "artifact/q4_fields.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <string_view>

namespace ninfer::artifact {

float HostValues::scalar_f32() const {
    if (format != QType::FP32 || elements != 1 || data.size() != 4) {
        throw ArtifactError("auxiliary requires one represented FP32 value");
    }
    return std::bit_cast<float>(read_u32_le(data.data()));
}

std::vector<std::int32_t> HostValues::integers() const {
    if (format != QType::INT32 || data.size() != checked_mul(elements, 4, "integer values")) {
        throw ArtifactError("semantic table requires INT32 values");
    }
    std::vector<std::int32_t> out;
    out.reserve(static_cast<std::size_t>(elements));
    for (std::size_t i = 0; i < elements; ++i) {
        out.push_back(std::bit_cast<std::int32_t>(read_u32_le(data.data() + i * 4)));
    }
    return out;
}

Binder::Binder(const Reader& reader)
    : reader_(reader), demands_(reader.directory().objects.size()) {}

ParameterReference Binder::parameter(std::string_view name, Shape shape, Residency residency,
                                     std::optional<QType> exact_format) {
    const auto found = reader_.directory().bindings.find(name);
    if (found == reader_.directory().bindings.end()) {
        throw ArtifactError("missing logical parameter " + std::string(name));
    }
    return binding(std::string(name), found->second, std::move(shape), residency, exact_format);
}

ParameterReference Binder::binding(std::string name, const Binding& binding, Shape shape,
                                   Residency residency, std::optional<QType> exact_format) {
    if (weight_element_count(shape) != binding.elements ||
        (binding.whole_object &&
         reader_.directory().tensor(binding.parts.at(0).object).shape != shape)) {
        throw ArtifactError(name + ": logical shape differs from Binding coverage");
    }
    for (const auto& part : binding.parts) {
        const auto& geometry = reader_.geometry(part.object);
        if (exact_format && geometry.format != *exact_format) {
            throw ArtifactError(name +
                                ": representation does not match its mathematical value type");
        }
        if (residency == Residency::Device) {
            require_device(part.object);
        } else if (residency == Residency::Pinned) {
            require_pinned(part.object);
        } else if (residency == Residency::Host) {
            (void)host_object(part.object);
        }
    }
    return {std::move(name), std::move(shape), binding, residency};
}

const Use& Binder::use(std::string_view parameter, std::string_view input) const {
    const auto found = reader_.directory().uses.find({std::string(parameter), std::string(input)});
    if (found == reader_.directory().uses.end()) {
        throw ArtifactError("missing Use " + std::string(parameter) + "@" + std::string(input));
    }
    return found->second;
}

bool Binder::contains(std::string_view parameter) const {
    return reader_.directory().bindings.contains(parameter);
}

void Binder::require_device(ObjectHandle object, std::uint64_t alignment) {
    const auto& geometry = reader_.geometry(object);
    if (!alignment || (alignment & (alignment - 1))) {
        throw ArtifactError("device alignment must be a power of two");
    }
    auto& demand     = demands_.at(object.index);
    demand.device    = true;
    demand.alignment = std::max({demand.alignment, alignment, geometry.alignment});
}

void Binder::transcode_device(ObjectHandle object, QType target) {
    const auto& geometry = reader_.geometry(object);
    const auto& tensor   = reader_.directory().tensor(object);
    auto& demand         = demands_.at(object.index);
    if (!demand.device) {
        throw ArtifactError(tensor.id + ": transcoding requires a device placement");
    }
    if (demand.host) {
        throw ArtifactError(tensor.id + ": a transcoded object cannot also keep Host bytes");
    }
    if (geometry.layout != QuantLayout::RowSplit ||
        !row_split_transcode_supported(geometry.format, target)) {
        throw ArtifactError(tensor.id + ": object format cannot be transcoded as requested");
    }
    if (demand.transcode && *demand.transcode != target) {
        throw ArtifactError(tensor.id + ": conflicting transcode targets");
    }
    demand.transcode = target;
}

void Binder::evict_device(ObjectHandle object, std::uint32_t rank) {
    const auto& tensor = reader_.directory().tensor(object);
    auto& demand       = demands_.at(object.index);
    if (!demand.device) {
        throw ArtifactError(tensor.id + ": an evictable placement requires a device placement");
    }
    if (rank == 0) { throw ArtifactError(tensor.id + ": evictable rank must be nonzero"); }
    demand.evict_rank = std::max(demand.evict_rank, rank);
}

void Binder::device_rank(ObjectHandle object, std::size_t rank) {
    const auto& tensor = reader_.directory().tensor(object);
    auto& demand       = demands_.at(object.index);
    if (!demand.device) {
        throw ArtifactError(tensor.id + ": a pipeline rank requires a device placement");
    }
    // Two callers asking for different devices means the object is shared across the split, which
    // no single placement can satisfy -- exactly the sort of binding mistake worth failing on.
    if (demand.device_rank && *demand.device_rank != rank) {
        throw ArtifactError(tensor.id + ": conflicting pipeline ranks");
    }
    demand.device_rank = rank;
}

void Binder::require_pinned(ObjectHandle object) {
    const auto& geometry = reader_.geometry(object);
    auto& demand         = demands_.at(object.index);
    demand.alignment     = std::max(demand.alignment, geometry.alignment);
    if (!demand.pinned_order) { demand.pinned_order = next_pinned_order_++; }
}

std::span<const std::byte> Binder::host_object(ObjectHandle object) {
    reader_.validate_object(object);
    auto& demand = demands_.at(object.index);
    if (!demand.host) {
        demand.host_data = reader_.read_object(object);
        read_bytes_      = checked_add(read_bytes_, demand.host_data.size(), "Host read bytes");
        demand.host      = true;
    }
    return demand.host_data;
}

ObjectHandle Binder::resource(std::string_view component, std::string_view role) {
    const auto& resources = reader_.directory().component(component).resources;
    const auto found      = resources.find(role);
    if (found == resources.end()) {
        throw ArtifactError(std::string(component) + ": missing resource " + std::string(role));
    }
    (void)host_object(found->second);
    return found->second;
}

HostValues Binder::values(const Binding& binding, std::optional<QType> format) {
    HostValues out;
    out.elements = binding.elements;
    if (binding.parts.empty()) { throw ArtifactError("value Binding is empty"); }
    out.format = format.value_or(reader_.geometry(binding.parts.front().object).format);
    std::uint64_t word_bytes = 0;
    if (out.format == QType::BF16) {
        word_bytes = 2;
    } else if (out.format == QType::FP32 || out.format == QType::INT32) {
        word_bytes = 4;
    } else {
        throw ArtifactError("owning Host values require a direct numeric format");
    }
    const auto total_bytes = checked_mul(out.elements, word_bytes, "Host value bytes");
    if (total_bytes > std::numeric_limits<std::size_t>::max()) {
        throw ArtifactError("Host values exceed size_t");
    }
    out.data.resize(static_cast<std::size_t>(total_bytes));
    std::size_t destination = 0;
    for (const auto& part : binding.parts) {
        const auto& geometry = reader_.geometry(part.object);
        if (geometry.layout != QuantLayout::Contiguous || geometry.format != out.format ||
            part.begin >= part.end || part.end > geometry.elements) {
            throw ArtifactError("Host value Binding has an incompatible representation or range");
        }
        const auto bytes = checked_mul(part.end - part.begin, word_bytes, "value range");
        if (bytes > out.data.size() - destination) {
            throw ArtifactError("Host value coverage exceeds declared size");
        }
        const auto source = checked_mul(part.begin, word_bytes, "value offset");
        auto target = std::span(out.data).subspan(destination, static_cast<std::size_t>(bytes));
        const auto& cached = demands_[part.object.index];
        if (cached.host) {
            std::memcpy(target.data(), cached.host_data.data() + source, target.size());
        } else {
            const auto& object = reader_.directory().tensor(part.object);
            reader_.read_into(checked_add(object.offset, source, "value file offset"), target);
            read_bytes_ = checked_add(read_bytes_, bytes, "Host read bytes");
        }
        destination += target.size();
    }
    if (destination != out.data.size()) {
        throw ArtifactError("Host value coverage is incomplete");
    }
    owned_value_bytes_ = checked_add(owned_value_bytes_, out.data.size(), "owning value bytes");
    return out;
}

MaterializationPlan Binder::finish(std::uint64_t evictable_alignment) && {
    MaterializationPlan plan;
    plan.source            = &reader_;
    plan.object_count      = demands_.size();
    plan.prior_read_bytes  = read_bytes_;
    plan.owned_value_bytes = owned_value_bytes_;
    std::vector<std::optional<WeightGeometry>> q6_geometries(demands_.size());
    std::vector<std::optional<WeightGeometry>> q4_geometries(demands_.size());
    if (q6_fields_mode_enabled()) {
        for (std::size_t index = 0; index < demands_.size(); ++index) {
            const auto& demand = demands_[index];
            if (!demand.device || demand.transcode || demand.host) { continue; }
            const ObjectHandle handle{index};
            const auto& stored = reader_.geometry(handle);
            if (stored.format != QType::GGUF_Q6_K) { continue; }
            const auto raw = reader_.read_object(handle);
            plan.prior_read_bytes = checked_add(plan.prior_read_bytes, raw.size(), "Q6 field plan reads");
            q6_geometries[index] = describe_q6_fields(stored, raw);
        }
    }
    if (q4_fields_mode_enabled()) {
        const auto has_text_projection_binding = [&](std::size_t index, std::string_view suffix) {
            for (const auto& [name, binding] : reader_.directory().bindings) {
                const std::string_view parameter{name};
                if (!parameter.starts_with("text/layers/") || !parameter.ends_with(suffix) ||
                    binding.parts.size() != 1) { continue; }
                const auto& part = binding.parts.front();
                if (part.object.index == index && part.begin == 0 &&
                    part.end == reader_.geometry(part.object).elements) { return true; }
            }
            return false;
        };
        const auto has_text_fused_mlp_binding = [&](std::size_t index, std::uint64_t elements) {
            constexpr std::string_view prefix = "text/layers/";
            if (elements == 0 || elements % 2 != 0) { return false; }
            const std::uint64_t half = elements / 2;
            for (const auto& [name, binding] : reader_.directory().bindings) {
                const std::string_view parameter{name};
                if (!parameter.starts_with(prefix)) { continue; }
                const auto layer_end = parameter.find('/', prefix.size());
                if (layer_end == std::string_view::npos || layer_end == prefix.size() ||
                    parameter.substr(layer_end) != "/mlp/gate" || binding.parts.size() != 1) {
                    continue;
                }
                const auto& gate = binding.parts.front();
                if (gate.object.index != index || gate.begin != 0 || gate.end != half) { continue; }
                const auto up_name = std::string(parameter.substr(0, layer_end)) + "/mlp/up";
                const auto up_found = reader_.directory().bindings.find(up_name);
                if (up_found == reader_.directory().bindings.end() || up_found->second.parts.size() != 1) {
                    continue;
                }
                const auto& up = up_found->second.parts.front();
                if (up.object.index == index && up.begin == half && up.end == elements) { return true; }
            }
            return false;
        };
        for (std::size_t index = 0; index < demands_.size(); ++index) {
            const auto& demand = demands_[index];
            if (!demand.device || demand.transcode || demand.host) { continue; }
            const ObjectHandle handle{index};
            const auto& stored = reader_.geometry(handle);
            if (stored.format != QType::GGUF_Q4_K || stored.shape.size() != 2) { continue; }
            const bool gdn = stored.shape[0] == 16384 && stored.shape[1] == 5120;
            const bool down = stored.shape[0] == 5120 && stored.shape[1] == 17408 &&
                              has_text_projection_binding(index, "/mlp/down");
            const bool fused_mlp = stored.shape[0] == 34816 && stored.shape[1] == 5120 &&
                                   has_text_fused_mlp_binding(index, stored.elements);
            if (!gdn && !down && !fused_mlp) { continue; }
            const auto raw = reader_.read_object(handle);
            plan.prior_read_bytes = checked_add(plan.prior_read_bytes, raw.size(), "Q4 field plan reads");
            q4_geometries[index] = describe_q4_fields(stored, raw);
        }
    }
    const auto device_bytes = [&](std::size_t index) {
        const ObjectHandle handle{index};
        const auto& demand = demands_[index];
        if (q6_geometries[index]) { return q6_geometries[index]->bytes; }
        if (q4_geometries[index]) { return q4_geometries[index]->bytes; }
        return demand.transcode ? weight_geometry(*demand.transcode, QuantLayout::RowSplit,
                                                  reader_.directory().tensor(handle).shape)
                                      .bytes
                                : reader_.geometry(handle).bytes;
    };
    std::size_t rank_count = 1;
    for (const auto& demand : demands_) {
        if (demand.device && demand.device_rank) {
            rank_count = std::max(rank_count, *demand.device_rank + 1);
        }
    }
    plan.device_capacity_by_rank.assign(rank_count, 0);
    const auto place_device = [&](std::size_t index) {
        const auto& demand  = demands_[index];
        const auto rank     = demand.device_rank.value_or(0);
        const auto bytes    = device_bytes(index);
        auto& capacity      = plan.device_capacity_by_rank[rank];
        const auto offset   = align_up(capacity, demand.alignment, "device offset");
        plan.device_objects.push_back(
            {ObjectHandle{index}, offset, bytes, demand.alignment, demand.transcode, rank,
             q6_geometries[index], q4_geometries[index]});
        capacity = checked_add(offset, bytes, "device capacity");
    };
    std::vector<std::size_t> evictable;
    std::vector<std::size_t> pinned;
    // Placements are emitted rank by rank so each rank's offsets stay ascending, which is what
    // materialization validates. Rank 0 first keeps a single-device plan byte-for-byte what it was.
    std::vector<std::vector<std::size_t>> resident(rank_count);
    for (std::size_t i = 0; i < demands_.size(); ++i) {
        const auto& demand = demands_[i];
        if (demand.pinned_order && (demand.device || demand.host)) {
            throw ArtifactError(reader_.directory().tensor(ObjectHandle{i}).id +
                                ": a pinned object cannot also have device or Host placement");
        }
        if (demand.pinned_order) { pinned.push_back(i); }
        if (!demand.device) { continue; }
        if (demand.evict_rank != 0) {
            // The evictable tail exists so a Vision encode window can borrow weight memory on the
            // card that runs Vision, which is the primary device. An offloaded rank holds expert
            // blocks and nothing else, so there is nothing there for a window to borrow.
            if (demand.device_rank.value_or(0) != 0) {
                throw ArtifactError(reader_.directory().tensor(ObjectHandle{i}).id +
                                    ": an evictable placement must stay on the primary device");
            }
            evictable.push_back(i);
        } else {
            resident[demand.device_rank.value_or(0)].push_back(i);
        }
    }
    for (const std::size_t index : resident[0]) { place_device(index); }
    if (!evictable.empty()) {
        std::stable_sort(evictable.begin(), evictable.end(), [&](std::size_t a, std::size_t b) {
            return demands_[a].evict_rank < demands_[b].evict_rank;
        });
        plan.evictable_tail_offset = align_up(plan.device_capacity_by_rank[0], evictable_alignment,
                                              "evictable tail offset");
        plan.device_capacity_by_rank[0] = plan.evictable_tail_offset;
        for (const std::size_t index : evictable) { place_device(index); }
        plan.evictable_tail_bytes = plan.device_capacity_by_rank[0] - plan.evictable_tail_offset;
    }
    for (std::size_t rank = 1; rank < rank_count; ++rank) {
        for (const std::size_t index : resident[rank]) { place_device(index); }
    }
    std::sort(pinned.begin(), pinned.end(), [&](std::size_t a, std::size_t b) {
        return *demands_[a].pinned_order < *demands_[b].pinned_order;
    });
    for (const std::size_t index : pinned) {
        const ObjectHandle handle{index};
        const auto& demand = demands_[index];
        const auto bytes   = reader_.geometry(handle).bytes;
        const auto offset =
            align_up(plan.pinned_capacity_bytes, demand.alignment, "pinned Host offset");
        plan.pinned_objects.push_back({handle, offset, bytes, demand.alignment});
        plan.pinned_capacity_bytes = checked_add(offset, bytes, "pinned Host capacity");
    }
    for (std::size_t i = 0; i < demands_.size(); ++i) {
        if (demands_[i].host) {
            plan.host_objects.push_back({ObjectHandle{i}, std::move(demands_[i].host_data)});
        }
    }
    return plan;
}

} // namespace ninfer::artifact
