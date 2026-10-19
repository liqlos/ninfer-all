#pragma once

#include "core/weight_view.h"

#include <cstddef>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::artifact {

bool q6_fields_mode_enabled();
std::optional<WeightGeometry> describe_q6_fields(const WeightGeometry& stored,
                                               std::span<const std::byte> source);
std::vector<std::byte> encode_q6_fields(const WeightGeometry& geometry,
                                      std::span<const std::byte> source);

}
