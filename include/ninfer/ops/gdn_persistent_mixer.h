#pragma once

#include "core/arena.h"
#include "core/gdn_replay_records.h"
#include "core/linear_attention_state.h"
#include "core/weight.h"
#include "core/tensor.h"
#include "ninfer/ops/weight_input.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

enum class GdnPersistentStateAction : std::uint8_t { Record = 1 };

struct GdnPersistentState {
    GdnPersistentStateAction action = GdnPersistentStateAction::Record;
    Tensor conv_states;
    Tensor recurrent_states;
    Tensor source_slots;
    Tensor destination_slots;
    Tensor valid_columns;
    GdnReplayRecordLayer records;
};

[[nodiscard]] std::size_t gdn_persistent_mixer_workspace_bytes(std::int32_t columns);

[[nodiscard]] bool gdn_persistent_mixer(
    const Tensor& hidden, const Tensor& g, const Tensor& beta,
    const GgufProjectionWeights& projection, const Tensor& convolution,
    const Tensor& norm_weight, const Weight& output_projection, float rms_norm_eps,
    GdnPersistentState& state, Tensor& residual, WorkspaceArena& workspace,
    cudaStream_t stream);

}
