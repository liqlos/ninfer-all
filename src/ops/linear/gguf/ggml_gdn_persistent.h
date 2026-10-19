#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::gguf::detail {

struct GdnPersistentProjectionPart {
    const std::uint8_t* weight;
    std::int64_t row_bytes;
    int type;
    int rows;
    int row;
    int output;
};

struct GdnPersistentLaunchArgs {
    GdnPersistentProjectionPart parts[8];
    int part_count;
    int columns;
    int action;
    bool state_fp16;
    float eps;
    const void* hidden;
    const float* g;
    const float* beta;
    const void* convolution;
    const void* norm_weight;
    const std::int32_t* output_columns;
    const std::uint8_t* output_weight;
    std::int64_t output_row_bytes;
    const void* residual;
    void* residual_out;
    const void* conv_states;
    const void* recurrent_states;
    const std::int32_t* source_slots;
    const std::int32_t* valid_columns;
    void* conv_record;
    void* key_record;
    void* value_record;
    void* gate_record;
    void* workspace;
};

[[nodiscard]] bool gdn_persistent_mixer_launch(const GdnPersistentLaunchArgs& args,
                                               cudaStream_t stream);

}
