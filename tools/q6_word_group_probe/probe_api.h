#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

#ifndef NINFER_Q6_WORD_GROUP_PROBE
#define NINFER_Q6_WORD_GROUP_PROBE 0
#endif

enum class Q6WGStorage : std::uint32_t {
    Native210 = 0,
    WordGroupSlab16 = 1,
};

enum class Q6WGOutputMode : std::uint32_t {
    PlainFP32 = 0,
    StoreBF16 = 1,
    AccumulateBF16 = 2,
};

struct Q6WGView {
    const std::uint8_t* base;
    std::uint64_t bytes;
    std::uint32_t parent_rows;
    std::uint32_t parent_B;
    std::uint32_t row_begin;
    std::uint32_t rows;
    std::uint32_t final_slab_blocks;
    Q6WGStorage storage;
};

struct Q6WGReadBuffer {
    const void* data;
    std::size_t bytes;
};

struct Q6WGWriteBuffer {
    void* data;
    std::size_t bytes;
};

struct Q6WGOutput {
    Q6WGWriteBuffer buffer;
    std::int64_t column_stride;
    Q6WGOutputMode mode;
};

struct Q6WGMatrixPlan {
    int rows;
    int k;
    int columns;
    int J;
    int I;
    int threads;
    std::size_t shared;
    int row_tiles;
    int column_tiles;
    bool stream_k;
    int blocks;
    bool fixup;
    std::size_t activation_bytes;
    std::size_t fixup_bytes;
};

bool q6wg_candidate_enabled();
bool q6wg_validate_view(const Q6WGView& view);
std::size_t q6wg_vector_activation_bytes(int k, int columns);
bool q6wg_vector_run(const Q6WGView& view, Q6WGReadBuffer activation, int columns,
                     const Q6WGOutput& output, cudaStream_t stream);
int q6wg_matrix_activation_layout();
bool q6wg_matrix_plan(int rows, int k, int columns, Q6WGMatrixPlan* plan);
bool q6wg_matrix_run(const Q6WGMatrixPlan& plan, const Q6WGView& view,
                     Q6WGReadBuffer activation, int activation_layout,
                     const Q6WGOutput& output, Q6WGWriteBuffer fixup,
                     cudaStream_t stream);
