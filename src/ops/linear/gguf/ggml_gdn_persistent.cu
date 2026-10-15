#include "ops/linear/gguf/ggml_gdn_persistent.h"

#include "core/dtype.h"
#include "ops/gdn_input_proj/gdn_conv.cuh"
#include "ops/kernel/rmsnorm.cuh"
#include "ops/linear/gguf/ggml_bridge_internal.cuh"
#include "ops/linear/gguf/ggml_bridge_vec.cuh"
#include "ops/linear_attention/gated_delta_net/recurrent.cuh"

#include <cooperative_groups.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

namespace cg = cooperative_groups;
namespace gguf = ninfer::ops::gguf;
namespace bridge = ninfer::ops::gguf::detail;
namespace vec = ninfer::ops::gguf::detail::vec;
namespace gdn = ninfer::ops::detail::gated_delta_net;

constexpr int kHidden = 5120;
constexpr int kQk = 2048;
constexpr int kValue = 6144;
constexpr int kQkv = 10240;
constexpr int kZ = 6144;
constexpr int kQkHeads = 16;
constexpr int kValueHeads = 48;
constexpr int kDim = 128;
constexpr int kConvTaps = 4;
constexpr int kMaxColumns = 4;
constexpr int kThreads = 128;
constexpr int kWarps = kThreads / 32;
constexpr int kRecurrentTasks = kValueHeads * 8;
constexpr int kMaxParts = 8;

struct ProjectionPart {
    const std::uint8_t* weight;
    std::int64_t row_bytes;
    int type;
    int rows;
    int row;
    int output;
};

struct ProjectionArgs {
    ProjectionPart parts[kMaxParts];
    int part_count;
    int columns;
    const std::int8_t* q8;
    const half2* ds;
    __nv_bfloat16* qkv;
    __nv_bfloat16* z;
};

__device__ __forceinline__ void load_activation(const ProjectionArgs& p, int column, int slice,
                                                int k, int (&a)[8], float& d, float& sum) {
    const std::int64_t at = static_cast<std::int64_t>(column) * k + 32LL * slice;
    const int4 lo = *reinterpret_cast<const int4*>(p.q8 + at);
    const int4 hi = *reinterpret_cast<const int4*>(p.q8 + at + 16);
    a[0] = lo.x;
    a[1] = lo.y;
    a[2] = lo.z;
    a[3] = lo.w;
    a[4] = hi.x;
    a[5] = hi.y;
    a[6] = hi.z;
    a[7] = hi.w;
    const float2 pair = __half22float2(p.ds[static_cast<std::int64_t>(column) * (k / 32) + slice]);
    d = pair.x;
    sum = pair.y;
}

template <ggml_type Type>
__device__ __forceinline__ void project_row(const ProjectionPart& part, int row,
                                            const ProjectionArgs& p, int k,
                                            float (&acc)[kMaxColumns]) {
    using D = vec::Decoder<Type>;
    constexpr int kPerBlock = D::kBlockElems / 32;
    const int lane = threadIdx.x & 31;
    const int sub = lane % kPerBlock;
    const std::int64_t lane_offset = std::int64_t(lane / kPerBlock) * D::kBlockBytes;
    const auto* weight = part.weight + std::int64_t(row) * part.row_bytes + lane_offset;
    for (int column = 0; column < p.columns; ++column) { acc[column] = 0.0f; }
    for (int it = 0; it < k / 1024; ++it) {
        vec::Slice slice;
        D::decode(weight + std::int64_t(32 / kPerBlock) * it * D::kBlockBytes, sub,
                  D::table(), slice);
        for (int column = 0; column < p.columns; ++column) {
            int a[8];
            float d;
            float sum;
            load_activation(p, column, lane + 32 * it, k, a, d, sum);
            acc[column] += D::dot(slice, a, d, sum);
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        for (int column = 0; column < p.columns; ++column) {
            acc[column] += __shfl_xor_sync(0xffffffffu, acc[column], offset);
        }
    }
}

__device__ __forceinline__ const ProjectionPart* find_part(const ProjectionArgs& p, int output,
                                                            int row) {
    for (int i = 0; i < p.part_count; ++i) {
        const ProjectionPart& part = p.parts[i];
        if (part.output == output && row >= part.row && row < part.row + part.rows) {
            return &part;
        }
    }
    return nullptr;
}

__device__ __forceinline__ void project_dispatch(const ProjectionPart& part, int row,
                                                 const ProjectionArgs& p,
                                                 float (&acc)[kMaxColumns]) {
    if (part.type == static_cast<int>(QType::GGUF_Q4_K)) {
        project_row<GGML_TYPE_Q4_K>(part, row - part.row, p, kHidden, acc);
    } else {
        project_row<GGML_TYPE_Q6_K>(part, row - part.row, p, kHidden, acc);
    }
}

__device__ __forceinline__ void store_projected(const ProjectionPart& part, int row,
                                                const ProjectionArgs& p,
                                                __nv_bfloat16* output, int stride) {
    float acc[kMaxColumns];
    project_dispatch(part, row, p, acc);
    if ((threadIdx.x & 31) == 0) {
        for (int column = 0; column < p.columns; ++column) {
            output[static_cast<std::int64_t>(column) * stride + row] = __float2bfloat16(acc[column]);
        }
    }
}


template <class StateT>
struct PersistentRecordAccess {
    const __nv_bfloat16* qkv;
    const float* g;
    const float* beta;
    const StateT* states;
    const std::int32_t* valid_columns;
    const std::int32_t* source_slots;
    __nv_bfloat16* key_record;
    __nv_bfloat16* value_record;
    uint2* gate_record;
    __nv_bfloat16* out;
    gdn::head_map heads;
    int width;
    std::int64_t state_slot_stride;
    float scale;

    __device__ __forceinline__ const StateT* state_read_base(
        const gdn::RecurrentCoordinates& coord) const {
        return states + static_cast<std::int64_t>(source_slots[0]) * state_slot_stride +
               static_cast<std::int64_t>(coord.value_head) * kDim * kDim;
    }

    __device__ __forceinline__ const __nv_bfloat16* key_ptr(
        const gdn::RecurrentCoordinates& coord, int token) const {
        return qkv + static_cast<std::int64_t>(token) * kQkv + kQk +
               static_cast<std::int64_t>(coord.qk_head) * kDim;
    }

    __device__ __forceinline__ const __nv_bfloat16* value_ptr(
        const gdn::RecurrentCoordinates& coord, int token) const {
        return qkv + static_cast<std::int64_t>(token) * kQkv + 2 * kQk +
               static_cast<std::int64_t>(coord.value_head) * kDim;
    }

    __device__ __forceinline__ const __nv_bfloat16* query_ptr(
        const gdn::RecurrentCoordinates& coord, int token) const {
        return qkv + static_cast<std::int64_t>(token) * kQkv +
               static_cast<std::int64_t>(coord.qk_head) * kDim;
    }

    __device__ __forceinline__ gdn::RawGatePair load_gate(
        const gdn::RecurrentCoordinates& coord, int token) const {
        const std::int64_t at = static_cast<std::int64_t>(token) * kValueHeads + coord.value_head;
        return gdn::load_source_gate(g, beta, at);
    }

    __device__ __forceinline__ __nv_bfloat16* output_ptr(
        const gdn::RecurrentCoordinates& coord, int token) const {
        return out + static_cast<std::int64_t>(token) * kValue +
               static_cast<std::int64_t>(coord.value_head) * kDim;
    }

    __device__ __forceinline__ void store_key(const gdn::RecurrentCoordinates& coord, int token,
                                              const gdn::RawQkLane& raw) const {
        if (coord.state_tile == 0 && coord.warp == 0 && coord.value_head % heads.group_size() == 0) {
            __nv_bfloat16* destination =
                key_record + (static_cast<std::int64_t>(token) * kQkHeads + coord.qk_head) * kDim;
            gdn::store_vec(destination + coord.dqk_base, raw.bits);
        }
    }

    __device__ __forceinline__ void store_value(const gdn::RecurrentCoordinates& coord, int token,
                                                const gdn::RawValueLane& raw) const {
        if (coord.lane < gdn::kDvPerWarp) {
            __nv_bfloat16* destination =
                value_record + (static_cast<std::int64_t>(token) * kValueHeads + coord.value_head) * kDim;
            destination[coord.dv_base + coord.lane] = raw.bits;
        }
    }

    __device__ __forceinline__ void store_gate(const gdn::RecurrentCoordinates& coord, int token,
                                               const gdn::RawGatePair& raw) const {
        if (coord.state_tile == 0 && coord.warp == 0 && coord.lane == 0) {
            gate_record[static_cast<std::int64_t>(token) * kValueHeads + coord.value_head] = raw.bits;
        }
    }
};

template <class StateT>
__global__ void persistent_gdn_kernel(ProjectionArgs projection, int columns,
                                      const __nv_bfloat16* hidden, const float* g,
                                      const float* beta, const __nv_bfloat16* convolution,
                                      const __nv_bfloat16* norm_weight,
                                      const std::int32_t* output_columns,
                                      const std::uint8_t* output_weight,
                                      std::int64_t output_row_bytes,
                                      const __nv_bfloat16* residual,
                                      __nv_bfloat16* residual_out,
                                      const __nv_bfloat16* conv_states,
                                      const StateT* recurrent_states,
                                      const std::int32_t* source_slots,
                                      const std::int32_t* valid_columns,
                                      __nv_bfloat16* conv_record,
                                      __nv_bfloat16* key_record,
                                      __nv_bfloat16* value_record,
                                      float* gate_record, __nv_bfloat16* q8,
                                      __nv_bfloat16* qkv_projection,
                                      __nv_bfloat16* qkv_convolved,
                                      __nv_bfloat16* z, __nv_bfloat16* recurrent_out,
                                      __nv_bfloat16* normalized, int action, float eps) {
    cg::grid_group grid = cg::this_grid();
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int valid = max(0, min(valid_columns[0], columns));
    auto* q8_bytes = reinterpret_cast<std::uint8_t*>(q8);
    const std::size_t q8_scale_offset = gguf::detail::vec_scales_offset(kHidden, columns);
    const auto* hidden_bf16 = hidden;
    for (int group = static_cast<int>(blockIdx.x) * kWarps + warp;
         group < columns * (kHidden / 32); group += static_cast<int>(gridDim.x) * kWarps) {
        const int column = group / (kHidden / 32);
        const int block = group - column * (kHidden / 32);
        const int index = block * 32 + lane;
        const float x = __bfloat162float(hidden_bf16[static_cast<std::int64_t>(column) * kHidden + index]);
        const float amax = ::warp_reduce_max<QK8_1>(fabsf(x));
        const float sum = ::warp_reduce_sum<QK8_1>(x);
        const float d = amax / 127.0f;
        q8_bytes[static_cast<std::int64_t>(column) * kHidden + index] =
            amax == 0.0f ? 0 : static_cast<std::int8_t>(roundf(x / d));
        if (lane == 0) {
            reinterpret_cast<half2*>(q8_bytes + q8_scale_offset)[
                static_cast<std::int64_t>(column) * (kHidden / 32) + block] = make_half2(d, sum);
        }
    }
    grid.sync();

    projection.q8 = reinterpret_cast<const std::int8_t*>(q8_bytes);
    projection.ds = reinterpret_cast<const half2*>(q8_bytes + q8_scale_offset);
    const int project_rows = kQkv + kZ;
    const int project_total = static_cast<int>(gridDim.x) * kWarps;
    for (int row = static_cast<int>(blockIdx.x) * kWarps + warp; row < project_rows;
         row += project_total) {
        const int output = row < kQkv ? 0 : 1;
        const int output_row = output == 0 ? row : row - kQkv;
        const ProjectionPart* part = find_part(projection, output, output_row);
        if (part == nullptr) { continue; }
        store_projected(*part, output_row, projection,
                        output == 0 ? qkv_projection : z, output == 0 ? kQkv : kZ);
    }
    grid.sync();

    for (int channel = static_cast<int>(blockIdx.x) * kThreads + threadIdx.x;
         channel < kQkv; channel += static_cast<int>(gridDim.x) * kThreads) {
        const int slot = source_slots[0];
        const std::int64_t state_base = static_cast<std::int64_t>(slot) * (3LL * kQkv);
        const __nv_bfloat16 s0 = conv_states[state_base + channel];
        const __nv_bfloat16 s1 = conv_states[state_base + kQkv + channel];
        const __nv_bfloat16 s2 = conv_states[state_base + 2LL * kQkv + channel];
        __nv_bfloat16 history[3] = {s0, s1, s2};
        __nv_bfloat16 projected[kMaxColumns];
        for (int token = 0; token < columns; ++token) {
            projected[token] = qkv_projection[static_cast<std::int64_t>(token) * kQkv + channel];
            if (token < valid) {
                const __nv_bfloat16 x0 = token >= 3
                    ? qkv_projection[static_cast<std::int64_t>(token - 3) * kQkv + channel]
                    : history[token];
                const __nv_bfloat16 x1 = token >= 2
                    ? qkv_projection[static_cast<std::int64_t>(token - 2) * kQkv + channel]
                    : history[token + 1];
                const __nv_bfloat16 x2 = token >= 1
                    ? qkv_projection[static_cast<std::int64_t>(token - 1) * kQkv + channel]
                    : history[token + 2];
                float acc = fmaf(__bfloat162float(convolution[channel]), __bfloat162float(x0), 0.0f);
                acc = fmaf(__bfloat162float(convolution[kQkv + channel]), __bfloat162float(x1), acc);
                acc = fmaf(__bfloat162float(convolution[2LL * kQkv + channel]), __bfloat162float(x2), acc);
                acc = fmaf(__bfloat162float(convolution[3LL * kQkv + channel]),
                           __bfloat162float(projected[token]), acc);
                qkv_convolved[static_cast<std::int64_t>(token) * kQkv + channel] =
                    __float2bfloat16_rn(silu(acc));
                if (action == 1) {
                    conv_record[static_cast<std::int64_t>(token) * kQkv + channel] = projected[token];
                }
            } else {
                qkv_convolved[static_cast<std::int64_t>(token) * kQkv + channel] =
                    __float2bfloat16_rn(0.0f);
            }
        }
    }
    grid.sync();

    const gdn::head_map heads = gdn::head_map::of(kQkHeads, kValueHeads);
    for (int task = static_cast<int>(blockIdx.x); task < kRecurrentTasks;
         task += static_cast<int>(gridDim.x)) {
        const int value_head = task / 8;
        const int state_tile = task % 8;
        const gdn::RecurrentCoordinates coord{
            lane, warp, 0, 0, state_tile, static_cast<std::uint32_t>(value_head),
            static_cast<std::uint32_t>(heads.qk_head(value_head)),
            static_cast<std::uint32_t>(state_tile * gdn::kBlockDv + warp * gdn::kDvPerWarp),
            static_cast<std::uint32_t>(lane * gdn::kQkPerLane)};
        PersistentRecordAccess<StateT> access{
            qkv_convolved, g, beta, recurrent_states, valid_columns, source_slots,
            key_record, value_record, reinterpret_cast<uint2*>(gate_record), recurrent_out,
            heads, columns, static_cast<std::int64_t>(kValueHeads) * kDim * kDim, 1.0f / sqrtf(128.0f)};
        __align__(16) float state[gdn::kDvPerWarp][gdn::kQkPerLane];
        gdn::load_state_tile(state, access.state_read_base(coord), coord);
        gdn::run_recurrent_sequence<true, gdn::RecordEffects>(state, access, coord, valid);
        gdn::zero_output_suffix(access, coord, valid, columns);
    }
    grid.sync();

    const int rows = columns * kValueHeads;
    for (int row = static_cast<int>(blockIdx.x) * kWarps + warp; row < rows;
         row += static_cast<int>(gridDim.x) * kWarps) {
        const int column = row / kValueHeads;
        const int head = row - column * kValueHeads;
        const std::int64_t offset = static_cast<std::int64_t>(column) * kValue + head * kDim;
        const int pair0 = lane;
        const int pair1 = lane + 32;
        const auto x0 = reinterpret_cast<const __nv_bfloat162*>(recurrent_out + offset)[pair0];
        const auto x1 = reinterpret_cast<const __nv_bfloat162*>(recurrent_out + offset)[pair1];
        const float2 f0 = __bfloat1622float2(x0);
        const float2 f1 = __bfloat1622float2(x1);
        float sum = f0.x * f0.x + f0.y * f0.y;
        sum += f1.x * f1.x + f1.y * f1.y;
        sum = warp_reduce_sum(sum);
        float inv = lane == 0 ? rsqrtf(sum * (1.0f / kDim) + eps) : 0.0f;
        inv = __shfl_sync(kFullWarpMask, inv, 0);
        const auto w0 = reinterpret_cast<const __nv_bfloat162*>(norm_weight)[pair0];
        const auto w1 = reinterpret_cast<const __nv_bfloat162*>(norm_weight)[pair1];
        const auto z0 = reinterpret_cast<const __nv_bfloat162*>(z + offset)[pair0];
        const auto z1 = reinterpret_cast<const __nv_bfloat162*>(z + offset)[pair1];
        const float2 wf0 = __bfloat1622float2(w0);
        const float2 wf1 = __bfloat1622float2(w1);
        const float2 gate0 = __bfloat1622float2(z0);
        const float2 gate1 = __bfloat1622float2(z1);
        const float y0 = f0.x * inv * wf0.x * silu(gate0.x);
        const float y1 = f0.y * inv * wf0.y * silu(gate0.y);
        const float y2 = f1.x * inv * wf1.x * silu(gate1.x);
        const float y3 = f1.y * inv * wf1.y * silu(gate1.y);
        auto* out_pair = reinterpret_cast<__nv_bfloat162*>(normalized + offset);
        out_pair[pair0] = __floats2bfloat162_rn(y0, y1);
        out_pair[pair1] = __floats2bfloat162_rn(y2, y3);
    }
    grid.sync();

    const std::size_t output_scale_offset = gguf::detail::vec_scales_offset(kValue, columns);
    auto* output_q8 = reinterpret_cast<std::uint8_t*>(q8);
    for (int group = static_cast<int>(blockIdx.x) * kWarps + warp;
         group < columns * (kValue / 32); group += static_cast<int>(gridDim.x) * kWarps) {
        const int column = group / (kValue / 32);
        const int block = group - column * (kValue / 32);
        const int index = block * 32 + lane;
        const int source_index = output_columns[index];
        const float x = __bfloat162float(normalized[static_cast<std::int64_t>(column) * kValue + source_index]);
        const float amax = ::warp_reduce_max<QK8_1>(fabsf(x));
        const float sum = ::warp_reduce_sum<QK8_1>(x);
        const float d = amax / 127.0f;
        output_q8[static_cast<std::int64_t>(column) * kValue + index] =
            amax == 0.0f ? 0 : static_cast<std::int8_t>(roundf(x / d));
        if (lane == 0) {
            reinterpret_cast<half2*>(output_q8 + output_scale_offset)[
                static_cast<std::int64_t>(column) * (kValue / 32) + block] = make_half2(d, sum);
        }
    }
    grid.sync();

    const int output_rows = kHidden;
    ProjectionArgs output_projection{};
    output_projection.part_count = 1;
    output_projection.columns = columns;
    output_projection.q8 = reinterpret_cast<const std::int8_t*>(output_q8);
    output_projection.ds = reinterpret_cast<const half2*>(output_q8 + output_scale_offset);
    for (int row = static_cast<int>(blockIdx.x) * kWarps + warp; row < output_rows;
         row += static_cast<int>(gridDim.x) * kWarps) {
        constexpr int kPerBlock = vec::Decoder<GGML_TYPE_Q4_K>::kBlockElems / 32;
        const std::int64_t lane_offset = std::int64_t(lane / kPerBlock) *
                                         vec::Decoder<GGML_TYPE_Q4_K>::kBlockBytes;
        const int sub = lane % kPerBlock;
        const auto* row_weight = output_weight + static_cast<std::int64_t>(row) * output_row_bytes + lane_offset;
        float acc[kMaxColumns] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int it = 0; it < kValue / 1024; ++it) {
            int a[kMaxColumns][8];
            float d[kMaxColumns];
            float sum[kMaxColumns];
            for (int column = 0; column < columns; ++column) {
                load_activation(output_projection, column, lane + 32 * it, kValue, a[column], d[column], sum[column]);
            }
            vec::Slice slice;
            vec::Decoder<GGML_TYPE_Q4_K>::decode(
                row_weight + std::int64_t(32 / kPerBlock) * it *
                                 vec::Decoder<GGML_TYPE_Q4_K>::kBlockBytes,
                sub, vec::Decoder<GGML_TYPE_Q4_K>::table(), slice);
            for (int column = 0; column < columns; ++column) {
                acc[column] += vec::Decoder<GGML_TYPE_Q4_K>::dot(slice, a[column], d[column], sum[column]);
            }
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            for (int column = 0; column < columns; ++column) {
                acc[column] += __shfl_xor_sync(0xffffffffu, acc[column], offset);
            }
        }
        if (lane == 0) {
            for (int column = 0; column < columns; ++column) {
                const std::int64_t at = static_cast<std::int64_t>(column) * output_rows + row;
                residual_out[at] = __float2bfloat16(acc[column] + __bfloat162float(residual[at]));
            }
        }
    }
    grid.sync();
}

}}

namespace ninfer::ops {
namespace {

std::size_t align_workspace(std::size_t n) { return (n + 255) & ~std::size_t(255); }

struct PersistentLaunchFacts {
    int multiprocessors;
    int active_float;
    int active_half;
};

bool persistent_launch_facts(cudaStream_t stream, PersistentLaunchFacts& result) {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) { return false; }
    static std::mutex mutex;
    static std::unordered_map<int, PersistentLaunchFacts> cache;
    thread_local int cached_device = -1;
    thread_local PersistentLaunchFacts cached_facts{};
    if (device == cached_device) {
        result = cached_facts;
        return true;
    }
    {
        const std::scoped_lock lock(mutex);
        const auto found = cache.find(device);
        if (found != cache.end()) {
            cached_device = device;
            cached_facts = found->second;
            result = cached_facts;
            return true;
        }
    }
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess ||
        capture != cudaStreamCaptureStatusNone) {
        return false;
    }
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, device) != cudaSuccess || properties.major != 8 ||
        properties.minor != 6 || !properties.cooperativeLaunch) {
        return false;
    }
    PersistentLaunchFacts facts{};
    facts.multiprocessors = properties.multiProcessorCount;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &facts.active_float, persistent_gdn_kernel<float>, kThreads, 0) != cudaSuccess ||
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &facts.active_half, persistent_gdn_kernel<__half>, kThreads, 0) != cudaSuccess ||
        facts.active_float <= 0 || facts.active_half <= 0 || facts.multiprocessors <= 0) {
        return false;
    }
    {
        const std::scoped_lock lock(mutex);
        cache.emplace(device, facts);
    }
    cached_device = device;
    cached_facts = facts;
    result = facts;
    return true;
}
template <class StateT>
cudaError_t launch_persistent_kernel(const gguf::detail::GdnPersistentLaunchArgs& args,
                                     ProjectionArgs projection, int blocks, cudaStream_t stream) {
    auto* q8 = static_cast<__nv_bfloat16*>(args.workspace);
    std::size_t offset = align_workspace(gguf::vector_activation_bytes(kValue, args.columns));
    auto* qkv_projection = reinterpret_cast<__nv_bfloat16*>(reinterpret_cast<std::uint8_t*>(args.workspace) + offset);
    offset += align_workspace(sizeof(__nv_bfloat16) * static_cast<std::size_t>(kQkv) * args.columns);
    auto* qkv_convolved = reinterpret_cast<__nv_bfloat16*>(reinterpret_cast<std::uint8_t*>(args.workspace) + offset);
    offset += align_workspace(sizeof(__nv_bfloat16) * static_cast<std::size_t>(kQkv) * args.columns);
    auto* z = reinterpret_cast<__nv_bfloat16*>(reinterpret_cast<std::uint8_t*>(args.workspace) + offset);
    offset += align_workspace(sizeof(__nv_bfloat16) * static_cast<std::size_t>(kZ) * args.columns);
    auto* recurrent_out = reinterpret_cast<__nv_bfloat16*>(reinterpret_cast<std::uint8_t*>(args.workspace) + offset);
    offset += align_workspace(sizeof(__nv_bfloat16) * static_cast<std::size_t>(kValue) * args.columns);
    auto* normalized = reinterpret_cast<__nv_bfloat16*>(reinterpret_cast<std::uint8_t*>(args.workspace) + offset);
    const int columns = args.columns;
    const auto* hidden = static_cast<const __nv_bfloat16*>(args.hidden);
    const auto* g = args.g;
    const auto* beta = args.beta;
    const auto* convolution = static_cast<const __nv_bfloat16*>(args.convolution);
    const auto* norm_weight = static_cast<const __nv_bfloat16*>(args.norm_weight);
    const auto* output_columns = args.output_columns;
    const auto* output_weight = args.output_weight;
    const auto output_row_bytes = args.output_row_bytes;
    const auto* residual = static_cast<const __nv_bfloat16*>(args.residual);
    auto* residual_out = static_cast<__nv_bfloat16*>(args.residual_out);
    const auto* conv_states = static_cast<const __nv_bfloat16*>(args.conv_states);
    const auto* recurrent_states = static_cast<const StateT*>(args.recurrent_states);
    const auto* source_slots = args.source_slots;
    const auto* valid_columns = args.valid_columns;
    auto* conv_record = static_cast<__nv_bfloat16*>(args.conv_record);
    auto* key_record = static_cast<__nv_bfloat16*>(args.key_record);
    auto* value_record = static_cast<__nv_bfloat16*>(args.value_record);
    auto* gate_record = static_cast<float*>(args.gate_record);
    const int action = args.action;
    const float eps = args.eps;
    void* launch_args[] = {&projection, &columns, &hidden, &g, &beta, &convolution,
                           &norm_weight, &output_columns, &output_weight, &output_row_bytes,
                           &residual, &residual_out, &conv_states, &recurrent_states,
                           &source_slots, &valid_columns, &conv_record, &key_record,
                           &value_record, &gate_record, &q8, &qkv_projection, &qkv_convolved,
                           &z, &recurrent_out, &normalized, &action, const_cast<float*>(&eps)};
    return cudaLaunchCooperativeKernel(
        reinterpret_cast<void*>(persistent_gdn_kernel<StateT>), blocks, kThreads, launch_args, 0, stream);
}

}

bool gguf::detail::gdn_persistent_mixer_launch(
    const gguf::detail::GdnPersistentLaunchArgs& args, cudaStream_t stream) {
    PersistentLaunchFacts facts{};
    if (!persistent_launch_facts(stream, facts)) { return false; }
    ProjectionArgs projection{};
    projection.part_count = args.part_count;
    projection.columns = args.columns;
    for (int i = 0; i < args.part_count; ++i) {
        const auto& part = args.parts[i];
        projection.parts[i] = {part.weight, part.row_bytes, part.type, part.rows, part.row, part.output};
    }
    const int active_blocks = args.state_fp16 ? facts.active_half : facts.active_float;
    const int blocks = active_blocks * facts.multiprocessors;
    if (blocks <= 0) { return false; }
    const cudaError_t status = args.state_fp16
        ? launch_persistent_kernel<__half>(args, projection, blocks, stream)
        : launch_persistent_kernel<float>(args, projection, blocks, stream);
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("gdn_persistent_mixer launch: ") + cudaGetErrorString(status));
    }
    return true;
}

}
