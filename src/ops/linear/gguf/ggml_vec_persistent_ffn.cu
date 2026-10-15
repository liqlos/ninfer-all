#include "ggml_bridge_internal.cuh"
#include "ggml_bridge_vec.cuh"

#include <cooperative_groups.h>

#include <algorithm>
#include <cstdint>

namespace ninfer::ops::gguf {
namespace {

struct PersistentFfnArgs {
    detail::VecArgs producer;
    const std::uint8_t* down_weight;
    std::int64_t down_row_bytes;
    int down_rows;
    std::int8_t* output_qs;
    half2* output_ds;
    __nv_bfloat16* residual;
    std::int64_t residual_stride;
};

template <int T>
__global__ void persistent_q4_q6_ffn_kernel(PersistentFfnArgs p) {
    namespace cg = cooperative_groups;
    using Q4 = detail::vec::Decoder<GGML_TYPE_Q4_K>;
    using Q6 = detail::vec::Decoder<GGML_TYPE_Q6_K>;
    constexpr int kWarps = detail::kVecWarps;
    constexpr int kThreads = kWarps * 32;
    constexpr int kIntermediate = 17408;
    constexpr int kDownRows = 5120;
    constexpr int kDownK = 17408;
    constexpr int kRowsPerWarp = 2;
    constexpr int kOutputGroups = kIntermediate / 32;
    __shared__ float rounded[T][32];
    __shared__ std::uint32_t table[4];
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int producer_slices = p.producer.k / 32;
    const int q4_per_block = Q4::kBlockElems / 32;
    const int q4_lane_offset = (lane / q4_per_block) * Q4::kBlockBytes;
    const int q4_sub = lane % q4_per_block;
    for (int group = blockIdx.x; group < kOutputGroups; group += gridDim.x) {
        for (int local_row = warp; local_row < 32; local_row += kWarps) {
            const int row = group * 32 + local_row;
            const std::uint8_t* gate = p.producer.weight + std::int64_t(row) * p.producer.row_bytes + q4_lane_offset;
            const std::uint8_t* up = p.producer.weight + std::int64_t(p.producer.rows + row) * p.producer.row_bytes + q4_lane_offset;
            float gate_acc[T];
            float up_acc[T];
#pragma unroll
            for (int j = 0; j < T; ++j) {
                gate_acc[j] = 0.0f;
                up_acc[j] = 0.0f;
            }
            const auto slice = [&](int it) {
                const int s = lane + 32 * it;
                int a[T][8];
                float d[T], sum[T];
#pragma unroll
                for (int j = 0; j < T; ++j) {
                    const float2 ds = __half22float2(p.producer.ds[std::int64_t(j) * producer_slices + s]);
                    d[j] = ds.x;
                    sum[j] = ds.y;
                    const auto* q = reinterpret_cast<const int4*>(p.producer.qs + std::int64_t(j) * p.producer.k + 32 * s);
                    const int4 v0 = q[0], v1 = q[1];
                    a[j][0] = v0.x; a[j][1] = v0.y; a[j][2] = v0.z; a[j][3] = v0.w;
                    a[j][4] = v1.x; a[j][5] = v1.y; a[j][6] = v1.z; a[j][7] = v1.w;
                }
                detail::vec::Slice decoded;
                Q4::decode(gate + std::int64_t(32 / q4_per_block) * it * Q4::kBlockBytes, q4_sub, table, decoded);
#pragma unroll
                for (int j = 0; j < T; ++j) { gate_acc[j] += Q4::dot(decoded, a[j], d[j], sum[j]); }
                Q4::decode(up + std::int64_t(32 / q4_per_block) * it * Q4::kBlockBytes, q4_sub, table, decoded);
#pragma unroll
                for (int j = 0; j < T; ++j) { up_acc[j] += Q4::dot(decoded, a[j], d[j], sum[j]); }
            };
#pragma unroll
            for (int it = 0; it < 5; ++it) { slice(it); }
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
                for (int j = 0; j < T; ++j) {
                    gate_acc[j] += __shfl_xor_sync(0xFFFFFFFFu, gate_acc[j], offset);
                    up_acc[j] += __shfl_xor_sync(0xFFFFFFFFu, up_acc[j], offset);
                }
            }
            if (lane == 0) {
#pragma unroll
                for (int j = 0; j < T; ++j) {
                    const float value = detail::vec_silu(gate_acc[j]) * up_acc[j];
                    rounded[j][local_row] = __bfloat162float(__float2bfloat16(value));
                }
            }
        }
        __syncthreads();
        if (warp == 0) {
#pragma unroll
            for (int j = 0; j < T; ++j) {
                const float xi = rounded[j][lane];
                const float amax = ::warp_reduce_max<QK8_1>(fabsf(xi));
                const float sum = ::warp_reduce_sum<QK8_1>(xi);
                const float d = amax / 127.0f;
                p.output_qs[std::int64_t(j) * kIntermediate + group * 32 + lane] =
                    amax == 0.0f ? 0 : static_cast<std::int8_t>(roundf(xi / d));
                if (lane == 0) { p.output_ds[std::int64_t(j) * kOutputGroups + group] = ::make_half2(d, sum); }
            }
        }
        __syncthreads();
    }
    cg::this_grid().sync();
    const int down_slices = kDownK / 32;
    constexpr int q6_per_block = Q6::kBlockElems / 32;
    const int q6_lane_offset = (lane / q6_per_block) * Q6::kBlockBytes;
    const int q6_sub = lane % q6_per_block;
    for (int row0 = (blockIdx.x * kWarps + warp) * kRowsPerWarp;
         row0 < kDownRows; row0 += gridDim.x * kWarps * kRowsPerWarp) {
        const std::uint8_t* rowp[kRowsPerWarp];
#pragma unroll
        for (int r = 0; r < kRowsPerWarp; ++r) {
            rowp[r] = p.down_weight + std::int64_t(min(row0 + r, kDownRows - 1)) * p.down_row_bytes + q6_lane_offset;
        }
        float acc[kRowsPerWarp][T];
#pragma unroll
        for (int r = 0; r < kRowsPerWarp; ++r) {
#pragma unroll
            for (int j = 0; j < T; ++j) { acc[r][j] = 0.0f; }
        }
#pragma unroll
        for (int it = 0; it < 17; ++it) {
            const int s = lane + 32 * it;
            int a[T][8];
            float d[T], sum[T];
#pragma unroll
            for (int j = 0; j < T; ++j) {
                const float2 ds = __half22float2(p.output_ds[std::int64_t(j) * (kIntermediate / 32) + s]);
                d[j] = ds.x;
                sum[j] = ds.y;
                const auto* q = reinterpret_cast<const int4*>(p.output_qs + std::int64_t(j) * kDownK + 32 * s);
                const int4 v0 = q[0], v1 = q[1];
                a[j][0] = v0.x; a[j][1] = v0.y; a[j][2] = v0.z; a[j][3] = v0.w;
                a[j][4] = v1.x; a[j][5] = v1.y; a[j][6] = v1.z; a[j][7] = v1.w;
            }
#pragma unroll
            for (int r = 0; r < kRowsPerWarp; ++r) {
                detail::vec::Slice decoded;
                Q6::decode(rowp[r] + std::int64_t(32 / q6_per_block) * it * Q6::kBlockBytes, q6_sub, table, decoded);
#pragma unroll
                for (int j = 0; j < T; ++j) { acc[r][j] += Q6::dot(decoded, a[j], d[j], sum[j]); }
            }
        }
#pragma unroll
        for (int r = 0; r < kRowsPerWarp; ++r) {
#pragma unroll
            for (int j = 0; j < T; ++j) {
#pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1) {
                    acc[r][j] += __shfl_xor_sync(0xFFFFFFFFu, acc[r][j], offset);
                }
            }
        }
        if (lane == 0) {
#pragma unroll
            for (int r = 0; r < kRowsPerWarp; ++r) {
                if (row0 + r < kDownRows) {
#pragma unroll
                    for (int j = 0; j < T; ++j) {
                        detail::VecStore destination{p.residual, nullptr, p.residual_stride,
                                                     Epilogue::Accumulate, nullptr, 0};
                        detail::vec_store(destination, row0 + r, j, acc[r][j]);
                    }
                }
            }
        }
    }
}

template <int T>
bool persistent_grid(int& blocks) {
    int device = 0;
    int cooperative = 0;
    int sms = 0;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaDeviceGetAttribute(&cooperative, cudaDevAttrCooperativeLaunch, device) != cudaSuccess ||
        cooperative == 0 ||
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) != cudaSuccess || sms <= 0) {
        return false;
    }
    int resident = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, persistent_q4_q6_ffn_kernel<T>,
                                                      detail::kVecWarps * 32, 0) != cudaSuccess || resident <= 0) {
        return false;
    }
    blocks = std::min(17408 / 32, resident * sms);
    return blocks > 0;
}

template <int T>
int persistent_blocks() {
    static const int blocks = [] {
        int count = 0;
        return persistent_grid<T>(count) ? count : 0;
    }();
    return blocks;
}

template <int T>
bool launch_persistent(const PersistentFfnArgs& args, cudaStream_t stream) {
    const int blocks = persistent_blocks<T>();
    if (blocks <= 0) { return false; }
    void* kernel_args[] = {const_cast<PersistentFfnArgs*>(&args)};
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    detail::check(cudaStreamIsCapturing(stream, &capture), "persistent FFN capture state");
    const cudaError_t status = cudaLaunchCooperativeKernel(
        reinterpret_cast<void*>(persistent_q4_q6_ffn_kernel<T>), dim3(blocks),
        dim3(detail::kVecWarps * 32), kernel_args, 0, stream);
    if (capture == cudaStreamCaptureStatusNone &&
        (status == cudaErrorCooperativeLaunchTooLarge ||
         status == cudaErrorLaunchOutOfResources || status == cudaErrorNotSupported)) {
        (void)cudaGetLastError();
        return false;
    }
    if (status != cudaSuccess) { detail::check(status, "persistent FFN cooperative launch"); }
    return true;
}

template <int T>
bool available(cudaStream_t) {
    return persistent_blocks<T>() > 0;
}

}

bool persistent_swiglu_q8_add_available(int columns, cudaStream_t stream) {
    switch (columns) {
    case 1: return available<1>(stream);
    case 2: return available<2>(stream);
    case 3: return available<3>(stream);
    case 4: return available<4>(stream);
    case 5: return available<5>(stream);
    case 6: return available<6>(stream);
    case 7: return available<7>(stream);
    case 8: return available<8>(stream);
    default: return false;
    }
}

bool persistent_swiglu_q8_add(GgmlType gate_type, const void* gate_weight, std::int64_t gate_row_bytes,
                              int intermediate_rows, int gate_k, const void* input_activation,
                              GgmlType down_type, const void* down_weight,
                              std::int64_t down_row_bytes, int down_rows, int down_k,
                              void* q8_workspace, __nv_bfloat16* residual,
                              std::int64_t residual_stride, int columns, cudaStream_t stream) {
    if (gate_type != GgmlType::Q4_K || down_type != GgmlType::Q6_K ||
        intermediate_rows != 17408 || gate_k != 5120 || down_rows != 5120 || down_k != 17408 ||
        gate_weight == nullptr || down_weight == nullptr || input_activation == nullptr ||
        q8_workspace == nullptr || residual == nullptr || residual_stride < down_rows ||
        columns < 1 || columns > 8) {
        return false;
    }
    const std::size_t q8_offset = detail::vec_scales_offset(gate_k, columns);
    const std::size_t output_offset = detail::vec_scales_offset(down_k, columns);
    const auto* input = static_cast<const std::uint8_t*>(input_activation);
    auto* output = static_cast<std::uint8_t*>(q8_workspace);
    detail::VecArgs producer{static_cast<const std::uint8_t*>(gate_weight), gate_row_bytes,
                             intermediate_rows, gate_k,
                             reinterpret_cast<const std::int8_t*>(input),
                             reinterpret_cast<const half2*>(input + q8_offset),
                             {nullptr, nullptr, 0, Epilogue::Store, nullptr, 0}};
    PersistentFfnArgs args{producer, static_cast<const std::uint8_t*>(down_weight), down_row_bytes,
                           down_rows, reinterpret_cast<std::int8_t*>(output),
                           reinterpret_cast<half2*>(output + output_offset), residual,
                           residual_stride};
    switch (columns) {
    case 1: return launch_persistent<1>(args, stream);
    case 2: return launch_persistent<2>(args, stream);
    case 3: return launch_persistent<3>(args, stream);
    case 4: return launch_persistent<4>(args, stream);
    case 5: return launch_persistent<5>(args, stream);
    case 6: return launch_persistent<6>(args, stream);
    case 7: return launch_persistent<7>(args, stream);
    case 8: return launch_persistent<8>(args, stream);
    default: return false;
    }
}

}
