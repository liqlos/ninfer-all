#include "ggml_bridge_internal.cuh"

#include "../../common/warp.cuh"

#include <cmath>
#include <stdexcept>

namespace ninfer::ops::gguf::detail {
namespace {

__launch_bounds__(256) __global__ void rmsnorm_q8_1_vector_kernel(
    const __nv_bfloat162* x, const __nv_bfloat162* weight, std::int8_t* qs, half2* ds,
    float eps, bool unit_offset) {
    constexpr int kWidth = 5120;
    constexpr int kPairs = kWidth / 2;
    constexpr int kPairsPerThread = kPairs / 256;
    constexpr int kBlockValuesPerThread = kWidth / 256;
    const int tid = static_cast<int>(threadIdx.x);
    const int column = static_cast<int>(blockIdx.x);
    const std::int64_t pair_base = std::int64_t(column) * kPairs;
    __nv_bfloat162 values[kPairsPerThread];
    float sum = 0.0f;
#pragma unroll
    for (int j = 0; j < kPairsPerThread; ++j) {
        const int pair = tid + j * 256;
        values[j] = x[pair_base + pair];
        const float2 v = __bfloat1622float2(values[j]);
        sum += v.x * v.x + v.y * v.y;
    }
    __shared__ float warp_sums[8];
    __shared__ float inv_shared;
    __shared__ __nv_bfloat162 stage[kPairs];
    const float norm_sum = ::ninfer::ops::block_reduce_sum<256>(sum, warp_sums);
    if (tid == 0) { inv_shared = rsqrtf(norm_sum / static_cast<float>(kWidth) + eps); }
    __syncthreads();
    const float inv = inv_shared;
#pragma unroll
    for (int j = 0; j < kPairsPerThread; ++j) {
        const int pair = tid + j * 256;
        const float2 xv = __bfloat1622float2(values[j]);
        const float2 wv = __bfloat1622float2(weight[pair]);
        const float gx = unit_offset ? wv.x + 1.0f : wv.x;
        const float gy = unit_offset ? wv.y + 1.0f : wv.y;
        stage[pair] = __floats2bfloat162_rn(xv.x * inv * gx, xv.y * inv * gy);
    }
    __syncthreads();
    const int lane = tid & 31;
    const std::int64_t qbase = std::int64_t(column) * kWidth;
    const std::int64_t dbase = std::int64_t(column) * (kWidth / QK8_1);
#pragma unroll
    for (int j = 0; j < kBlockValuesPerThread; ++j) {
        const int i = tid + j * 256;
        const float xi = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(stage)[i]);
        const float amax = ::warp_reduce_max<QK8_1>(fabsf(xi));
        const float block_sum = ::warp_reduce_sum<QK8_1>(xi);
        const float d = amax / 127.0f;
        qs[qbase + i] = amax == 0.0f ? 0 : static_cast<std::int8_t>(roundf(xi / d));
        if (lane == 0) { ds[dbase + i / QK8_1] = make_half2(d, block_sum); }
    }
}

template <mmq_q8_1_ds_layout ds_layout>
__launch_bounds__(256) __global__ void rmsnorm_q8_1_matrix_kernel(
    const __nv_bfloat162* x, const __nv_bfloat162* weight, block_q8_1_mmq* y, int columns,
    float eps, bool unit_offset) {
    constexpr int kWidth = 5120;
    constexpr int kPairs = kWidth / 2;
    constexpr int kPairsPerThread = kPairs / 256;
    constexpr int kPackThreads = 128;
    constexpr int kPackIterations = kWidth / (kPackThreads * 4);
    constexpr int vals_per_scale = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 64 : 32;
    constexpr int vals_per_sum = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 16 : 32;
    const int tid = static_cast<int>(threadIdx.x);
    const int column = static_cast<int>(blockIdx.x);
    const std::int64_t pair_base = std::int64_t(column) * kPairs;
    __nv_bfloat162 values[kPairsPerThread];
    float sum = 0.0f;
#pragma unroll
    for (int j = 0; j < kPairsPerThread; ++j) {
        const int pair = tid + j * 256;
        values[j] = x[pair_base + pair];
        const float2 v = __bfloat1622float2(values[j]);
        sum += v.x * v.x + v.y * v.y;
    }
    __shared__ float warp_sums[8];
    __shared__ float inv_shared;
    __shared__ __nv_bfloat162 stage[kPairs];
    const float norm_sum = ::ninfer::ops::block_reduce_sum<256>(sum, warp_sums);
    if (tid == 0) { inv_shared = rsqrtf(norm_sum / static_cast<float>(kWidth) + eps); }
    __syncthreads();
    const float inv = inv_shared;
#pragma unroll
    for (int j = 0; j < kPairsPerThread; ++j) {
        const int pair = tid + j * 256;
        const float2 xv = __bfloat1622float2(values[j]);
        const float2 wv = __bfloat1622float2(weight[pair]);
        const float gx = unit_offset ? wv.x + 1.0f : wv.x;
        const float gy = unit_offset ? wv.y + 1.0f : wv.y;
        stage[pair] = __floats2bfloat162_rn(xv.x * inv * gx, xv.y * inv * gy);
    }
    __syncthreads();
    if (tid >= kPackThreads) { return; }
    for (int iteration = 0; iteration < kPackIterations; ++iteration) {
        const std::int64_t i0 = (std::int64_t(iteration) * kPackThreads + tid) * 4;
        const std::int64_t k_block = i0 / QK8_1_MMQ;
        const std::int64_t iqs = i0 % QK8_1_MMQ;
        const float2 lo = __bfloat1622float2(stage[i0 / 2]);
        const float2 hi = __bfloat1622float2(stage[i0 / 2 + 1]);
        const float4 xi = make_float4(lo.x, lo.y, hi.x, hi.y);
        float amax = fabsf(xi.x);
        amax = fmaxf(amax, fabsf(xi.y));
        amax = fmaxf(amax, fabsf(xi.z));
        amax = fmaxf(amax, fabsf(xi.w));
#pragma unroll
        for (int offset = vals_per_scale / 8; offset > 0; offset >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
        }
        float block_sum = 0.0f;
        if (ds_layout != MMQ_Q8_1_DS_LAYOUT_D4) {
            block_sum = xi.x + xi.y + xi.z + xi.w;
#pragma unroll
            for (int offset = vals_per_sum / 8; offset > 0; offset >>= 1) {
                block_sum += __shfl_xor_sync(0xFFFFFFFF, block_sum, offset, WARP_SIZE);
            }
        }
        const float d_inv = 127.0f / amax;
        char4 q;
        q.x = roundf(xi.x * d_inv);
        q.y = roundf(xi.y * d_inv);
        q.z = roundf(xi.z * d_inv);
        q.w = roundf(xi.w * d_inv);
        const float d = 1.0f / d_inv;
        const std::int64_t ib = k_block * columns + column;
        reinterpret_cast<char4*>(y[ib].qs)[iqs / 4] = q;
        if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
            if (iqs % 16 == 0 && iqs < 96) {
                y[ib].d2s6[2 + iqs / 16] = block_sum;
                if (iqs % 64 == 0) { y[ib].d2s6[iqs / 64] = d; }
            }
        } else if (iqs % 32 == 0) {
            if (ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4) {
                y[ib].ds4[iqs / 32] = make_half2(d, block_sum);
            } else {
                y[ib].d4[iqs / 32] = d;
            }
        }
    }
    if (column == 0) {
        auto* slack = reinterpret_cast<uint4*>(y + std::int64_t(kWidth / QK8_1_MMQ) * columns);
        for (int i = tid; i < 128 * (sizeof(block_q8_1_mmq) / sizeof(uint4)); i += kPackThreads) {
            slack[i] = make_uint4(0, 0, 0, 0);
        }
    }
}

}

}

namespace ninfer::ops::gguf {

void rmsnorm_q8_1_vector_activation(const __nv_bfloat16* x, const __nv_bfloat16* weight, int k,
                                    int columns, float eps, bool unit_offset, void* out,
                                    cudaStream_t stream) {
    if (x == nullptr || weight == nullptr || out == nullptr || k != 5120 || columns <= 0 ||
        columns > 8 || !(eps > 0.0f) || !std::isfinite(eps) ||
        (reinterpret_cast<std::uintptr_t>(x) & (alignof(__nv_bfloat162) - 1)) != 0 ||
        (reinterpret_cast<std::uintptr_t>(weight) & (alignof(__nv_bfloat162) - 1)) != 0 ||
        (reinterpret_cast<std::uintptr_t>(out) & (alignof(half2) - 1)) != 0) {
        throw std::invalid_argument("gguf rmsnorm q8_1: unsupported operands");
    }
    auto* base = static_cast<std::uint8_t*>(out);
    detail::rmsnorm_q8_1_vector_kernel<<<static_cast<unsigned>(columns), 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat162*>(x), reinterpret_cast<const __nv_bfloat162*>(weight),
        reinterpret_cast<std::int8_t*>(base),
        reinterpret_cast<half2*>(base + detail::vec_scales_offset(k, columns)), eps, unit_offset);
    detail::check(cudaGetLastError(), "rmsnorm q8_1 launch");
}

void rmsnorm_q8_1_matrix_activation(const __nv_bfloat16* x, const __nv_bfloat16* weight, int k,
                                    int columns, float eps, bool unit_offset, GgmlType type,
                                    void* out, cudaStream_t stream) {
    if (x == nullptr || weight == nullptr || out == nullptr || k != 5120 || columns <= 8 ||
        columns > 1024 || !(eps > 0.0f) || !std::isfinite(eps) ||
        (reinterpret_cast<std::uintptr_t>(x) & (alignof(__nv_bfloat162) - 1)) != 0 ||
        (reinterpret_cast<std::uintptr_t>(weight) & (alignof(__nv_bfloat162) - 1)) != 0 ||
        (reinterpret_cast<std::uintptr_t>(out) & (alignof(uint4) - 1)) != 0) {
        throw std::invalid_argument("gguf rmsnorm matrix q8_1: unsupported operands");
    }
    if (!has_matrix_kernel(type)) {
        throw std::invalid_argument("gguf rmsnorm matrix q8_1: unsupported matrix type");
    }
    auto* y = static_cast<block_q8_1_mmq*>(out);
    const dim3 blocks(columns, 1, 1);
    switch (mmq_get_q8_1_ds_layout(detail::to_ggml(type))) {
    case MMQ_Q8_1_DS_LAYOUT_D4:
        detail::rmsnorm_q8_1_matrix_kernel<MMQ_Q8_1_DS_LAYOUT_D4>
            <<<blocks, 256, 0, stream>>>(reinterpret_cast<const __nv_bfloat162*>(x),
                                         reinterpret_cast<const __nv_bfloat162*>(weight), y,
                                         columns, eps, unit_offset);
        break;
    case MMQ_Q8_1_DS_LAYOUT_DS4:
        detail::rmsnorm_q8_1_matrix_kernel<MMQ_Q8_1_DS_LAYOUT_DS4>
            <<<blocks, 256, 0, stream>>>(reinterpret_cast<const __nv_bfloat162*>(x),
                                         reinterpret_cast<const __nv_bfloat162*>(weight), y,
                                         columns, eps, unit_offset);
        break;
    case MMQ_Q8_1_DS_LAYOUT_D2S6:
        detail::rmsnorm_q8_1_matrix_kernel<MMQ_Q8_1_DS_LAYOUT_D2S6>
            <<<blocks, 256, 0, stream>>>(reinterpret_cast<const __nv_bfloat162*>(x),
                                         reinterpret_cast<const __nv_bfloat162*>(weight), y,
                                         columns, eps, unit_offset);
        break;
    }
    detail::check(cudaGetLastError(), "rmsnorm matrix q8_1 launch");
}

}
