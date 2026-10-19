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

}

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
    rmsnorm_q8_1_vector_kernel<<<static_cast<unsigned>(columns), 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat162*>(x), reinterpret_cast<const __nv_bfloat162*>(weight),
        reinterpret_cast<std::int8_t*>(base),
        reinterpret_cast<half2*>(base + vec_scales_offset(k, columns)), eps, unit_offset);
    check(cudaGetLastError(), "rmsnorm q8_1 launch");
}

}
