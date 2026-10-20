#include "native_q4.cuh"
#include "probe_api.h"

namespace ninfer::ops::gguf::detail::q4tc {

#if NINFER_Q4_T4_CHAIN_PROBE

template <int R, int P>
__device__ __forceinline__ void pass(
    const VecArgs& a, int warp, int gid, int tig, int iteration,
    unsigned v0, unsigned v1, unsigned v2, unsigned v3,
    float scale0, float min0, float scale1, float min1, float (&partial)[4][4]) {
    constexpr unsigned mask = 0x0F0F0F0Fu;
    const int stripe = 32 * iteration + 8 * R + 2 * warp + P;
    const auto* x = reinterpret_cast<const unsigned*>(
        a.qs + (gid & 3) * 5120 + 32 * stripe);
    const unsigned b0 = x[tig];
    const unsigned b1 = x[tig + 4];
    int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    ninfer::ops::mma_s8(c0, c1, c2, c3,
                       (v0 >> (4 * P)) & mask, (v1 >> (4 * P)) & mask,
                       (v2 >> (4 * P)) & mask, (v3 >> (4 * P)) & mask, b0, b1);
    if ((tig >> 1) == P) {
        const int token = 2 * (tig & 1);
        const float2 d0 = __half22float2(a.ds[token * 160 + stripe]);
        const float2 d1 = __half22float2(a.ds[(token + 1) * 160 + stripe]);
        partial[R][0] += native_dot_integer(scale0, min0, d0.x, d0.y, c0);
        partial[R][1] += native_dot_integer(scale0, min0, d1.x, d1.y, c1);
        partial[R][2] += native_dot_integer(scale1, min1, d0.x, d0.y, c2);
        partial[R][3] += native_dot_integer(scale1, min1, d1.x, d1.y, c3);
    }
}

template <int R>
__device__ __forceinline__ void pair(
    const VecArgs& a, int row, int warp, int gid, int tig, int iteration,
    float (&partial)[4][4]) {
    const int block = R + 4 * iteration;
    const auto* row0 = a.weight + std::int64_t(row + gid) * a.row_bytes + block * 144;
    const auto* row1 = row0 + 8 * a.row_bytes;
    const auto* q0 = reinterpret_cast<const unsigned*>(row0 + 16 + 32 * warp);
    const auto* q1 = reinterpret_cast<const unsigned*>(row1 + 16 + 32 * warp);
    const unsigned v0 = q0[tig];
    const unsigned v1 = q1[tig];
    const unsigned v2 = q0[tig + 4];
    const unsigned v3 = q1[tig + 4];
    const int sub = 2 * warp + (tig >> 1);
    const float2 dm0 = __half22float2(*reinterpret_cast<const half2*>(row0));
    const float2 dm1 = __half22float2(*reinterpret_cast<const half2*>(row1));
    int sc0, m0, sc1, m1;
    vec::scale_min_k4(row0 + 4, sub, sc0, m0);
    vec::scale_min_k4(row1 + 4, sub, sc1, m1);
    const float scale0 = dm0.x * float(sc0);
    const float min0 = dm0.y * float(m0);
    const float scale1 = dm1.x * float(sc1);
    const float min1 = dm1.y * float(m1);
    pass<R, 0>(a, warp, gid, tig, iteration, v0, v1, v2, v3,
               scale0, min0, scale1, min1, partial);
    pass<R, 1>(a, warp, gid, tig, iteration, v0, v1, v2, v3,
               scale0, min0, scale1, min1, partial);
}

__launch_bounds__(128) __global__ void q4_t4_chain_kernel(VecArgs a) {
    __shared__ float terminal[8][64];
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int gid = lane >> 2;
    const int tig = lane & 3;
    const int p = tig >> 1;
    const int token = 2 * (tig & 1);
    const int row = 16 * blockIdx.x;
    float partial[4][4];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
#pragma unroll
        for (int c = 0; c < 4; ++c) partial[r][c] = 0.0f;
    }
#pragma unroll 1
    for (int iteration = 0; iteration < 5; ++iteration) {
        pair<0>(a, row, warp, gid, tig, iteration, partial);
        pair<1>(a, row, warp, gid, tig, iteration, partial);
        pair<2>(a, row, warp, gid, tig, iteration, partial);
        pair<3>(a, row, warp, gid, tig, iteration, partial);
    }
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        const int out = 4 * (gid + 8 * (c >> 1)) + token + (c & 1);
        const float xor16_0 = partial[0][c] + partial[2][c];
        const float xor16_1 = partial[1][c] + partial[3][c];
        const float xor8 = xor16_0 + xor16_1;
        terminal[2 * warp + p][out] = xor8;
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const int out = 4 * (gid + 8 * (c >> 1)) + token + (c & 1);
            const float xor4_0 = terminal[p][out] + terminal[p ^ 4][out];
            const float xor4_1 = terminal[p ^ 2][out] + terminal[p ^ 6][out];
            const float xor2 = xor4_0 + xor4_1;
            const float xor1 = xor2 + __shfl_xor_sync(0xFFFFFFFFu, xor2, 2);
            if (p == 0) {
                vec_store(a.out, row + gid + 8 * (c >> 1), token + (c & 1), xor1);
            }
        }
    }
}

#endif

}

bool q4tc_candidate_enabled() {
    return NINFER_Q4_T4_CHAIN_PROBE == 1;
}

bool q4tc_candidate_shape(int rows, int k, int columns) {
    return (rows == 6144 || rows == 17408 || rows == 34816) && k == 5120 && columns == 4;
}

bool q4tc_run(const Q4TCArgs& a, Q4TCRoute route, cudaStream_t stream) {
    using namespace ninfer::ops::gguf;
    using namespace ninfer::ops::gguf::detail;
    if (a.rows <= 0 || a.k != 5120 || a.columns != 4 || a.row_bytes != 2880 ||
        !a.weight || !a.qs || !a.ds || !a.output || a.output_column_stride < a.rows ||
        (reinterpret_cast<std::uintptr_t>(a.weight) & 15) ||
        (reinterpret_cast<std::uintptr_t>(a.qs) & 15) ||
        (reinterpret_cast<std::uintptr_t>(a.ds) & 3) ||
        (a.output_mode != Q4TCOutput::PlainFP32 && a.output_mode != Q4TCOutput::StoreBF16 &&
         a.output_mode != Q4TCOutput::AccumulateBF16)) return false;
    const bool fp32 = a.output_mode == Q4TCOutput::PlainFP32;
    if (reinterpret_cast<std::uintptr_t>(a.output) & (fp32 ? 3 : 1)) return false;
    VecArgs native{a.weight, a.row_bytes, a.rows, a.k, a.qs, static_cast<const half2*>(a.ds),
                   {fp32 ? nullptr : static_cast<__nv_bfloat16*>(a.output),
                    fp32 ? static_cast<float*>(a.output) : nullptr,
                    a.output_column_stride,
                    a.output_mode == Q4TCOutput::AccumulateBF16 ? Epilogue::Accumulate
                                                              : Epilogue::Store,
                    nullptr, 0}};
    if (route == Q4TCRoute::NativeW) {
        vec::launch<GGML_TYPE_Q4_K, 4, false, 5120>(native, stream);
        return true;
    }
#if NINFER_Q4_T4_CHAIN_PROBE
    if (route == Q4TCRoute::Chain && q4tc_candidate_shape(a.rows, a.k, a.columns) &&
        device_facts().cc == 860) {
        q4tc::q4_t4_chain_kernel<<<a.rows / 16, 128, 0, stream>>>(native);
        check(cudaGetLastError(), "Q4 T4 chain probe launch");
        return true;
    }
#endif
    return false;
}
