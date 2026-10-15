#include "ggml_bridge_internal.cuh"
#include "ggml_bridge_vec.cuh"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::gguf {
namespace {

template <ggml_type type, int T, int KS, int kWarps>
__launch_bounds__(kWarps * 32) __global__ void swiglu_q8_kernel(const detail::VecArgs p, std::int8_t* out_qs, half2* out_ds,
                                 int out_rows) {
    using D = detail::vec::Decoder<type>;
    constexpr int kBlockElems = D::kBlockElems;
    constexpr int kBlockBytes = D::kBlockBytes;
    constexpr int kPerBlock = kBlockElems / 32;
    __shared__ float rounded[T][32];
    __shared__ __align__(16) std::uint32_t table[D::kTableWords > 0 ? D::kTableWords : 4];
    if constexpr (D::kTableWords > 0) {
        const std::uint32_t* src = D::table();
        for (int i = threadIdx.x; i < D::kTableWords; i += kWarps * 32) { table[i] = src[i]; }
        __syncthreads();
    }
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int k = KS != 0 ? KS : p.k;
    const int slices = k / 32;
    const std::int64_t lane_offset = std::int64_t(lane / kPerBlock) * kBlockBytes;
    const int sub = lane % kPerBlock;
    const int group = blockIdx.x;
    for (int local_row = warp; local_row < 32; local_row += kWarps) {
        const int row = group * 32 + local_row;
        const std::uint8_t* gate = p.weight + std::int64_t(row) * p.row_bytes + lane_offset;
        const std::uint8_t* up = p.weight + std::int64_t(p.rows + row) * p.row_bytes + lane_offset;
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
                const float2 ds = __half22float2(p.ds[std::int64_t(j) * slices + s]);
                d[j] = ds.x;
                sum[j] = ds.y;
                const auto* q = reinterpret_cast<const int4*>(p.qs + std::int64_t(j) * k + 32 * s);
                const int4 v0 = q[0], v1 = q[1];
                a[j][0] = v0.x; a[j][1] = v0.y; a[j][2] = v0.z; a[j][3] = v0.w;
                a[j][4] = v1.x; a[j][5] = v1.y; a[j][6] = v1.z; a[j][7] = v1.w;
            }
            detail::vec::Slice decoded;
            D::decode(gate + std::int64_t(32 / kPerBlock) * it * kBlockBytes, sub, table, decoded);
#pragma unroll
            for (int j = 0; j < T; ++j) {
                gate_acc[j] += D::dot(decoded, a[j], d[j], sum[j]);
            }
            D::decode(up + std::int64_t(32 / kPerBlock) * it * kBlockBytes, sub, table, decoded);
#pragma unroll
            for (int j = 0; j < T; ++j) {
                up_acc[j] += D::dot(decoded, a[j], d[j], sum[j]);
            }
        };
        if constexpr (KS != 0) {
            static_assert(KS % 1024 == 0);
#pragma unroll
            for (int it = 0; it < KS / 1024; ++it) { slice(it); }
        } else {
            for (int it = 0; lane + 32 * it < slices; ++it) { slice(it); }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
#pragma unroll
            for (int j = 0; j < T; ++j) {
                gate_acc[j] += __shfl_xor_sync(0xFFFFFFFFu, gate_acc[j], o);
                up_acc[j] += __shfl_xor_sync(0xFFFFFFFFu, up_acc[j], o);
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
            const float amax = warp_reduce_max<QK8_1>(fabsf(xi));
            const float sum = warp_reduce_sum<QK8_1>(xi);
            const float d = amax / 127.0f;
            out_qs[std::int64_t(j) * out_rows + group * 32 + lane] =
                amax == 0.0f ? 0 : static_cast<std::int8_t>(roundf(xi / d));
            if (lane == 0) {
                out_ds[std::int64_t(j) * (out_rows / 32) + group] = make_half2(d, sum);
            }
        }
    }
}

template <ggml_type type, int T, int KS, int kWarps>
void launch(const detail::VecArgs& args, std::int8_t* out_qs, half2* out_ds, int out_rows,
            cudaStream_t stream) {
    if (args.rows % 32 != 0 || out_rows % 32 != 0) {
        throw std::invalid_argument("gguf fused swiglu q8: rows must be divisible by 32");
    }
    const int groups = out_rows / 32;
    swiglu_q8_kernel<type, T, KS, kWarps><<<groups, kWarps * 32, 0, stream>>>(args, out_qs,
                                                                                out_ds, out_rows);
    detail::check(cudaGetLastError(), "fused SwiGLU q8 launch");
}

template <ggml_type type, int kWarps>
void launch_columns(const detail::VecArgs& args, int columns, std::int8_t* out_qs, half2* out_ds,
                    int out_rows, cudaStream_t stream) {
    switch (columns) {
    case 1: return args.k == 5120 ? launch<type, 1, 5120, kWarps>(args, out_qs, out_ds, out_rows, stream)
                                  : launch<type, 1, 0, kWarps>(args, out_qs, out_ds, out_rows, stream);
    case 2: return args.k == 5120 ? launch<type, 2, 5120, kWarps>(args, out_qs, out_ds, out_rows, stream)
                                  : launch<type, 2, 0, kWarps>(args, out_qs, out_ds, out_rows, stream);
    case 3: return args.k == 5120 ? launch<type, 3, 5120, kWarps>(args, out_qs, out_ds, out_rows, stream)
                                  : launch<type, 3, 0, kWarps>(args, out_qs, out_ds, out_rows, stream);
    case 4: return args.k == 5120 ? launch<type, 4, 5120, kWarps>(args, out_qs, out_ds, out_rows, stream)
                                  : launch<type, 4, 0, kWarps>(args, out_qs, out_ds, out_rows, stream);
    case 5: return args.k == 5120 ? launch<type, 5, 5120, kWarps>(args, out_qs, out_ds, out_rows, stream)
                                  : launch<type, 5, 0, kWarps>(args, out_qs, out_ds, out_rows, stream);
    case 6: return args.k == 5120 ? launch<type, 6, 5120, kWarps>(args, out_qs, out_ds, out_rows, stream)
                                  : launch<type, 6, 0, kWarps>(args, out_qs, out_ds, out_rows, stream);
    case 7: return args.k == 5120 ? launch<type, 7, 5120, kWarps>(args, out_qs, out_ds, out_rows, stream)
                                  : launch<type, 7, 0, kWarps>(args, out_qs, out_ds, out_rows, stream);
    case 8: return args.k == 5120 ? launch<type, 8, 5120, kWarps>(args, out_qs, out_ds, out_rows, stream)
                                  : launch<type, 8, 0, kWarps>(args, out_qs, out_ds, out_rows, stream);
    default: throw std::invalid_argument("gguf fused swiglu q8: columns must be 1..8");
    }
}

template <ggml_type type>
void launch_policy(const detail::VecArgs& args, int columns, std::int8_t* out_qs, half2* out_ds,
                   int rows, cudaStream_t stream, int warps) {
    switch (warps) {
    case 4: return launch_columns<type, 4>(args, columns, out_qs, out_ds, rows, stream);
    case 8: return launch_columns<type, 8>(args, columns, out_qs, out_ds, rows, stream);
    case 16: return launch_columns<type, 16>(args, columns, out_qs, out_ds, rows, stream);
    default: throw std::invalid_argument("gguf fused swiglu q8: warps must be 4, 8 or 16");
    }
}

}

void vector_swiglu_q8(GgmlType type, const void* weight, std::int64_t row_bytes, int rows, int k,
                      const void* activation, int columns, void* out, cudaStream_t stream, int warps) {
    const BlockShape block = block_shape(type);
    if (rows <= 0 || rows % 32 != 0 || k <= 0 || k % 256 != 0 || columns < 1 || columns > 8 ||
        weight == nullptr || activation == nullptr || out == nullptr || block.elements <= 0 ||
        k % block.elements != 0 || row_bytes < std::int64_t(k / block.elements) * block.bytes ||
        row_bytes % block.bytes != 0) {
        throw std::invalid_argument("gguf fused swiglu q8: invalid geometry");
    }
    const auto* input = static_cast<const std::uint8_t*>(activation);
    const detail::VecArgs args{static_cast<const std::uint8_t*>(weight), row_bytes, rows, k,
                               reinterpret_cast<const std::int8_t*>(input),
                               reinterpret_cast<const half2*>(input + detail::vec_scales_offset(k, columns)),
                               {nullptr, nullptr, 0, Epilogue::Store, nullptr, 0}};
    auto* output = static_cast<std::uint8_t*>(out);
    auto* out_qs = reinterpret_cast<std::int8_t*>(output);
    auto* out_ds = reinterpret_cast<half2*>(output + detail::vec_scales_offset(rows, columns));
    switch (type) {
    case GgmlType::Q4_K: return launch_policy<GGML_TYPE_Q4_K>(args, columns, out_qs, out_ds, rows, stream, warps);
    case GgmlType::Q6_K: return launch_policy<GGML_TYPE_Q6_K>(args, columns, out_qs, out_ds, rows, stream, warps);
    default: throw std::invalid_argument("gguf fused swiglu q8: unsupported gate type");
    }
}

}
