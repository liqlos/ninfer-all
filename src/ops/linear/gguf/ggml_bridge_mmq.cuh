#pragma once

// Host side of ggml's integer tensor-core matrix kernel (mmq.cuh launch_mul_mat_q), without ggml's
// backend context: the caller owns the activation, the FP32 output and the stream-k fixup plane.

#if defined(NINFER_GGUF_Q4_MMQ_ASYNC_Y) || defined(NINFER_GGUF_Q4_MMQ_ASYNC_X) || defined(NINFER_GGUF_Q6_MMQ_ASYNC_Y)
#include "../../common/memory.cuh"
#endif

#include "ggml_bridge_internal.cuh"

#include <climits>
#include <limits>
#include <mutex>

namespace ninfer::ops::gguf::detail {

struct MatrixLaunch {
    int J                   = 0;
    int I                   = 0;
    int threads             = 0;
    std::size_t shared      = 0;
    int row_tiles           = 0;
    int column_tiles        = 0;
    bool stream_k           = false;
    int blocks              = 0;
    bool fixup              = false;
};

template <ggml_type type>
static MatrixLaunch select_matrix_launch(int rows, int columns) {
    const DeviceFacts& device = device_facts();
    MatrixLaunch best;
    int best_tiles = INT_MAX;
    for (const int J : kMatrixColumnTiles) {
#if defined(NINFER_GGUF_Q6_MMQ_J64_ALIGNED)
        if constexpr (type == GGML_TYPE_Q6_K) {
            if (device.cc == 860 && columns >= 128 && columns % 128 == 0 && J > 64) {
                continue;
            }
        }
#endif
#if defined(NINFER_GGUF_SM86_MMQ_TILE_CAP)
        if constexpr (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q6_K) {
            if (device.cc == 860 && columns >= 64 && J > NINFER_GGUF_SM86_MMQ_MAX_J) {
                continue;
            }
        }
#endif
        const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(type, J, false, device.cc);
        if (config.type == GGML_TYPE_COUNT) { continue; }
        const std::size_t shared = mmq_get_nbytes_shared(config, device.cc);
        if (shared > device.shared_per_block_optin) { continue; }
        const int tiles = (columns + J - 1) / J;
        if (tiles < best_tiles) {
            best_tiles        = tiles;
            best.J            = J;
            best.I            = config.I;
            best.threads      = config.nthreads;
            best.shared       = shared;
            best.stream_k     = config.stream_k;
        }
        if (tiles == 1) { break; }
    }
    if (best.J == 0) { throw std::invalid_argument("gguf matrix product: no tile fits this device"); }
    best.row_tiles    = (rows + best.I - 1) / best.I;
    best.column_tiles = (columns + best.J - 1) / best.J;
    const int tiles   = best.row_tiles * best.column_tiles;
    if (best.stream_k) {
        const int waves      = (tiles + device.sm_count - 1) / device.sm_count;
        const int efficiency = 100 * tiles / (device.sm_count * waves);
        best.blocks          = efficiency >= 90 ? tiles : device.sm_count;
        best.fixup           = tiles % best.blocks != 0;
    } else {
        best.blocks = tiles;
    }
    return best;
}

template <ggml_type type>
std::size_t matrix_fixup_bytes_impl(int rows, int columns) {
    const MatrixLaunch launch = select_matrix_launch<type>(rows, columns);
    return launch.fixup ? std::size_t(launch.blocks) * launch.J * launch.I * sizeof(float) : 0;
}

template <ggml_type type, int J, bool bf16_accumulate = false, bool bf16_gate_product = false>
void launch_matrix(const MatrixLaunch& launch, const char* weight, int stride_row_blocks, int rows,
                   int k, const int* activation, int columns, float* out,
                   __nv_bfloat16* out_bf16, std::int64_t out_column_stride, float* fixup,
                   cudaStream_t stream, const float* gate = nullptr,
                   std::int64_t gate_column_stride = 0) {
    static std::once_flag raised[16];
#if defined(NINFER_GGUF_MMQ_STORE_FUSION) || defined(NINFER_GGUF_MMQ_GATE_EPILOGUE_FUSION)
    static std::once_flag raised_bf16[16];
#endif
    int device = 0;
    check(cudaGetDevice(&device), "device");
    std::call_once(raised[device & 15], [&] {
        check(cudaFuncSetAttribute(mul_mat_q<type, J, false>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   static_cast<int>(launch.shared)),
              "matrix shared memory");
    });
#if defined(NINFER_GGUF_MMQ_STORE_FUSION) || defined(NINFER_GGUF_MMQ_GATE_EPILOGUE_FUSION)
    if (out_bf16 != nullptr) {
        std::call_once(raised_bf16[device & 15], [&] {
            check(cudaFuncSetAttribute(mul_mat_q<type, J, false, GGML_PREC_Q8, true,
                                                 bf16_accumulate, bf16_gate_product>,
                                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       static_cast<int>(launch.shared)),
                  "matrix BF16 shared memory");
        });
    }
#endif
    constexpr int qk        = ggml_cuda_type_traits<type>::qk;
    const uint3 blocks_fd   = init_fastdiv_values(k / qk);
    const uint3 one_fd      = init_fastdiv_values(1);
    const uint3 ntx_fd      = init_fastdiv_values(launch.column_tiles);
    const dim3 block_dims(WARP_SIZE, launch.threads / WARP_SIZE, 1);
    const int stride        = static_cast<int>(out_column_stride);
    if (!launch.stream_k) {
#if defined(NINFER_GGUF_MMQ_STORE_FUSION) || defined(NINFER_GGUF_MMQ_GATE_EPILOGUE_FUSION)
        if (out_bf16 != nullptr) {
            mul_mat_q<type, J, false, GGML_PREC_Q8, true, bf16_accumulate, bf16_gate_product>
                <<<dim3(launch.row_tiles, launch.column_tiles, 1), block_dims, launch.shared, stream>>>(
                    weight, activation, nullptr, nullptr, out, out_bf16, nullptr,
                    bf16_gate_product ? gate : nullptr, blocks_fd, rows,
                    columns, stride_row_blocks, columns, stride, one_fd, one_fd, 0, 0,
                    bf16_gate_product ? static_cast<int>(gate_column_stride) : 0, one_fd,
                    one_fd, 0, 0, 0, ntx_fd);
        } else {
            mul_mat_q<type, J, false>
                <<<dim3(launch.row_tiles, launch.column_tiles, 1), block_dims, launch.shared, stream>>>(
                    weight, activation, nullptr, nullptr, out, nullptr, nullptr, nullptr, blocks_fd, rows,
                    columns, stride_row_blocks, columns, stride, one_fd, one_fd, 0, 0, 0, one_fd,
                    one_fd, 0, 0, 0, ntx_fd);
        }
#else
        mul_mat_q<type, J, false>
            <<<dim3(launch.row_tiles, launch.column_tiles, 1), block_dims, launch.shared, stream>>>(
                weight, activation, nullptr, nullptr, out, nullptr, nullptr, nullptr, blocks_fd, rows,
                columns, stride_row_blocks, columns, stride, one_fd, one_fd, 0, 0, 0, one_fd,
                one_fd, 0, 0, 0, ntx_fd);
#endif
        check(cudaGetLastError(), "matrix product launch");
        return;
    }
    if (launch.fixup && fixup == nullptr) {
        throw std::invalid_argument("gguf matrix product: this launch needs a fixup plane");
    }
#if defined(NINFER_GGUF_MMQ_STORE_FUSION) || defined(NINFER_GGUF_MMQ_GATE_EPILOGUE_FUSION)
    if (out_bf16 != nullptr) {
        if (launch.fixup) { throw std::invalid_argument("gguf BF16 matrix product requires no fixup"); }
        mul_mat_q<type, J, false, GGML_PREC_Q8, true, bf16_accumulate, bf16_gate_product><<<dim3(launch.blocks, 1, 1), block_dims, launch.shared, stream>>>(
            weight, activation, nullptr, nullptr, out, out_bf16, nullptr,
            bf16_gate_product ? gate : nullptr,
            blocks_fd, rows, columns, stride_row_blocks, columns, stride, one_fd, one_fd, 0, 0,
            bf16_gate_product ? static_cast<int>(gate_column_stride) : 0,
            one_fd, one_fd, 0, 0, 0, ntx_fd);
    } else {
        mul_mat_q<type, J, false><<<dim3(launch.blocks, 1, 1), block_dims, launch.shared, stream>>>(
            weight, activation, nullptr, nullptr, out, nullptr, launch.fixup ? fixup : nullptr, nullptr,
            blocks_fd, rows, columns, stride_row_blocks, columns, stride, one_fd, one_fd, 0, 0, 0,
            one_fd, one_fd, 0, 0, 0, ntx_fd);
    }
#else
    mul_mat_q<type, J, false><<<dim3(launch.blocks, 1, 1), block_dims, launch.shared, stream>>>(
        weight, activation, nullptr, nullptr, out, nullptr, launch.fixup ? fixup : nullptr, nullptr,
        blocks_fd, rows, columns, stride_row_blocks, columns, stride, one_fd, one_fd, 0, 0, 0,
        one_fd, one_fd, 0, 0, 0, ntx_fd);
#endif
    check(cudaGetLastError(), "matrix product launch");
    if (!launch.fixup) { return; }
    const dim3 fixup_blocks(launch.blocks, launch.I / WARP_SIZE, 1);
    const dim3 fixup_dims(block_dims.x, block_dims.y / 2, 1);
    mul_mat_q_stream_k_fixup<type, J, false><<<fixup_blocks, fixup_dims, 0, stream>>>(
        nullptr, nullptr, out, fixup, blocks_fd, rows, columns, stride, one_fd, 0, one_fd, 0,
        ntx_fd);
    check(cudaGetLastError(), "matrix fixup launch");
}

template <ggml_type type>
void matrix_product_impl(const void* weight, std::int64_t row_bytes, int rows, int k,
                         const void* activation, int columns, float* out,
                         std::int64_t out_column_stride, void* fixup, cudaStream_t stream) {
    constexpr int qk = ggml_cuda_type_traits<type>::qk;
    constexpr int block_bytes = ggml_cuda_type_traits<type>::bs;
    if (k % 256 != 0 || rows % 128 != 0 || row_bytes % block_bytes != 0 ||
        row_bytes < std::int64_t(k / qk) * block_bytes) {
        throw std::invalid_argument("gguf matrix product: unsupported geometry");
    }
    const MatrixLaunch launch = select_matrix_launch<type>(rows, columns);
    const auto* x             = static_cast<const char*>(weight);
    const auto* y             = static_cast<const int*>(activation);
    auto* f                   = static_cast<float*>(fixup);
    const int stride_row      = static_cast<int>(row_bytes / block_bytes);
    switch (launch.J) {
    case 16:
        launch_matrix<type, 16>(launch, x, stride_row, rows, k, y, columns, out, nullptr, out_column_stride, f, stream);
        return;
    case 32:
        launch_matrix<type, 32>(launch, x, stride_row, rows, k, y, columns, out, nullptr, out_column_stride, f, stream);
        return;
    case 64:
        launch_matrix<type, 64>(launch, x, stride_row, rows, k, y, columns, out, nullptr, out_column_stride, f, stream);
        return;
    case 128:
        launch_matrix<type, 128>(launch, x, stride_row, rows, k, y, columns, out, nullptr, out_column_stride, f, stream);
        return;
    default:
        break;
    }
    throw std::invalid_argument("gguf matrix product: unexpected column tile");
}

template <ggml_type type>
bool matrix_bf16_store_plan_impl(int rows, int columns, int& J, int& I, bool& stream_k) {
    const DeviceFacts& device = device_facts();
    if (device.cc != 860 || rows <= 0 || columns <= kMaxVectorColumns || rows % 128 != 0) {
        return false;
    }
    const MatrixLaunch launch = select_matrix_launch<type>(rows, columns);
    J = launch.J;
    I = launch.I;
    stream_k = launch.stream_k;
    return !launch.fixup;
}

template <ggml_type type, bool accumulate>
bool launch_matrix_bf16_store(const MatrixLaunch& launch, const char* weight,
                              int stride_row, int rows, int k, const int* activation,
                              int columns, __nv_bfloat16* out,
                              std::int64_t out_column_stride, cudaStream_t stream) {
    switch (launch.J) {
    case 16:
        launch_matrix<type, 16, accumulate>(launch, weight, stride_row, rows, k, activation,
            columns, nullptr, out, out_column_stride, nullptr, stream);
        return true;
    case 32:
        launch_matrix<type, 32, accumulate>(launch, weight, stride_row, rows, k, activation,
            columns, nullptr, out, out_column_stride, nullptr, stream);
        return true;
    case 64:
        launch_matrix<type, 64, accumulate>(launch, weight, stride_row, rows, k, activation,
            columns, nullptr, out, out_column_stride, nullptr, stream);
        return true;
    case 128:
        launch_matrix<type, 128, accumulate>(launch, weight, stride_row, rows, k, activation,
            columns, nullptr, out, out_column_stride, nullptr, stream);
        return true;
    default:
        return false;
    }
}

template <ggml_type type>
bool launch_matrix_bf16_gate_store(const MatrixLaunch& launch, const char* weight,
                                   int stride_row, int rows, int k, const int* activation,
                                   int columns, __nv_bfloat16* out,
                                   std::int64_t out_column_stride, const float* gate,
                                   std::int64_t gate_column_stride, cudaStream_t stream) {
    switch (launch.J) {
    case 16:
        launch_matrix<type, 16, false, true>(launch, weight, stride_row, rows, k, activation,
            columns, nullptr, out, out_column_stride, nullptr, stream, gate, gate_column_stride);
        return true;
    case 32:
        launch_matrix<type, 32, false, true>(launch, weight, stride_row, rows, k, activation,
            columns, nullptr, out, out_column_stride, nullptr, stream, gate, gate_column_stride);
        return true;
    case 64:
        launch_matrix<type, 64, false, true>(launch, weight, stride_row, rows, k, activation,
            columns, nullptr, out, out_column_stride, nullptr, stream, gate, gate_column_stride);
        return true;
    case 128:
        launch_matrix<type, 128, false, true>(launch, weight, stride_row, rows, k, activation,
            columns, nullptr, out, out_column_stride, nullptr, stream, gate, gate_column_stride);
        return true;
    default:
        return false;
    }
}

template <ggml_type type>
bool matrix_product_bf16_store_impl(const void* weight, std::int64_t row_bytes, int rows, int k,
                                    const void* activation, int columns, __nv_bfloat16* out,
                                    std::int64_t out_column_stride, cudaStream_t stream,
                                    bool accumulate) {
    constexpr int qk = ggml_cuda_type_traits<type>::qk;
    constexpr int block_bytes = ggml_cuda_type_traits<type>::bs;
    if (weight == nullptr || activation == nullptr || out == nullptr || k <= 0 || k % 256 != 0 ||
        rows <= 0 || rows % 128 != 0 || columns <= kMaxVectorColumns ||
        out_column_stride < rows || row_bytes <= 0 || row_bytes % block_bytes != 0 ||
        row_bytes < std::int64_t(k / qk) * block_bytes) {
        return false;
    }
    int J = 0;
    int I = 0;
    bool stream_k = false;
    try {
        if (!matrix_bf16_store_plan_impl<type>(rows, columns, J, I, stream_k)) { return false; }
    } catch (const std::invalid_argument&) {
        return false;
    }
    const MatrixLaunch launch = select_matrix_launch<type>(rows, columns);
    if (launch.J != J || launch.I != I || launch.fixup) { return false; }
    const auto* x = static_cast<const char*>(weight);
    const auto* y = static_cast<const int*>(activation);
    const int stride_row = static_cast<int>(row_bytes / block_bytes);
    if (accumulate) {
        return launch_matrix_bf16_store<type, true>(launch, x, stride_row, rows, k, y, columns,
                                                   out, out_column_stride, stream);
    }
    return launch_matrix_bf16_store<type, false>(launch, x, stride_row, rows, k, y, columns,
                                                 out, out_column_stride, stream);
}

template <ggml_type type>
bool matrix_product_bf16_gate_store_impl(
    const void* weight, std::int64_t row_bytes, int rows, int k, const void* activation,
    int columns, __nv_bfloat16* out, std::int64_t out_column_stride, const float* gate,
    std::int64_t gate_column_stride, cudaStream_t stream) {
    constexpr int qk = ggml_cuda_type_traits<type>::qk;
    constexpr int block_bytes = ggml_cuda_type_traits<type>::bs;
    if (type != GGML_TYPE_Q4_K || weight == nullptr || activation == nullptr || out == nullptr ||
        gate == nullptr || k <= 0 || k % 256 != 0 || rows <= 0 || rows % 128 != 0 ||
        columns <= kMaxVectorColumns || columns > kMaxGateMatrixColumns || out_column_stride < rows ||
        out_column_stride > INT_MAX || gate_column_stride < rows || gate_column_stride > INT_MAX ||
        row_bytes / block_bytes > INT_MAX ||
        (reinterpret_cast<std::uintptr_t>(weight) & 15U) != 0U ||
        (reinterpret_cast<std::uintptr_t>(activation) & (alignof(std::uint32_t) - 1)) != 0U ||
        (reinterpret_cast<std::uintptr_t>(out) & (alignof(__nv_bfloat16) - 1)) != 0 ||
        (reinterpret_cast<std::uintptr_t>(gate) & (alignof(float) - 1)) != 0 ||
        row_bytes <= 0 || row_bytes % block_bytes != 0 ||
        row_bytes < std::int64_t(k / qk) * block_bytes) {
        return false;
    }
    const std::size_t out_elements =
        std::size_t(columns - 1) * static_cast<std::size_t>(out_column_stride) + rows;
    const std::size_t gate_elements =
        std::size_t(columns - 1) * static_cast<std::size_t>(gate_column_stride) + rows;
    if (out_elements > static_cast<std::size_t>(INT_MAX) ||
        gate_elements > static_cast<std::size_t>(INT_MAX)) {
        return false;
    }
    const std::uintptr_t out_begin = reinterpret_cast<std::uintptr_t>(out);
    const std::uintptr_t gate_begin = reinterpret_cast<std::uintptr_t>(gate);
    const std::size_t out_bytes = out_elements * sizeof(__nv_bfloat16);
    const std::size_t gate_bytes = gate_elements * sizeof(float);
    if (out_bytes > std::numeric_limits<std::uintptr_t>::max() - out_begin ||
        gate_bytes > std::numeric_limits<std::uintptr_t>::max() - gate_begin ||
        (out_begin < gate_begin + gate_bytes && gate_begin < out_begin + out_bytes)) {
        return false;
    }
    int J = 0;
    int I = 0;
    bool stream_k = false;
    try {
        if (!matrix_bf16_store_plan_impl<type>(rows, columns, J, I, stream_k)) { return false; }
    } catch (const std::invalid_argument&) {
        return false;
    }
    const MatrixLaunch launch = select_matrix_launch<type>(rows, columns);
    if (launch.J != J || launch.I != I || launch.fixup) { return false; }
    const auto* x = static_cast<const char*>(weight);
    const auto* y = static_cast<const int*>(activation);
    const int stride_row = static_cast<int>(row_bytes / block_bytes);
    return launch_matrix_bf16_gate_store<type>(launch, x, stride_row, rows, k, y, columns, out,
                                               out_column_stride, gate, gate_column_stride,
                                               stream);
}

} // namespace ninfer::ops::gguf::detail

#define NINFER_GGUF_MATRIX_INSTANCE(TYPE)                                                          \
    template std::size_t ninfer::ops::gguf::detail::matrix_fixup_bytes_impl<TYPE>(int, int);      \
    template void ninfer::ops::gguf::detail::matrix_product_impl<TYPE>(                           \
        const void*, std::int64_t, int, int, const void*, int, float*, std::int64_t, void*,        \
        cudaStream_t)

#define NINFER_GGUF_MMQ_BF16_STORE_INSTANCE(TYPE)                                                   \
    template bool ninfer::ops::gguf::detail::matrix_bf16_store_plan_impl<TYPE>(                    \
        int, int, int&, int&, bool&);                                                               \
    template bool ninfer::ops::gguf::detail::matrix_product_bf16_store_impl<TYPE>(                 \
        const void*, std::int64_t, int, int, const void*, int, __nv_bfloat16*, std::int64_t,       \
        cudaStream_t, bool)
