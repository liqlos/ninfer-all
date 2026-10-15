#include "ggml_bridge_mmq.cuh"
#include "ggml_bridge_dual_mmq.h"

#include <mutex>
#include <stdexcept>

using namespace ninfer::ops::gguf::detail;

namespace ninfer::ops::gguf {
namespace {

constexpr int kRows = 6144;
constexpr int kK = 5120;
constexpr int kMinColumns = 9;
constexpr int kMaxColumns = 1024;

template <int J>
__launch_bounds__(ggml_cuda_mmq_get_nthreads(GGML_TYPE_Q4_K, J, false), 1)
static __global__ void dual_q4_k_matrix_product_kernel(
    const char* __restrict__ weight_a, const char* __restrict__ weight_b,
    const int* __restrict__ activation, float* __restrict__ out_a,
    std::int64_t out_column_stride_a, float* __restrict__ out_b,
    std::int64_t out_column_stride_b, float* __restrict__ fixup_a,
    float* __restrict__ fixup_b, int rows, int columns, int stride_row_a,
    int stride_row_b, int blocks_per_k, int column_tiles, int row_tiles,
    int x_tile_ints,
    int blocks, bool stream_k, bool fixup) {
    constexpr ggml_type type = GGML_TYPE_Q4_K;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps = ggml_cuda_mmq_get_nthreads(type, J, false) / warp_size;
    constexpr int I = ggml_cuda_mmq_get_I(type, J, false);
    constexpr int qk = ggml_cuda_type_traits<GGML_TYPE_Q4_K>::qk;
    constexpr int blocks_per_iter =
        ggml_cuda_mmq_get_K_vram(type, J, false) / qk;
    constexpr auto load_tiles = ggml_cuda_mmq_get_load_tiles<type, J, false>();
    constexpr auto vec_dot = ggml_cuda_mmq_get_vec_dot<type, J, false>();
    constexpr auto write_back = ggml_cuda_mmq_get_write_back<type, J, false>();
    constexpr int activation_block_ints = sizeof(block_q8_1_mmq) / sizeof(int);

    extern __shared__ int shared[];
    int* ids = shared;
    int* tile_y = shared + J;
    constexpr int y_tile_ints = GGML_PAD(J * MMQ_TILE_Y_K, nwarps * warp_size);
    int* tile_x_a = tile_y + y_tile_ints;
    int* tile_x_b = tile_x_a + x_tile_ints;

    for (int j = threadIdx.y * warp_size + threadIdx.x; j < J;
         j += nwarps * warp_size) {
        ids[j] = j;
    }
    __syncthreads();

    auto compute_tile = [&](int it, int jt, int kb0_start, int kb0_stop,
                            bool write_fixup) {
        float sum_a[J * I / (nwarps * warp_size)] = {0.0f};
        float sum_b[J * I / (nwarps * warp_size)] = {0.0f};
        const int tile_x_max_i = rows - it * I - 1;
        const int tile_y_max_j = columns - jt * J - 1;
        const int offset_x_a = it * I * stride_row_a;
        const int offset_x_b = it * I * stride_row_b;
        for (int kb0 = kb0_start; kb0 < kb0_stop; kb0 += blocks_per_iter) {
            load_tiles(weight_a, tile_x_a, offset_x_a + kb0, tile_x_max_i,
                       stride_row_a);
            load_tiles(weight_b, tile_x_b, offset_x_b + kb0, tile_x_max_i,
                       stride_row_b);
            for (int half = 0; half < 2; ++half) {
                const int* y = activation + columns *
                    ((kb0 * qk / QK8_1_MMQ) * activation_block_ints + half * activation_block_ints) +
                    jt * J * activation_block_ints;
                for (int i = threadIdx.y * warp_size + threadIdx.x;
                     i < J * MMQ_TILE_Y_K; i += nwarps * warp_size) {
                    tile_y[i] = y[i];
                }
                __syncthreads();
                vec_dot(tile_x_a, tile_y, sum_a, half * MMQ_TILE_NE_K);
                vec_dot(tile_x_b, tile_y, sum_b, half * MMQ_TILE_NE_K);
                __syncthreads();
            }
        }
        if (write_fixup) {
            write_back(sum_a, ids, fixup_a + blockIdx.x * (J * I), nullptr,
                       I, I, J);
            write_back(sum_b, ids, fixup_b + blockIdx.x * (J * I), nullptr,
                       I, I, J);
            return;
        }
        const int dst_offset_a = jt * J * out_column_stride_a + it * I;
        const int dst_offset_b = jt * J * out_column_stride_b + it * I;
        write_back(sum_a, ids, out_a + dst_offset_a, nullptr,
                   out_column_stride_a, tile_x_max_i, tile_y_max_j);
        write_back(sum_b, ids, out_b + dst_offset_b, nullptr,
                   out_column_stride_b, tile_x_max_i, tile_y_max_j);
    };

    if (!stream_k) {
        const int it = blockIdx.x;
        const int jt = blockIdx.y;
        compute_tile(it, jt, 0, blocks_per_k, false);
        return;
    }

    int kbc = static_cast<int>(std::int64_t(blockIdx.x) *
                               (row_tiles * column_tiles * blocks_per_k) / blocks);
    int kbc_stop = static_cast<int>(std::int64_t(blockIdx.x + 1) *
                                    (row_tiles * column_tiles * blocks_per_k) / blocks);
    kbc -= (kbc % blocks_per_k) % blocks_per_iter;
    kbc_stop -= (kbc_stop % blocks_per_k) % blocks_per_iter;

    int kb0_start = kbc % blocks_per_k;
    int kb0_stop = min(blocks_per_k, kb0_start + kbc_stop - kbc);
    while (kbc < kbc_stop && kb0_stop == blocks_per_k) {
        const int tile = kbc / blocks_per_k;
        compute_tile(tile / column_tiles, tile % column_tiles,
                     kb0_start, kb0_stop, false);
        kbc += blocks_per_k;
        kbc -= kbc % blocks_per_k;
        kb0_start = 0;
        kb0_stop = min(blocks_per_k, kbc_stop - kbc);
    }
    if (kbc >= kbc_stop) { return; }
    const int tile = kbc / blocks_per_k;
    compute_tile(tile / column_tiles, tile % column_tiles,
                 kb0_start, kb0_stop, true);
}

std::size_t dual_shared_bytes(const ggml_cuda_mmq_config& config, int cc) {
    const std::size_t ids = config.J * sizeof(int);
    const std::size_t y = GGML_PAD(config.J * sizeof(block_q8_1_mmq),
                                   config.nthreads * sizeof(int));
    const std::size_t x = ggml_cuda_mmq_get_nbytes_shared_x(config, cc);
    return ids + y + 2 * x;
}

MatrixLaunch select_dual_matrix_launch(int rows, int columns) {
    const DeviceFacts& device = device_facts();
    MatrixLaunch best;
    int best_tiles = INT_MAX;
    for (const int J : kMatrixColumnTiles) {
        const ggml_cuda_mmq_config config =
            ggml_cuda_mmq_get_config(GGML_TYPE_Q4_K, J, false, device.cc);
        if (config.type == GGML_TYPE_COUNT ||
            dual_shared_bytes(config, device.cc) > device.shared_per_block_optin) {
            continue;
        }
        const int tiles = (columns + J - 1) / J;
        if (tiles < best_tiles) {
            best_tiles = tiles;
            best.J = J;
            best.I = config.I;
            best.threads = config.nthreads;
            best.shared = dual_shared_bytes(config, device.cc);
            best.stream_k = config.stream_k;
        }
        if (tiles == 1) { break; }
    }
    if (best.J == 0) { return best; }
    best.row_tiles = (rows + best.I - 1) / best.I;
    best.column_tiles = (columns + best.J - 1) / best.J;
    const int tiles = best.row_tiles * best.column_tiles;
    if (best.stream_k) {
        const int waves = (tiles + device.sm_count - 1) / device.sm_count;
        const int efficiency = 100 * tiles / (device.sm_count * waves);
        best.blocks = efficiency >= 90 ? tiles : device.sm_count;
        best.fixup = tiles % best.blocks != 0;
    } else {
        best.blocks = tiles;
    }
    return best;
}

template <int J>
void launch_dual(const MatrixLaunch& launch, const void* weight_a,
                 std::int64_t row_bytes_a, const void* weight_b,
                 std::int64_t row_bytes_b, int rows, int k,
                 const void* activation, int columns, float* out_a,
                 std::int64_t out_column_stride_a, float* out_b,
                 std::int64_t out_column_stride_b, void* fixup_a,
                 void* fixup_b, cudaStream_t stream) {
    int device = 0;
    check(cudaGetDevice(&device), "dual matrix device");
    static std::once_flag raised[16];
    std::call_once(raised[device & 15], [&] {
        check(cudaFuncSetAttribute(dual_q4_k_matrix_product_kernel<J>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   static_cast<int>(launch.shared)),
              "dual matrix shared memory");
    });

    constexpr int qk = ggml_cuda_type_traits<GGML_TYPE_Q4_K>::qk;
    const int blocks_per_k = k / qk;
    const int stride_row_a = static_cast<int>(row_bytes_a / sizeof(block_q4_K));
    const int stride_row_b = static_cast<int>(row_bytes_b / sizeof(block_q4_K));
    const dim3 block_dims(WARP_SIZE, launch.threads / WARP_SIZE, 1);
    const auto config = ggml_cuda_mmq_get_config(GGML_TYPE_Q4_K, J, false,
                                                  device_facts().cc);
    const int x_tile_ints = static_cast<int>(
        ggml_cuda_mmq_get_nbytes_shared_x(config, device_facts().cc) / sizeof(int));
    const std::size_t shared = launch.shared;
    const dim3 grid = launch.stream_k
        ? dim3(launch.blocks, 1, 1)
        : dim3(launch.row_tiles, launch.column_tiles, 1);
    dual_q4_k_matrix_product_kernel<J><<<grid, block_dims, shared, stream>>>(
        static_cast<const char*>(weight_a), static_cast<const char*>(weight_b),
        static_cast<const int*>(activation), out_a, out_column_stride_a,
        out_b, out_column_stride_b, static_cast<float*>(fixup_a),
        static_cast<float*>(fixup_b), rows, columns, stride_row_a, stride_row_b,
        blocks_per_k, launch.column_tiles, launch.row_tiles, x_tile_ints,
        launch.stream_k ? launch.blocks : launch.row_tiles * launch.column_tiles,
        launch.stream_k, launch.fixup);
    check(cudaGetLastError(), "dual matrix product launch");
    if (!launch.stream_k || !launch.fixup) { return; }

    const auto blocks_fd = init_fastdiv_values(blocks_per_k);
    const auto one_fd = init_fastdiv_values(1);
    const auto column_tiles_fd = init_fastdiv_values(launch.column_tiles);
    const dim3 fixup_blocks(launch.blocks, launch.I / WARP_SIZE, 1);
    const dim3 fixup_dims(block_dims.x, block_dims.y / 2, block_dims.z);
    mul_mat_q_stream_k_fixup<GGML_TYPE_Q4_K, J, false>
        <<<fixup_blocks, fixup_dims, 0, stream>>>(
            nullptr, nullptr, out_a, static_cast<float*>(fixup_a), blocks_fd,
            rows, columns, static_cast<int>(out_column_stride_a), one_fd, 0,
            one_fd, 0, column_tiles_fd);
    check(cudaGetLastError(), "dual matrix first fixup launch");
    mul_mat_q_stream_k_fixup<GGML_TYPE_Q4_K, J, false>
        <<<fixup_blocks, fixup_dims, 0, stream>>>(
            nullptr, nullptr, out_b, static_cast<float*>(fixup_b), blocks_fd,
            rows, columns, static_cast<int>(out_column_stride_b), one_fd, 0,
            one_fd, 0, column_tiles_fd);
    check(cudaGetLastError(), "dual matrix second fixup launch");
}

}

std::size_t dual_q4_k_matrix_fixup_bytes(int rows, int columns) {
    if (rows != kRows || columns < kMinColumns || columns > kMaxColumns) {
        throw std::invalid_argument("dual Q4_K matrix: unsupported geometry");
    }
    const MatrixLaunch launch = select_dual_matrix_launch(rows, columns);
    return launch.fixup
        ? std::size_t(launch.blocks) * launch.J * launch.I * sizeof(float)
        : 0;
}

bool dual_q4_k_matrix_supported(int rows, int columns) {
    if (rows != kRows || columns < kMinColumns || columns > kMaxColumns) { return false; }
    return select_dual_matrix_launch(rows, columns).J != 0;
}

bool dual_q4_k_matrix_product(const void* weight_a, std::int64_t row_bytes_a,
                              const void* weight_b, std::int64_t row_bytes_b,
                              int rows, int k, const void* activation, int columns,
                              float* out_a, std::int64_t out_column_stride_a,
                              float* out_b, std::int64_t out_column_stride_b,
                              void* fixup_a, void* fixup_b, cudaStream_t stream) {
    constexpr int qk = ggml_cuda_type_traits<GGML_TYPE_Q4_K>::qk;
    constexpr int block_bytes = ggml_cuda_type_traits<GGML_TYPE_Q4_K>::bs;
    if (rows != kRows || k != kK || columns < kMinColumns || columns > kMaxColumns ||
        weight_a == nullptr || weight_b == nullptr || activation == nullptr ||
        out_a == nullptr || out_b == nullptr || out_a == out_b ||
        (reinterpret_cast<std::uintptr_t>(weight_a) & (alignof(std::uint32_t) - 1)) != 0 ||
        (reinterpret_cast<std::uintptr_t>(weight_b) & (alignof(std::uint32_t) - 1)) != 0 ||
        (reinterpret_cast<std::uintptr_t>(activation) & (alignof(std::uint32_t) - 1)) != 0 ||
        k % qk != 0 || rows % 128 != 0 || row_bytes_a != row_bytes_b ||
        row_bytes_a % block_bytes != 0 ||
        row_bytes_a < std::int64_t(k / qk) * block_bytes ||
        out_column_stride_a < rows || out_column_stride_b < rows) {
        throw std::invalid_argument("dual Q4_K matrix: unsupported geometry");
    }
    const MatrixLaunch launch = select_dual_matrix_launch(rows, columns);
    if (launch.J == 0 || (launch.fixup && (fixup_a == nullptr || fixup_b == nullptr))) {
        return false;
    }
    switch (launch.J) {
    case 16:
        launch_dual<16>(launch, weight_a, row_bytes_a, weight_b, row_bytes_b,
                               rows, k, activation, columns, out_a, out_column_stride_a,
                               out_b, out_column_stride_b, fixup_a, fixup_b, stream);
        return true;
    case 32:
        launch_dual<32>(launch, weight_a, row_bytes_a, weight_b, row_bytes_b,
                               rows, k, activation, columns, out_a, out_column_stride_a,
                               out_b, out_column_stride_b, fixup_a, fixup_b, stream);
        return true;
    case 64:
        launch_dual<64>(launch, weight_a, row_bytes_a, weight_b, row_bytes_b,
                               rows, k, activation, columns, out_a, out_column_stride_a,
                               out_b, out_column_stride_b, fixup_a, fixup_b, stream);
        return true;
    case 128:
        launch_dual<128>(launch, weight_a, row_bytes_a, weight_b, row_bytes_b,
                                rows, k, activation, columns, out_a, out_column_stride_a,
                                out_b, out_column_stride_b, fixup_a, fixup_b, stream);
        return true;
    default:
        return false;
    }
}

}
