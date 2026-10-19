#include "ggml_bridge_mmq.cuh"
#include "ggml_bridge_q4_mmq_private.h"

#include <limits>
#include <mutex>
#include <stdexcept>

namespace ninfer::ops::gguf {
namespace {

template <int J>
void launch_q4_fields(const detail::MatrixLaunch& launch, const Weight& weight,
                      const void* activation, int columns, float* out,
                      std::int64_t out_column_stride, void* fixup, cudaStream_t stream,
                      const ggml_cuda_q4_fields_args& fields) {
    static std::once_flag raised[16];
    int device = 0;
    detail::check(cudaGetDevice(&device), "device");
    std::call_once(raised[device & 15], [&] {
        detail::check(cudaFuncSetAttribute(mul_mat_q4_fields<GGML_TYPE_Q4_K, J, false>,
                                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                                           static_cast<int>(launch.shared)),
                      "Q4 fields matrix shared memory");
    });
    const uint3 blocks_fd = init_fastdiv_values(weight.k / ggml_cuda_type_traits<GGML_TYPE_Q4_K>::qk);
    const uint3 one_fd = init_fastdiv_values(1);
    const uint3 ntx_fd = init_fastdiv_values(launch.column_tiles);
    const dim3 block_dims(WARP_SIZE, launch.threads / WARP_SIZE, 1);
    const int stride_row_blocks = weight.k / 256;
    if (!launch.stream_k) {
        mul_mat_q4_fields<GGML_TYPE_Q4_K, J, false>
            <<<dim3(launch.row_tiles, launch.column_tiles, 1), block_dims, launch.shared, stream>>>(
                static_cast<const char*>(weight.qdata), static_cast<const int*>(activation), nullptr,
                nullptr, out, nullptr, nullptr, blocks_fd, weight.n, columns, stride_row_blocks,
                columns, static_cast<int>(out_column_stride), one_fd, one_fd, 0, 0, 0, one_fd,
                one_fd, 0, 0, 0, ntx_fd, fields);
        detail::check(cudaGetLastError(), "Q4 fields matrix product launch");
        return;
    }
    if (launch.fixup && fixup == nullptr) {
        throw std::invalid_argument("Q4 fields matrix product needs a fixup plane");
    }
    mul_mat_q4_fields<GGML_TYPE_Q4_K, J, false>
        <<<dim3(launch.blocks, 1, 1), block_dims, launch.shared, stream>>>(
            static_cast<const char*>(weight.qdata), static_cast<const int*>(activation), nullptr,
            nullptr, out, launch.fixup ? static_cast<float*>(fixup) : nullptr, nullptr, blocks_fd,
            weight.n, columns, stride_row_blocks, columns, static_cast<int>(out_column_stride),
            one_fd, one_fd, 0, 0, 0, one_fd, one_fd, 0, 0, 0, ntx_fd, fields);
    detail::check(cudaGetLastError(), "Q4 fields matrix product launch");
    if (!launch.fixup) { return; }
    const dim3 fixup_blocks(launch.blocks, launch.I / WARP_SIZE, 1);
    const dim3 fixup_dims(block_dims.x, block_dims.y / 2, block_dims.z);
    mul_mat_q_stream_k_fixup<GGML_TYPE_Q4_K, J, false><<<fixup_blocks, fixup_dims, 0, stream>>>(
        nullptr, nullptr, out, static_cast<float*>(fixup), blocks_fd, weight.n, columns,
        static_cast<int>(out_column_stride), one_fd, 0, one_fd, 0, ntx_fd);
    detail::check(cudaGetLastError(), "Q4 fields matrix fixup launch");
}

}

void matrix_q4_fields_private(const Weight& weight, const void* activation, int columns,
                              float* out, std::int64_t out_column_stride, void* fixup,
                              cudaStream_t stream) {
    const std::uint64_t root_rows = weight.k == 5120 ? 16384 : weight.k == 17408 ? 5120 : 0;
    const auto& encoding = weight.q4_fields;
    if (weight.layout != QuantLayout::GgufQ4Fields || weight.qtype != QType::GGUF_Q4_K ||
        weight.n <= 0 || weight.k <= 0 || weight.k % 256 || columns <= 0 ||
        !weight.qdata || !weight.scales || !weight.q4_d || !weight.q4_dmin || !activation || !out ||
        out_column_stride < weight.n || out_column_stride > std::numeric_limits<int>::max() ||
        !root_rows || weight.q4_field_row_begin + std::uint64_t(weight.n) > root_rows ||
        encoding.d_exponent_base != 0 || encoding.d_exponent_bits != (weight.k == 5120 ? 2 : 3) ||
        encoding.d_width != (weight.k == 5120 ? 12 : 13) || encoding.dmin_width != 13 ||
        encoding.dmin_exponent_base != 2 || encoding.dmin_exponent_bits != 3 ||
        (weight.k == 5120 && weight.n % 128 != 0)) {
        throw std::invalid_argument("invalid private Q4 fields matrix product");
    }
    const ggml_cuda_q4_fields_args fields{
        static_cast<const unsigned char*>(weight.scales),
        static_cast<const unsigned char*>(weight.q4_d),
        static_cast<const unsigned char*>(weight.q4_dmin),
        static_cast<unsigned int>(weight.k / 256),
        static_cast<unsigned int>(weight.q4_field_row_begin),
        encoding.d_width, encoding.d_exponent_base, encoding.d_exponent_bits,
        encoding.dmin_width, encoding.dmin_exponent_base, encoding.dmin_exponent_bits};
    const auto launch = detail::select_matrix_launch<GGML_TYPE_Q4_K>(weight.n, columns);
    if (weight.n % launch.I != 0) {
        throw std::invalid_argument("private Q4 row range must fill MMQ tiles");
    }
    switch (launch.J) {
    case 16: launch_q4_fields<16>(launch, weight, activation, columns, out, out_column_stride, fixup, stream, fields); return;
    case 32: launch_q4_fields<32>(launch, weight, activation, columns, out, out_column_stride, fixup, stream, fields); return;
    case 64: launch_q4_fields<64>(launch, weight, activation, columns, out, out_column_stride, fixup, stream, fields); return;
    case 128: launch_q4_fields<128>(launch, weight, activation, columns, out, out_column_stride, fixup, stream, fields); return;
    default: throw std::invalid_argument("private Q4 MMQ column tile is unsupported");
    }
}

}
