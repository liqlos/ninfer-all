// The integer tensor-core product for one block type, in its own unit so the heavy template builds in
// parallel with the others.
#include "ggml_bridge_mmq.cuh"
#include "ggml_bridge_q6_mmq.h"
#include "q6_matrix_partition.h"

#include <limits>

NINFER_GGUF_MATRIX_INSTANCE(GGML_TYPE_Q6_K);

namespace ninfer::ops::gguf {

Q6MatrixPartition q6_matrix_partition(int rows, int columns, int maximum_rows) {
    const auto full = detail::select_matrix_launch<GGML_TYPE_Q6_K>(rows, columns);
    if (full.fixup || rows <= 0 || rows % full.I || maximum_rows < full.I) {
        throw std::invalid_argument("Q6 row partition cannot preserve the full MMQ reduction");
    }
    const auto tiles = rows / full.I;
    const auto maximum_tiles = maximum_rows / full.I;
    for (int parts = (tiles + maximum_tiles - 1) / maximum_tiles; parts <= tiles; ++parts) {
        const auto low_tiles = tiles / parts;
        const auto high_tiles = low_tiles + (tiles % parts != 0);
        if (high_tiles > maximum_tiles) { continue; }
        const auto low = detail::select_matrix_launch<GGML_TYPE_Q6_K>(low_tiles * full.I, columns);
        const auto high = detail::select_matrix_launch<GGML_TYPE_Q6_K>(high_tiles * full.I, columns);
        if (!low.fixup && !high.fixup && low.J == full.J && high.J == full.J &&
            low.I == full.I && high.I == full.I) { return {parts, full.I}; }
    }
    throw std::invalid_argument("Q6 bounded row partition preserving MMQ reduction is unavailable");
}

namespace {

std::uint64_t q6_encoded_bytes(std::uint64_t blocks, std::uint8_t width) {
    return blocks * 208 + (blocks * width + 7) / 8;
}

std::uint64_t q6_parent_blocks(const Weight& weight, std::uint32_t blocks_per_row) {
    if (weight.payload_bytes < 32) { throw std::invalid_argument("invalid Q6 fields payload size"); }
    const std::uint64_t encoded = weight.payload_bytes - 32;
    std::uint64_t low = 0;
    std::uint64_t high = encoded / 208;
    while (low < high) {
        const auto middle = low + (high - low + 1) / 2;
        if (q6_encoded_bytes(middle, weight.q6_fields.width) <= encoded) { low = middle; }
        else { high = middle - 1; }
    }
    if (q6_encoded_bytes(low, weight.q6_fields.width) != encoded || low % blocks_per_row != 0) {
        throw std::invalid_argument("invalid Q6 fields encoded extent");
    }
    return low;
}

template <int J>
void launch_q6_fields_matrix(const detail::MatrixLaunch& launch, const Weight& weight,
                             const void* activation, int columns, float* out,
                             std::int64_t out_column_stride, void* fixup, cudaStream_t stream,
                             const ggml_cuda_q6_fields_args& fields) {
    static std::once_flag raised[16];
    int device = 0;
    detail::check(cudaGetDevice(&device), "device");
    std::call_once(raised[device & 15], [&] {
        detail::check(cudaFuncSetAttribute(mul_mat_q_fields<GGML_TYPE_Q6_K, J, false>,
                                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                                           static_cast<int>(launch.shared)),
                      "Q6 fields matrix shared memory");
    });
    const uint3 blocks_fd = init_fastdiv_values(weight.k / ggml_cuda_type_traits<GGML_TYPE_Q6_K>::qk);
    const uint3 one_fd = init_fastdiv_values(1);
    const uint3 ntx_fd = init_fastdiv_values(launch.column_tiles);
    const dim3 block_dims(WARP_SIZE, launch.threads / WARP_SIZE, 1);
    const int stride_row_blocks = weight.k / 256;
    if (!launch.stream_k) {
        mul_mat_q_fields<GGML_TYPE_Q6_K, J, false>
            <<<dim3(launch.row_tiles, launch.column_tiles, 1), block_dims, launch.shared, stream>>>(
                static_cast<const char*>(weight.qdata), static_cast<const int*>(activation), nullptr,
                nullptr, out, nullptr, nullptr, blocks_fd, weight.n, columns, stride_row_blocks,
                columns, static_cast<int>(out_column_stride), one_fd, one_fd, 0, 0, 0, one_fd,
                one_fd, 0, 0, 0, ntx_fd, fields);
        detail::check(cudaGetLastError(), "Q6 fields matrix product launch");
        return;
    }
    if (launch.fixup && fixup == nullptr) {
        throw std::invalid_argument("Q6 fields matrix product needs a fixup plane");
    }
    mul_mat_q_fields<GGML_TYPE_Q6_K, J, false>
        <<<dim3(launch.blocks, 1, 1), block_dims, launch.shared, stream>>>(
            static_cast<const char*>(weight.qdata), static_cast<const int*>(activation), nullptr,
            nullptr, out, launch.fixup ? static_cast<float*>(fixup) : nullptr, nullptr, blocks_fd,
            weight.n, columns, stride_row_blocks, columns, static_cast<int>(out_column_stride),
            one_fd, one_fd, 0, 0, 0, one_fd, one_fd, 0, 0, 0, ntx_fd, fields);
    detail::check(cudaGetLastError(), "Q6 fields matrix product launch");
    if (!launch.fixup) { return; }
    const dim3 fixup_blocks(launch.blocks, launch.I / WARP_SIZE, 1);
    const dim3 fixup_dims(block_dims.x, block_dims.y / 2, block_dims.z);
    mul_mat_q_stream_k_fixup<GGML_TYPE_Q6_K, J, false><<<fixup_blocks, fixup_dims, 0, stream>>>(
        nullptr, nullptr, out, static_cast<float*>(fixup), blocks_fd, weight.n, columns,
        static_cast<int>(out_column_stride), one_fd, 0, one_fd, 0, ntx_fd);
    detail::check(cudaGetLastError(), "Q6 fields matrix fixup launch");
}

}

void matrix_q6_fields(const Weight& weight, const void* activation, int columns, float* out,
                      std::int64_t out_column_stride, void* fixup, cudaStream_t stream) {
    constexpr std::uint32_t header_bytes = 32;
    if (weight.layout != QuantLayout::GgufQ6Fields || weight.qtype != QType::GGUF_Q6_K ||
        weight.n <= 0 || weight.n % 128 || weight.k <= 0 || weight.k % 1024 || columns <= 0 ||
        !weight.qdata || !weight.scales || !activation || !out || out_column_stride < weight.n ||
        out_column_stride > std::numeric_limits<int>::max() ||
        weight.q6_fields.width < 10 || weight.q6_fields.width > 16 ||
        weight.q6_fields.exponent_bits > 5 || weight.q6_fields.exponent_base > 31 ||
        weight.q6_fields.sign_variable > 1 || weight.q6_fields.sign_value > 1 ||
        weight.payload_bytes < header_bytes) {
        throw std::invalid_argument("invalid Q6 fields matrix product");
    }
    const auto blocks_per_row = std::uint32_t(weight.k / 256);
    const auto parent_blocks = q6_parent_blocks(weight, blocks_per_row);
    const auto parent_rows = parent_blocks / blocks_per_row;
    if (weight.q6_field_row_begin + std::uint64_t(weight.n) > parent_rows ||
        (10 + weight.q6_fields.exponent_bits + weight.q6_fields.sign_variable) > weight.q6_fields.width) {
        throw std::invalid_argument("Q6 fields matrix view exceeds its encoded parent");
    }
    const auto d_bytes = (parent_blocks * weight.q6_fields.width + 7) / 8;
    if (d_bytes > std::numeric_limits<std::uint32_t>::max() ||
        weight.q6_field_row_begin > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("Q6 fields matrix metadata exceeds device indexing");
    }
    const ggml_cuda_q6_fields_args fields{
        static_cast<const unsigned char*>(weight.scales), blocks_per_row,
        static_cast<unsigned int>(weight.q6_field_row_begin), static_cast<unsigned int>(d_bytes),
        weight.q6_fields.width, weight.q6_fields.exponent_base, weight.q6_fields.exponent_bits,
        weight.q6_fields.sign_variable, weight.q6_fields.sign_value};
    const auto launch = detail::select_matrix_launch<GGML_TYPE_Q6_K>(weight.n, columns);
    if (weight.n % launch.I != 0) { throw std::invalid_argument("Q6 fields matrix rows must fill MMQ tiles"); }
    switch (launch.J) {
    case 16: launch_q6_fields_matrix<16>(launch, weight, activation, columns, out, out_column_stride, fixup, stream, fields); return;
    case 32: launch_q6_fields_matrix<32>(launch, weight, activation, columns, out, out_column_stride, fixup, stream, fields); return;
    case 64: launch_q6_fields_matrix<64>(launch, weight, activation, columns, out, out_column_stride, fixup, stream, fields); return;
    case 128: launch_q6_fields_matrix<128>(launch, weight, activation, columns, out, out_column_stride, fixup, stream, fields); return;
    default: throw std::invalid_argument("Q6 fields matrix product has unsupported column tile");
    }
}

}
