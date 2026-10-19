#include "ops/linear/gguf/ggml_bridge_mmq.cuh"
#include "probe_api.h"

#include <stdexcept>

namespace ffn_pair_probe {

constexpr int kRows = 17408;
constexpr int kK = 5120;

std::size_t matrix_fixup_bytes(int columns) {
    return ninfer::ops::gguf::detail::matrix_fixup_bytes_impl<GGML_TYPE_Q4_K>(kRows, columns);
}

void matrix_product(const void* weight, const void* activation, float* output,
                    void* fixup, cudaStream_t stream) {
    ninfer::ops::gguf::detail::matrix_product_impl<GGML_TYPE_Q4_K>(
        weight, 20 * 144, kRows, kK, activation, 1280, output, kRows, fixup, stream);
}

__global__ void gate_product_store_plane(const float* up, const float* gate,
                                         __nv_bfloat16* output) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int column = blockIdx.y;
    if (row >= kRows) { return; }
    const ninfer::ops::gguf::detail::VecStore destination{
        output, nullptr, kRows, ninfer::ops::gguf::Epilogue::GateProduct, gate, kRows};
    ninfer::ops::gguf::detail::vec_store(
        destination, row, column, up[std::int64_t(column) * kRows + row]);
}

void store_plane_gate_product(const float* up, const float* gate,
                              __nv_bfloat16* output, cudaStream_t stream) {
    gate_product_store_plane<<<dim3((kRows + 255) / 256, 1280, 1), 256, 0, stream>>>(
        up, gate, output);
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(status)); }
}

}

template std::size_t ninfer::ops::gguf::detail::matrix_fixup_bytes_impl<GGML_TYPE_Q4_K>(int, int);
template void ninfer::ops::gguf::detail::matrix_product_impl<GGML_TYPE_Q4_K>(
    const void*, std::int64_t, int, int, const void*, int, float*, std::int64_t, void*,
    cudaStream_t);
