#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ops/linear/gguf/ggml_bridge.h"
#include "ops/linear/gguf/gguf_linear.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::ops::detail;
namespace gguf = ninfer::ops::gguf;

void check(cudaError_t status, const char* label) {
    if (status != cudaSuccess) { throw std::runtime_error(std::string(label) + ": " + cudaGetErrorString(status)); }
}

struct Case {
    QType qtype;
    gguf::GgmlType type;
    int scale_offset;
};

std::vector<std::uint8_t> make_weight(const Case& c, int rows, int k, std::uint32_t seed) {
    const auto block = gguf::block_shape(c.type);
    std::vector<std::uint8_t> bytes(std::size_t(rows) * (k / block.elements) * block.bytes);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> byte(0, 255);
    for (auto& value : bytes) { value = static_cast<std::uint8_t>(byte(rng)); }
    const std::uint16_t scale = 0x3980;
    for (std::size_t i = 0; i < bytes.size() / block.bytes; ++i) {
        std::memcpy(bytes.data() + i * block.bytes + c.scale_offset, &scale, sizeof(scale));
        if (c.qtype == QType::GGUF_Q4_K) {
            std::memcpy(bytes.data() + i * block.bytes + 2, &scale, sizeof(scale));
        }
    }
    return bytes;
}

bool run_case(const Case& c, bool accumulate, int k, int columns, int padding) {
    const int rows = 128 * gguf::multiprocessor_count();
    const int output_rows = rows + 2 * padding;
    const auto host_weight = make_weight(c, rows, k, 0xE190u + static_cast<unsigned>(c.qtype));
    DeviceBuffer weight(host_weight.size());
    weight.copy_from_host(host_weight.data(), host_weight.size());

    std::mt19937 rng(0xA119u + static_cast<unsigned>(c.qtype));
    std::uniform_real_distribution<float> sample(-0.25f, 0.25f);
    std::vector<__nv_bfloat16> host_input(std::size_t(k) * columns);
    std::vector<__nv_bfloat16> initial(std::size_t(output_rows) * columns);
    for (auto& value : host_input) { value = __float2bfloat16(sample(rng)); }
    for (auto& value : initial) { value = __float2bfloat16(sample(rng)); }
    DeviceBuffer input(host_input.size() * sizeof(__nv_bfloat16));
    input.copy_from_host(host_input.data(), input.bytes);
    DeviceBuffer output(initial.size() * sizeof(__nv_bfloat16));
    output.copy_from_host(initial.data(), output.bytes);

    DeviceBuffer activation(gguf::matrix_activation_bytes(k, columns));
    gguf::quantize_matrix_activation(c.type, static_cast<const __nv_bfloat16*>(input.p), k,
                                     columns, nullptr, activation.p, nullptr);
    DeviceBuffer reference(std::size_t(rows) * columns * sizeof(float));
    DeviceBuffer fixup(gguf::matrix_fixup_bytes(c.type, rows, columns));
    const auto block = gguf::block_shape(c.type);
    gguf::matrix_product(c.type, weight.p, std::int64_t(k / block.elements) * block.bytes,
                         rows, k, activation.p, columns, static_cast<float*>(reference.p), rows,
                         fixup.p, nullptr);

    Tensor x(input.p, DType::BF16, {k, columns});
    Tensor y(output.p, DType::BF16, {output_rows, columns});
    Weight w;
    w.qtype = c.qtype;
    w.layout = QuantLayout::GgufBlocks;
    w.qdata = weight.p;
    w.payload = weight.p;
    w.n = w.shape[0] = w.padded_shape[0] = rows;
    w.k = w.shape[1] = w.padded_shape[1] = k;
    w.ndim = 2;
    w.group_size = static_cast<std::uint32_t>(block.elements);
    w.group = block.elements;
    const GgufProduct product{&w, &y, padding,
                              accumulate ? GgufEpilogue::Accumulate : GgufEpilogue::Store};
    const GgufShape shape{c.qtype, rows, k};
    WorkspaceArena workspace(gguf_project_workspace_bytes({&shape, 1}, columns, columns) + (1U << 20));
    const auto admitted = [&](gguf::GgmlType type, int n, int t) {
        return gguf::matrix_product_bf16_store(type, weight.p,
            std::int64_t(k / block.elements) * block.bytes, n, k, activation.p, t,
            static_cast<__nv_bfloat16*>(output.p) + padding, output_rows, nullptr, accumulate);
    };
    if (admitted(gguf::GgmlType::Q8_0, rows, columns) ||
        admitted(c.type, rows - 1, columns) || admitted(c.type, rows, 8)) {
        throw std::runtime_error("GGUF MMQ BF16 store admitted an unsupported route");
    }
    bool used = false;
    gguf_project(x, {&product, 1}, workspace, nullptr, &used);
    check(cudaDeviceSynchronize(), "GGUF MMQ store synchronization");
#if defined(NINFER_GGUF_MMQ_STORE_FUSION)
    if (!used) { throw std::runtime_error("enabled MMQ BF16 store rejected its qualified geometry"); }
#else
    if (used) { throw std::runtime_error("disabled MMQ BF16 store was selected"); }
#endif

    std::vector<float> fp32(std::size_t(rows) * columns);
    std::vector<__nv_bfloat16> actual(initial.size());
    reference.copy_to_host(fp32.data(), reference.bytes);
    output.copy_to_host(actual.data(), output.bytes);
    for (int column = 0; column < columns; ++column) {
        for (int row = 0; row < output_rows; ++row) {
            const std::size_t at = std::size_t(column) * output_rows + row;
            const bool guard = row < padding || row >= padding + rows;
            float sum = __bfloat162float(initial[at]);
            if (!guard) {
                sum = fp32[std::size_t(column) * rows + row - padding];
                if (accumulate) { sum += __bfloat162float(initial[at]); }
            }
            std::uint16_t expected_bits = 0;
            std::uint16_t actual_bits = 0;
            const __nv_bfloat16 expected = __float2bfloat16(sum);
            std::memcpy(&expected_bits, &expected, sizeof(expected_bits));
            std::memcpy(&actual_bits, &actual[at], sizeof(actual_bits));
            if (expected_bits != actual_bits) { throw std::runtime_error("GGUF MMQ BF16 output mismatch"); }
        }
    }
    return true;
}

}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, 0) != cudaSuccess ||
        properties.major != 8 || properties.minor != 6) {
        std::cout << "SKIP: this qualification requires SM86\n";
        return 77;
    }
    try {
        const Case cases[] = {{QType::GGUF_Q4_K, gguf::GgmlType::Q4_K, 0},
                              {QType::GGUF_Q6_K, gguf::GgmlType::Q6_K, 208}};
        for (const Case& c : cases) {
            for (const int k : {512, 5120}) {
                for (const int columns : {9, 16, 17, 33, 65, 129}) {
                    for (const int padding : {0, 7}) {
                        run_case(c, false, k, columns, padding);
                        run_case(c, true, k, columns, padding);
                    }
                }
            }
        }
        std::cout << "OK GGUF MMQ BF16 store\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "GGUF MMQ BF16 store: " << error.what() << '\n';
        return 1;
    }
}
