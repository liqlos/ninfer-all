#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ops/linear/gguf/ggml_bridge.h"
#include "ops/linear/gguf/gguf_linear.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::ops;

void check(cudaError_t status, const char* label) {
    if (status != cudaSuccess) { throw std::runtime_error(std::string(label) + ": " + cudaGetErrorString(status)); }
}

struct Case {
    QType qtype;
    gguf::GgmlType type;
    int rows;
    int k;
    int scale_offset;
};

Case q4_case() { return {QType::GGUF_Q4_K, gguf::GgmlType::Q4_K, 6144, 5120, 0}; }
Case q6_case() { return {QType::GGUF_Q6_K, gguf::GgmlType::Q6_K, 5120, 17408, 208}; }

std::vector<std::uint8_t> make_weight(const Case& c, std::uint32_t seed) {
    const auto block = gguf::block_shape(c.type);
    const std::size_t blocks = std::size_t(c.rows) * (c.k / block.elements);
    std::vector<std::uint8_t> bytes(blocks * block.bytes);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> byte(0, 255);
    for (auto& value : bytes) { value = static_cast<std::uint8_t>(byte(rng)); }
    for (std::size_t i = 0; i < blocks; ++i) {
        const std::uint16_t scale = 0x3980;
        std::memcpy(bytes.data() + i * block.bytes + c.scale_offset, &scale, sizeof(scale));
        if (c.qtype == QType::GGUF_Q4_K) {
            std::memcpy(bytes.data() + i * block.bytes + 2, &scale, sizeof(scale));
        }
    }
    return bytes;
}

Weight make_gguf_weight(const Case& c, const void* data) {
    Weight weight;
    weight.qtype = c.qtype;
    weight.layout = QuantLayout::GgufBlocks;
    weight.qdata = data;
    weight.payload = data;
    weight.n = weight.shape[0] = weight.padded_shape[0] = c.rows;
    weight.k = weight.shape[1] = weight.padded_shape[1] = c.k;
    weight.ndim = 2;
    const auto block = gguf::block_shape(c.type);
    weight.group_size = static_cast<std::uint32_t>(block.elements);
    weight.group = block.elements;
    return weight;
}

std::vector<__nv_bfloat16> make_input(int k, int columns, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> value(-0.25f, 0.25f);
    std::vector<__nv_bfloat16> input(std::size_t(k) * columns);
    for (auto& element : input) { element = __float2bfloat16(value(rng)); }
    return input;
}

bool compare_rounding(const std::vector<float>& fp32, const std::vector<__nv_bfloat16>& bf16,
                      int rows, int columns) {
    for (std::size_t i = 0; i < std::size_t(rows) * columns; ++i) {
        const __nv_bfloat16 expected = __float2bfloat16(fp32[i]);
        std::uint16_t expected_bits = 0;
        std::uint16_t actual_bits = 0;
        std::memcpy(&expected_bits, &expected, sizeof(expected_bits));
        std::memcpy(&actual_bits, &bf16[i], sizeof(actual_bits));
        if (expected_bits != actual_bits) { return false; }
    }
    return true;
}

bool run_case(const Case& c, int columns, bool prepared, bool expect_fused) {
    const auto host_weight = make_weight(c, 0xE190u + c.k + columns);
    DeviceBuffer weight(host_weight.size());
    weight.copy_from_host(host_weight.data(), host_weight.size());

    const auto host_input = make_input(c.k, columns, 0xA119u + columns);
    DeviceBuffer input(host_input.size() * sizeof(__nv_bfloat16));
    input.copy_from_host(host_input.data(), input.bytes);

    const std::size_t activation_bytes = gguf::matrix_activation_bytes(c.k, columns);
    DeviceBuffer activation(activation_bytes);
    gguf::quantize_matrix_activation(c.type, static_cast<const __nv_bfloat16*>(input.p), c.k,
                                     columns, nullptr, activation.p, nullptr);

    gguf::MatrixBf16StorePlan plan;
    const bool plan_supported = gguf::matrix_bf16_store_plan(c.type, c.rows, columns, plan);
    if (expect_fused && (!plan_supported || plan.J != 128 || plan.I != 128 || !plan.stream_k)) {
        throw std::runtime_error("native MMQ store plan did not select the expected J/I tile");
    }
    if (!expect_fused && plan_supported) {
        throw std::runtime_error("unsupported width unexpectedly admitted BF16 MMQ store");
    }

    DeviceBuffer actual(std::size_t(c.rows) * columns * sizeof(__nv_bfloat16));
    DeviceBuffer fp32(std::size_t(c.rows) * columns * sizeof(float));
    const std::size_t fixup_bytes = gguf::matrix_fixup_bytes(c.type, c.rows, columns);
    DeviceBuffer fixup(fixup_bytes);
    gguf::matrix_product(c.type, weight.p, std::int64_t(c.k / 256) * gguf::block_shape(c.type).bytes,
                         c.rows, c.k, activation.p, columns, static_cast<float*>(fp32.p), c.rows,
                         fixup.p, nullptr);

    Tensor output_tensor(actual.p, DType::BF16, {c.rows, columns});
    Weight product_weight = make_gguf_weight(c, weight.p);
    GgufProduct product{&product_weight, &output_tensor, 0, GgufEpilogue::Store};
    bool fused = false;
    if (prepared) {
        const gguf::MatrixActivationPlane plane{c.type, activation.p};
        WorkspaceArena workspace(std::size_t(512) << 20);
        fused = detail::gguf_project_matrix_q8_1_activations(
            c.k, columns, {&plane, 1}, {&product, 1}, workspace, nullptr);
    } else {
        const GgufShape shape{c.qtype, c.rows, c.k};
        const std::size_t workspace_bytes = detail::gguf_project_workspace_bytes({&shape, 1}, columns, columns);
        WorkspaceArena workspace(workspace_bytes + (1U << 20));
        Tensor x(input.p, DType::BF16, {c.k, columns});
        fused = detail::gguf_project(x, {&product, 1}, workspace, nullptr);
    }
    check(cudaDeviceSynchronize(), "MMQ store test sync");
    if (fused != expect_fused) { throw std::runtime_error("GGUF projector reported the wrong MMQ store route"); }

    std::vector<float> host_fp32(std::size_t(c.rows) * columns);
    std::vector<__nv_bfloat16> host_bf16(std::size_t(c.rows) * columns);
    fp32.copy_to_host(host_fp32.data(), fp32.bytes);
    actual.copy_to_host(host_bf16.data(), actual.bytes);
    if (!compare_rounding(host_fp32, host_bf16, c.rows, columns)) {
        throw std::runtime_error("fused BF16 MMQ store differs from the native FP32 accumulator rounding");
    }
    return true;
}

}

int main() {
    try {
        int device = 0;
        check(cudaGetDevice(&device), "get device");
        cudaDeviceProp properties{};
        check(cudaGetDeviceProperties(&properties, device), "device properties");
        if (properties.major != 8 || properties.minor != 6) { return 77; }
        if (!run_case(q4_case(), 1024, false, true) ||
            !run_case(q4_case(), 1024, true, true) ||
            !run_case(q6_case(), 1024, false, true) ||
            !run_case(q4_case(), 130, false, false)) {
            return 1;
        }
        std::cout << "native MMQ BF16 store exactness and fallback checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
