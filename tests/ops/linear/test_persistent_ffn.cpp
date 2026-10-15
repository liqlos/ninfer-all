#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"
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

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) { throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status)); }
}

struct Buffer {
    void* p = nullptr;
    std::size_t bytes = 0;
    explicit Buffer(std::size_t size) : bytes(size) { check(cudaMalloc(&p, size), "allocate"); }
    ~Buffer() { if (p != nullptr) { (void)cudaFree(p); } }
    void copy_from(const void* source, std::size_t size) { check(cudaMemcpy(p, source, size, cudaMemcpyHostToDevice), "copy to device"); }
    void copy_to(void* destination, std::size_t size) const { check(cudaMemcpy(destination, p, size, cudaMemcpyDeviceToHost), "copy from device"); }
};

Weight weight(QType type, const void* data, int rows, int k) {
    Weight w;
    w.qtype = type;
    w.layout = QuantLayout::GgufBlocks;
    w.qdata = data;
    w.payload = data;
    w.n = w.shape[0] = w.padded_shape[0] = rows;
    w.k = w.shape[1] = w.padded_shape[1] = k;
    w.ndim = 2;
    w.group = w.group_size = gguf_block_shape(type).elements;
    return w;
}

std::vector<std::uint8_t> packed(QType type, int rows, int k, std::mt19937& random) {
    const auto block = gguf_block_shape(type);
    const std::size_t row_bytes = std::size_t(k / block.elements) * block.bytes;
    std::vector<std::uint8_t> bytes(std::size_t(rows) * row_bytes);
    std::uniform_int_distribution<int> byte(0, 255);
    for (std::size_t i = 0; i < bytes.size(); ++i) { bytes[i] = static_cast<std::uint8_t>(byte(random)); }
    const std::uint16_t one = 0x3c00;
    if (type == QType::GGUF_Q4_K) {
        for (std::size_t at = 0; at < bytes.size(); at += block.bytes) {
            std::memcpy(bytes.data() + at, &one, sizeof(one));
            std::memcpy(bytes.data() + at + 2, &one, sizeof(one));
        }
    } else {
        for (std::size_t at = 0; at < bytes.size(); at += block.bytes) {
            std::memcpy(bytes.data() + at + 208, &one, sizeof(one));
        }
    }
    return bytes;
}

int run_persistent_ffn() {
    constexpr int k = 5120;
    constexpr int intermediate = 17408;
    constexpr int down_rows = 5120;
    std::mt19937 random(31821);
    auto gate_host = packed(QType::GGUF_Q4_K, intermediate * 2, k, random);
    auto down_host = packed(QType::GGUF_Q6_K, down_rows, intermediate, random);
    Buffer gate_data(gate_host.size()), down_data(down_host.size());
    gate_data.copy_from(gate_host.data(), gate_host.size());
    down_data.copy_from(down_host.data(), down_host.size());
    const Weight gate = weight(QType::GGUF_Q4_K, gate_data.p, intermediate * 2, k);
    const Weight down = weight(QType::GGUF_Q6_K, down_data.p, down_rows, intermediate);
    for (const int t : {1, 4}) {
        std::vector<__nv_bfloat16> x_host(std::size_t(k) * t), residual_host(std::size_t(down_rows) * t);
        for (std::size_t i = 0; i < x_host.size(); ++i) { x_host[i] = __float2bfloat16(float(int(i % 31) - 15) / 32.0f); }
        for (std::size_t i = 0; i < residual_host.size(); ++i) { residual_host[i] = __float2bfloat16(float(int(i % 17) - 8) / 16.0f); }
        Buffer x_data(x_host.size() * sizeof(__nv_bfloat16));
        Buffer expected(std::size_t(down_rows) * t * sizeof(__nv_bfloat16));
        Buffer actual(expected.bytes), activation(std::size_t(intermediate) * t * sizeof(__nv_bfloat16));
        x_data.copy_from(x_host.data(), x_data.bytes);
        expected.copy_from(residual_host.data(), expected.bytes);
        actual.copy_from(residual_host.data(), actual.bytes);
        Tensor x(x_data.p, DType::BF16, {k, t});
        Tensor expect(expected.p, DType::BF16, {down_rows, t});
        Tensor got(actual.p, DType::BF16, {down_rows, t});
        Tensor intermediate_bf16(activation.p, DType::BF16, {intermediate, t});
        const auto route = ops::detail::gguf_swiglu_linear_add_route(gate, nullptr, down, t);
        const bool cooperative = ops::gguf::persistent_swiglu_q8_add_available(t, nullptr);
        if (cooperative && route != ops::detail::GgufSwiGluRoute::PersistentFfn) {
            throw std::runtime_error("available cooperative route did not select persistent FFN");
        }
        if (!cooperative && route == ops::detail::GgufSwiGluRoute::PersistentFfn) {
            throw std::runtime_error("persistent route selected without cooperative launch support");
        }
        const auto bytes = ops::detail::gguf_swiglu_linear_add_workspace_capacity_bytes(gate, nullptr, down, t, t);
        if ((route == ops::detail::GgufSwiGluRoute::Fallback) != (bytes == 0)) {
            throw std::runtime_error("route and workspace admission disagree");
        }
        const auto baseline_bytes = std::max(
            ops::linear_swiglu_workspace_capacity_bytes(gate.qtype, gate.n, gate.k,
                                                        ops::LinearPolicy::A16Only, t, t),
            ops::linear_add_workspace_capacity_bytes(down.qtype, down.n, down.k,
                                                     ops::LinearPolicy::A16Only, t, t));
        WorkspaceArena workspace(std::max(bytes, baseline_bytes) + (1U << 20));
        ops::linear_swiglu(x, gate, intermediate_bf16, ops::LinearPolicy::A16Only, workspace, nullptr);
        ops::linear_add(intermediate_bf16, down, expect, ops::LinearPolicy::A16Only, workspace, nullptr);
        check(cudaDeviceSynchronize(), "reference completion");
        const bool selected = ops::detail::gguf_swiglu_linear_add(x, gate, nullptr, down, got, workspace, nullptr);
        if (selected != (route != ops::detail::GgufSwiGluRoute::Fallback)) {
            throw std::runtime_error("runtime route differs from route query");
        }
        if (!selected) {
            ops::linear_swiglu(x, gate, intermediate_bf16, ops::LinearPolicy::A16Only, workspace, nullptr);
            ops::linear_add(intermediate_bf16, down, got, ops::LinearPolicy::A16Only, workspace, nullptr);
        }
        check(cudaDeviceSynchronize(), "candidate completion");
        std::vector<__nv_bfloat16> expected_host(residual_host.size()), actual_host(residual_host.size());
        expected.copy_to(expected_host.data(), expected.bytes);
        actual.copy_to(actual_host.data(), actual.bytes);
        if (std::memcmp(expected_host.data(), actual_host.data(), expected.bytes) != 0) {
            throw std::runtime_error("candidate changed exact BF16 residual values");
        }
        if (route == ops::detail::GgufSwiGluRoute::PersistentFfn) {
            actual.copy_from(residual_host.data(), actual.bytes);
            cudaStream_t stream = nullptr;
            cudaGraph_t graph = nullptr;
            cudaGraphExec_t executable = nullptr;
            check(cudaStreamCreate(&stream), "create persistent graph stream");
            check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "begin persistent graph capture");
            const bool captured = ops::detail::gguf_swiglu_linear_add(x, gate, nullptr, down, got, workspace, stream);
            if (!captured) { throw std::runtime_error("persistent route was not selected during capture"); }
            check(cudaStreamEndCapture(stream, &graph), "end persistent graph capture");
            check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), "instantiate persistent graph");
            check(cudaGraphLaunch(executable, stream), "launch persistent graph");
            check(cudaStreamSynchronize(stream), "synchronize persistent graph");
            actual.copy_to(actual_host.data(), actual.bytes);
            if (std::memcmp(expected_host.data(), actual_host.data(), expected.bytes) != 0) {
                throw std::runtime_error("persistent graph replay changed exact BF16 residual values");
            }
            check(cudaGraphExecDestroy(executable), "destroy persistent graph executable");
            check(cudaGraphDestroy(graph), "destroy persistent graph");
            check(cudaStreamDestroy(stream), "destroy persistent graph stream");
        }
    }
    auto gathered = down;
    std::vector<std::int32_t> ids(intermediate);
    for (int i = 0; i < intermediate; ++i) { ids[i] = i; }
    Buffer id_data(ids.size() * sizeof(std::int32_t));
    id_data.copy_from(ids.data(), id_data.bytes);
    gathered.input_columns = static_cast<const std::int32_t*>(id_data.p);
    if (ops::detail::gguf_swiglu_linear_add_route(gate, nullptr, gathered, 1) !=
            ops::detail::GgufSwiGluRoute::Fallback ||
        ops::detail::gguf_swiglu_linear_add_workspace_capacity_bytes(gate, nullptr, gathered, 1, 1) != 0) {
        throw std::runtime_error("gathered down matrix must use the standard fallback");
    }
    std::cout << "persistent_ffn route=" << static_cast<int>(ops::detail::gguf_swiglu_linear_add_route(gate, nullptr, down, 1)) << "\n";
    return 0;
}

}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) { return 77; }
    try { return run_persistent_ffn(); }
    catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
