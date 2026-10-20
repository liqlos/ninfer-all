#include "ops/linear/gguf/gguf_linear.h"
#include "ops/linear/gguf/ggml_bridge.h"
#include "../q4_folded_ffn_probe/pack_api.h"

#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

#ifndef NINFER_LAB_FOLDED16_RUNTIME
#define NINFER_LAB_FOLDED16_RUNTIME 0
#endif

#if NINFER_LAB_FOLDED16_RUNTIME != 0 && NINFER_LAB_FOLDED16_RUNTIME != 1
#error NINFER_LAB_FOLDED16_RUNTIME_must_be_zero_or_one
#endif

#define NINFER_LAB_FFN_SYMBOL \
    "_ZN6ninfer3ops6detail22gguf_swiglu_linear_addERKNS_6TensorERKNS_6WeightEPS6_S7_RS2_RNS_11DeviceArenaEP11CUstream_st"

extern "C" bool ninfer_lab_real_ffn(
    const ninfer::Tensor&, const ninfer::Weight&, const ninfer::Weight*, const ninfer::Weight&,
    ninfer::Tensor&, ninfer::WorkspaceArena&, cudaStream_t) asm("__real_" NINFER_LAB_FFN_SYMBOL);

extern "C" bool ninfer_lab_wrap_ffn(
    const ninfer::Tensor&, const ninfer::Weight&, const ninfer::Weight*, const ninfer::Weight&,
    ninfer::Tensor&, ninfer::WorkspaceArena&, cudaStream_t) asm("__wrap_" NINFER_LAB_FFN_SYMBOL);

namespace {

struct HostAdmissions {
    const bool enabled = [] {
        const char* value = std::getenv("NINFER_LAB_FOLDED16_HOST_ADMISSIONS");
        return value != nullptr && std::strcmp(value, "1") == 0;
    }();
    std::atomic<std::uint64_t> q4_ds4{0};
    std::atomic<std::uint64_t> q6_d4{0};
    std::atomic<std::uint64_t> forwarded{0};

    ~HostAdmissions() {
        if (!enabled) { return; }
        std::fprintf(stderr,
            "{\"kind\":\"ninfer_lab_folded16_host_admissions\",\"arm\":\"%s\","
            "\"q4_ds4\":%llu,\"q6_d4\":%llu,\"forwarded\":%llu,"
            "\"counts_gpu_replays\":false}\n",
            NINFER_LAB_FOLDED16_RUNTIME ? "on" : "off",
            static_cast<unsigned long long>(q4_ds4.load(std::memory_order_relaxed)),
            static_cast<unsigned long long>(q6_d4.load(std::memory_order_relaxed)),
            static_cast<unsigned long long>(forwarded.load(std::memory_order_relaxed)));
    }
};

HostAdmissions host_admissions;

#if NINFER_LAB_FOLDED16_RUNTIME

namespace gguf = ninfer::ops::gguf;

constexpr int kHidden = 5120;
constexpr int kIntermediate = 17408;
constexpr int kTokens = 1280;
constexpr std::size_t kGateRowBytes = 2880;
constexpr std::size_t kGateHalfBytes = std::size_t(kIntermediate) * kGateRowBytes;
constexpr std::size_t kTensorBytes = std::size_t(kHidden) * kTokens * sizeof(__nv_bfloat16);
constexpr std::size_t kInputBytes = 7391232;
constexpr std::size_t kPackedBytes = 25085952;
constexpr std::size_t kAlignment = 256;

static_assert(kInputBytes % kAlignment == 0 && kPackedBytes % kAlignment == 0);

struct AddressRange {
    std::uintptr_t begin = 0;
    std::uintptr_t end = 0;
};

bool address_range(const void* pointer, std::size_t bytes, AddressRange& range) {
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (begin == 0 || (begin & 15U) != 0 ||
        bytes > std::numeric_limits<std::uintptr_t>::max() - begin) {
        return false;
    }
    range = {begin, begin + bytes};
    return true;
}

bool overlaps(const AddressRange& a, const AddressRange& b) {
    return a.begin < b.end && b.begin < a.end;
}

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

bool device_admits() {
    struct DeviceSupport {
        int device = -1;
        bool supported = false;
    };
    static thread_local DeviceSupport cached;
    int device = -1;
    check_cuda(cudaGetDevice(&device), "folded16 runtime device");
    if (device < 0 || device >= 16) { return false; }
    if (cached.device != device) {
        cudaDeviceProp properties{};
        check_cuda(cudaGetDeviceProperties(&properties, device), "folded16 runtime properties");
        cached.supported = properties.major == 8 && properties.minor == 6 &&
                           properties.sharedMemPerBlockOptin >= 57856;
        cached.device = device;
    }
    return cached.supported;
}

bool tensor_admits(const ninfer::Tensor& tensor) {
    return tensor.dtype == ninfer::DType::BF16 && tensor.ne[0] == kHidden &&
           tensor.ne[1] == kTokens && tensor.ne[2] == 1 && tensor.ne[3] == 1 &&
           tensor.is_contiguous();
}

bool try_folded(const ninfer::Tensor& x, const ninfer::Weight& gate,
                const ninfer::Weight* up, const ninfer::Weight& down,
                ninfer::Tensor& residual, ninfer::WorkspaceArena& workspace,
                cudaStream_t stream) {
    if (up != nullptr || gate.qtype != ninfer::QType::GGUF_Q4_K ||
        gate.layout != ninfer::QuantLayout::GgufBlocks || gate.n != 2 * kIntermediate ||
        gate.k != kHidden || gate.input_columns != nullptr ||
        (down.qtype != ninfer::QType::GGUF_Q4_K && down.qtype != ninfer::QType::GGUF_Q6_K) ||
        down.layout != ninfer::QuantLayout::GgufBlocks || down.n != kHidden ||
        down.k != kIntermediate || down.input_columns != nullptr ||
        !tensor_admits(x) || !tensor_admits(residual)) {
        return false;
    }
    const bool q6 = down.qtype == ninfer::QType::GGUF_Q6_K;
    const auto down_type = q6 ? gguf::GgmlType::Q6_K : gguf::GgmlType::Q4_K;
    const std::size_t down_row_bytes = q6 ? 14280 : 9792;
    std::array<AddressRange, 4> live{};
    if (!address_range(gate.qdata, 2 * kGateHalfBytes, live[0]) ||
        !address_range(down.qdata, std::size_t(kHidden) * down_row_bytes, live[1]) ||
        !address_range(x.data, kTensorBytes, live[2]) ||
        !address_range(residual.data, kTensorBytes, live[3]) ||
        overlaps(live[3], live[0]) || overlaps(live[3], live[1])) {
        return false;
    }
    if (!device_admits()) { return false; }
    gguf::MatrixBf16StorePlan plan;
    const int layout = gguf::matrix_activation_layout(down_type);
    if (gguf::matrix_activation_layout(gguf::GgmlType::Q4_K) != 1 ||
        layout != (q6 ? 0 : 1) ||
        !gguf::matrix_bf16_store_plan(down_type, kHidden, kTokens, plan) ||
        plan.J != 128 || plan.I != 128 || plan.stream_k ||
        gguf::matrix_fixup_bytes(down_type, kHidden, kTokens) != 0 ||
        gguf::matrix_activation_bytes(kHidden, kTokens) != kInputBytes ||
        gguf::matrix_activation_bytes(kIntermediate, kTokens) != kPackedBytes) {
        return false;
    }
    const auto base = reinterpret_cast<std::uintptr_t>(workspace.base());
    const auto used = workspace.used();
    const auto capacity = workspace.capacity();
    if (base == 0 || used > capacity ||
        used > std::numeric_limits<std::uintptr_t>::max() - base) {
        return false;
    }
    const auto padding = (kAlignment - ((base + used) & (kAlignment - 1))) & (kAlignment - 1);
    const auto available = capacity - used;
    constexpr auto required = kInputBytes + kPackedBytes;
    if (padding > available || required > available - padding ||
        padding > std::numeric_limits<std::uintptr_t>::max() - (base + used) ||
        required > std::numeric_limits<std::uintptr_t>::max() - (base + used + padding)) {
        return false;
    }
    auto scope = workspace.scope();
    void* input = workspace.alloc_bytes(kInputBytes, kAlignment).data;
    void* packed = workspace.alloc_bytes(kPackedBytes, kAlignment).data;
    AddressRange input_range, packed_range;
    if (!address_range(input, kInputBytes, input_range) ||
        !address_range(packed, kPackedBytes, packed_range) ||
        overlaps(input_range, packed_range)) {
        return false;
    }
    for (const auto& range : live) {
        if (overlaps(input_range, range) || overlaps(packed_range, range)) { return false; }
    }
    const void* up_data = static_cast<const std::byte*>(gate.qdata) + kGateHalfBytes;
    gguf::quantize_matrix_activation(gguf::GgmlType::Q4_K,
        static_cast<const __nv_bfloat16*>(x.data), kHidden, kTokens, nullptr, input, stream);
    if (!q4_folded_ffn_pack_run(gate.qdata, up_data, kGateRowBytes, input, kTokens, packed,
                              layout, nullptr, nullptr, nullptr, stream)) {
        throw std::logic_error("folded16 runtime producer declined after input pack");
    }
    if (!gguf::matrix_product_bf16_store(down_type, down.qdata, down_row_bytes, kHidden,
            kIntermediate, packed, kTokens, static_cast<__nv_bfloat16*>(residual.data),
            kHidden, stream, true)) {
        throw std::logic_error("folded16 runtime down declined after producer");
    }
    if (host_admissions.enabled) {
        (q6 ? host_admissions.q6_d4 : host_admissions.q4_ds4)
            .fetch_add(1, std::memory_order_relaxed);
    }
    return true;
}

#endif

}

extern "C" bool ninfer_lab_wrap_ffn(
    const ninfer::Tensor& x, const ninfer::Weight& gate, const ninfer::Weight* up,
    const ninfer::Weight& down, ninfer::Tensor& residual, ninfer::WorkspaceArena& workspace,
    cudaStream_t stream) {
#if NINFER_LAB_FOLDED16_RUNTIME
    if (try_folded(x, gate, up, down, residual, workspace, stream)) { return true; }
#endif
    if (host_admissions.enabled) {
        host_admissions.forwarded.fetch_add(1, std::memory_order_relaxed);
    }
    return ninfer_lab_real_ffn(x, gate, up, down, residual, workspace, stream);
}
