#include <cupti.h>
#include <cuda.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <set>
#include <utility>
#include <vector>

namespace {
CUpti_SubscriberHandle subscriber{};
std::mutex mutex;
std::set<std::pair<int, int>> captured;
const char* folder = nullptr;
void* driver = nullptr;

struct Pending { bool ready = false; int k = 0, t = 0; CUdeviceptr source = 0; const int* columns = nullptr; cudaStream_t stream = nullptr; char symbol[512]{}; };
thread_local Pending pending;

void capture(const Pending& p) {
    const int k = p.k, t = p.t;
    const auto* columns = p.columns;
    std::lock_guard<std::mutex> lock(mutex);
    if (captured.count({k, t})) { return; }
    auto capture_status = reinterpret_cast<CUresult(*)(CUstream, CUstreamCaptureStatus*)>(dlsym(driver, "cuStreamIsCapturing"));
    auto sync = reinterpret_cast<CUresult(*)(CUstream)>(dlsym(driver, "cuStreamSynchronize"));
    auto copy = reinterpret_cast<CUresult(*)(void*, CUdeviceptr, size_t)>(dlsym(driver, "cuMemcpyDtoH_v2"));
    if (!capture_status || !sync || !copy) { return; }
    CUstreamCaptureStatus status{};
    if (capture_status(reinterpret_cast<CUstream>(p.stream), &status) != CUDA_SUCCESS || status != CU_STREAM_CAPTURE_STATUS_NONE) { return; }
    if (sync(reinterpret_cast<CUstream>(p.stream)) != CUDA_SUCCESS) { return; }
    const auto source = p.source;
    std::vector<unsigned short> values(size_t(k) * t);
    if (copy(values.data(), source, values.size() * sizeof(unsigned short)) != CUDA_SUCCESS) { return; }
    if (columns) {
        std::vector<int> indices(k);
        if (copy(indices.data(), reinterpret_cast<CUdeviceptr>(columns), size_t(k) * sizeof(int)) != CUDA_SUCCESS) { return; }
        auto gathered = values;
        for (int i = 0; i < k; ++i) {
            if (indices[i] < 0 || indices[i] >= k) { return; }
            for (int j = 0; j < t; ++j) { gathered[size_t(j) * k + i] = values[size_t(j) * k + indices[i]]; }
        }
        values.swap(gathered);
    }
    size_t nonzero = 0, nonfinite = 0;
    for (const auto value : values) {
        nonzero += (value & 0x7fff) != 0;
        nonfinite += (value & 0x7f80) == 0x7f80;
    }
    if (!nonzero || nonfinite) { return; }
    const auto stem = std::string(folder) + "/bf16-k" + std::to_string(k) + "-t" + std::to_string(t);
    if (FILE* file = std::fopen((stem + ".bin").c_str(), "wb")) {
        const auto wrote = std::fwrite(values.data(), sizeof(unsigned short), values.size(), file);
        std::fclose(file);
        if (wrote != values.size()) { return; }
    } else { return; }
    if (FILE* file = std::fopen((stem + ".json").c_str(), "wb")) {
        std::fprintf(file, "{\"K\":%d,\"T\":%d,\"bytes\":%zu,\"nonzero_values\":%zu,\"nonfinite_values\":%zu,\"startup_arm_required\":true,\"gather_columns\":%s,\"capture_status\":\"none\",\"symbol\":\"%s\"}\n", k, t, values.size() * sizeof(unsigned short), nonzero, nonfinite, columns ? "true" : "false", p.symbol);
        std::fclose(file);
    }
    captured.insert({k, t});
}

void CUPTIAPI callback(void*, CUpti_CallbackDomain domain, CUpti_CallbackId id, const void* data) {
    if (domain != CUPTI_CB_DOMAIN_RUNTIME_API || id != CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000) { return; }
    const auto* event = static_cast<const CUpti_CallbackData*>(data);
    if (event->callbackSite != CUPTI_API_ENTER || !event->symbolName || !std::strstr(event->symbolName, "quantize_matrix_kernel")) { return; }
    if (!std::filesystem::exists(std::string(folder) + "/arm")) { return; }
    const auto* params = static_cast<const cudaLaunchKernel_v7000_params*>(event->functionParams);
    if (!params || !params->args) { return; }
    const int k = *static_cast<const int*>(params->args[3]);
    const int t = *static_cast<const int*>(params->args[4]);
    if ((k != 5120 && k != 17408) || t < 9 || t > 1024) { return; }
    pending.ready = true;
    pending.k = k; pending.t = t;
    pending.source = *static_cast<const CUdeviceptr*>(params->args[0]);
    pending.columns = *static_cast<const int* const*>(params->args[1]);
    pending.stream = params->stream;
    std::snprintf(pending.symbol, sizeof(pending.symbol), "%s", event->symbolName);
}

__attribute__((constructor)) void initialize() {
    folder = std::getenv("NINFER_MATRIX_SNAPSHOTS");
    if (!folder || !*folder) { return; }
    driver = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!driver) { return; }
    const auto subscribe = cuptiSubscribe(&subscriber, callback, nullptr);
    const auto enable = subscribe == CUPTI_SUCCESS ? cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000) : subscribe;
    std::fprintf(stderr, "matrix_snapshot subscribe=%d enable=%d\n", int(subscribe), int(enable));
}
}

extern "C" cudaError_t cudaLaunchKernel(const void* function, dim3 grid, dim3 block, void** args, size_t shared, cudaStream_t stream) {
    using Launch = cudaError_t(*)(const void*, dim3, dim3, void**, size_t, cudaStream_t);
    static auto original = reinterpret_cast<Launch>(dlsym(RTLD_NEXT, "cudaLaunchKernel"));
    if (!original) { return cudaErrorUnknown; }
    const auto result = original(function, grid, block, args, shared, stream);
    if (result == cudaSuccess && pending.ready) {
        const auto saved = pending;
        pending.ready = false;
        capture(saved);
    }
    return result;
}
