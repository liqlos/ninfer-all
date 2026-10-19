#include <cupti.h>
#include <cupti_activity.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unistd.h>
#include <time.h>

namespace {

std::FILE* output = nullptr;
std::mutex output_mutex;
std::atomic<bool> active{false};
CUpti_SubscriberHandle subscriber{};

uint64_t CUPTIAPI realtime_timestamp() {
    timespec sample{};
    clock_gettime(CLOCK_REALTIME, &sample);
    return static_cast<uint64_t>(sample.tv_sec) * 1000000000ull + sample.tv_nsec;
}

void clock_sample(const char* label) {
    uint64_t cupti_ns = 0;
    timespec before{}, after{};
    clock_gettime(CLOCK_REALTIME, &before);
    CUptiResult status = cuptiGetTimestamp(&cupti_ns);
    clock_gettime(CLOCK_REALTIME, &after);
    if (output != nullptr) {
        std::fprintf(output, "{\"type\":\"clock_sample\",\"label\":\"%s\",\"cupti_ns\":%llu,\"realtime_before_ns\":%llu,\"realtime_after_ns\":%llu,\"status\":%d}\n", label,
                     static_cast<unsigned long long>(cupti_ns),
                     static_cast<unsigned long long>(before.tv_sec) * 1000000000ull + before.tv_nsec,
                     static_cast<unsigned long long>(after.tv_sec) * 1000000000ull + after.tv_nsec, int(status));
    }
}

void CUPTIAPI launch_callback(void*, CUpti_CallbackDomain domain, CUpti_CallbackId id, const void* data) {
    if (domain != CUPTI_CB_DOMAIN_RUNTIME_API || id != CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000) { return; }
    const auto* callback = static_cast<const CUpti_CallbackData*>(data);
    if (callback->callbackSite != CUPTI_API_ENTER || callback->symbolName == nullptr || std::strstr(callback->symbolName, "3vec6kernel") == nullptr) { return; }
    const auto* params = static_cast<const cudaLaunchKernel_v7000_params*>(callback->functionParams);
    if (params == nullptr || params->args == nullptr || params->args[0] == nullptr) { return; }
    struct Geometry { uint64_t weight; int64_t row_bytes; int rows; int k; };
    Geometry geometry{};
    std::memcpy(&geometry, params->args[0], sizeof(geometry));
    std::lock_guard<std::mutex> lock(output_mutex);
    if (output != nullptr) {
        clock_sample("vec_launch");
        std::fputs("{\"type\":\"vec_geometry\",\"name\":", output);
        std::fputc('"', output); std::fputs(callback->symbolName, output); std::fputc('"', output);
        std::fprintf(output, ",\"correlation_id\":%u,\"rows\":%d,\"K\":%d,\"row_bytes\":%lld,\"weight_address\":%llu}\n", callback->correlationId,
                     geometry.rows, geometry.k, static_cast<long long>(geometry.row_bytes), static_cast<unsigned long long>(geometry.weight));
    }
}

void write_json_string(std::FILE* file, const char* value) {
    std::fputc('"', file);
    if (value != nullptr) {
        for (const unsigned char* p = reinterpret_cast<const unsigned char*>(value); *p; ++p) {
            switch (*p) {
            case '"': std::fputs("\\\"", file); break;
            case '\\': std::fputs("\\\\", file); break;
            case '\n': std::fputs("\\n", file); break;
            case '\r': std::fputs("\\r", file); break;
            case '\t': std::fputs("\\t", file); break;
            default:
                if (*p < 0x20) { std::fprintf(file, "\\u%04x", unsigned(*p)); }
                else { std::fputc(*p, file); }
            }
        }
    }
    std::fputc('"', file);
}

void CUPTIAPI request_buffer(uint8_t** buffer, size_t* size, size_t* max_records) {
    constexpr size_t capacity = 8 * 1024 * 1024;
    void* memory = nullptr;
    if (posix_memalign(&memory, 8, capacity) != 0) { memory = nullptr; }
    *buffer = static_cast<uint8_t*>(memory);
    *size = memory == nullptr ? 0 : capacity;
    *max_records = 0;
}

void CUPTIAPI complete_buffer(CUcontext, uint32_t, uint8_t* buffer, size_t, size_t valid_size) {
    if (buffer == nullptr) { return; }
    CUpti_Activity* record = nullptr;
    while (cuptiActivityGetNextRecord(buffer, valid_size, &record) == CUPTI_SUCCESS) {
        if (record->kind == CUPTI_ACTIVITY_KIND_MARKER) {
            const auto* marker = reinterpret_cast<const CUpti_ActivityMarker2*>(record);
            std::lock_guard<std::mutex> lock(output_mutex);
            if (output == nullptr) { continue; }
            std::fputs("{\"type\":\"marker\",\"name\":", output);
            write_json_string(output, marker->name);
            std::fputs(",\"domain\":", output);
            write_json_string(output, marker->domain);
            std::fprintf(output, ",\"timestamp_ns\":%llu,\"id\":%u,\"flags\":%u}\n",
                         static_cast<unsigned long long>(marker->timestamp), marker->id,
                         unsigned(marker->flags));
            continue;
        }
        if (record->kind == CUPTI_ACTIVITY_KIND_MARKER_DATA) {
            const auto* marker = reinterpret_cast<const CUpti_ActivityMarkerData*>(record);
            std::lock_guard<std::mutex> lock(output_mutex);
            if (output != nullptr) {
                std::fprintf(output, "{\"type\":\"marker_data\",\"id\":%u,\"category\":%u,\"color\":%u}\n",
                             marker->id, marker->category, marker->color);
            }
            continue;
        }
        if (record->kind != CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL &&
            record->kind != CUPTI_ACTIVITY_KIND_KERNEL) {
            continue;
        }
        const auto* kernel = reinterpret_cast<const CUpti_ActivityKernel11*>(record);
        std::lock_guard<std::mutex> lock(output_mutex);
        if (output == nullptr) { continue; }
        std::fputs("{\"type\":\"kernel\",\"name\":", output);
        write_json_string(output, kernel->name);
        std::fprintf(output,
                     ",\"start_ns\":%llu,\"end_ns\":%llu,\"duration_ns\":%llu,\"device\":%u,\"context\":%u,\"stream\":%u,\"correlation_id\":%u,\"grid\":[%d,%d,%d],\"block\":[%d,%d,%d],\"registers_per_thread\":%u,\"static_shared_bytes\":%d,\"dynamic_shared_bytes\":%d,\"local_bytes_per_thread\":%u,\"graph_id\":%u,\"graph_node_id\":%llu}\n",
                     static_cast<unsigned long long>(kernel->start),
                     static_cast<unsigned long long>(kernel->end),
                     static_cast<unsigned long long>(kernel->end >= kernel->start ? kernel->end - kernel->start : 0),
                     kernel->deviceId, kernel->contextId, kernel->streamId, kernel->correlationId,
                     kernel->gridX, kernel->gridY, kernel->gridZ,
                     kernel->blockX, kernel->blockY, kernel->blockZ,
                     unsigned(kernel->registersPerThread), kernel->staticSharedMemory,
                     kernel->dynamicSharedMemory, kernel->localMemoryPerThread,
                     kernel->graphId, static_cast<unsigned long long>(kernel->graphNodeId));
    }
    size_t dropped = 0;
    cuptiActivityGetNumDroppedRecords(nullptr, 0, &dropped);
    if (dropped != 0) {
        std::lock_guard<std::mutex> lock(output_mutex);
        if (output != nullptr) {
            std::fprintf(output, "{\"type\":\"dropped\",\"count\":%zu}\n", dropped);
        }
    }
    std::free(buffer);
}

void initialize() {
    const char* path = std::getenv("NINFER_CUPTI_TRACE");
    if (path == nullptr || *path == '\0') { return; }
    const char* executable_filter = std::getenv("NINFER_CUPTI_EXECUTABLE");
    if (executable_filter != nullptr) {
        char executable[4096]{};
        ssize_t size = readlink("/proc/self/exe", executable, sizeof(executable) - 1);
        if (size <= 0) { return; }
        const char* basename = std::strrchr(executable, '/');
        if (std::strcmp(basename == nullptr ? executable : basename + 1, executable_filter) != 0) { return; }
    }
    output = std::fopen(path, "a");
    if (output == nullptr) { return; }
    uint32_t version = 0;
    const CUptiResult version_status = cuptiGetVersion(&version);
    std::fprintf(output,
                 "{\"type\":\"trace_start\",\"pid\":%ld,\"header_api\":%u,\"cupti_api\":%u,\"cupti_version_status\":%d}\n",
                 static_cast<long>(getpid()), unsigned(CUPTI_API_VERSION), unsigned(version), int(version_status));
    if (version_status != CUPTI_SUCCESS) {
        std::fflush(output);
        return;
    }
    CUptiResult status = cuptiActivityRegisterTimestampCallback(realtime_timestamp);
    if (status == CUPTI_SUCCESS) { status = cuptiActivityRegisterCallbacks(request_buffer, complete_buffer); }
    if (status == CUPTI_SUCCESS) { status = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL); }
    if (status == CUPTI_SUCCESS) { status = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MARKER); }
    if (status == CUPTI_SUCCESS) { status = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MARKER_DATA); }
    if (status != CUPTI_SUCCESS) {
        std::fprintf(output, "{\"type\":\"trace_error\",\"operation\":\"enable_kernel_activity\",\"status\":%d}\n", int(status));
        std::fflush(output);
        return;
    }
    CUptiResult callback_status = cuptiSubscribe(&subscriber, launch_callback, nullptr);
    if (callback_status == CUPTI_SUCCESS) {
        callback_status = cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000);
    }
    std::fprintf(output, "{\"type\":\"callback_status\",\"status\":%d}\n", int(callback_status));
    active.store(true, std::memory_order_release);
    clock_sample("initialize");
    std::fflush(output);
}

void shutdown() {
    if (active.exchange(false, std::memory_order_acq_rel)) {
        clock_sample("shutdown");
        cuptiActivityFlushAll(0);
        cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
        cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MARKER);
        cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MARKER_DATA);
        cuptiUnsubscribe(subscriber);
    }
    std::lock_guard<std::mutex> lock(output_mutex);
    if (output != nullptr) {
        std::fputs("{\"type\":\"trace_end\"}\n", output);
        std::fflush(output);
        std::fclose(output);
        output = nullptr;
    }
}

}

__attribute__((constructor)) static void ninfer_cupti_initialize() { initialize(); }
__attribute__((destructor)) static void ninfer_cupti_shutdown() { shutdown(); }
