#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace ninfer::ops::gguf::detail {

struct MatrixActivationSnapshotSlot {
    int k;
    int columns;
    bool scheduled;
    unsigned* flag;
    unsigned* status;
    std::uint16_t* values;
};

inline std::vector<MatrixActivationSnapshotSlot> matrix_activation_snapshot_slots;
inline std::mutex matrix_activation_snapshot_mutex;
inline unsigned* matrix_activation_snapshot_armed = nullptr;
inline std::atomic<unsigned> matrix_activation_snapshot_host_armed{0};
inline bool matrix_activation_snapshot_initialized = false;

__global__ void matrix_activation_snapshot_prepare(unsigned* flag, unsigned* status,
                                                   const unsigned* armed) {
    if (threadIdx.x == 0) {
        *status = (*armed != 0 && atomicCAS(flag, 0u, 1u) == 0u) ? 4u : 0u;
    }
}

__global__ void matrix_activation_snapshot_validate(const __nv_bfloat16* x,
                                                    const std::int32_t* input_columns,
                                                    int k, int columns, unsigned* status) {
    if ((*status & 4u) == 0u) { return; }
    const auto* bits = reinterpret_cast<const std::uint16_t*>(x);
    const std::int64_t count = std::int64_t(k) * columns;
    unsigned flags = 0;
    for (std::int64_t i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += std::int64_t(blockDim.x) * gridDim.x) {
        const int row = int(i % k);
        const int column = int(i / k);
        const int source = input_columns == nullptr ? row : input_columns[row];
        if (source < 0 || source >= k) { flags |= 2u; continue; }
        const std::uint16_t value = bits[std::int64_t(column) * k + source];
        flags |= (value & 0x7fffu) != 0 ? 1u : 0u;
        flags |= (value & 0x7f80u) == 0x7f80u ? 2u : 0u;
    }
    if (flags != 0u) { atomicOr(status, flags); }
}

__global__ void matrix_activation_snapshot_copy(const __nv_bfloat16* x,
                                                const std::int32_t* input_columns,
                                                int k, int columns, unsigned* status,
                                                std::uint16_t* values) {
    if ((*status & 7u) != 5u) { return; }
    const auto* bits = reinterpret_cast<const std::uint16_t*>(x);
    const std::int64_t count = std::int64_t(k) * columns;
    for (std::int64_t i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += std::int64_t(blockDim.x) * gridDim.x) {
        const int row = int(i % k);
        const int column = int(i / k);
        const int source = input_columns == nullptr ? row : input_columns[row];
        values[i] = bits[std::int64_t(column) * k + source];
    }
}

__global__ void matrix_activation_snapshot_finish(unsigned* flag, const unsigned* status) {
    if (threadIdx.x == 0) { atomicExch(flag, (*status & 7u) == 5u ? 2u : 0u); }
}

inline void flush_matrix_activation_snapshots() {
    const char* folder = std::getenv("NINFER_MATRIX_ACTIVATION_SNAPSHOTS");
    if (!folder) { return; }
    std::filesystem::create_directories(folder);
    for (const auto& slot : matrix_activation_snapshot_slots) {
        unsigned flag = 0;
        if (cudaMemcpy(&flag, slot.flag, sizeof(flag), cudaMemcpyDeviceToHost) != cudaSuccess || flag != 2u) { continue; }
        const std::size_t count = std::size_t(slot.k) * slot.columns;
        std::vector<std::uint16_t> values(count);
        if (cudaMemcpy(values.data(), slot.values, count * sizeof(std::uint16_t), cudaMemcpyDeviceToHost) != cudaSuccess) { continue; }
        const std::string stem = std::string(folder) + "/bf16-k" + std::to_string(slot.k) + "-t" + std::to_string(slot.columns);
        if (FILE* file = std::fopen((stem + ".bf16").c_str(), "wb")) {
            std::fwrite(values.data(), sizeof(std::uint16_t), values.size(), file);
            std::fclose(file);
        }
        if (FILE* file = std::fopen((stem + ".json").c_str(), "wb")) {
            std::fprintf(file, "{\"format\":\"headerless-bf16-le\",\"version\":1,\"k\":%d,\"columns\":%d,\"order\":\"column-major-gathered\",\"elements\":%zu,\"bytes\":%zu,\"finite\":true,\"nonzero\":true}\n",
                         slot.k, slot.columns, count, count * sizeof(std::uint16_t));
            std::fclose(file);
        }
    }
}

inline void initialize_matrix_activation_snapshots(const std::string& folder) {
    if (matrix_activation_snapshot_initialized) { return; }
    int device = 0;
    check(cudaGetDevice(&device), "matrix snapshot device");
    check(cudaMalloc(&matrix_activation_snapshot_armed, sizeof(unsigned)), "matrix snapshot armed flag");
    check(cudaMemset(matrix_activation_snapshot_armed, 0, sizeof(unsigned)), "matrix snapshot disarm");
    std::filesystem::create_directories(folder);
    matrix_activation_snapshot_initialized = true;
    if (FILE* file = std::fopen((folder + "/ready").c_str(), "wb")) { std::fclose(file); }
    std::thread([folder, device] {
        while (!std::filesystem::exists(folder + "/arm")) { std::this_thread::sleep_for(std::chrono::milliseconds(20)); }
        if (cudaSetDevice(device) != cudaSuccess) { return; }
        const unsigned value = 1;
        if (cudaMemcpy(matrix_activation_snapshot_armed, &value, sizeof(value), cudaMemcpyHostToDevice) != cudaSuccess) { return; }
        matrix_activation_snapshot_host_armed.store(1, std::memory_order_release);
        if (FILE* file = std::fopen((folder + "/armed").c_str(), "wb")) { std::fclose(file); }
    }).detach();
}

inline void capture_matrix_activation_snapshot(const __nv_bfloat16* x, int k, int columns,
                                               const std::int32_t* input_columns,
                                               cudaStream_t stream) {
    if (k != 5120 && k != 17408) { return; }
    const char* configured_folder = std::getenv("NINFER_MATRIX_ACTIVATION_SNAPSHOTS");
    if (!configured_folder) { return; }
    cudaStreamCaptureStatus capture;
    check(cudaStreamIsCapturing(stream, &capture), "matrix snapshot capture status");
    if (capture != cudaStreamCaptureStatusNone) { return; }
    std::lock_guard<std::mutex> lock(matrix_activation_snapshot_mutex);
    const std::string folder(configured_folder);
    initialize_matrix_activation_snapshots(folder);
    if (columns != 1024) { return; }
    if (matrix_activation_snapshot_host_armed.load(std::memory_order_acquire) == 0) { return; }
    MatrixActivationSnapshotSlot* found = nullptr;
    for (auto& slot : matrix_activation_snapshot_slots) {
        if (slot.k == k && slot.columns == columns) { found = &slot; break; }
    }
    if (!found) {
        if (matrix_activation_snapshot_slots.empty()) { std::atexit(flush_matrix_activation_snapshots); }
        MatrixActivationSnapshotSlot slot{k, columns, false, nullptr, nullptr, nullptr};
        const std::size_t count = std::size_t(k) * columns;
        check(cudaMalloc(&slot.flag, sizeof(unsigned)), "matrix snapshot flag");
        check(cudaMemsetAsync(slot.flag, 0, sizeof(unsigned), stream), "matrix snapshot flag clear");
        check(cudaMalloc(&slot.status, sizeof(unsigned)), "matrix snapshot status");
        check(cudaMalloc(&slot.values, count * sizeof(std::uint16_t)), "matrix snapshot values");
        matrix_activation_snapshot_slots.push_back(slot);
        found = &matrix_activation_snapshot_slots.back();
    }
    if (found->scheduled) { return; }
    found->scheduled = true;
    matrix_activation_snapshot_prepare<<<1, 1, 0, stream>>>(found->flag, found->status, matrix_activation_snapshot_armed);
    const std::int64_t count = std::int64_t(k) * columns;
    const int blocks = int((count + 255) / 256);
    matrix_activation_snapshot_validate<<<blocks, 256, 0, stream>>>(x, input_columns, k, columns, found->status);
    matrix_activation_snapshot_copy<<<blocks, 256, 0, stream>>>(x, input_columns, k, columns, found->status, found->values);
    matrix_activation_snapshot_finish<<<1, 1, 0, stream>>>(found->flag, found->status);
    check(cudaGetLastError(), "matrix activation snapshot launch");
}

}
