#pragma once

#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <thread>
#include <chrono>
#include <vector>

namespace ninfer::ops::gguf::detail {

struct VectorSnapshotSlot {
    int type;
    int k;
    int columns;
    int fused;
    unsigned* flag;
    std::uint8_t* bytes;
};

inline std::vector<VectorSnapshotSlot> snapshot_slots;
inline std::mutex snapshot_mutex;
inline unsigned* snapshot_armed = nullptr;

__global__ void snapshot_vector_kernel(const std::int8_t* qs, const half2* ds, int count,
                                      unsigned* flag, std::uint8_t* bytes, const unsigned* armed) {
    __shared__ unsigned take;
    __shared__ unsigned nonzero;
    if (threadIdx.x == 0) { take = *armed && *flag == 0; nonzero = 0; }
    __syncthreads();
    if (!take) { return; }
    bool found = false;
    for (int i = threadIdx.x; i < count; i += blockDim.x) { found |= qs[i] != 0; }
    if (found) { atomicOr(&nonzero, 1u); }
    __syncthreads();
    if (threadIdx.x == 0 && nonzero) { take = atomicCAS(flag, 0u, 1u) == 0; }
    __syncthreads();
    if (!take) { return; }
    for (int i = threadIdx.x; i < count; i += blockDim.x) { bytes[i] = std::uint8_t(qs[i]); }
    const auto* scales = reinterpret_cast<const std::uint8_t*>(ds);
    for (int i = threadIdx.x; i < count / 8; i += blockDim.x) { bytes[count + i] = scales[i]; }
    __syncthreads();
    if (threadIdx.x == 0) { __threadfence(); atomicExch(flag, 2u); }
}

inline void flush_vector_snapshots() {
    const char* folder = std::getenv("NINFER_VECTOR_SNAPSHOTS");
    if (!folder) { return; }
    std::filesystem::create_directories(folder);
    for (const auto& slot : snapshot_slots) {
        unsigned flag = 0;
        if (cudaMemcpy(&flag, slot.flag, sizeof(flag), cudaMemcpyDeviceToHost) != cudaSuccess || flag != 2) { continue; }
        const std::size_t count = std::size_t(slot.k) * slot.columns;
        std::vector<std::uint8_t> bytes(count + count / 8);
        if (cudaMemcpy(bytes.data(), slot.bytes, bytes.size(), cudaMemcpyDeviceToHost) != cudaSuccess) { continue; }
        const std::array<std::uint32_t, 8> header{0x31534156u, 1u, std::uint32_t(slot.type),
            std::uint32_t(slot.k), std::uint32_t(slot.columns), std::uint32_t(slot.fused),
            std::uint32_t(bytes.size()), 0u};
        const std::string path = std::string(folder) + "/q" + std::to_string(slot.type) + "-k" +
            std::to_string(slot.k) + "-t" + std::to_string(slot.columns) + "-f" + std::to_string(slot.fused) + ".bin";
        if (FILE* file = std::fopen(path.c_str(), "wb")) {
            std::fwrite(header.data(), sizeof(header), 1, file);
            std::fwrite(bytes.data(), 1, bytes.size(), file);
            std::fclose(file);
        }
    }
}

inline void capture_vector_snapshot(int type, const VecArgs& args, int columns, bool fused,
                                    cudaStream_t stream) {
    static const bool enabled = std::getenv("NINFER_VECTOR_SNAPSHOTS") != nullptr;
    if (!enabled) { return; }
    std::lock_guard<std::mutex> lock(snapshot_mutex);
    if (!snapshot_armed) {
        check(cudaMalloc(&snapshot_armed, sizeof(unsigned)), "snapshot armed flag");
        check(cudaMemset(snapshot_armed, 0, sizeof(unsigned)), "snapshot disarm");
        const std::string folder = std::getenv("NINFER_VECTOR_SNAPSHOTS");
        std::thread([folder] {
            while (!std::filesystem::exists(folder + "/arm")) { std::this_thread::sleep_for(std::chrono::milliseconds(20)); }
            if (cudaSetDevice(0) != cudaSuccess) { return; }
            const unsigned value = 1;
            if (cudaMemcpy(snapshot_armed, &value, sizeof(value), cudaMemcpyHostToDevice) != cudaSuccess) { return; }
            if (FILE* file = std::fopen((folder + "/armed").c_str(), "wb")) { std::fclose(file); }
        }).detach();
    }
    VectorSnapshotSlot* found = nullptr;
    for (auto& slot : snapshot_slots) {
        if (slot.type == type && slot.k == args.k && slot.columns == columns && slot.fused == int(fused)) { found = &slot; break; }
    }
    if (!found) {
        cudaStreamCaptureStatus capture;
        check(cudaStreamIsCapturing(stream, &capture), "snapshot capture status");
        if (capture != cudaStreamCaptureStatusNone) { return; }
        if (snapshot_slots.empty()) { std::atexit(flush_vector_snapshots); }
        VectorSnapshotSlot slot{type, args.k, columns, int(fused), nullptr, nullptr};
        check(cudaMalloc(&slot.flag, sizeof(unsigned)), "snapshot flag");
        check(cudaMemsetAsync(slot.flag, 0, sizeof(unsigned), stream), "snapshot clear");
        const std::size_t count = std::size_t(args.k) * columns;
        check(cudaMalloc(&slot.bytes, count + count / 8), "snapshot bytes");
        snapshot_slots.push_back(slot);
        found = &snapshot_slots.back();
    }
    snapshot_vector_kernel<<<1, 128, 0, stream>>>(args.qs, args.ds, args.k * columns, found->flag, found->bytes, snapshot_armed);
    check(cudaGetLastError(), "snapshot kernel");
}

}
