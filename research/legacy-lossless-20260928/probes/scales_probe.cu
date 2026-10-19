#include "ops/linear/gguf/ggml_bridge_vec.cuh"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace ninfer::ops::gguf::detail {
const DeviceFacts& device_facts() {
    static DeviceFacts facts = [] {
        cudaDeviceProp properties;
        check(cudaGetDeviceProperties(&properties, 0), "device");
        return DeviceFacts{properties.major * 100 + properties.minor * 10,
                           properties.multiProcessorCount, properties.sharedMemPerBlockOptin};
    }();
    return facts;
}
}

using namespace ninfer::ops::gguf::detail;
using namespace ninfer::ops::gguf::detail::vec;

struct PackedScales {
    int rows = 0;
    int k = 0;
    std::vector<uint8_t> dm;
    std::vector<uint8_t> qs;
    std::vector<uint8_t> raw_metadata;
    std::vector<uint64_t> eligible;
    std::vector<uint32_t> offsets;
    std::vector<uint8_t> metadata;
    uint64_t scale_eligible = 0;
    uint64_t min_eligible = 0;
    uint64_t both_eligible = 0;
    uint64_t eligible_groups = 0;
    uint64_t multiple_scale_max = 0;
    uint64_t multiple_min_max = 0;
};

static void scale_min_host(const uint8_t* s, int j, int& sc, int& mn) {
    if (j < 4) {
        sc = s[j] & 63;
        mn = s[j + 4] & 63;
    } else {
        sc = (s[j + 4] & 15) | ((s[j - 4] >> 6) << 4);
        mn = (s[j + 4] >> 4) | ((s[j] >> 6) << 4);
    }
}

static void write_bits(std::vector<uint8_t>& out, uint64_t at, unsigned width, uint32_t value) {
    for (unsigned i = 0; i < width; ++i) {
        const uint64_t bit = at + i;
        const uint8_t mask = uint8_t(1u << (bit & 7));
        if ((value >> i) & 1u) { out[bit >> 3] |= mask; }
        else { out[bit >> 3] &= uint8_t(~mask); }
    }
}

static uint32_t read_bits_host(const uint8_t* in, uint64_t at, unsigned width) {
    uint32_t value = 0;
    for (unsigned i = 0; i < width; ++i) {
        const uint64_t bit = at + i;
        value |= uint32_t((in[bit >> 3] >> (bit & 7)) & 1u) << i;
    }
    return value;
}

static bool unpack_scale_min_host(const PackedScales& packed, uint64_t block,
                                  uint8_t* scales, uint8_t* mins) {
    const uint64_t group = block / 64;
    const uint32_t local = uint32_t(block % 64);
    const uint64_t mask = packed.eligible[group];
    const uint64_t prefix = local == 0 ? 0 : mask & ((uint64_t(1) << local) - 1);
    const uint64_t bit = uint64_t(packed.offsets[group]) * 8 + uint64_t(local) * 96 - 6 * __builtin_popcountll(prefix);
    const uint8_t* data = packed.metadata.data();
    if ((mask >> local) & 1u) {
        const uint32_t scale_pos = read_bits_host(data, bit, 3);
        const uint32_t min_pos = read_bits_host(data, bit + 45, 3);
        for (uint32_t j = 0; j < 8; ++j) {
            scales[j] = j == scale_pos ? 63 : uint8_t(read_bits_host(data, bit + 3 + 6 * (j - (j > scale_pos)), 6));
            mins[j] = j == min_pos ? 63 : uint8_t(read_bits_host(data, bit + 48 + 6 * (j - (j > min_pos)), 6));
        }
    } else {
        for (uint32_t j = 0; j < 8; ++j) {
            scales[j] = uint8_t(read_bits_host(data, bit + 6 * j, 6));
            mins[j] = uint8_t(read_bits_host(data, bit + 48 + 6 * j, 6));
        }
    }
    return true;
}

static PackedScales pack_scales(const std::vector<uint8_t>& raw, int rows, int k) {
    const uint64_t blocks = uint64_t(rows) * (k / 256);
    const uint64_t groups = (blocks + 63) / 64;
    if (raw.size() != blocks * 144) { throw std::runtime_error("Q4_K matrix payload size"); }
    PackedScales out;
    out.rows = rows;
    out.k = k;
    out.dm.resize(blocks * 4);
    out.qs.resize(blocks * 128);
    out.raw_metadata.resize(blocks * 12);
    out.eligible.resize(groups);
    out.offsets.resize(groups);
    for (uint64_t b = 0; b < blocks; ++b) {
        const uint8_t* src = raw.data() + b * 144;
        std::memcpy(out.dm.data() + b * 4, src, 4);
        std::memcpy(out.raw_metadata.data() + b * 12, src + 4, 12);
        std::memcpy(out.qs.data() + b * 128, src + 16, 128);
    }
    for (uint64_t group = 0; group < groups; ++group) {
        uint64_t mask = 0;
        const uint64_t first = group * 64;
        const uint64_t count = std::min<uint64_t>(64, blocks - first);
        for (uint64_t local = 0; local < count; ++local) {
            const uint8_t* s = raw.data() + (first + local) * 144 + 4;
            int scales[8], mins[8], scale_count = 0, min_count = 0;
            for (int j = 0; j < 8; ++j) {
                scale_min_host(s, j, scales[j], mins[j]);
                scale_count += scales[j] == 63;
                min_count += mins[j] == 63;
            }
            const bool has_scale = scale_count != 0;
            const bool has_min = min_count != 0;
            out.scale_eligible += has_scale;
            out.min_eligible += has_min;
            out.both_eligible += has_scale && has_min;
            out.multiple_scale_max += scale_count > 1;
            out.multiple_min_max += min_count > 1;
            if (has_scale && has_min) { mask |= uint64_t(1) << local; }
        }
        out.eligible[group] = mask;
        out.eligible_groups += mask != 0;
        const size_t aligned = (out.metadata.size() + 15) & ~size_t(15);
        out.metadata.resize(aligned, 0);
        out.offsets[group] = uint32_t(aligned);
        const uint64_t bit_count = count * 96 - 6 * __builtin_popcountll(mask);
        const size_t group_bytes = size_t((bit_count + 7) / 8);
        out.metadata.resize(aligned + group_bytes, 0);
        for (uint64_t local = 0; local < count; ++local) {
            const uint64_t b = first + local;
            const uint8_t* s = raw.data() + b * 144 + 4;
            int scales[8], mins[8];
            for (int j = 0; j < 8; ++j) { scale_min_host(s, j, scales[j], mins[j]); }
            const uint64_t prefix = local == 0 ? 0 : mask & ((uint64_t(1) << local) - 1);
            const uint64_t bit = uint64_t(aligned) * 8 + local * 96 - 6 * __builtin_popcountll(prefix);
            if ((mask >> local) & 1u) {
                const int scale_pos = int(std::find(scales, scales + 8, 63) - scales);
                const int min_pos = int(std::find(mins, mins + 8, 63) - mins);
                write_bits(out.metadata, bit, 3, uint32_t(scale_pos));
                for (int j = 0, slot = 0; j < 8; ++j) {
                    if (j != scale_pos) { write_bits(out.metadata, bit + 3 + 6 * slot++, 6, uint32_t(scales[j])); }
                }
                write_bits(out.metadata, bit + 45, 3, uint32_t(min_pos));
                for (int j = 0, slot = 0; j < 8; ++j) {
                    if (j != min_pos) { write_bits(out.metadata, bit + 48 + 6 * slot++, 6, uint32_t(mins[j])); }
                }
            } else {
                for (int j = 0; j < 8; ++j) {
                    write_bits(out.metadata, bit + 6 * j, 6, uint32_t(scales[j]));
                    write_bits(out.metadata, bit + 48 + 6 * j, 6, uint32_t(mins[j]));
                }
            }
        }
    }
    out.metadata.resize((out.metadata.size() + 15) & ~size_t(15), 0);
    if (out.metadata.size() * 8 >= (uint64_t(1) << 32)) {
        throw std::runtime_error("packed metadata exceeds 32-bit device bit addressing");
    }
    return out;
}

static std::vector<uint8_t> unpack_cpu(const PackedScales& packed) {
    const uint64_t blocks = uint64_t(packed.rows) * (packed.k / 256);
    std::vector<uint8_t> raw(blocks * 144);
    for (uint64_t b = 0; b < blocks; ++b) {
        uint8_t* dst = raw.data() + b * 144;
        std::memcpy(dst, packed.dm.data() + b * 4, 4);
        uint8_t scales[8], mins[8];
        unpack_scale_min_host(packed, b, scales, mins);
        uint8_t* s = dst + 4;
        for (int j = 0; j < 4; ++j) {
            s[j] = uint8_t(scales[j] & 63);
            s[j + 4] = uint8_t(mins[j] & 63);
        }
        for (int j = 4; j < 8; ++j) {
            s[j + 4] = uint8_t((scales[j] & 15) | ((mins[j] & 15) << 4));
            s[j - 4] |= uint8_t((scales[j] >> 4) << 6);
            s[j] |= uint8_t((mins[j] >> 4) << 6);
        }
        std::memcpy(dst + 16, packed.qs.data() + b * 128, 128);
    }
    return raw;
}

__device__ __forceinline__ uint32_t read_bits_device(const uint8_t* data, uint32_t at, uint32_t width) {
    const uint32_t shift = uint32_t(at & 7);
    const uint32_t value = uint32_t(data[at >> 3]) | (uint32_t(data[(at >> 3) + 1]) << 8);
    return (value >> shift) & ((1u << width) - 1u);
}

__device__ __forceinline__ void unpack_scale_min_device(const uint64_t* eligible,
                                                        const uint32_t* offsets,
                                                        const uint8_t* metadata,
                                                        uint32_t block, int u, int& sc, int& mn) {
    const uint32_t group = block / 64;
    const uint32_t local = block % 64;
    const uint64_t mask = eligible[group];
    const uint64_t prefix = local == 0 ? 0 : mask & ((uint64_t(1) << local) - 1);
    const uint32_t bit = offsets[group] * 8u + local * 96u - 6u * __popcll(prefix);
    const uint8_t* data = metadata;
    if ((mask >> local) & 1u) {
        const uint32_t scale_pos = read_bits_device(data, bit, 3);
        const uint32_t min_pos = read_bits_device(data, bit + 45, 3);
        sc = u == int(scale_pos) ? 63 : int(read_bits_device(data, bit + 3 + 6 * (uint32_t(u) - (u > int(scale_pos))), 6));
        mn = u == int(min_pos) ? 63 : int(read_bits_device(data, bit + 48 + 6 * (uint32_t(u) - (u > int(min_pos))), 6));
    } else {
        sc = int(read_bits_device(data, bit + 6 * uint32_t(u), 6));
        mn = int(read_bits_device(data, bit + 48 + 6 * uint32_t(u), 6));
    }
}

struct ScaleArgs {
    VecArgs vec;
    const uint8_t* dm;
    const uint8_t* qs;
    const uint8_t* raw_metadata;
    const uint64_t* eligible;
    const uint32_t* offsets;
    const uint8_t* metadata;
};

template <int T, bool RawPlanar>
__launch_bounds__(128) __global__ void scale_kernel(ScaleArgs p) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    constexpr int k = 5120;
    constexpr int blocks_per_row = k / 256;
    for (int row0 = (blockIdx.x * 4 + warp) * 2; row0 < p.vec.rows;
         row0 += gridDim.x * 8) {
        float acc[2][T];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
#pragma unroll
            for (int j = 0; j < T; ++j) { acc[r][j] = 0.0f; }
        }
#pragma unroll
        for (int it = 0; it < k / 1024; ++it) {
            int a[T][8];
            float d[T], sum[T];
#pragma unroll
            for (int j = 0; j < T; ++j) {
                const float2 ds = __half22float2(p.vec.ds[int64_t(j) * (k / 32) + lane + 32 * it]);
                d[j] = ds.x;
                sum[j] = ds.y;
                const int4* q = reinterpret_cast<const int4*>(p.vec.qs + int64_t(j) * k + 32 * (lane + 32 * it));
                const int4 v0 = q[0], v1 = q[1];
                a[j][0] = v0.x; a[j][1] = v0.y; a[j][2] = v0.z; a[j][3] = v0.w;
                a[j][4] = v1.x; a[j][5] = v1.y; a[j][6] = v1.z; a[j][7] = v1.w;
            }
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                const int row = row0 + r;
                if (row < p.vec.rows) {
                    const uint32_t block = uint32_t(row) * uint32_t(blocks_per_row) + uint32_t(4 * it + lane / 8);
                    int sc, mn;
                    if constexpr (RawPlanar) {
                        scale_min_k4(p.raw_metadata + block * 12, lane % 8, sc, mn);
                    } else {
                        unpack_scale_min_device(p.eligible, p.offsets, p.metadata, block, lane % 8, sc, mn);
                    }
                    const float2 dm = __half22float2(*reinterpret_cast<const half2*>(p.dm + block * 4));
                    const uint4* q = reinterpret_cast<const uint4*>(p.qs + block * 128 + 32 * ((lane % 8) / 2));
                    const uint4 q0 = q[0], q1 = q[1];
                    const int shift = 4 * ((lane % 8) & 1);
                    Slice sl;
                    sl.f[0] = dm.x * float(sc);
                    sl.f[1] = dm.y * float(mn);
                    const uint32_t words[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
#pragma unroll
                    for (int i = 0; i < 8; ++i) { sl.w[i] = int((words[i] >> shift) & 0x0F0F0F0Fu); }
#pragma unroll
                    for (int j = 0; j < T; ++j) { acc[r][j] += Decoder<GGML_TYPE_Q4_K>::dot(sl, a[j], d[j], sum[j]); }
                }
            }
        }
#pragma unroll
        for (int r = 0; r < 2; ++r) {
#pragma unroll
            for (int j = 0; j < T; ++j) {
#pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1) {
                    acc[r][j] += __shfl_xor_sync(0xffffffffu, acc[r][j], offset);
                }
                if (lane == 0 && row0 + r < p.vec.rows) { vec_store(p.vec.out, row0 + r, j, acc[r][j]); }
            }
        }
    }
}

__global__ void reconstruct_kernel(int rows, const uint8_t* dm, const uint8_t* qs,
                                   const uint64_t* eligible, const uint32_t* offsets,
                                   const uint8_t* metadata, uint8_t* raw) {
    const uint32_t block = uint32_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint32_t blocks = uint32_t(rows) * (5120 / 256);
    if (block >= blocks) { return; }
    const uint32_t row_bytes = (5120 / 256) * 144;
    const uint32_t at = (block / (5120 / 256)) * row_bytes + (block % (5120 / 256)) * 144;
    for (int j = 0; j < 4; ++j) { raw[at + j] = dm[block * 4 + j]; }
    int scales[8], mins[8];
    for (int j = 0; j < 8; ++j) { unpack_scale_min_device(eligible, offsets, metadata, block, j, scales[j], mins[j]); }
    uint8_t s[12]{};
    for (int j = 0; j < 4; ++j) {
        s[j] = uint8_t(scales[j] & 63);
        s[j + 4] = uint8_t(mins[j] & 63);
    }
    for (int j = 4; j < 8; ++j) {
        s[j + 4] = uint8_t((scales[j] & 15) | ((mins[j] & 15) << 4));
        s[j - 4] |= uint8_t((scales[j] >> 4) << 6);
        s[j] |= uint8_t((mins[j] >> 4) << 6);
    }
    for (int j = 0; j < 12; ++j) { raw[at + 4 + j] = s[j]; }
    for (int j = 0; j < 128; ++j) { raw[at + 16 + j] = qs[block * 128 + j]; }
}

template <int T, bool RawPlanar>
void launch_candidate(const VecArgs& vec, const PackedScales& packed,
                      const uint8_t* dm, const uint8_t* qs, const uint8_t* raw_metadata,
                      const uint64_t* eligible,
                      const uint32_t* offsets, const uint8_t* metadata, cudaStream_t stream) {
    ScaleArgs args{vec, dm, qs, raw_metadata, eligible, offsets, metadata};
    static const int resident = [] {
        int blocks = 0;
        check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, scale_kernel<T, RawPlanar>, 128, 0), "scale occupancy");
        return std::max(blocks, 1);
    }();
    const int groups = (packed.rows + 7) / 8;
    const int grid = std::min(groups, resident * device_facts().sm_count);
    scale_kernel<T, RawPlanar><<<grid, 128, 0, stream>>>(args);
    check(cudaGetLastError(), "scale candidate launch");
}

struct KernelResource {
    int registers = 0;
    int shared_bytes = 0;
    int local_bytes = 0;
    int active_blocks = 0;
    int active_warps = 0;
    float occupancy = 0.0f;
};

template <typename Kernel>
static KernelResource kernel_resource(Kernel kernel) {
    cudaFuncAttributes attributes{};
    check(cudaFuncGetAttributes(&attributes, kernel), "kernel attributes");
    int active = 0;
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, kernel, 128, 0), "kernel occupancy");
    KernelResource result;
    result.registers = attributes.numRegs;
    result.shared_bytes = int(attributes.sharedSizeBytes);
    result.local_bytes = int(attributes.localSizeBytes);
    result.active_blocks = active;
    result.active_warps = active * 4;
    result.occupancy = float(result.active_warps) / 64.0f;
    return result;
}

static void print_resource(const char* name, const KernelResource& resource) {
    std::cout << "\"" << name << "\":{\"registers_per_thread\":" << resource.registers
              << ",\"static_shared_bytes\":" << resource.shared_bytes
              << ",\"local_bytes_per_thread\":" << resource.local_bytes
              << ",\"active_blocks_per_sm\":" << resource.active_blocks
              << ",\"active_warps_per_sm\":" << resource.active_warps
              << ",\"warp_occupancy\":" << resource.occupancy << '}';
}

template <int T>
void measure(const std::vector<uint8_t>& raw, const PackedScales& packed, const std::string& label) {
    constexpr int k = 5120;
    const int n = packed.rows;
    const size_t weight_bytes = raw.size();
    uint8_t* weights[3];
    uint8_t* device_dm[3];
    uint8_t* device_qs[3];
    uint8_t* device_raw_metadata[3];
    uint64_t* device_eligible[3];
    uint32_t* device_offsets[3];
    uint8_t* device_metadata[3];
    uint8_t* device_debug;
    int8_t* x;
    half2* ds;
    float* y;
    auto alloc = [](auto** ptr, size_t bytes) { check(cudaMalloc(ptr, bytes), "alloc"); };
    for (int r = 0; r < 3; ++r) {
        alloc(&weights[r], raw.size());
        check(cudaMemcpy(weights[r], raw.data(), raw.size(), cudaMemcpyHostToDevice), "raw upload");
        alloc(&device_dm[r], packed.dm.size());
        check(cudaMemcpy(device_dm[r], packed.dm.data(), packed.dm.size(), cudaMemcpyHostToDevice), "dm upload");
        alloc(&device_qs[r], packed.qs.size());
        check(cudaMemcpy(device_qs[r], packed.qs.data(), packed.qs.size(), cudaMemcpyHostToDevice), "qs upload");
        alloc(&device_raw_metadata[r], packed.raw_metadata.size());
        check(cudaMemcpy(device_raw_metadata[r], packed.raw_metadata.data(), packed.raw_metadata.size(), cudaMemcpyHostToDevice), "raw metadata upload");
        alloc(&device_eligible[r], packed.eligible.size() * sizeof(uint64_t));
        check(cudaMemcpy(device_eligible[r], packed.eligible.data(), packed.eligible.size() * sizeof(uint64_t), cudaMemcpyHostToDevice), "eligible upload");
        alloc(&device_offsets[r], packed.offsets.size() * sizeof(uint32_t));
        check(cudaMemcpy(device_offsets[r], packed.offsets.data(), packed.offsets.size() * sizeof(uint32_t), cudaMemcpyHostToDevice), "offsets upload");
        alloc(&device_metadata[r], packed.metadata.size());
        check(cudaMemcpy(device_metadata[r], packed.metadata.data(), packed.metadata.size(), cudaMemcpyHostToDevice), "metadata upload");
    }
    alloc(&device_debug, raw.size());
    const uint64_t blocks = uint64_t(n) * (k / 256);
    reconstruct_kernel<<<(blocks + 255) / 256, 256>>>(n, device_dm[0], device_qs[0], device_eligible[0],
                                                       device_offsets[0], device_metadata[0], device_debug);
    check(cudaGetLastError(), "reconstruct launch");
    std::vector<uint8_t> back(raw.size());
    check(cudaMemcpy(back.data(), device_debug, back.size(), cudaMemcpyDeviceToHost), "reconstruct copy");
    if (back != raw) { throw std::runtime_error("GPU scale/quant payload full bit mismatch"); }
    check(cudaFree(device_debug), "debug free");

    std::vector<int8_t> hostx(size_t(k) * T);
    std::vector<half2> hostds(size_t(k / 32) * T);
    for (size_t i = 0; i < hostx.size(); ++i) { hostx[i] = int8_t((i * 37 + i / 7 * 11) % 255 - 127); }
    for (size_t i = 0; i < hostds.size(); ++i) {
        int sum = 0;
        for (int j = 0; j < 32; ++j) { sum += hostx[i * 32 + j]; }
        hostds[i] = __floats2half2_rn(0.0625f, sum * 0.0625f);
    }
    alloc(&x, hostx.size());
    check(cudaMemcpy(x, hostx.data(), hostx.size(), cudaMemcpyHostToDevice), "x upload");
    alloc(&ds, hostds.size() * sizeof(half2));
    check(cudaMemcpy(ds, hostds.data(), hostds.size() * sizeof(half2), cudaMemcpyHostToDevice), "ds upload");
    alloc(&y, size_t(n) * T * sizeof(float));
    VecArgs args{weights[0], k / 256 * 144, n, k, x, ds,
                 {nullptr, y, n, ninfer::ops::gguf::Epilogue::Store, nullptr, 0}};
    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
    auto baseline = [&](int rotation) {
        args.weight = weights[rotation];
        vec::launch<GGML_TYPE_Q4_K, T, false, k>(args, stream);
    };
    auto candidate = [&](int rotation) {
        args.weight = weights[rotation];
        launch_candidate<T, false>(args, packed, device_dm[rotation], device_qs[rotation], device_raw_metadata[rotation], device_eligible[rotation],
                            device_offsets[rotation], device_metadata[rotation], stream);
    };
    auto raw_planar = [&](int rotation) {
        args.weight = weights[rotation];
        launch_candidate<T, true>(args, packed, device_dm[rotation], device_qs[rotation], device_raw_metadata[rotation], device_eligible[rotation],
                            device_offsets[rotation], device_metadata[rotation], stream);
    };
    std::vector<float> reference(size_t(n) * T), actual(reference.size());
    baseline(0);
    check(cudaStreamSynchronize(stream), "baseline sync");
    check(cudaMemcpy(reference.data(), y, reference.size() * sizeof(float), cudaMemcpyDeviceToHost), "baseline copy");
    candidate(0);
    check(cudaStreamSynchronize(stream), "candidate sync");
    check(cudaMemcpy(actual.data(), y, actual.size() * sizeof(float), cudaMemcpyDeviceToHost), "candidate copy");
    if (std::memcmp(reference.data(), actual.data(), reference.size() * sizeof(float)) != 0) {
        throw std::runtime_error("FP32 output bit mismatch");
    }
    __nv_bfloat16* bfout;
    alloc(&bfout, size_t(n) * T * sizeof(__nv_bfloat16));
    std::vector<uint16_t> bf_reference(size_t(n) * T), bf_actual(bf_reference.size());
    args.out.bf16 = bfout;
    args.out.f32 = nullptr;
    baseline(0);
    check(cudaStreamSynchronize(stream), "baseline bf16 sync");
    check(cudaMemcpy(bf_reference.data(), bfout, bf_reference.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost), "baseline bf16 copy");
    candidate(0);
    check(cudaStreamSynchronize(stream), "candidate bf16 sync");
    check(cudaMemcpy(bf_actual.data(), bfout, bf_actual.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost), "candidate bf16 copy");
    if (bf_reference != bf_actual) { throw std::runtime_error("BF16 output bit mismatch"); }
    check(cudaFree(bfout), "bf16 free");
    args.out.bf16 = nullptr;
    args.out.f32 = y;

    cudaGraphExec_t graphs[3];
    for (int mode = 0; mode < 3; ++mode) {
        cudaGraph_t graph;
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "capture");
        for (int r = 0; r < 3; ++r) {
            if (mode == 0) { baseline(r); }
            else if (mode == 1) { raw_planar(r); }
            else { candidate(r); }
        }
        check(cudaStreamEndCapture(stream, &graph), "capture end");
        check(cudaGraphInstantiate(&graphs[mode], graph, 0), "instantiate");
        check(cudaGraphDestroy(graph), "graph destroy");
    }
    for (int i = 0; i < 4; ++i) {
        for (int mode = 0; mode < 3; ++mode) { check(cudaGraphLaunch(graphs[mode], stream), "graph warmup"); }
    }
    check(cudaStreamSynchronize(stream), "warmup sync");
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start), "start event");
    check(cudaEventCreate(&stop), "stop event");
    const uint64_t blocks_count = uint64_t(n) * (k / 256);
    const uint64_t groups = (blocks_count + 63) / 64;
    const uint64_t aligned_meta = packed.metadata.size();
    const uint64_t format_bytes = packed.dm.size() + packed.qs.size() + packed.eligible.size() * 8 +
                                  packed.offsets.size() * 4 + aligned_meta + 64;
    const uint64_t planar_bytes = packed.dm.size() + packed.qs.size() + packed.raw_metadata.size() + 64;
    const KernelResource baseline_resource = kernel_resource(vec::kernel<GGML_TYPE_Q4_K, T, false, k>);
    const KernelResource raw_resource = kernel_resource(scale_kernel<T, true>);
    const KernelResource coded_resource = kernel_resource(scale_kernel<T, false>);
    std::cout << "{\"matrix\":\"" << label << "\",\"T\":" << T << ",\"N\":" << n << ",\"K\":" << k
              << ",\"baseline_bytes\":" << weight_bytes << ",\"raw_planar_bytes_including_headers\":" << planar_bytes
              << ",\"scale_format_bytes\":" << format_bytes
              << ",\"metadata_groups\":" << groups << ",\"eligible_blocks\":" << packed.both_eligible
              << ",\"eligible_rate\":" << double(packed.both_eligible) / double(blocks_count)
              << ",\"fp32_output_bit_exact\":true,\"bf16_output_bit_exact\":true,\"gpu_all_payload_bytes_exact\":true,\"rotations\":3,\"graphs\":true,\"resources\":{";
    print_resource("baseline", baseline_resource);
    std::cout << ',';
    print_resource("raw_planar", raw_resource);
    std::cout << ',';
    print_resource("coded", coded_resource);
    std::cout << "},\"samples_ms\":[";
    int sample = 0;
    for (int round = 0; round < 5; ++round) {
        for (int order = 0; order < 3; ++order) {
            const int mode = (round + order) % 3;
            check(cudaEventRecord(start, stream), "start record");
            for (int repeat = 0; repeat < 10; ++repeat) { check(cudaGraphLaunch(graphs[mode], stream), "graph launch"); }
            check(cudaEventRecord(stop, stream), "stop record");
            check(cudaEventSynchronize(stop), "event sync");
            float ms;
            check(cudaEventElapsedTime(&ms, start, stop), "elapsed time");
            if (sample++) { std::cout << ','; }
            const char* mode_name = mode == 0 ? "baseline" : mode == 1 ? "raw_planar" : "coded";
            std::cout << "{\"mode\":\"" << mode_name << "\",\"ms\":" << ms / 30 << '}';
        }
    }
    std::cout << "]}\n";
    for (auto graph : graphs) { check(cudaGraphExecDestroy(graph), "graph free"); }
    check(cudaEventDestroy(start), "event free");
    check(cudaEventDestroy(stop), "event free");
    for (int r = 0; r < 3; ++r) {
        check(cudaFree(weights[r]), "free");
        check(cudaFree(device_dm[r]), "free");
        check(cudaFree(device_qs[r]), "free");
        check(cudaFree(device_raw_metadata[r]), "free");
        check(cudaFree(device_eligible[r]), "free");
        check(cudaFree(device_offsets[r]), "free");
        check(cudaFree(device_metadata[r]), "free");
    }
    check(cudaFree(x), "free");
    check(cudaFree(ds), "free");
    check(cudaFree(y), "free");
    check(cudaStreamDestroy(stream), "stream free");
}

static void run_scan(const std::vector<uint8_t>& raw, const PackedScales& packed, int rows, int k,
                     const char* label) {
    const std::vector<uint8_t> restored = unpack_cpu(packed);
    if (restored != raw) { throw std::runtime_error("CPU full Q4_K payload bit mismatch"); }
    const uint64_t blocks = uint64_t(rows) * (k / 256);
    const uint64_t groups = (blocks + 63) / 64;
    const uint64_t raw_bytes = raw.size();
    const uint64_t bytes = packed.dm.size() + packed.qs.size() + packed.eligible.size() * 8 +
                           packed.offsets.size() * 4 + packed.metadata.size() + 64;
    std::cout << "{\"matrix\":\"" << label << "\",\"N\":" << rows << ",\"K\":" << k
              << ",\"blocks\":" << blocks << ",\"groups\":" << groups
              << ",\"scale_has_63_blocks\":" << packed.scale_eligible
              << ",\"min_has_63_blocks\":" << packed.min_eligible
              << ",\"both_have_63_blocks\":" << packed.both_eligible
              << ",\"scale_has_multiple_63_blocks\":" << packed.multiple_scale_max
              << ",\"min_has_multiple_63_blocks\":" << packed.multiple_min_max
              << ",\"eligible_rate\":" << double(packed.both_eligible) / double(blocks)
              << ",\"eligible_groups\":" << packed.eligible_groups
              << ",\"cpu_all_payload_bytes_exact\":true,\"baseline_bytes\":" << raw_bytes
              << ",\"candidate_bytes_including_headers_and_alignment\":" << bytes
              << ",\"net_savings_bytes\":" << (raw_bytes > bytes ? raw_bytes - bytes : 0)
              << ",\"candidate_larger_bytes\":" << (bytes > raw_bytes ? bytes - raw_bytes : 0) << "}\n";
}

static std::vector<uint8_t> synthetic_payload() {
    constexpr int rows = 131;
    std::vector<uint8_t> raw(size_t(rows) * 144);
    for (size_t i = 0; i < raw.size(); ++i) {
        if (i % 144 < 4 || i % 144 >= 16) { raw[i] = uint8_t(i * 73 + i / 11 * 19); }
    }
    auto encode = [&](int block, const int* scales, const int* mins) {
        uint8_t* s = raw.data() + block * 144 + 4;
        for (int j = 0; j < 4; ++j) {
            s[j] = uint8_t(scales[j] & 63);
            s[j + 4] = uint8_t(mins[j] & 63);
        }
        for (int j = 4; j < 8; ++j) {
            s[j + 4] = uint8_t((scales[j] & 15) | ((mins[j] & 15) << 4));
            s[j - 4] = uint8_t((s[j - 4] & 63) | ((scales[j] >> 4) << 6));
            s[j] = uint8_t((s[j] & 63) | ((mins[j] >> 4) << 6));
        }
    };
    int scales[8] = {1, 2, 63, 4, 5, 6, 63, 8};
    int mins[8] = {9, 10, 11, 63, 13, 14, 15, 63};
    encode(0, scales, mins);
    encode(63, scales, mins);
    encode(128, scales, mins);
    int only_scale[8] = {1, 2, 63, 4, 5, 6, 7, 8};
    int no_max[8] = {9, 10, 11, 12, 13, 14, 15, 16};
    encode(129, only_scale, no_max);
    int no_scale[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    int only_min[8] = {9, 10, 11, 63, 13, 14, 15, 16};
    encode(130, no_scale, only_min);
    return raw;
}

static void run_cpu_edge_checks() {
    const std::vector<uint8_t> synthetic = synthetic_payload();
    const PackedScales packed = pack_scales(synthetic, 131, 256);
    if (unpack_cpu(packed) != synthetic) { throw std::runtime_error("synthetic CPU full payload mismatch"); }
    if (packed.eligible.size() != 3 || packed.eligible[0] != ((uint64_t(1) << 0) | (uint64_t(1) << 63)) ||
        packed.eligible[1] != 0 || packed.eligible[2] != uint64_t(1)) {
        throw std::runtime_error("synthetic eligibility boundary mismatch");
    }
    if (packed.multiple_scale_max < 3 || packed.multiple_min_max < 2 ||
        packed.scale_eligible != 4 || packed.min_eligible != 4 || packed.both_eligible != 3) {
        throw std::runtime_error("synthetic eligibility edge counts mismatch");
    }
}

int main(int argc, char** argv) {
    if (argc != 7 || (std::string(argv[6]) != "scan" && std::string(argv[6]) != "bench")) {
        std::cerr << "model offset N K label scan|bench\n";
        return 2;
    }
    const int rows = std::stoi(argv[3]);
    const int k = std::stoi(argv[4]);
    if (k != 5120 || rows % 8 != 0) { throw std::runtime_error("requires K5120 and complete eight-row tiles"); }
    run_cpu_edge_checks();
    std::vector<uint8_t> raw(size_t(rows) * (k / 256) * 144);
    std::ifstream file(argv[1], std::ios::binary);
    file.seekg(std::stoull(argv[2]));
    file.read(reinterpret_cast<char*>(raw.data()), raw.size());
    if (size_t(file.gcount()) != raw.size()) { throw std::runtime_error("truncated model tensor"); }
    PackedScales packed = pack_scales(raw, rows, k);
    const std::vector<uint8_t> restored = unpack_cpu(packed);
    if (restored != raw) { throw std::runtime_error("CPU full Q4_K payload bit mismatch"); }
    if (std::string(argv[6]) == "scan") {
        run_scan(raw, packed, rows, k, argv[5]);
        return 0;
    }
    measure<1>(raw, packed, argv[5]);
    measure<2>(raw, packed, argv[5]);
    measure<3>(raw, packed, argv[5]);
    measure<4>(raw, packed, argv[5]);
    measure<8>(raw, packed, argv[5]);
    return 0;
}
