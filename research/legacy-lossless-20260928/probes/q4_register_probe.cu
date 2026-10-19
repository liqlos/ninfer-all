#include "ops/linear/gguf/ggml_bridge_vec.cuh"

#include <algorithm>
#include <array>
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

struct alignas(16) GroupMeta {
    uint64_t eligible = 0;
    uint32_t raw_offset = 0;
    uint32_t coded_offset = 0;
};

struct PackedScales {
    int rows = 0;
    int k = 0;
    std::vector<uint8_t> dm;
    std::vector<uint8_t> qs;
    std::vector<uint8_t> raw_metadata;
    std::vector<uint8_t> raw_fallback;
    std::vector<uint8_t> coded_metadata;
    std::vector<GroupMeta> groups;
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
    const uint32_t group = uint32_t(block / 64);
    const uint32_t local = uint32_t(block % 64);
    const GroupMeta& meta = packed.groups[group];
    const uint64_t prefix = local == 0 ? 0 : meta.eligible & ((uint64_t(1) << local) - 1);
    const uint32_t rank = uint32_t(__builtin_popcountll(prefix));
    if ((meta.eligible >> local) & 1u) {
        const uint64_t bit = uint64_t(meta.coded_offset) * 8 + uint64_t(rank) * 90;
        const uint32_t scale_pos = read_bits_host(packed.coded_metadata.data(), bit, 3);
        const uint32_t min_pos = read_bits_host(packed.coded_metadata.data(), bit + 45, 3);
        for (uint32_t j = 0; j < 8; ++j) {
            scales[j] = j == scale_pos ? 63 : uint8_t(read_bits_host(packed.coded_metadata.data(), bit + 3 + 6 * (j - (j > scale_pos)), 6));
            mins[j] = j == min_pos ? 63 : uint8_t(read_bits_host(packed.coded_metadata.data(), bit + 48 + 6 * (j - (j > min_pos)), 6));
        }
    } else {
        const uint32_t fallback = local - rank;
        const uint8_t* values = packed.raw_fallback.data() + meta.raw_offset + uint64_t(fallback) * 12;
        for (int j = 0; j < 8; ++j) {
            int sc, mn;
            scale_min_host(values, j, sc, mn);
            scales[j] = uint8_t(sc);
            mins[j] = uint8_t(mn);
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
    out.groups.resize(groups);
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
            out.both_eligible += scale_count == 1 && min_count == 1;
            out.multiple_scale_max += scale_count > 1;
            out.multiple_min_max += min_count > 1;
            if (scale_count == 1 && min_count == 1) { mask |= uint64_t(1) << local; }
        }
        GroupMeta& meta = out.groups[group];
        meta.eligible = mask;
        out.eligible_groups += mask != 0;
        const size_t raw_aligned = (out.raw_fallback.size() + 15) & ~size_t(15);
        out.raw_fallback.resize(raw_aligned, 0);
        meta.raw_offset = uint32_t(raw_aligned);
        const size_t coded_aligned = (out.coded_metadata.size() + 15) & ~size_t(15);
        out.coded_metadata.resize(coded_aligned, 0);
        meta.coded_offset = uint32_t(coded_aligned);
        const uint32_t eligible_count = uint32_t(__builtin_popcountll(mask));
        const uint32_t fallback_count = uint32_t(count) - eligible_count;
        out.raw_fallback.resize(raw_aligned + size_t(fallback_count) * 12, 0);
        out.coded_metadata.resize(coded_aligned + size_t(eligible_count) * 90 / 8 + ((eligible_count * 90) % 8 != 0), 0);
        uint32_t coded_rank = 0;
        uint32_t raw_rank = 0;
        for (uint64_t local = 0; local < count; ++local) {
            const uint64_t b = first + local;
            const uint8_t* s = raw.data() + b * 144 + 4;
            int scales[8], mins[8];
            for (int j = 0; j < 8; ++j) { scale_min_host(s, j, scales[j], mins[j]); }
            if ((mask >> local) & 1u) {
                const uint64_t bit = uint64_t(coded_aligned) * 8 + uint64_t(coded_rank++) * 90;
                const int scale_pos = int(std::find(scales, scales + 8, 63) - scales);
                const int min_pos = int(std::find(mins, mins + 8, 63) - mins);
                write_bits(out.coded_metadata, bit, 3, uint32_t(scale_pos));
                for (int j = 0, slot = 0; j < 8; ++j) {
                    if (j != scale_pos) { write_bits(out.coded_metadata, bit + 3 + 6 * slot++, 6, uint32_t(scales[j])); }
                }
                write_bits(out.coded_metadata, bit + 45, 3, uint32_t(min_pos));
                for (int j = 0, slot = 0; j < 8; ++j) {
                    if (j != min_pos) { write_bits(out.coded_metadata, bit + 48 + 6 * slot++, 6, uint32_t(mins[j])); }
                }
            } else {
                std::memcpy(out.raw_fallback.data() + raw_aligned + uint64_t(raw_rank++) * 12, s, 12);
            }
        }
    }
    out.raw_fallback.resize(std::max<size_t>(16, (out.raw_fallback.size() + 15) & ~size_t(15)), 0);
    out.coded_metadata.resize(std::max<size_t>(16, (out.coded_metadata.size() + 15) & ~size_t(15)), 0);
    if (out.coded_metadata.size() * 8 >= (uint64_t(1) << 32)) {
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
    const uint32_t word = at >> 5;
    const uint32_t shift = at & 31;
    const uint32_t* words = reinterpret_cast<const uint32_t*>(data);
    const uint64_t value = uint64_t(words[word]) | (uint64_t(words[word + 1]) << 32);
    return uint32_t((value >> shift) & ((uint64_t(1) << width) - 1));
}

__device__ __forceinline__ uint32_t read_bits_registers(uint32_t w0, uint32_t w1, uint32_t w2,
                                                        uint32_t w3, uint32_t bit, uint32_t width) {
    const uint32_t index = bit >> 5;
    const uint32_t shift = bit & 31;
    const uint32_t low = index == 0 ? w0 : index == 1 ? w1 : index == 2 ? w2 : w3;
    const uint32_t high = index == 0 ? w1 : index == 1 ? w2 : w3;
    const uint64_t pair = uint64_t(low) | (uint64_t(high) << 32);
    return uint32_t((pair >> shift) & ((uint64_t(1) << width) - 1));
}

__device__ __forceinline__ void unpack_scale_min_device(const GroupMeta* groups,
                                                        const uint8_t* raw_fallback,
                                                        const uint8_t* coded_metadata,
                                                        uint32_t block, int u, int& sc, int& mn) {
    const uint32_t group = block / 64;
    const uint32_t local = block % 64;
    const GroupMeta meta = groups[group];
    const uint64_t mask = meta.eligible;
    const uint64_t prefix = local == 0 ? 0 : mask & ((uint64_t(1) << local) - 1);
    if ((mask >> local) & 1u) {
        const uint32_t bit = meta.coded_offset * 8u + uint32_t(__popcll(prefix)) * 90u;
        const uint32_t word = bit >> 5;
        const uint32_t shift = bit & 31;
        const uint32_t* src = reinterpret_cast<const uint32_t*>(coded_metadata) + word;
        const uint32_t w0 = src[0], w1 = src[1], w2 = src[2], w3 = src[3];
        const uint32_t scale_pos = read_bits_registers(w0, w1, w2, w3, shift, 3);
        const uint32_t min_pos = read_bits_registers(w0, w1, w2, w3, shift + 45, 3);
        sc = u == int(scale_pos) ? 63 : int(read_bits_registers(w0, w1, w2, w3, shift + 3 + 6 * (uint32_t(u) - (u > int(scale_pos))), 6));
        mn = u == int(min_pos) ? 63 : int(read_bits_registers(w0, w1, w2, w3, shift + 48 + 6 * (uint32_t(u) - (u > int(min_pos))), 6));
    } else {
        const uint32_t fallback = local - uint32_t(__popcll(prefix));
        scale_min_k4(raw_fallback + meta.raw_offset + fallback * 12, u, sc, mn);
    }
}

struct ScaleArgs {
    VecArgs vec;
    const uint8_t* dm;
    const uint8_t* qs;
    const uint8_t* raw_metadata;
    const GroupMeta* groups;
    const uint8_t* raw_fallback;
    const uint8_t* coded_metadata;
};

template <int T, bool RawPlanar, bool Fused = false>
__launch_bounds__(128) __global__ void scale_kernel(ScaleArgs p) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    constexpr int k = 5120;
    constexpr int blocks_per_row = k / 256;
    constexpr int step = Fused ? 1 : 2;
    for (int row0 = (blockIdx.x * 4 + warp) * step; row0 < p.vec.rows;
         row0 += gridDim.x * 4 * step) {
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
                const int row = Fused ? row0 + r * p.vec.rows : row0 + r;
                if (Fused || row < p.vec.rows) {
                    const uint32_t block = uint32_t(row) * uint32_t(blocks_per_row) + uint32_t(4 * it + lane / 8);
                    int sc, mn;
                    if constexpr (RawPlanar) {
                        scale_min_k4(p.raw_metadata + block * 12, lane % 8, sc, mn);
                    } else {
                        unpack_scale_min_device(p.groups, p.raw_fallback, p.coded_metadata, block, lane % 8, sc, mn);
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
            }
        }
        if (lane == 0) {
            if constexpr (Fused) {
#pragma unroll
                for (int j = 0; j < T; ++j) { vec_store(p.vec.out, row0, j, vec_silu(acc[0][j]) * acc[1][j]); }
            } else {
#pragma unroll
                for (int r = 0; r < 2; ++r) {
                    if (row0 + r < p.vec.rows) {
#pragma unroll
                        for (int j = 0; j < T; ++j) { vec_store(p.vec.out, row0 + r, j, acc[r][j]); }
                    }
                }
            }
        }
    }
}

__global__ void reconstruct_kernel(int rows, const uint8_t* dm, const uint8_t* qs,
                                   const GroupMeta* groups, const uint8_t* raw_fallback,
                                   const uint8_t* coded_metadata, uint8_t* raw) {
    const uint32_t block = uint32_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint32_t blocks = uint32_t(rows) * (5120 / 256);
    if (block >= blocks) { return; }
    const uint32_t row_bytes = (5120 / 256) * 144;
    const uint32_t at = (block / (5120 / 256)) * row_bytes + (block % (5120 / 256)) * 144;
    for (int j = 0; j < 4; ++j) { raw[at + j] = dm[block * 4 + j]; }
    int scales[8], mins[8];
    for (int j = 0; j < 8; ++j) { unpack_scale_min_device(groups, raw_fallback, coded_metadata, block, j, scales[j], mins[j]); }
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

template <int T, bool RawPlanar, bool Fused = false>
void launch_candidate(const VecArgs& vec, const PackedScales& packed,
                      const uint8_t* dm, const uint8_t* qs, const uint8_t* raw_metadata,
                      const GroupMeta* group_meta, const uint8_t* raw_fallback,
                      const uint8_t* coded_metadata, cudaStream_t stream) {
    ScaleArgs args{vec, dm, qs, raw_metadata, group_meta, raw_fallback, coded_metadata};
    static const int resident = [] {
        int blocks = 0;
        check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, scale_kernel<T, RawPlanar, Fused>, 128, 0), "scale occupancy");
        return std::max(blocks, 1);
    }();
    const int groups = Fused ? (vec.rows + 3) / 4 : (packed.rows + 7) / 8;
    const int grid = std::min(groups, resident * device_facts().sm_count);
    scale_kernel<T, RawPlanar, Fused><<<grid, 128, 0, stream>>>(args);
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

struct Activation {
    int qtype = 0;
    int k = 0;
    int t = 0;
    int fused = 0;
    std::vector<int8_t> qs;
    std::vector<half2> ds;
};

static Activation load_activation(const char* path) {
    std::ifstream file(path, std::ios::binary);
    uint32_t header[8]{};
    file.read(reinterpret_cast<char*>(header), sizeof(header));
    if (!file || header[0] != 0x31534156u || header[1] != 1 || header[2] != 12 ||
        header[3] != 5120 || (header[4] != 1 && header[4] != 4) || header[7] != 0) {
        throw std::runtime_error("activation header mismatch");
    }
    Activation result;
    result.qtype = int(header[2]);
    result.k = int(header[3]);
    result.t = int(header[4]);
    result.fused = int(header[5]);
    const size_t qbytes = size_t(result.k) * result.t;
    const size_t dbytes = size_t(result.k / 32) * result.t * sizeof(half2);
    if (header[6] != qbytes + dbytes) { throw std::runtime_error("activation payload size header mismatch"); }
    result.qs.resize(qbytes);
    result.ds.resize(dbytes / sizeof(half2));
    file.read(reinterpret_cast<char*>(result.qs.data()), qbytes);
    file.read(reinterpret_cast<char*>(result.ds.data()), dbytes);
    if (!file || file.peek() != std::char_traits<char>::eof()) { throw std::runtime_error("activation payload framing mismatch"); }
    return result;
}

template <int T, bool Fused = false>
void measure(const std::vector<uint8_t>& raw, const PackedScales& packed, const std::string& label,
             const Activation* activation = nullptr) {
    constexpr int k = 5120;
    const int n = packed.rows;
    const int out_rows = Fused ? n / 2 : n;
    if constexpr (Fused) { if ((n & 1) != 0) { throw std::runtime_error("fused Q4_K tensor needs an even row count"); } }
    const size_t weight_bytes = raw.size();
    uint8_t* weights[3];
    uint8_t* device_dm[3];
    uint8_t* device_qs[3];
    uint8_t* device_raw_metadata[3];
    GroupMeta* device_groups[3];
    uint8_t* device_raw_fallback[3];
    uint8_t* device_coded_metadata[3];
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
        alloc(&device_groups[r], packed.groups.size() * sizeof(GroupMeta));
        check(cudaMemcpy(device_groups[r], packed.groups.data(), packed.groups.size() * sizeof(GroupMeta), cudaMemcpyHostToDevice), "group upload");
        alloc(&device_raw_fallback[r], packed.raw_fallback.size());
        check(cudaMemcpy(device_raw_fallback[r], packed.raw_fallback.data(), packed.raw_fallback.size(), cudaMemcpyHostToDevice), "fallback upload");
        alloc(&device_coded_metadata[r], packed.coded_metadata.size());
        check(cudaMemcpy(device_coded_metadata[r], packed.coded_metadata.data(), packed.coded_metadata.size(), cudaMemcpyHostToDevice), "coded metadata upload");
    }
    alloc(&device_debug, raw.size());
    const uint64_t blocks = uint64_t(n) * (k / 256);
    reconstruct_kernel<<<(blocks + 255) / 256, 256>>>(n, device_dm[0], device_qs[0], device_groups[0],
                                                       device_raw_fallback[0], device_coded_metadata[0], device_debug);
    check(cudaGetLastError(), "reconstruct launch");
    std::vector<uint8_t> back(raw.size());
    check(cudaMemcpy(back.data(), device_debug, back.size(), cudaMemcpyDeviceToHost), "reconstruct copy");
    if (back != raw) { throw std::runtime_error("GPU scale/quant payload full bit mismatch"); }
    check(cudaFree(device_debug), "debug free");

    if (activation && (activation->k != k || activation->t != T)) { throw std::runtime_error("activation T/K mismatch"); }
    std::vector<int8_t> hostx(size_t(k) * T);
    std::vector<half2> hostds(size_t(k / 32) * T);
    if (activation) {
        hostx = activation->qs;
        hostds = activation->ds;
    } else {
        for (size_t i = 0; i < hostx.size(); ++i) { hostx[i] = int8_t((i * 37 + i / 7 * 11) % 255 - 127); }
        for (size_t i = 0; i < hostds.size(); ++i) {
            int sum = 0;
            for (int j = 0; j < 32; ++j) { sum += hostx[i * 32 + j]; }
            hostds[i] = __floats2half2_rn(0.0625f, sum * 0.0625f);
        }
    }
    alloc(&x, hostx.size());
    check(cudaMemcpy(x, hostx.data(), hostx.size(), cudaMemcpyHostToDevice), "x upload");
    alloc(&ds, hostds.size() * sizeof(half2));
    check(cudaMemcpy(ds, hostds.data(), hostds.size() * sizeof(half2), cudaMemcpyHostToDevice), "ds upload");
    const size_t float_output_bytes = size_t(out_rows) * T * sizeof(float);
    const size_t output_guard_bytes = 64;
    alloc(&y, float_output_bytes + output_guard_bytes);
    VecArgs args{weights[0], k / 256 * 144, out_rows, k, x, ds,
                 {nullptr, y, out_rows, ninfer::ops::gguf::Epilogue::Store, nullptr, 0}};
    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
    auto reset_guard = [&](void* output, size_t bytes) {
        check(cudaMemset(reinterpret_cast<uint8_t*>(output) + bytes, 0xA5, output_guard_bytes), "guard reset");
    };
    auto verify_guard = [&](void* output, size_t bytes, const char* name) {
        std::array<uint8_t, 64> guard{};
        check(cudaMemcpy(guard.data(), reinterpret_cast<uint8_t*>(output) + bytes, guard.size(), cudaMemcpyDeviceToHost), "guard copy");
        if (!std::all_of(guard.begin(), guard.end(), [](uint8_t value) { return value == 0xA5; })) {
            throw std::runtime_error(std::string(name) + " output canary overwritten");
        }
    };
    auto baseline = [&](int rotation) {
        args.weight = weights[rotation];
        if constexpr (Fused) { vec::launch<GGML_TYPE_Q4_K, T, true, k>(args, stream); }
        else { vec::launch<GGML_TYPE_Q4_K, T, false, k>(args, stream); }
    };
    auto candidate = [&](int rotation) {
        args.weight = weights[rotation];
        launch_candidate<T, false, Fused>(args, packed, device_dm[rotation], device_qs[rotation], device_raw_metadata[rotation], device_groups[rotation],
                            device_raw_fallback[rotation], device_coded_metadata[rotation], stream);
    };
    auto raw_planar = [&](int rotation) {
        args.weight = weights[rotation];
        launch_candidate<T, true, Fused>(args, packed, device_dm[rotation], device_qs[rotation], device_raw_metadata[rotation], device_groups[rotation],
                            device_raw_fallback[rotation], device_coded_metadata[rotation], stream);
    };
    std::vector<float> reference(size_t(out_rows) * T), actual(reference.size());
    reset_guard(y, float_output_bytes);
    baseline(0);
    check(cudaStreamSynchronize(stream), "baseline sync");
    verify_guard(y, float_output_bytes, "baseline FP32");
    check(cudaMemcpy(reference.data(), y, reference.size() * sizeof(float), cudaMemcpyDeviceToHost), "baseline copy");
    reset_guard(y, float_output_bytes);
    candidate(0);
    check(cudaStreamSynchronize(stream), "candidate sync");
    verify_guard(y, float_output_bytes, "candidate FP32");
    check(cudaMemcpy(actual.data(), y, actual.size() * sizeof(float), cudaMemcpyDeviceToHost), "candidate copy");
    if (std::memcmp(reference.data(), actual.data(), reference.size() * sizeof(float)) != 0) {
        throw std::runtime_error("FP32 output bit mismatch");
    }
    __nv_bfloat16* bfout;
    const size_t bf_output_bytes = size_t(out_rows) * T * sizeof(__nv_bfloat16);
    alloc(&bfout, bf_output_bytes + output_guard_bytes);
    std::vector<uint16_t> bf_reference(size_t(out_rows) * T), bf_actual(bf_reference.size());
    args.out.bf16 = bfout;
    args.out.f32 = nullptr;
    reset_guard(bfout, bf_output_bytes);
    baseline(0);
    check(cudaStreamSynchronize(stream), "baseline bf16 sync");
    verify_guard(bfout, bf_output_bytes, "baseline BF16");
    check(cudaMemcpy(bf_reference.data(), bfout, bf_reference.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost), "baseline bf16 copy");
    reset_guard(bfout, bf_output_bytes);
    candidate(0);
    check(cudaStreamSynchronize(stream), "candidate bf16 sync");
    verify_guard(bfout, bf_output_bytes, "candidate BF16");
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
    verify_guard(y, float_output_bytes, "graph FP32");
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start), "start event");
    check(cudaEventCreate(&stop), "stop event");
    const uint64_t blocks_count = uint64_t(n) * (k / 256);
    const uint64_t groups = (blocks_count + 63) / 64;
    const uint64_t format_bytes = packed.dm.size() + packed.qs.size() + packed.groups.size() * sizeof(GroupMeta) +
                                  packed.raw_fallback.size() + packed.coded_metadata.size() + 64;
    const uint64_t planar_bytes = packed.dm.size() + packed.qs.size() + packed.raw_metadata.size() + 64;
    const KernelResource baseline_resource = kernel_resource(vec::kernel<GGML_TYPE_Q4_K, T, Fused, k>);
    const KernelResource raw_resource = kernel_resource(scale_kernel<T, true, Fused>);
    const KernelResource coded_resource = kernel_resource(scale_kernel<T, false, Fused>);
    std::cout << "{\"matrix\":\"" << label << "\",\"T\":" << T << ",\"N\":" << n << ",\"K\":" << k
              << ",\"output_rows\":" << out_rows << ",\"fused_swiglu\":" << (Fused ? "true" : "false")
              << ",\"baseline_bytes\":" << weight_bytes << ",\"raw_planar_bytes_including_headers\":" << planar_bytes
              << ",\"weight_halves_included\":" << (Fused ? 2 : 1)
              << ",\"scale_format_bytes\":" << format_bytes
              << ",\"metadata_groups\":" << groups << ",\"eligible_blocks\":" << packed.both_eligible
              << ",\"eligible_rate\":" << double(packed.both_eligible) / double(blocks_count)
              << ",\"activation_snapshot\":" << (activation ? "true" : "false")
              << ",\"activation_fused\":" << (activation ? activation->fused : -1)
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
    verify_guard(y, float_output_bytes, "timed graph FP32");
    for (auto graph : graphs) { check(cudaGraphExecDestroy(graph), "graph free"); }
    check(cudaEventDestroy(start), "event free");
    check(cudaEventDestroy(stop), "event free");
    for (int r = 0; r < 3; ++r) {
        check(cudaFree(weights[r]), "free");
        check(cudaFree(device_dm[r]), "free");
        check(cudaFree(device_qs[r]), "free");
        check(cudaFree(device_raw_metadata[r]), "free");
        check(cudaFree(device_groups[r]), "free");
        check(cudaFree(device_raw_fallback[r]), "free");
        check(cudaFree(device_coded_metadata[r]), "free");
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
    const uint64_t bytes = packed.dm.size() + packed.qs.size() + packed.groups.size() * sizeof(GroupMeta) +
                           packed.raw_fallback.size() + packed.coded_metadata.size() + 64;
    std::cout << "{\"matrix\":\"" << label << "\",\"N\":" << rows << ",\"K\":" << k
              << ",\"blocks\":" << blocks << ",\"groups\":" << groups
              << ",\"scale_has_63_blocks\":" << packed.scale_eligible
              << ",\"min_has_63_blocks\":" << packed.min_eligible
              << ",\"both_single_63_blocks\":" << packed.both_eligible
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
    int one_scale[8] = {1, 2, 63, 4, 5, 6, 7, 8};
    int one_min[8] = {9, 10, 11, 12, 13, 63, 15, 16};
    encode(64, one_scale, one_min);
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
    if (packed.groups.size() != 3 || packed.groups[0].eligible != 0 ||
        packed.groups[1].eligible != uint64_t(1) || packed.groups[2].eligible != 0) {
        throw std::runtime_error("synthetic eligibility boundary mismatch");
    }
    if (packed.multiple_scale_max < 3 || packed.multiple_min_max < 2 ||
        packed.scale_eligible != 5 || packed.min_eligible != 5 || packed.both_eligible != 1) {
        throw std::runtime_error("synthetic eligibility edge counts mismatch");
    }
}

int main(int argc, char** argv) {
    if ((argc != 7 && argc != 8) || (std::string(argv[6]) != "scan" && std::string(argv[6]) != "bench")) {
        std::cerr << "model offset N K label scan|bench [activation]\n";
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
        if (argc == 8) { throw std::runtime_error("activation is bench-only"); }
        run_scan(raw, packed, rows, k, argv[5]);
        return 0;
    }
    if (argc == 8) {
        const Activation activation = load_activation(argv[7]);
        if (activation.t == 1 && activation.fused == 0) { measure<1>(raw, packed, argv[5], &activation); }
        else if (activation.t == 4 && activation.fused == 0) { measure<4>(raw, packed, argv[5], &activation); }
        else if (activation.t == 1 && activation.fused == 1) { measure<1, true>(raw, packed, argv[5], &activation); }
        else if (activation.t == 4 && activation.fused == 1) { measure<4, true>(raw, packed, argv[5], &activation); }
        return 0;
    }
    measure<1>(raw, packed, argv[5]);
    measure<2>(raw, packed, argv[5]);
    measure<3>(raw, packed, argv[5]);
    measure<4>(raw, packed, argv[5]);
    measure<8>(raw, packed, argv[5]);
    return 0;
}
