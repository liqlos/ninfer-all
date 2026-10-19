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

struct PackedScales {
    int rows = 0;
    int k = 0;
    std::vector<uint8_t> dm;
    std::vector<uint8_t> qs;
    std::vector<uint8_t> raw_metadata;
    std::vector<uint8_t> pair_metadata;
    std::vector<uint8_t> wide_pair_metadata;
    std::vector<uint8_t> d_code;
    std::vector<uint8_t> dmin_code;
    uint8_t d_exp_min = 0;
    uint8_t dmin_exp_min = 0;
    uint8_t d_exp_bits = 0;
    uint8_t dmin_exp_bits = 0;
    uint8_t sign_bits = 0;
    uint8_t field_bits = 16;
    bool coded = false;
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

static uint16_t pair_code_host(const uint8_t* data, uint64_t block, unsigned index) {
    const uint64_t bit = uint64_t(index) * 12;
    const uint64_t at = block * 12 + (bit >> 3);
    const unsigned shift = unsigned(bit & 7);
    const uint16_t word = uint16_t(data[at]) | (uint16_t(data[at + 1]) << 8);
    return uint16_t((word >> shift) & 0x0FFFu);
}

static uint16_t wide_pair_code_host(const uint8_t* data, uint64_t block, unsigned index) {
    const uint32_t bit = index * 12u;
    const uint32_t word_index = bit >> 5;
    const uint32_t shift = bit & 31u;
    uint32_t lo = 0, hi = 0;
    std::memcpy(&lo, data + block * 12 + word_index * 4, 4);
    std::memcpy(&hi, data + block * 12 + (word_index + 1) * 4, 4);
    const uint64_t pair = uint64_t(lo) | (uint64_t(hi) << 32);
    return uint16_t((pair >> shift) & 0x0FFFu);
}

static void validate_wide_pair_codec() {
    std::vector<uint8_t> stream(12 * 2 + 4, 0);
    for (unsigned index = 0; index < 8; ++index) {
        for (unsigned scale = 0; scale < 64; ++scale) {
            for (unsigned minimum = 0; minimum < 64; ++minimum) {
                std::fill(stream.begin(), stream.end(), 0);
                const unsigned bit = index * 12;
                const unsigned at = bit >> 3;
                const unsigned shift = bit & 7u;
                const uint16_t code = uint16_t(scale | (minimum << 6));
                const uint32_t shifted = uint32_t(code) << shift;
                stream[at] = uint8_t(shifted);
                stream[at + 1] = uint8_t(shifted >> 8);
                if (at + 2 < 12) { stream[at + 2] |= uint8_t(shifted >> 16); }
                const uint16_t decoded = wide_pair_code_host(stream.data(), 0, index);
                if (decoded != code || (decoded & 63u) != scale || ((decoded >> 6) & 63u) != minimum) {
                    throw std::runtime_error("wide pair metadata exhaustive value fixture failed");
                }
            }
        }
    }
    std::fill(stream.begin(), stream.end(), 0);
    for (unsigned block = 0; block < 2; ++block) {
        for (unsigned index = 0; index < 8; ++index) {
            const unsigned scale = (block * 19 + index * 23) & 63u;
            const unsigned minimum = (block * 37 + index * 11) & 63u;
            const uint16_t code = uint16_t(scale | (minimum << 6));
            const unsigned bit = index * 12;
            const unsigned at = block * 12 + (bit >> 3);
            const unsigned shift = bit & 7u;
            const uint32_t shifted = uint32_t(code) << shift;
            stream[at] = uint8_t(shifted);
            stream[at + 1] = uint8_t(shifted >> 8);
            if ((bit >> 3) + 2 < 12) { stream[at + 2] |= uint8_t(shifted >> 16); }
            if (wide_pair_code_host(stream.data(), block, index) != code) {
                throw std::runtime_error("wide pair metadata crossing-block fixture failed");
            }
        }
    }
}

static void store_scale_min_host(uint8_t* out, const uint8_t scales[8], const uint8_t minima[8]) {
    for (int j = 0; j < 4; ++j) {
        out[j] = uint8_t((scales[j] & 63u) | ((scales[j + 4] >> 4) << 6));
        out[j + 4] = uint8_t((minima[j] & 63u) | ((minima[j + 4] >> 4) << 6));
        out[j + 8] = uint8_t((scales[j + 4] & 15u) | ((minima[j + 4] & 15u) << 4));
    }
}

static void pack_pair_metadata(const uint8_t* raw, uint64_t block, uint8_t* out) {
    uint8_t scales[8], minima[8];
    const uint8_t* source = raw + block * 144 + 4;
    for (int j = 0; j < 8; ++j) {
        int scale, minimum;
        scale_min_host(source, j, scale, minimum);
        scales[j] = uint8_t(scale);
        minima[j] = uint8_t(minimum);
    }
    for (int j = 0; j < 8; ++j) {
        const uint16_t code = uint16_t(scales[j] | (uint16_t(minima[j]) << 6));
        const uint64_t bit = uint64_t(j) * 12;
        const uint64_t at = block * 12 + (bit >> 3);
        const unsigned shift = unsigned(bit & 7);
        const uint16_t shifted = uint16_t(code << shift);
        out[at] |= uint8_t(shifted);
        out[at + 1] |= uint8_t(shifted >> 8);
    }
}

static void unpack_pair_metadata(const uint8_t* packed, uint64_t block, uint8_t* out) {
    uint8_t scales[8], minima[8];
    for (int j = 0; j < 8; ++j) {
        const uint16_t code = pair_code_host(packed, block, unsigned(j));
        scales[j] = uint8_t(code & 63u);
        minima[j] = uint8_t((code >> 6) & 63u);
    }
    store_scale_min_host(out, scales, minima);
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

static unsigned width_for_range(unsigned count) {
    unsigned bits = 0;
    unsigned limit = 1;
    while (limit < count) { ++bits; limit <<= 1; }
    return bits;
}

static uint16_t decode_half_field(const std::vector<uint8_t>& code, uint64_t block,
                                  unsigned exp_min, unsigned exp_bits, unsigned sign_bits) {
    const uint32_t at = uint32_t(block * (10 + exp_bits + sign_bits));
    const uint16_t sign = sign_bits ? uint16_t(read_bits_host(code.data(), at, 1)) : 0;
    const uint16_t exp = uint16_t(exp_min + read_bits_host(code.data(), at + sign_bits, exp_bits));
    const uint16_t mantissa = uint16_t(read_bits_host(code.data(), at + sign_bits + exp_bits, 10));
    return uint16_t((sign << 15) | (exp << 10) | mantissa);
}

static PackedScales pack_scales(const std::vector<uint8_t>& raw, int rows, int k) {
    const uint64_t blocks = uint64_t(rows) * (k / 256);
    if (raw.size() != blocks * 144) { throw std::runtime_error("Q4_K matrix payload size"); }
    PackedScales out;
    out.rows = rows;
    out.k = k;
    out.dm.resize(blocks * 4);
    out.qs.resize(blocks * 128);
    out.raw_metadata.resize(blocks * 12);
    out.pair_metadata.resize(blocks * 12, 0);
    unsigned exp_min[2] = {31, 31};
    unsigned exp_max[2] = {0, 0};
    bool any_sign = false;
    for (uint64_t b = 0; b < blocks; ++b) {
        const uint8_t* src = raw.data() + b * 144;
        std::memcpy(out.dm.data() + b * 4, src, 4);
        std::memcpy(out.raw_metadata.data() + b * 12, src + 4, 12);
        pack_pair_metadata(raw.data(), b, out.pair_metadata.data());
        std::memcpy(out.qs.data() + b * 128, src + 16, 128);
        for (int field = 0; field < 2; ++field) {
            const uint16_t half = uint16_t(src[field * 2]) | (uint16_t(src[field * 2 + 1]) << 8);
            any_sign |= (half & 0x8000u) != 0;
            const unsigned exp = (half >> 10) & 31u;
            exp_min[field] = std::min(exp_min[field], exp);
            exp_max[field] = std::max(exp_max[field], exp);
        }
    }
    out.wide_pair_metadata = out.pair_metadata;
    out.wide_pair_metadata.resize(out.pair_metadata.size() + 4, 0);
    out.d_exp_min = uint8_t(exp_min[0]);
    out.dmin_exp_min = uint8_t(exp_min[1]);
    out.d_exp_bits = uint8_t(width_for_range(exp_max[0] - exp_min[0] + 1));
    out.dmin_exp_bits = uint8_t(width_for_range(exp_max[1] - exp_min[1] + 1));
    out.sign_bits = any_sign ? 1 : 0;
    const unsigned d_width = 10 + out.d_exp_bits + out.sign_bits;
    const unsigned dmin_width = 10 + out.dmin_exp_bits + out.sign_bits;
    out.field_bits = uint8_t(std::max(d_width, dmin_width));
    out.coded = out.field_bits < 16;
    if (out.coded) {
        out.d_code.resize(std::max<size_t>(16, ((blocks * d_width + 7) / 8 + 15) & ~size_t(15)), 0);
        out.dmin_code.resize(std::max<size_t>(16, ((blocks * dmin_width + 7) / 8 + 15) & ~size_t(15)), 0);
        for (uint64_t b = 0; b < blocks; ++b) {
            const uint8_t* src = raw.data() + b * 144;
            for (int field = 0; field < 2; ++field) {
                const uint16_t half = uint16_t(src[field * 2]) | (uint16_t(src[field * 2 + 1]) << 8);
                const unsigned min_exp = field == 0 ? out.d_exp_min : out.dmin_exp_min;
                const unsigned exp_bits = field == 0 ? out.d_exp_bits : out.dmin_exp_bits;
                const unsigned width = 10 + exp_bits + out.sign_bits;
                std::vector<uint8_t>& stream = field == 0 ? out.d_code : out.dmin_code;
                const uint64_t at = b * width;
                if (out.sign_bits) { write_bits(stream, at, 1, (half >> 15) & 1u); }
                write_bits(stream, at + out.sign_bits, exp_bits, ((half >> 10) & 31u) - min_exp);
                write_bits(stream, at + out.sign_bits + exp_bits, 10, half & 1023u);
            }
        }
    } else {
        out.d_code.resize(16, 0);
        out.dmin_code.resize(16, 0);
    }
    return out;
}

static std::vector<uint8_t> unpack_cpu(const PackedScales& packed, bool pairmeta = false, bool wide_pairmeta = false) {
    const uint64_t blocks = uint64_t(packed.rows) * (packed.k / 256);
    std::vector<uint8_t> raw(blocks * 144);
    for (uint64_t b = 0; b < blocks; ++b) {
        uint8_t* dst = raw.data() + b * 144;
        std::memcpy(dst, packed.dm.data() + b * 4, 4);
        if (packed.coded) {
            const uint16_t d = decode_half_field(packed.d_code, b, packed.d_exp_min, packed.d_exp_bits, packed.sign_bits);
            const uint16_t dmin = decode_half_field(packed.dmin_code, b, packed.dmin_exp_min, packed.dmin_exp_bits, packed.sign_bits);
            dst[0] = uint8_t(d); dst[1] = uint8_t(d >> 8); dst[2] = uint8_t(dmin); dst[3] = uint8_t(dmin >> 8);
        } else { std::memcpy(dst, packed.dm.data() + b * 4, 4); }
        if (wide_pairmeta) {
            uint8_t scales[8], minima[8];
            for (unsigned j = 0; j < 8; ++j) {
                const uint16_t code = wide_pair_code_host(packed.wide_pair_metadata.data(), b, j);
                scales[j] = uint8_t(code & 63u);
                minima[j] = uint8_t((code >> 6) & 63u);
            }
            store_scale_min_host(dst + 4, scales, minima);
        } else if (pairmeta) { unpack_pair_metadata(packed.pair_metadata.data(), b, dst + 4); }
        else { std::memcpy(dst + 4, packed.raw_metadata.data() + b * 12, 12); }
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

__device__ __forceinline__ uint16_t decode_half_field_device(const uint8_t* code, uint32_t block,
                                                              uint32_t exp_min, uint32_t exp_bits,
                                                              uint32_t sign_bits) {
    const uint32_t width = 10 + exp_bits + sign_bits;
    const uint32_t at = block * width;
    const uint16_t sign = sign_bits ? uint16_t(read_bits_device(code, at, 1)) : 0;
    const uint16_t exp = uint16_t(exp_min + read_bits_device(code, at + sign_bits, exp_bits));
    const uint16_t mantissa = uint16_t(read_bits_device(code, at + sign_bits + exp_bits, 10));
    return uint16_t((sign << 15) | (exp << 10) | mantissa);
}

__device__ __forceinline__ uint16_t pair_code_device(const uint8_t* data, uint32_t block,
                                                     uint32_t index) {
    const uint32_t bit = index * 12;
    const uint32_t at = block * 12 + (bit >> 3);
    const uint32_t shift = bit & 7;
    const uint32_t word = uint32_t(data[at]) | (uint32_t(data[at + 1]) << 8);
    return uint16_t((word >> shift) & 0x0FFFu);
}

__device__ __forceinline__ void pair_scale_min_device(const uint8_t* data, uint32_t block,
                                                       uint32_t index, int& sc, int& mn) {
    const uint16_t code = pair_code_device(data, block, index);
    sc = code & 63u;
    mn = (code >> 6) & 63u;
}

__device__ __forceinline__ void wide_pair_scale_min_device(const uint8_t* data, uint32_t block,
                                                            uint32_t index, int& sc, int& mn) {
    const uint32_t bit = index * 12;
    const uint32_t word_index = bit >> 5;
    const uint32_t shift = bit & 31u;
    const uint32_t* words = reinterpret_cast<const uint32_t*>(data + block * 12);
    const uint32_t packed = __funnelshift_r(words[word_index], words[word_index + 1], shift) & 0x0FFFu;
    sc = int(packed & 63u);
    mn = int((packed >> 6) & 63u);
}

struct ScaleArgs {
    VecArgs vec;
    const uint8_t* dm;
    const uint8_t* qs;
    const uint8_t* raw_metadata;
    const uint8_t* pair_metadata;
    const uint8_t* d_code;
    const uint8_t* dmin_code;
    uint32_t d_exp_min;
    uint32_t dmin_exp_min;
    uint32_t d_exp_bits;
    uint32_t dmin_exp_bits;
    uint32_t sign_bits;
    bool coded;
};

template <int T, bool RawPlanar, bool Fused = false, bool PairMeta = false, bool WidePairMeta = false>
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
                    if constexpr (WidePairMeta) { wide_pair_scale_min_device(p.pair_metadata, block, lane % 8, sc, mn); }
                    else if constexpr (PairMeta) { pair_scale_min_device(p.pair_metadata, block, lane % 8, sc, mn); }
                    else { scale_min_k4(p.raw_metadata + block * 12, lane % 8, sc, mn); }
                    const uint16_t dword = RawPlanar || !p.coded ? *reinterpret_cast<const uint16_t*>(p.dm + block * 4) :
                        decode_half_field_device(p.d_code, block, p.d_exp_min, p.d_exp_bits, p.sign_bits);
                    const uint16_t dminword = RawPlanar || !p.coded ? *reinterpret_cast<const uint16_t*>(p.dm + block * 4 + 2) :
                        decode_half_field_device(p.dmin_code, block, p.dmin_exp_min, p.dmin_exp_bits, p.sign_bits);
                    const float2 dm = make_float2(__half2float(__ushort_as_half(dword)), __half2float(__ushort_as_half(dminword)));
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

template <int T, bool RawPlanar, bool PairMeta = false, bool WidePairMeta = false>
__launch_bounds__(256) __global__ void fused_split_kernel(ScaleArgs p) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row_slot = warp >> 1;
    const int side = warp & 1;
    constexpr int k = 5120;
    constexpr int blocks_per_row = k / 256;
    __shared__ float gate[4][T];
    for (int tile0 = blockIdx.x * 4; tile0 < p.vec.rows; tile0 += gridDim.x * 4) {
        const int row0 = tile0 + row_slot;
        float acc[T];
#pragma unroll
        for (int j = 0; j < T; ++j) { acc[j] = 0.0f; }
        if (row0 < p.vec.rows) {
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
                const int row = row0 + side * p.vec.rows;
                const uint32_t block = uint32_t(row) * uint32_t(blocks_per_row) + uint32_t(4 * it + lane / 8);
                int sc, mn;
                if constexpr (WidePairMeta) { wide_pair_scale_min_device(p.pair_metadata, block, lane % 8, sc, mn); }
                else if constexpr (PairMeta) { pair_scale_min_device(p.pair_metadata, block, lane % 8, sc, mn); }
                else { scale_min_k4(p.raw_metadata + block * 12, lane % 8, sc, mn); }
                const uint16_t dword = RawPlanar ? *reinterpret_cast<const uint16_t*>(p.dm + block * 4) :
                    decode_half_field_device(p.d_code, block, p.d_exp_min, p.d_exp_bits, p.sign_bits);
                const uint16_t dminword = RawPlanar ? *reinterpret_cast<const uint16_t*>(p.dm + block * 4 + 2) :
                    decode_half_field_device(p.dmin_code, block, p.dmin_exp_min, p.dmin_exp_bits, p.sign_bits);
                const float dscale = __half2float(__ushort_as_half(dword));
                const float dmin = __half2float(__ushort_as_half(dminword));
                const float2 dm = make_float2(dscale, dmin);
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
                for (int j = 0; j < T; ++j) { acc[j] += Decoder<GGML_TYPE_Q4_K>::dot(sl, a[j], d[j], sum[j]); }
            }
        }
#pragma unroll
        for (int j = 0; j < T; ++j) {
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) { acc[j] += __shfl_xor_sync(0xffffffffu, acc[j], offset); }
        }
        if (lane == 0 && row0 < p.vec.rows && side == 0) {
#pragma unroll
            for (int j = 0; j < T; ++j) { gate[row_slot][j] = acc[j]; }
        }
        __syncthreads();
        if (lane == 0 && row0 < p.vec.rows && side == 1) {
#pragma unroll
            for (int j = 0; j < T; ++j) { vec_store(p.vec.out, row0, j, vec_silu(gate[row_slot][j]) * acc[j]); }
        }
        __syncthreads();
    }
}

__global__ void reconstruct_kernel(int rows, const uint8_t* dm, const uint8_t* qs,
                                   const uint8_t* raw_metadata, const uint8_t* pair_metadata, const uint8_t* d_code,
                                   const uint8_t* dmin_code, uint32_t d_exp_min,
                                   uint32_t dmin_exp_min, uint32_t d_exp_bits,
                                   uint32_t dmin_exp_bits, uint32_t sign_bits, bool coded,
                                   bool pairmeta, uint8_t* raw) {
    const uint32_t block = uint32_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint32_t blocks = uint32_t(rows) * (5120 / 256);
    if (block >= blocks) { return; }
    const uint32_t row_bytes = (5120 / 256) * 144;
    const uint32_t at = (block / (5120 / 256)) * row_bytes + (block % (5120 / 256)) * 144;
    const uint16_t d = coded ? decode_half_field_device(d_code, block, d_exp_min, d_exp_bits, sign_bits) :
        *reinterpret_cast<const uint16_t*>(dm + block * 4);
    const uint16_t dmin = coded ? decode_half_field_device(dmin_code, block, dmin_exp_min, dmin_exp_bits, sign_bits) :
        *reinterpret_cast<const uint16_t*>(dm + block * 4 + 2);
    raw[at] = uint8_t(d); raw[at + 1] = uint8_t(d >> 8); raw[at + 2] = uint8_t(dmin); raw[at + 3] = uint8_t(dmin >> 8);
    if (pairmeta) {
        uint8_t scales[8], minima[8];
        for (int j = 0; j < 8; ++j) {
            int scale, minimum;
            pair_scale_min_device(pair_metadata, block, uint32_t(j), scale, minimum);
            scales[j] = uint8_t(scale);
            minima[j] = uint8_t(minimum);
        }
        for (int j = 0; j < 4; ++j) {
            raw[at + 4 + j] = uint8_t((scales[j] & 63u) | ((scales[j + 4] >> 4) << 6));
            raw[at + 8 + j] = uint8_t((minima[j] & 63u) | ((minima[j + 4] >> 4) << 6));
            raw[at + 12 + j] = uint8_t((scales[j + 4] & 15u) | ((minima[j + 4] & 15u) << 4));
        }
    } else {
        for (int j = 0; j < 12; ++j) { raw[at + 4 + j] = raw_metadata[block * 12 + j]; }
    }
    for (int j = 0; j < 128; ++j) { raw[at + 16 + j] = qs[block * 128 + j]; }
}

template <int T, bool RawPlanar, bool Fused = false, bool SplitFused = (Fused && !RawPlanar), bool PairMeta = false, bool WidePairMeta = false>
void launch_candidate(const VecArgs& vec, const PackedScales& packed,
                      const uint8_t* dm, const uint8_t* qs, const uint8_t* raw_metadata,
                      const uint8_t* pair_metadata, const uint8_t* d_code, const uint8_t* dmin_code,
                      cudaStream_t stream) {
    ScaleArgs args{vec, dm, qs, raw_metadata, pair_metadata, d_code, dmin_code, packed.d_exp_min, packed.dmin_exp_min,
                   packed.d_exp_bits, packed.dmin_exp_bits, packed.sign_bits, packed.coded};
    static const int resident = [] {
        int blocks = 0;
        if constexpr (SplitFused) {
            check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, fused_split_kernel<T, RawPlanar, PairMeta, WidePairMeta>, 256, 0), "split occupancy");
        } else {
            check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, scale_kernel<T, RawPlanar, Fused, PairMeta, WidePairMeta>, 128, 0), "scale occupancy");
        }
        return std::max(blocks, 1);
    }();
    const int groups = Fused ? (vec.rows + 3) / 4 : (packed.rows + 7) / 8;
    const int grid = std::min(groups, resident * device_facts().sm_count);
    if constexpr (SplitFused) { fused_split_kernel<T, RawPlanar, PairMeta, WidePairMeta><<<grid, 256, 0, stream>>>(args); }
    else { scale_kernel<T, RawPlanar, Fused, PairMeta, WidePairMeta><<<grid, 128, 0, stream>>>(args); }
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
static KernelResource kernel_resource(Kernel kernel, int block_threads = 128) {
    cudaFuncAttributes attributes{};
    check(cudaFuncGetAttributes(&attributes, kernel), "kernel attributes");
    int active = 0;
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, kernel, block_threads, 0), "kernel occupancy");
    int max_threads = 0;
    check(cudaDeviceGetAttribute(&max_threads, cudaDevAttrMaxThreadsPerMultiProcessor, 0), "device max threads");
    KernelResource result;
    result.registers = attributes.numRegs;
    result.shared_bytes = int(attributes.sharedSizeBytes);
    result.local_bytes = int(attributes.localSizeBytes);
    result.active_blocks = active;
    result.active_warps = active * block_threads / 32;
    result.occupancy = float(active * block_threads) / float(max_threads);
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
    uint8_t* device_pair_metadata[3];
    uint8_t* device_wide_pair_metadata[3];
    uint8_t* device_d_code[3];
    uint8_t* device_dmin_code[3];
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
        alloc(&device_pair_metadata[r], packed.pair_metadata.size());
        check(cudaMemcpy(device_pair_metadata[r], packed.pair_metadata.data(), packed.pair_metadata.size(), cudaMemcpyHostToDevice), "pair metadata upload");
        alloc(&device_wide_pair_metadata[r], packed.wide_pair_metadata.size());
        check(cudaMemcpy(device_wide_pair_metadata[r], packed.wide_pair_metadata.data(), packed.wide_pair_metadata.size(), cudaMemcpyHostToDevice), "wide pair metadata upload");
        alloc(&device_d_code[r], packed.d_code.size());
        check(cudaMemcpy(device_d_code[r], packed.d_code.data(), packed.d_code.size(), cudaMemcpyHostToDevice), "d stream upload");
        alloc(&device_dmin_code[r], packed.dmin_code.size());
        check(cudaMemcpy(device_dmin_code[r], packed.dmin_code.data(), packed.dmin_code.size(), cudaMemcpyHostToDevice), "dmin stream upload");
    }
    alloc(&device_debug, raw.size());
    const uint64_t blocks = uint64_t(n) * (k / 256);
    reconstruct_kernel<<<(blocks + 255) / 256, 256>>>(n, device_dm[0], device_qs[0], device_raw_metadata[0], device_pair_metadata[0],
        device_d_code[0], device_dmin_code[0], packed.d_exp_min, packed.dmin_exp_min, packed.d_exp_bits,
        packed.dmin_exp_bits, packed.sign_bits, packed.coded, true, device_debug);
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
        launch_candidate<T, false, Fused, false, true>(args, packed, device_dm[rotation], device_qs[rotation], device_raw_metadata[rotation], device_pair_metadata[rotation],
                            device_d_code[rotation], device_dmin_code[rotation], stream);
    };
    auto raw_planar = [&](int rotation) {
        args.weight = weights[rotation];
        launch_candidate<T, true, Fused, false, false>(args, packed, device_dm[rotation], device_qs[rotation], device_raw_metadata[rotation], device_pair_metadata[rotation],
                            device_d_code[rotation], device_dmin_code[rotation], stream);
    };
    auto raw_split = [&](int rotation) {
        args.weight = weights[rotation];
        launch_candidate<T, true, true, true, false>(args, packed, device_dm[rotation], device_qs[rotation], device_raw_metadata[rotation], device_pair_metadata[rotation],
                            device_d_code[rotation], device_dmin_code[rotation], stream);
    };
    auto raw_pairmeta = [&](int rotation) {
        args.weight = weights[rotation];
        launch_candidate<T, true, Fused, false, true>(args, packed, device_dm[rotation], device_qs[rotation], device_raw_metadata[rotation], device_pair_metadata[rotation],
                            device_d_code[rotation], device_dmin_code[rotation], stream);
    };
    auto coded_original = [&](int rotation) {
        args.weight = weights[rotation];
        launch_candidate<T, false, true, false, false>(args, packed, device_dm[rotation], device_qs[rotation], device_raw_metadata[rotation], device_pair_metadata[rotation],
                            device_d_code[rotation], device_dmin_code[rotation], stream);
    };
    auto coded_split = [&](int rotation) {
        args.weight = weights[rotation];
        launch_candidate<T, false, true, true, false>(args, packed, device_dm[rotation], device_qs[rotation], device_raw_metadata[rotation], device_pair_metadata[rotation],
                            device_d_code[rotation], device_dmin_code[rotation], stream);
    };
    auto raw_pairmeta_split = [&](int rotation) {
        args.weight = weights[rotation];
        launch_candidate<T, true, true, true, true>(args, packed, device_dm[rotation], device_qs[rotation], device_raw_metadata[rotation], device_pair_metadata[rotation],
                            device_d_code[rotation], device_dmin_code[rotation], stream);
    };
    auto coded_pairmeta_split = [&](int rotation) {
        args.weight = weights[rotation];
        launch_candidate<T, false, true, true, true>(args, packed, device_dm[rotation], device_qs[rotation], device_raw_metadata[rotation], device_pair_metadata[rotation],
                            device_d_code[rotation], device_dmin_code[rotation], stream);
    };
    auto coded_wide_pairmeta = [&](int rotation) {
        args.weight = weights[rotation];
        launch_candidate<T, false, Fused, false, true, true>(args, packed, device_dm[rotation], device_qs[rotation], device_raw_metadata[rotation], device_wide_pair_metadata[rotation],
                            device_d_code[rotation], device_dmin_code[rotation], stream);
    };
    constexpr int mode_count = Fused ? 10 : 5;
    auto launch_mode = [&](int mode, int rotation) {
        if (mode == 0) { baseline(rotation); }
        else if constexpr (Fused) {
            if (mode == 1) { raw_planar(rotation); }
            else if (mode == 2) { raw_split(rotation); }
            else if (mode == 3) { coded_original(rotation); }
            else if (mode == 4) { coded_split(rotation); }
            else if (mode == 5) { raw_pairmeta(rotation); }
            else if (mode == 6) { candidate(rotation); }
            else if (mode == 7) { raw_pairmeta_split(rotation); }
            else if (mode == 8) { coded_pairmeta_split(rotation); }
            else { coded_wide_pairmeta(rotation); }
        } else {
            if (mode == 1) { raw_planar(rotation); }
            else if (mode == 2) { raw_pairmeta(rotation); }
            else if (mode == 3) { candidate(rotation); }
            else { coded_wide_pairmeta(rotation); }
        }
    };
    std::vector<float> reference(size_t(out_rows) * T), actual(reference.size());
    reset_guard(y, float_output_bytes);
    baseline(0);
    check(cudaStreamSynchronize(stream), "baseline sync");
    verify_guard(y, float_output_bytes, "baseline FP32");
    check(cudaMemcpy(reference.data(), y, reference.size() * sizeof(float), cudaMemcpyDeviceToHost), "baseline copy");
    for (int mode = 1; mode < mode_count; ++mode) {
        reset_guard(y, float_output_bytes);
        launch_mode(mode, 0);
        check(cudaStreamSynchronize(stream), "control fp32 sync");
        verify_guard(y, float_output_bytes, "control FP32");
        check(cudaMemcpy(actual.data(), y, actual.size() * sizeof(float), cudaMemcpyDeviceToHost), "control fp32 copy");
        if (std::memcmp(reference.data(), actual.data(), reference.size() * sizeof(float)) != 0) {
            throw std::runtime_error("control FP32 output bit mismatch at mode " + std::to_string(mode));
        }
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
    for (int mode = 1; mode < mode_count; ++mode) {
        reset_guard(bfout, bf_output_bytes);
        launch_mode(mode, 0);
        check(cudaStreamSynchronize(stream), "control bf16 sync");
        verify_guard(bfout, bf_output_bytes, "control BF16");
        check(cudaMemcpy(bf_actual.data(), bfout, bf_actual.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost), "control bf16 copy");
        if (bf_reference != bf_actual) { throw std::runtime_error("control BF16 output bit mismatch at mode " + std::to_string(mode)); }
    }
    check(cudaFree(bfout), "bf16 free");
    args.out.bf16 = nullptr;
    args.out.f32 = y;

    cudaGraphExec_t graphs[mode_count];
    for (int mode = 0; mode < mode_count; ++mode) {
        cudaGraph_t graph;
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "capture");
        for (int r = 0; r < 3; ++r) {
            launch_mode(mode, r);
        }
        check(cudaStreamEndCapture(stream, &graph), "capture end");
        check(cudaGraphInstantiate(&graphs[mode], graph, 0), "instantiate");
        check(cudaGraphDestroy(graph), "graph destroy");
    }
    for (int i = 0; i < 4; ++i) {
        for (int mode = 0; mode < mode_count; ++mode) { check(cudaGraphLaunch(graphs[mode], stream), "graph warmup"); }
    }
    check(cudaStreamSynchronize(stream), "warmup sync");
    for (int mode = 0; mode < mode_count; ++mode) {
        reset_guard(y, float_output_bytes);
        check(cudaGraphLaunch(graphs[mode], stream), "graph verification launch");
        check(cudaStreamSynchronize(stream), "graph verification sync");
        verify_guard(y, float_output_bytes, "graph FP32");
        check(cudaMemcpy(actual.data(), y, actual.size() * sizeof(float), cudaMemcpyDeviceToHost), "graph output copy");
        if (std::memcmp(reference.data(), actual.data(), reference.size() * sizeof(float)) != 0) {
            throw std::runtime_error("graph FP32 output bit mismatch at mode " + std::to_string(mode));
        }
    }
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start), "start event");
    check(cudaEventCreate(&stop), "stop event");
    const uint64_t format_bytes = packed.qs.size() + packed.pair_metadata.size() + packed.d_code.size() +
                                  packed.dmin_code.size() + 64;
    const uint64_t wide_format_bytes = packed.qs.size() + packed.wide_pair_metadata.size() + packed.d_code.size() +
                                       packed.dmin_code.size() + 64;
    const uint64_t planar_bytes = packed.dm.size() + packed.qs.size() + packed.raw_metadata.size() + 64;
    const KernelResource baseline_resource = kernel_resource(vec::kernel<GGML_TYPE_Q4_K, T, Fused, k>);
    const KernelResource raw_resource = kernel_resource(scale_kernel<T, true, Fused>);
    const KernelResource raw_split_resource = [] {
        if constexpr (Fused) { return kernel_resource(fused_split_kernel<T, true>, 256); }
        else { return KernelResource{}; }
    }();
    const KernelResource coded_original_resource = [] {
        if constexpr (Fused) { return kernel_resource(scale_kernel<T, false, true>); }
        else { return KernelResource{}; }
    }();
    const KernelResource coded_resource = [] {
        if constexpr (Fused) { return kernel_resource(fused_split_kernel<T, false>, 256); }
        else { return kernel_resource(scale_kernel<T, false, false>); }
    }();
    const KernelResource raw_pair_resource = kernel_resource(scale_kernel<T, true, Fused, true>);
    const KernelResource coded_pair_resource = kernel_resource(scale_kernel<T, false, Fused, true>);
    const KernelResource coded_wide_pair_resource = kernel_resource(scale_kernel<T, false, Fused, true, true>);
    const KernelResource raw_pair_split_resource = [] {
        if constexpr (Fused) { return kernel_resource(fused_split_kernel<T, true, true>, 256); }
        else { return KernelResource{}; }
    }();
    const KernelResource coded_pair_split_resource = [] {
        if constexpr (Fused) { return kernel_resource(fused_split_kernel<T, false, true>, 256); }
        else { return KernelResource{}; }
    }();
    std::cout << "{\"matrix\":\"" << label << "\",\"T\":" << T << ",\"N\":" << n << ",\"K\":" << k
              << ",\"output_rows\":" << out_rows << ",\"fused_swiglu\":" << (Fused ? "true" : "false")
              << ",\"baseline_bytes\":" << weight_bytes << ",\"raw_planar_bytes_including_headers\":" << planar_bytes
              << ",\"raw_pairmeta_bytes_including_headers\":" << (packed.dm.size() + packed.qs.size() + packed.pair_metadata.size() + 64)
              << ",\"coded_pairmeta_bytes_including_headers\":" << format_bytes
              << ",\"coded_wide_pairmeta_bytes_including_headers_and_padding\":" << wide_format_bytes
              << ",\"wide_pairmeta_padding_bytes\":" << (packed.wide_pair_metadata.size() - packed.pair_metadata.size())
              << ",\"weight_halves_included\":" << (Fused ? 2 : 1)
              << ",\"scale_format_bytes\":" << format_bytes
              << ",\"d_exp_min\":" << unsigned(packed.d_exp_min) << ",\"d_exp_bits\":" << unsigned(packed.d_exp_bits)
              << ",\"dmin_exp_min\":" << unsigned(packed.dmin_exp_min) << ",\"dmin_exp_bits\":" << unsigned(packed.dmin_exp_bits)
              << ",\"sign_bits\":" << unsigned(packed.sign_bits) << ",\"half_field_bits_max\":" << unsigned(packed.field_bits)
              << ",\"coded\":" << (packed.coded ? "true" : "false")
              << ",\"activation_snapshot\":" << (activation ? "true" : "false")
              << ",\"activation_fused\":" << (activation ? activation->fused : -1)
              << ",\"fp32_output_bit_exact\":true,\"bf16_output_bit_exact\":true,\"gpu_all_payload_bytes_exact\":true,\"rotations\":3,\"graphs\":true,\"resources\":{";
    print_resource("baseline", baseline_resource);
    std::cout << ',';
    print_resource("raw_planar", raw_resource);
    std::cout << ',';
    print_resource(Fused ? "coded_split_warp" : "coded", coded_resource);
    std::cout << ',';
    print_resource("raw_pairmeta", raw_pair_resource);
    std::cout << ',';
    print_resource("coded_pairmeta", coded_pair_resource);
    std::cout << ',';
    print_resource("coded_wide_pairmeta", coded_wide_pair_resource);
    if constexpr (Fused) {
        std::cout << ',';
        print_resource("raw_split_warp", raw_split_resource);
        std::cout << ',';
        print_resource("coded_original_fused", coded_original_resource);
        std::cout << ',';
        print_resource("raw_pairmeta_split", raw_pair_split_resource);
        std::cout << ',';
        print_resource("coded_pairmeta_split", coded_pair_split_resource);
    }
    std::cout << "},\"samples_ms\":[";
    int sample = 0;
    for (int round = 0; round < 5; ++round) {
        for (int order = 0; order < mode_count; ++order) {
            const int mode = (round + order) % mode_count;
            check(cudaEventRecord(start, stream), "start record");
            for (int repeat = 0; repeat < 10; ++repeat) { check(cudaGraphLaunch(graphs[mode], stream), "graph launch"); }
            check(cudaEventRecord(stop, stream), "stop record");
            check(cudaEventSynchronize(stop), "event sync");
            float ms;
            check(cudaEventElapsedTime(&ms, start, stop), "elapsed time");
            if (sample++) { std::cout << ','; }
            const char* mode_name;
            if (mode == 0) { mode_name = "baseline"; }
            else if (mode == 1) { mode_name = "raw_planar_old_metadata"; }
            else if (!Fused && mode == 2) { mode_name = "raw_pairmeta"; }
            else if (!Fused && mode == 3) { mode_name = "coded_pairmeta"; }
            else if (!Fused) { mode_name = "coded_wide_pairmeta"; }
            else if (mode == 2) { mode_name = "raw_split_old_metadata"; }
            else if (mode == 3) { mode_name = "coded_old_metadata"; }
            else if (mode == 4) { mode_name = "coded_split_old_metadata"; }
            else if (mode == 5) { mode_name = "raw_pairmeta"; }
            else if (mode == 6) { mode_name = "coded_pairmeta"; }
            else if (mode == 7) { mode_name = "raw_pairmeta_split"; }
            else if (mode == 8) { mode_name = "coded_pairmeta_split"; }
            else { mode_name = "coded_wide_pairmeta_fused"; }
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
        check(cudaFree(device_pair_metadata[r]), "free");
        check(cudaFree(device_wide_pair_metadata[r]), "free");
        check(cudaFree(device_d_code[r]), "free");
        check(cudaFree(device_dmin_code[r]), "free");
    }
    check(cudaFree(x), "free");
    check(cudaFree(ds), "free");
    check(cudaFree(y), "free");
    check(cudaStreamDestroy(stream), "stream free");
}

static void run_scan(const std::vector<uint8_t>& raw, const PackedScales& packed, int rows, int k,
                     const char* label) {
    const std::vector<uint8_t> restored = unpack_cpu(packed);
    const std::vector<uint8_t> pair_restored = unpack_cpu(packed, true);
    const std::vector<uint8_t> wide_pair_restored = unpack_cpu(packed, true, true);
    if (restored != raw || pair_restored != raw || wide_pair_restored != raw) { throw std::runtime_error("CPU full Q4_K payload bit mismatch"); }
    const uint64_t blocks = uint64_t(rows) * (k / 256);
    const uint64_t raw_bytes = raw.size();
    const uint64_t bytes = packed.qs.size() + packed.pair_metadata.size() + packed.d_code.size() +
                           packed.dmin_code.size() + 64;
    const uint64_t wide_bytes = packed.qs.size() + packed.wide_pair_metadata.size() + packed.d_code.size() +
                                packed.dmin_code.size() + 64;
    const uint64_t raw_pair_bytes = packed.dm.size() + packed.qs.size() + packed.pair_metadata.size() + 64;
    std::cout << "{\"matrix\":\"" << label << "\",\"N\":" << rows << ",\"K\":" << k
              << ",\"blocks\":" << blocks << ",\"d_exp_min\":" << unsigned(packed.d_exp_min)
              << ",\"d_exp_bits\":" << unsigned(packed.d_exp_bits) << ",\"dmin_exp_min\":" << unsigned(packed.dmin_exp_min)
              << ",\"dmin_exp_bits\":" << unsigned(packed.dmin_exp_bits) << ",\"sign_bits\":" << unsigned(packed.sign_bits)
              << ",\"half_field_bits_max\":" << unsigned(packed.field_bits) << ",\"coded\":" << (packed.coded ? "true" : "false")
              << ",\"pair_metadata_bytes\":" << packed.pair_metadata.size()
              << ",\"pair_metadata_all_64x64_values_exact\":true,\"raw_pairmeta_bytes_including_headers\":" << raw_pair_bytes
              << ",\"wide_pairmeta_padding_bytes\":" << (packed.wide_pair_metadata.size() - packed.pair_metadata.size())
              << ",\"wide_pairmeta_bytes_including_padding_and_headers\":" << wide_bytes
              << ",\"cpu_all_payload_bytes_exact\":true,\"cpu_pairmeta_full_payload_bytes_exact\":true,\"cpu_wide_pairmeta_full_payload_bytes_exact\":true,\"baseline_bytes\":" << raw_bytes
              << ",\"candidate_bytes_including_headers_and_alignment\":" << bytes
              << ",\"net_savings_bytes\":" << (raw_bytes > bytes ? raw_bytes - bytes : 0)
              << ",\"candidate_larger_bytes\":" << (bytes > raw_bytes ? bytes - raw_bytes : 0)
              << ",\"wide_net_savings_bytes\":" << (raw_bytes > wide_bytes ? raw_bytes - wide_bytes : 0)
              << ",\"wide_candidate_larger_bytes\":" << (wide_bytes > raw_bytes ? wide_bytes - raw_bytes : 0) << "}\n";
}

static std::vector<uint8_t> synthetic_payload() {
    constexpr int rows = 131;
    std::vector<uint8_t> raw(size_t(rows) * 144);
    for (size_t i = 0; i < raw.size(); ++i) {
        if (i % 144 >= 4) { raw[i] = uint8_t(i * 73 + i / 11 * 19); }
    }
    for (int b = 0; b < rows; ++b) {
        const uint16_t d = uint16_t(((b % 5) << 10) | ((b * 19) & 1023));
        const uint16_t dmin = uint16_t(((2 + b % 6) << 10) | ((b * 37) & 1023));
        uint8_t* dst = raw.data() + size_t(b) * 144;
        dst[0] = uint8_t(d); dst[1] = uint8_t(d >> 8); dst[2] = uint8_t(dmin); dst[3] = uint8_t(dmin >> 8);
    }
    raw[130 * 144 + 1] |= 0x80;
    return raw;
}

static std::vector<uint8_t> pair_metadata_fixture() {
    constexpr int blocks = 512;
    std::vector<uint8_t> raw(size_t(blocks) * 144, 0);
    for (int b = 0; b < blocks; ++b) {
        uint8_t scales[8], minima[8];
        for (int j = 0; j < 8; ++j) {
            const int pair = b * 8 + j;
            scales[j] = uint8_t((pair >> 6) & 63);
            minima[j] = uint8_t(pair & 63);
        }
        uint8_t* dst = raw.data() + size_t(b) * 144;
        const uint16_t d = uint16_t(((b % 5) << 10) | ((b * 19) & 1023));
        const uint16_t dmin = uint16_t(((2 + b % 6) << 10) | ((b * 37) & 1023));
        dst[0] = uint8_t(d); dst[1] = uint8_t(d >> 8); dst[2] = uint8_t(dmin); dst[3] = uint8_t(dmin >> 8);
        store_scale_min_host(dst + 4, scales, minima);
        for (int q = 0; q < 128; ++q) { dst[16 + q] = uint8_t(b * 73 + q * 29); }
    }
    return raw;
}

static void run_cpu_edge_checks() {
    const std::vector<uint8_t> synthetic = synthetic_payload();
    const PackedScales packed = pack_scales(synthetic, 131, 256);
    if (unpack_cpu(packed) != synthetic) { throw std::runtime_error("synthetic CPU full payload mismatch"); }
    if (!packed.coded || packed.d_exp_bits != 3 || packed.dmin_exp_bits != 3 ||
        packed.sign_bits != 1 || packed.field_bits != 14) {
        throw std::runtime_error("synthetic exponent-window framing mismatch");
    }
    std::vector<uint8_t> fallback = synthetic;
    fallback[0] = uint8_t(fallback[0] & 0xFF);
    fallback[1] = uint8_t((fallback[1] & 0x83) | (31u << 2));
    const PackedScales raw_fallback = pack_scales(fallback, 131, 256);
    if (raw_fallback.coded || unpack_cpu(raw_fallback) != fallback) {
        throw std::runtime_error("synthetic full-width raw fallback mismatch");
    }
    const std::vector<uint8_t> pair_fixture = pair_metadata_fixture();
    const PackedScales pair_packed = pack_scales(pair_fixture, 512, 256);
    if (unpack_cpu(pair_packed) != pair_fixture || unpack_cpu(pair_packed, true) != pair_fixture ||
        unpack_cpu(pair_packed, true, true) != pair_fixture) {
        throw std::runtime_error("pair metadata 64x64 mixed-values round trip mismatch");
    }
    for (uint64_t block = 0; block < 512; ++block) {
        for (unsigned j = 0; j < 8; ++j) {
            const uint16_t code = pair_code_host(pair_packed.pair_metadata.data(), block, j);
            const uint16_t wide_code = wide_pair_code_host(pair_packed.wide_pair_metadata.data(), block, j);
            const uint16_t expected = uint16_t((((block * 8 + j) >> 6) & 63) |
                                               (((block * 8 + j) & 63) << 6));
            if (code != expected || wide_code != expected) { throw std::runtime_error("pair metadata code mismatch"); }
        }
    }
    validate_wide_pair_codec();
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
