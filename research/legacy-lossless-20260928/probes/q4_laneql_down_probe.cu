#include "ops/linear/gguf/ggml_bridge_vec.cuh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>

#ifndef Q4_HALFWORD_K
#define Q4_HALFWORD_K 17408
#endif

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
    std::vector<uint8_t> laneql_qs;
    std::vector<uint8_t> raw_metadata;
    std::vector<uint8_t> d_code;
    std::vector<uint8_t> dmin_code;
    std::vector<uint8_t> fixed_d_code;
    std::vector<uint8_t> fixed_dmin_code;
    std::vector<uint8_t> fixed_d_word_code;
    std::vector<uint8_t> fixed_dmin_word_code;
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

static uint16_t decode_half_field_fixed(const std::vector<uint8_t>& code, uint64_t block,
                                        unsigned exp_min, unsigned exp_bits) {
    const uint32_t value = read_bits_host(code.data(), uint32_t(block * (10 + exp_bits)), 10 + exp_bits);
    return uint16_t((((value >> 10) + exp_min) << 10) | (value & 1023u));
}

static uint32_t read_half13_word_host(const std::vector<uint8_t>& code, uint32_t block, bool conditional) {
    const uint32_t bit = block * 13;
    const uint32_t word = bit >> 5;
    const uint32_t shift = bit & 31;
    uint32_t lo = 0;
    std::memcpy(&lo, code.data() + size_t(word) * 4, sizeof(lo));
    if (conditional && shift + 13 <= 32) { return (lo >> shift) & 0x1FFFu; }
    uint32_t hi = 0;
    std::memcpy(&hi, code.data() + size_t(word + 1) * 4, sizeof(hi));
    const uint32_t value = shift == 0 ? lo : (lo >> shift) | (hi << (32 - shift));
    return value & 0x1FFFu;
}

static uint32_t read_half13_byte_host(const std::vector<uint8_t>& code, uint32_t block) {
    const uint32_t bit = block * 13;
    const uint32_t at = bit >> 3;
    const uint32_t shift = bit & 7;
    uint32_t value = uint32_t(code[at]) | (uint32_t(code[at + 1]) << 8);
    if (shift > 3) { value |= uint32_t(code[at + 2]) << 16; }
    return (value >> shift) & 0x1FFFu;
}

static PackedScales pack_scales(const std::vector<uint8_t>& raw, int rows, int k) {
    const uint64_t blocks = uint64_t(rows) * (k / 256);
    if (raw.size() != blocks * 144) { throw std::runtime_error("Q4_K matrix payload size"); }
    PackedScales out;
    out.rows = rows;
    out.k = k;
    out.dm.resize(blocks * 4);
    out.qs.resize(blocks * 128);
    out.laneql_qs.resize(blocks * 128);
    out.raw_metadata.resize(blocks * 12);
    unsigned exp_min[2] = {31, 31};
    unsigned exp_max[2] = {0, 0};
    bool any_sign = false;
    for (uint64_t b = 0; b < blocks; ++b) {
        const uint8_t* src = raw.data() + b * 144;
        std::memcpy(out.dm.data() + b * 4, src, 4);
        std::memcpy(out.raw_metadata.data() + b * 12, src + 4, 12);
        std::memcpy(out.qs.data() + b * 128, src + 16, 128);
        for (int lane = 0; lane < 8; ++lane) {
            const int group = lane / 2;
            const int nibble_shift = 4 * (lane & 1);
            uint8_t* lane_bytes = out.laneql_qs.data() + b * 128 + lane * 16;
            for (int i = 0; i < 32; ++i) {
                const uint8_t code = uint8_t((src[16 + group * 32 + i] >> nibble_shift) & 15u);
                lane_bytes[i % 16] |= uint8_t(code << (4 * (i / 16)));
            }
        }
        for (int field = 0; field < 2; ++field) {
            const uint16_t half = uint16_t(src[field * 2]) | (uint16_t(src[field * 2 + 1]) << 8);
            any_sign |= (half & 0x8000u) != 0;
            const unsigned exp = (half >> 10) & 31u;
            exp_min[field] = std::min(exp_min[field], exp);
            exp_max[field] = std::max(exp_max[field], exp);
        }
    }
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
    if (out.coded && !out.sign_bits) {
        const unsigned d_width = 10 + out.d_exp_bits;
        const unsigned dmin_width = 10 + out.dmin_exp_bits;
        out.fixed_d_code.resize(std::max<size_t>(16, ((blocks * d_width + 7) / 8 + 15) & ~size_t(15)), 0);
        out.fixed_dmin_code.resize(std::max<size_t>(16, ((blocks * dmin_width + 7) / 8 + 15) & ~size_t(15)), 0);
        for (uint64_t b = 0; b < blocks; ++b) {
            const uint8_t* src = raw.data() + b * 144;
            const uint16_t d = uint16_t(src[0]) | (uint16_t(src[1]) << 8);
            const uint16_t dmin = uint16_t(src[2]) | (uint16_t(src[3]) << 8);
            write_bits(out.fixed_d_code, b * d_width, 10, d & 1023u);
            write_bits(out.fixed_d_code, b * d_width + 10, out.d_exp_bits, ((d >> 10) & 31u) - out.d_exp_min);
            write_bits(out.fixed_dmin_code, b * dmin_width, 10, dmin & 1023u);
            write_bits(out.fixed_dmin_code, b * dmin_width + 10, out.dmin_exp_bits, ((dmin >> 10) & 31u) - out.dmin_exp_min);
        }
        out.fixed_d_word_code = out.fixed_d_code;
        out.fixed_dmin_word_code = out.fixed_dmin_code;
        out.fixed_d_word_code.resize(out.fixed_d_word_code.size() + 4, 0);
        out.fixed_dmin_word_code.resize(out.fixed_dmin_word_code.size() + 4, 0);
    }
    return out;
}

static std::vector<uint8_t> unpack_cpu(const PackedScales& packed) {
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
        std::memcpy(dst + 4, packed.raw_metadata.data() + b * 12, 12);
        std::memcpy(dst + 16, packed.qs.data() + b * 128, 128);
    }
    return raw;
}

static std::vector<uint8_t> unpack_laneql_cpu(const PackedScales& packed) {
    const uint64_t blocks = uint64_t(packed.rows) * (packed.k / 256);
    std::vector<uint8_t> raw(blocks * 144);
    for (uint64_t b = 0; b < blocks; ++b) {
        uint8_t* dst = raw.data() + b * 144;
        if (packed.coded) {
            const uint16_t d = decode_half_field(packed.d_code, b, packed.d_exp_min, packed.d_exp_bits, packed.sign_bits);
            const uint16_t dmin = decode_half_field(packed.dmin_code, b, packed.dmin_exp_min, packed.dmin_exp_bits, packed.sign_bits);
            dst[0] = uint8_t(d); dst[1] = uint8_t(d >> 8); dst[2] = uint8_t(dmin); dst[3] = uint8_t(dmin >> 8);
        } else { std::memcpy(dst, packed.dm.data() + b * 4, 4); }
        std::memcpy(dst + 4, packed.raw_metadata.data() + b * 12, 12);
        for (int group = 0; group < 4; ++group) {
            const uint8_t* low = packed.laneql_qs.data() + b * 128 + group * 2 * 16;
            const uint8_t* high = packed.laneql_qs.data() + b * 128 + (group * 2 + 1) * 16;
            for (int i = 0; i < 32; ++i) {
                const int shift = 4 * (i / 16);
                const uint8_t lo = uint8_t((low[i % 16] >> shift) & 15u);
                const uint8_t hi = uint8_t((high[i % 16] >> shift) & 15u);
                dst[16 + group * 32 + i] = uint8_t(lo | (hi << 4));
            }
        }
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

__device__ __forceinline__ uint16_t read_half13_device(const uint8_t* code, uint32_t block) {
    const uint32_t bit = block * 13;
    const uint32_t at = bit >> 3;
    const uint32_t shift = bit & 7;
    uint32_t value = uint32_t(code[at]) | (uint32_t(code[at + 1]) << 8);
    if (shift > 3) { value |= uint32_t(code[at + 2]) << 16; }
    return uint16_t((value >> shift) & 0x1FFFu);
}

template <bool ConditionalCross>
__device__ __forceinline__ uint16_t read_half13_word_device(const uint8_t* code, uint32_t block) {
    const uint32_t bit = block * 13;
    const uint32_t word = bit >> 5;
    const uint32_t shift = bit & 31;
    const uint32_t* words = reinterpret_cast<const uint32_t*>(code);
    const uint32_t lo = words[word];
    uint32_t value;
    if constexpr (ConditionalCross) {
        if (shift + 13 > 32) {
            value = __funnelshift_r(lo, words[word + 1], shift);
        } else {
            value = lo >> shift;
        }
    } else {
        value = __funnelshift_r(lo, words[word + 1], shift);
    }
    return uint16_t(value & 0x1FFFu);
}

template <unsigned ExpBits, unsigned ExpBase, int HalfReadMode = 0>
__device__ __forceinline__ uint16_t decode_half_field_fixed_device(const uint8_t* code, uint32_t block) {
    constexpr unsigned width = 10 + ExpBits;
    uint32_t value;
    if constexpr (ExpBits == 3) {
        if constexpr (HalfReadMode == 1) { value = read_half13_word_device<false>(code, block); }
        else if constexpr (HalfReadMode == 2) { value = read_half13_word_device<true>(code, block); }
        else { value = read_half13_device(code, block); }
    } else {
        const uint32_t bit = block * width;
        const uint32_t at = bit >> 3;
        const uint32_t shift = bit & 7;
        value = uint32_t(code[at]) | (uint32_t(code[at + 1]) << 8);
        if (shift + width > 16) { value |= uint32_t(code[at + 2]) << 16; }
        value = (value >> shift) & ((1u << width) - 1u);
    }
    return uint16_t((((value >> 10) + ExpBase) << 10) | (value & 1023u));
}

struct ScaleArgs {
    VecArgs vec;
    const uint8_t* dm;
    const uint8_t* qs;
    const uint8_t* raw_metadata;
    const uint8_t* d_code;
    const uint8_t* dmin_code;
    uint32_t d_exp_min;
    uint32_t dmin_exp_min;
    uint32_t d_exp_bits;
    uint32_t dmin_exp_bits;
    uint32_t sign_bits;
    bool coded;
};

template <int T, bool RawPlanar, bool Fused = false, int DExpBits = -1, int DExpBase = 0,
          int DMinExpBits = -1, int DMinExpBase = 0, bool DeferredActivation = false,
          int HalfReadMode = 0, int DMinHalfReadMode = HalfReadMode, bool LaneQL = false>
__launch_bounds__(128) __global__ void scale_kernel(ScaleArgs p) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    constexpr int k = Q4_HALFWORD_K;
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
                    scale_min_k4(p.raw_metadata + block * 12, lane % 8, sc, mn);
                uint16_t dword, dminword;
                if constexpr (DExpBits >= 0) {
                    dword = decode_half_field_fixed_device<DExpBits, DExpBase, HalfReadMode>(p.d_code, block);
                    dminword = decode_half_field_fixed_device<DMinExpBits, DMinExpBase, DMinHalfReadMode>(p.dmin_code, block);
                } else {
                    dword = RawPlanar || !p.coded ? *reinterpret_cast<const uint16_t*>(p.dm + block * 4) :
                        decode_half_field_device(p.d_code, block, p.d_exp_min, p.d_exp_bits, p.sign_bits);
                    dminword = RawPlanar || !p.coded ? *reinterpret_cast<const uint16_t*>(p.dm + block * 4 + 2) :
                        decode_half_field_device(p.dmin_code, block, p.dmin_exp_min, p.dmin_exp_bits, p.sign_bits);
                }
                    const float2 dm = make_float2(__half2float(__ushort_as_half(dword)), __half2float(__ushort_as_half(dminword)));
                    Slice sl;
                    sl.f[0] = dm.x * float(sc);
                    sl.f[1] = dm.y * float(mn);
                    if constexpr (LaneQL) {
                        static_assert(!Fused);
                        const uint4 q = *reinterpret_cast<const uint4*>(p.qs + block * 128 + (lane % 8) * 16);
                        const uint32_t words[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
                        for (int i = 0; i < 8; ++i) { sl.w[i] = int((words[i & 3] >> (4 * (i / 4))) & 0x0F0F0F0Fu); }
                    } else {
                        const uint4* q = reinterpret_cast<const uint4*>(p.qs + block * 128 + 32 * ((lane % 8) / 2));
                        const uint4 q0 = q[0], q1 = q[1];
                        const int shift = 4 * ((lane % 8) & 1);
                        const uint32_t words[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
#pragma unroll
                        for (int i = 0; i < 8; ++i) { sl.w[i] = int((words[i] >> shift) & 0x0F0F0F0Fu); }
                    }
                    if constexpr (DeferredActivation) {
#pragma unroll
                        for (int j = 0; j < T; ++j) {
                            float d, sum;
                            int activation[8];
                            const float2 ds = __half22float2(p.vec.ds[int64_t(j) * (k / 32) + lane + 32 * it]);
                            d = ds.x;
                            sum = ds.y;
                            const int4* q = reinterpret_cast<const int4*>(p.vec.qs + int64_t(j) * k + 32 * (lane + 32 * it));
                            const int4 v0 = q[0], v1 = q[1];
                            activation[0] = v0.x; activation[1] = v0.y; activation[2] = v0.z; activation[3] = v0.w;
                            activation[4] = v1.x; activation[5] = v1.y; activation[6] = v1.z; activation[7] = v1.w;
                            acc[r][j] += Decoder<GGML_TYPE_Q4_K>::dot(sl, activation, d, sum);
                        }
                    } else {
#pragma unroll
                        for (int j = 0; j < T; ++j) { acc[r][j] += Decoder<GGML_TYPE_Q4_K>::dot(sl, a[j], d[j], sum[j]); }
                    }
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

template <int T, bool RawPlanar>
__launch_bounds__(256) __global__ void fused_split_kernel(ScaleArgs p) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row_slot = warp >> 1;
    const int side = warp & 1;
    constexpr int k = Q4_HALFWORD_K;
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
                scale_min_k4(p.raw_metadata + block * 12, lane % 8, sc, mn);
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
                                   const uint8_t* raw_metadata, const uint8_t* d_code,
                                   const uint8_t* dmin_code, uint32_t d_exp_min,
                                   uint32_t dmin_exp_min, uint32_t d_exp_bits,
                                   uint32_t dmin_exp_bits, uint32_t sign_bits, bool coded, uint8_t* raw) {
    const uint32_t block = uint32_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint32_t blocks = uint32_t(rows) * (Q4_HALFWORD_K / 256);
    if (block >= blocks) { return; }
    const uint32_t row_bytes = (Q4_HALFWORD_K / 256) * 144;
    const uint32_t at = (block / (Q4_HALFWORD_K / 256)) * row_bytes + (block % (Q4_HALFWORD_K / 256)) * 144;
    const uint16_t d = coded ? decode_half_field_device(d_code, block, d_exp_min, d_exp_bits, sign_bits) :
        *reinterpret_cast<const uint16_t*>(dm + block * 4);
    const uint16_t dmin = coded ? decode_half_field_device(dmin_code, block, dmin_exp_min, dmin_exp_bits, sign_bits) :
        *reinterpret_cast<const uint16_t*>(dm + block * 4 + 2);
    raw[at] = uint8_t(d); raw[at + 1] = uint8_t(d >> 8); raw[at + 2] = uint8_t(dmin); raw[at + 3] = uint8_t(dmin >> 8);
    for (int j = 0; j < 12; ++j) { raw[at + 4 + j] = raw_metadata[block * 12 + j]; }
    for (int j = 0; j < 128; ++j) { raw[at + 16 + j] = qs[block * 128 + j]; }
}

__global__ void reconstruct_laneql_kernel(int rows, const uint8_t* dm, const uint8_t* laneql_qs,
                                          const uint8_t* raw_metadata, const uint8_t* d_code,
                                          const uint8_t* dmin_code, uint32_t d_exp_min,
                                          uint32_t dmin_exp_min, uint32_t d_exp_bits,
                                          uint32_t dmin_exp_bits, uint32_t sign_bits,
                                          bool coded, uint8_t* raw) {
    const uint32_t block = uint32_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint32_t blocks = uint32_t(rows) * (Q4_HALFWORD_K / 256);
    if (block >= blocks) { return; }
    const uint32_t row_bytes = (Q4_HALFWORD_K / 256) * 144;
    const uint32_t at = (block / (Q4_HALFWORD_K / 256)) * row_bytes + (block % (Q4_HALFWORD_K / 256)) * 144;
    const uint16_t d = coded ? decode_half_field_device(d_code, block, d_exp_min, d_exp_bits, sign_bits) :
        *reinterpret_cast<const uint16_t*>(dm + block * 4);
    const uint16_t dmin = coded ? decode_half_field_device(dmin_code, block, dmin_exp_min, dmin_exp_bits, sign_bits) :
        *reinterpret_cast<const uint16_t*>(dm + block * 4 + 2);
    raw[at] = uint8_t(d); raw[at + 1] = uint8_t(d >> 8); raw[at + 2] = uint8_t(dmin); raw[at + 3] = uint8_t(dmin >> 8);
    for (int j = 0; j < 12; ++j) { raw[at + 4 + j] = raw_metadata[block * 12 + j]; }
    for (int group = 0; group < 4; ++group) {
        const uint8_t* low = laneql_qs + block * 128 + group * 2 * 16;
        const uint8_t* high = laneql_qs + block * 128 + (group * 2 + 1) * 16;
        for (int i = 0; i < 32; ++i) {
            const int shift = 4 * (i / 16);
            const uint8_t lo = uint8_t((low[i % 16] >> shift) & 15u);
            const uint8_t hi = uint8_t((high[i % 16] >> shift) & 15u);
            raw[at + 16 + group * 32 + i] = uint8_t(lo | (hi << 4));
        }
    }
}

template <int T, bool RawPlanar, bool Fused = false, bool SplitFused = (Fused && !RawPlanar),
          int DExpBits = -1, int DExpBase = 0, int DMinExpBits = -1, int DMinExpBase = 0,
          bool DeferredActivation = false, int HalfReadMode = 0, int DMinHalfReadMode = HalfReadMode,
          bool LaneQL = false>
void launch_candidate(const VecArgs& vec, const PackedScales& packed,
                      const uint8_t* dm, const uint8_t* qs, const uint8_t* raw_metadata,
                      const uint8_t* d_code, const uint8_t* dmin_code, cudaStream_t stream) {
    if constexpr (DExpBits >= 0) {
        if (!packed.coded || packed.sign_bits != 0 || packed.d_exp_bits != DExpBits ||
            packed.d_exp_min != DExpBase || packed.dmin_exp_bits != DMinExpBits ||
            packed.dmin_exp_min != DMinExpBase) {
            throw std::runtime_error("fixed Q4 half metadata does not match tensor");
        }
    }
    ScaleArgs args{vec, dm, qs, raw_metadata, d_code, dmin_code, packed.d_exp_min, packed.dmin_exp_min,
                   packed.d_exp_bits, packed.dmin_exp_bits, packed.sign_bits, packed.coded};
    static const int resident = [] {
        int blocks = 0;
        if constexpr (SplitFused) {
            check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, fused_split_kernel<T, RawPlanar>, 256, 0), "split occupancy");
        } else {
            check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, scale_kernel<T, RawPlanar, Fused, DExpBits, DExpBase, DMinExpBits, DMinExpBase, DeferredActivation, HalfReadMode, DMinHalfReadMode, LaneQL>, 128, 0), "scale occupancy");
        }
        return std::max(blocks, 1);
    }();
    const int groups = Fused ? (vec.rows + 3) / 4 : (packed.rows + 7) / 8;
    const int grid = std::min(groups, resident * device_facts().sm_count);
    if constexpr (SplitFused) { fused_split_kernel<T, RawPlanar><<<grid, 256, 0, stream>>>(args); }
    else { scale_kernel<T, RawPlanar, Fused, DExpBits, DExpBase, DMinExpBits, DMinExpBase, DeferredActivation, HalfReadMode, DMinHalfReadMode, LaneQL><<<grid, 128, 0, stream>>>(args); }
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
        header[3] != Q4_HALFWORD_K || (header[4] != 1 && header[4] != 3 && header[4] != 4) ||
        (header[5] != 0 && header[5] != 1) || header[7] != 0) {
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
    constexpr int k = Q4_HALFWORD_K;
    const int n = packed.rows;
    const int out_rows = Fused ? n / 2 : n;
    if constexpr (Fused) { if ((n & 1) != 0) { throw std::runtime_error("fused Q4_K tensor needs an even row count"); } }
    const size_t weight_bytes = raw.size();
    uint8_t* weights[3];
    uint8_t* device_dm[3];
    uint8_t* device_qs[3];
    uint8_t* device_laneql_qs[3];
    uint8_t* device_raw_metadata[3];
    uint8_t* device_d_code[3];
    uint8_t* device_dmin_code[3];
    uint8_t* device_fixed_d_code[3];
    uint8_t* device_fixed_dmin_code[3];
    uint8_t* device_fixed_d_word_code[3];
    uint8_t* device_fixed_dmin_word_code[3];
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
        alloc(&device_laneql_qs[r], packed.laneql_qs.size());
        check(cudaMemcpy(device_laneql_qs[r], packed.laneql_qs.data(), packed.laneql_qs.size(), cudaMemcpyHostToDevice), "laneql qs upload");
        alloc(&device_raw_metadata[r], packed.raw_metadata.size());
        check(cudaMemcpy(device_raw_metadata[r], packed.raw_metadata.data(), packed.raw_metadata.size(), cudaMemcpyHostToDevice), "raw metadata upload");
        alloc(&device_d_code[r], packed.d_code.size());
        check(cudaMemcpy(device_d_code[r], packed.d_code.data(), packed.d_code.size(), cudaMemcpyHostToDevice), "d stream upload");
        alloc(&device_dmin_code[r], packed.dmin_code.size());
        check(cudaMemcpy(device_dmin_code[r], packed.dmin_code.data(), packed.dmin_code.size(), cudaMemcpyHostToDevice), "dmin stream upload");
        alloc(&device_fixed_d_code[r], packed.fixed_d_code.size());
        check(cudaMemcpy(device_fixed_d_code[r], packed.fixed_d_code.data(), packed.fixed_d_code.size(), cudaMemcpyHostToDevice), "fixed d stream upload");
        alloc(&device_fixed_dmin_code[r], packed.fixed_dmin_code.size());
        check(cudaMemcpy(device_fixed_dmin_code[r], packed.fixed_dmin_code.data(), packed.fixed_dmin_code.size(), cudaMemcpyHostToDevice), "fixed dmin stream upload");
        alloc(&device_fixed_d_word_code[r], packed.fixed_d_word_code.size());
        check(cudaMemcpy(device_fixed_d_word_code[r], packed.fixed_d_word_code.data(), packed.fixed_d_word_code.size(), cudaMemcpyHostToDevice), "fixed d word stream upload");
        alloc(&device_fixed_dmin_word_code[r], packed.fixed_dmin_word_code.size());
        check(cudaMemcpy(device_fixed_dmin_word_code[r], packed.fixed_dmin_word_code.data(), packed.fixed_dmin_word_code.size(), cudaMemcpyHostToDevice), "fixed dmin word stream upload");
    }
    alloc(&device_debug, raw.size());
    const uint64_t blocks = uint64_t(n) * (k / 256);
    reconstruct_kernel<<<(blocks + 255) / 256, 256>>>(n, device_dm[0], device_qs[0], device_raw_metadata[0],
        device_d_code[0], device_dmin_code[0], packed.d_exp_min, packed.dmin_exp_min, packed.d_exp_bits,
        packed.dmin_exp_bits, packed.sign_bits, packed.coded, device_debug);
    check(cudaGetLastError(), "reconstruct launch");
    std::vector<uint8_t> back(raw.size());
    check(cudaMemcpy(back.data(), device_debug, back.size(), cudaMemcpyDeviceToHost), "reconstruct copy");
    if (back != raw) { throw std::runtime_error("GPU scale/quant payload full bit mismatch"); }
    reconstruct_laneql_kernel<<<(blocks + 255) / 256, 256>>>(n, device_dm[0], device_laneql_qs[0], device_raw_metadata[0],
        device_d_code[0], device_dmin_code[0], packed.d_exp_min, packed.dmin_exp_min, packed.d_exp_bits,
        packed.dmin_exp_bits, packed.sign_bits, packed.coded, device_debug);
    check(cudaGetLastError(), "laneql reconstruction launch");
    check(cudaMemcpy(back.data(), device_debug, back.size(), cudaMemcpyDeviceToHost), "laneql reconstruction copy");
    if (back != raw) { throw std::runtime_error("GPU laneQL full payload reconstruction mismatch"); }
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
        launch_candidate<T, false, Fused>(args, packed, device_dm[rotation], device_qs[rotation], device_raw_metadata[rotation],
                            device_d_code[rotation], device_dmin_code[rotation], stream);
    };
    auto raw_planar = [&](int rotation) {
        args.weight = weights[rotation];
        launch_candidate<T, true, Fused>(args, packed, device_dm[rotation], device_qs[rotation], device_raw_metadata[rotation],
                            device_d_code[rotation], device_dmin_code[rotation], stream);
    };
    auto raw_split = [&](int rotation) {
        args.weight = weights[rotation];
        launch_candidate<T, true, true, true>(args, packed, device_dm[rotation], device_qs[rotation], device_raw_metadata[rotation],
                            device_d_code[rotation], device_dmin_code[rotation], stream);
    };
    auto coded_original = [&](int rotation) {
        args.weight = weights[rotation];
        launch_candidate<T, false, true, false>(args, packed, device_dm[rotation], device_qs[rotation], device_raw_metadata[rotation],
                            device_d_code[rotation], device_dmin_code[rotation], stream);
    };
    auto fast_fields = [&](int rotation, bool deferred) {
        args.weight = weights[rotation];
        if (packed.d_exp_bits == 3 && packed.d_exp_min == 0 && packed.dmin_exp_bits == 3 && packed.dmin_exp_min == 2) {
            if (deferred) {
                launch_candidate<T, false, Fused, false, 3, 0, 3, 2, true>(args, packed, device_dm[rotation], device_qs[rotation], device_raw_metadata[rotation],
                    device_fixed_d_code[rotation], device_fixed_dmin_code[rotation], stream);
            } else {
                launch_candidate<T, false, Fused, false, 3, 0, 3, 2>(args, packed, device_dm[rotation], device_qs[rotation], device_raw_metadata[rotation],
                    device_fixed_d_code[rotation], device_fixed_dmin_code[rotation], stream);
            }
        } else if (packed.d_exp_bits == 2 && packed.d_exp_min == 0 && packed.dmin_exp_bits == 3 && packed.dmin_exp_min == 2) {
            if (deferred) {
                launch_candidate<T, false, Fused, false, 2, 0, 3, 2, true>(args, packed, device_dm[rotation], device_qs[rotation], device_raw_metadata[rotation],
                    device_fixed_d_code[rotation], device_fixed_dmin_code[rotation], stream);
            } else {
                launch_candidate<T, false, Fused, false, 2, 0, 3, 2>(args, packed, device_dm[rotation], device_qs[rotation], device_raw_metadata[rotation],
                    device_fixed_d_code[rotation], device_fixed_dmin_code[rotation], stream);
            }
        } else {
            throw std::runtime_error("unsupported fixed Q4 half metadata window");
        }
    };
    auto fast_fields_word = [&](int rotation, int mode) {
        args.weight = weights[rotation];
        if (packed.d_exp_min != 0 || packed.dmin_exp_bits != 3 || packed.dmin_exp_min != 2) {
            throw std::runtime_error("word fixed13 decoder requires exponent bases 0/2 and dmin width 13");
        }
        if (packed.d_exp_bits == 3) {
            if (mode == 1) {
                launch_candidate<T, false, Fused, false, 3, 0, 3, 2, false, 1>(args, packed, device_dm[rotation], device_qs[rotation], device_raw_metadata[rotation],
                    device_fixed_d_word_code[rotation], device_fixed_dmin_word_code[rotation], stream);
            } else {
                launch_candidate<T, false, Fused, false, 3, 0, 3, 2, false, 2>(args, packed, device_dm[rotation], device_qs[rotation], device_raw_metadata[rotation],
                    device_fixed_d_word_code[rotation], device_fixed_dmin_word_code[rotation], stream);
            }
        } else if (packed.d_exp_bits == 2) {
            if (mode == 1) {
                launch_candidate<T, false, Fused, false, 2, 0, 3, 2, false, 0, 1>(args, packed, device_dm[rotation], device_qs[rotation], device_raw_metadata[rotation],
                    device_fixed_d_code[rotation], device_fixed_dmin_word_code[rotation], stream);
            } else {
                launch_candidate<T, false, Fused, false, 2, 0, 3, 2, false, 0, 2>(args, packed, device_dm[rotation], device_qs[rotation], device_raw_metadata[rotation],
                    device_fixed_d_code[rotation], device_fixed_dmin_word_code[rotation], stream);
            }
        } else {
            throw std::runtime_error("unsupported mixed fixed13 Q4 half fields");
        }
    };
    auto laneql_word_mode2 = [&](int rotation) {
        args.weight = weights[rotation];
        if (packed.d_exp_bits != 3 || packed.d_exp_min != 0 || packed.dmin_exp_bits != 3 || packed.dmin_exp_min != 2) {
            throw std::runtime_error("Q4 down laneQL requires wordreadmode2 for 13-bit d and dmin");
        }
        launch_candidate<T, false, false, false, 3, 0, 3, 2, false, 2, 2, true>(args, packed,
            device_dm[rotation], device_laneql_qs[rotation], device_raw_metadata[rotation],
            device_fixed_d_word_code[rotation], device_fixed_dmin_word_code[rotation], stream);
    };
    auto raw_deferred = [&](int rotation) {
        args.weight = weights[rotation];
        launch_candidate<T, true, Fused, false, -1, 0, -1, 0, true>(args, packed, device_dm[rotation], device_qs[rotation], device_raw_metadata[rotation],
            device_d_code[rotation], device_dmin_code[rotation], stream);
    };
    constexpr int mode_count = Fused ? 10 : 9;
    auto launch_mode = [&](int mode, int rotation) {
        if (mode == 0) { baseline(rotation); }
        else if (mode == 1) { raw_planar(rotation); }
        else if constexpr (Fused) {
            if (mode == 2) { raw_split(rotation); }
            else if (mode == 3) { coded_original(rotation); }
            else if (mode == 4) { candidate(rotation); }
            else if (mode == 5) { fast_fields(rotation, false); }
            else if (mode == 6) { raw_deferred(rotation); }
            else if (mode == 7) { fast_fields(rotation, true); }
            else if (mode == 8) { fast_fields_word(rotation, 1); }
            else { fast_fields_word(rotation, 2); }
        } else {
            if (mode == 2) { candidate(rotation); }
            else if (mode == 3) { fast_fields(rotation, false); }
            else if (mode == 4) { raw_deferred(rotation); }
            else if (mode == 5) { fast_fields(rotation, true); }
            else if (mode == 6) { fast_fields_word(rotation, 1); }
            else if (mode == 7) { fast_fields_word(rotation, 2); }
            else { laneql_word_mode2(rotation); }
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
    bool laneql_bf16_graph_exact = false;
    if constexpr (!Fused) {
        __nv_bfloat16* laneql_bf_graph_output = nullptr;
        alloc(&laneql_bf_graph_output, bf_output_bytes + output_guard_bytes);
        args.out.f32 = nullptr;
        args.out.bf16 = laneql_bf_graph_output;
        reset_guard(laneql_bf_graph_output, bf_output_bytes);
        cudaGraph_t laneql_bf_graph;
        cudaGraphExec_t laneql_bf_exec;
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "laneQL BF16 graph capture");
        launch_mode(8, 0);
        check(cudaStreamEndCapture(stream, &laneql_bf_graph), "laneQL BF16 graph capture end");
        check(cudaGraphInstantiate(&laneql_bf_exec, laneql_bf_graph, 0), "laneQL BF16 graph instantiate");
        check(cudaGraphDestroy(laneql_bf_graph), "laneQL BF16 graph destroy");
        check(cudaGraphLaunch(laneql_bf_exec, stream), "laneQL BF16 graph replay");
        check(cudaStreamSynchronize(stream), "laneQL BF16 graph sync");
        verify_guard(laneql_bf_graph_output, bf_output_bytes, "laneQL BF16 graph");
        std::vector<uint16_t> laneql_bf_graph_actual(bf_reference.size());
        check(cudaMemcpy(laneql_bf_graph_actual.data(), laneql_bf_graph_output,
                         laneql_bf_graph_actual.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost), "laneQL BF16 graph readback");
        if (laneql_bf_graph_actual != bf_reference) { throw std::runtime_error("laneQL BF16 graph output mismatch"); }
        laneql_bf16_graph_exact = true;
        check(cudaGraphExecDestroy(laneql_bf_exec), "laneQL BF16 graph executable destroy");
        check(cudaFree(laneql_bf_graph_output), "laneQL BF16 graph output free");
        args.out.bf16 = bfout;
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
    verify_guard(y, float_output_bytes, "graph FP32");
    for (int mode = 0; mode < mode_count; ++mode) {
        reset_guard(y, float_output_bytes);
        check(cudaGraphLaunch(graphs[mode], stream), "graph verification launch");
        check(cudaStreamSynchronize(stream), "graph verification sync");
        verify_guard(y, float_output_bytes, "graph FP32");
        check(cudaMemcpy(actual.data(), y, actual.size() * sizeof(float), cudaMemcpyDeviceToHost), "graph output copy");
        if (std::memcmp(reference.data(), actual.data(), reference.size() * sizeof(float)) != 0) {
            throw std::runtime_error("graph output bit mismatch at mode " + std::to_string(mode));
        }
    }
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start), "start event");
    check(cudaEventCreate(&stop), "stop event");
    const uint64_t format_bytes = packed.qs.size() + packed.raw_metadata.size() + packed.d_code.size() +
                                  packed.dmin_code.size() + 64;
    const uint64_t planar_bytes = packed.dm.size() + packed.qs.size() + packed.raw_metadata.size() + 64;
    const KernelResource baseline_resource = kernel_resource(vec::kernel<GGML_TYPE_Q4_K, T, Fused, k>);
    const KernelResource raw_resource = kernel_resource(scale_kernel<T, true, Fused>);
    const KernelResource raw_deferred_resource = kernel_resource(scale_kernel<T, true, Fused, -1, 0, -1, 0, true>);
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
    const KernelResource fast_resource = packed.d_exp_bits == 3
        ? kernel_resource(scale_kernel<T, false, Fused, 3, 0, 3, 2>)
        : kernel_resource(scale_kernel<T, false, Fused, 2, 0, 3, 2>);
    const KernelResource fast_deferred_resource = packed.d_exp_bits == 3
        ? kernel_resource(scale_kernel<T, false, Fused, 3, 0, 3, 2, true>)
        : kernel_resource(scale_kernel<T, false, Fused, 2, 0, 3, 2, true>);
    const KernelResource word_always_resource = packed.d_exp_bits == 3
        ? kernel_resource(scale_kernel<T, false, Fused, 3, 0, 3, 2, false, 1>)
        : kernel_resource(scale_kernel<T, false, Fused, 2, 0, 3, 2, false, 0, 1>);
    const KernelResource word_conditional_resource = packed.d_exp_bits == 3
        ? kernel_resource(scale_kernel<T, false, Fused, 3, 0, 3, 2, false, 2>)
        : kernel_resource(scale_kernel<T, false, Fused, 2, 0, 3, 2, false, 0, 2>);
    const uint64_t word_tail_bytes = packed.d_exp_bits == 3 ? 8 : 4;
    const uint64_t laneql_candidate_bytes = packed.laneql_qs.size() + packed.raw_metadata.size() +
        packed.fixed_d_word_code.size() + packed.fixed_dmin_word_code.size() + 64;
    std::cout << "{\"matrix\":\"" << label << "\",\"T\":" << T << ",\"N\":" << n << ",\"K\":" << k
              << ",\"output_rows\":" << out_rows << ",\"fused_swiglu\":" << (Fused ? "true" : "false")
              << ",\"baseline_bytes\":" << weight_bytes << ",\"raw_planar_bytes_including_headers\":" << planar_bytes
              << ",\"weight_halves_included\":" << (Fused ? 2 : 1)
              << ",\"scale_format_bytes\":" << format_bytes
              << ",\"fast_fields_byte_decoder_bytes\":" << format_bytes
              << ",\"word_decoder_planes\":\"" << (packed.d_exp_bits == 3 ? "d+dmin" : "dmin_only") << "\""
              << ",\"word_decoder_tail_padding_bytes\":" << word_tail_bytes
              << ",\"fast_fields_word_decoder_bytes\":" << (format_bytes + word_tail_bytes)
              << ",\"laneql_down_format_version\":2"
              << ",\"laneql_down_layout\":\"per_lane_16B_first16_low_last16_high\""
              << ",\"laneql_down_candidate_bytes_including_word_tails_and_headers\":" << laneql_candidate_bytes
              << ",\"d_exp_min\":" << unsigned(packed.d_exp_min) << ",\"d_exp_bits\":" << unsigned(packed.d_exp_bits)
              << ",\"dmin_exp_min\":" << unsigned(packed.dmin_exp_min) << ",\"dmin_exp_bits\":" << unsigned(packed.dmin_exp_bits)
              << ",\"sign_bits\":" << unsigned(packed.sign_bits) << ",\"half_field_bits_max\":" << unsigned(packed.field_bits)
              << ",\"coded\":" << (packed.coded ? "true" : "false")
              << ",\"activation_snapshot\":" << (activation ? "true" : "false")
              << ",\"activation_fused\":" << (activation ? activation->fused : -1)
              << ",\"fp32_output_bit_exact\":true,\"bf16_output_bit_exact\":true,\"gpu_all_payload_bytes_exact\":true"
              << ",\"laneql_down_gpu_full_payload_reconstruction_exact\":true"
              << ",\"laneql_down_cpu_full_roundtrip_exact\":true"
              << ",\"laneql_down_fp32_bit_exact\":true,\"laneql_down_bf16_bit_exact\":true"
              << ",\"laneql_down_graph_fp32_bit_exact\":true"
              << ",\"laneql_down_graph_bf16_bit_exact\":" << (laneql_bf16_graph_exact ? "true" : "false")
              << ",\"output_canaries_exact\":true,\"rotations\":3,\"graphs\":true,\"resources\":{";
    print_resource("baseline", baseline_resource);
    std::cout << ',';
    print_resource("raw_planar", raw_resource);
    std::cout << ',';
    print_resource(Fused ? "coded_split_warp" : "coded", coded_resource);
    std::cout << ',';
    print_resource("raw_deferred_activation", raw_deferred_resource);
    std::cout << ',';
    print_resource("coded_fixed_fields", fast_resource);
    std::cout << ',';
    print_resource("coded_fixed_fields_deferred_activation", fast_deferred_resource);
    std::cout << ',';
    print_resource("fixed13_aligned_word_always2", word_always_resource);
    std::cout << ',';
    print_resource("fixed13_aligned_word_conditional_cross", word_conditional_resource);
    std::cout << ',';
    print_resource("laneql16B_fixed13_wordreadmode2", kernel_resource(scale_kernel<T, false, Fused, 3, 0, 3, 2, false, 2, 2, true>));
    if constexpr (Fused) {
        std::cout << ',';
        print_resource("raw_split_warp", raw_split_resource);
        std::cout << ',';
        print_resource("coded_original_fused", coded_original_resource);
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
            else if (mode == 1) { mode_name = "raw_planar"; }
            else if constexpr (Fused) {
                if (mode == 2) { mode_name = "raw_split_warp"; }
                else if (mode == 3) { mode_name = "coded_original_fused"; }
                else if (mode == 4) { mode_name = "coded_split_warp"; }
                else if (mode == 5) { mode_name = "coded_fixed_fields"; }
                else if (mode == 6) { mode_name = "raw_deferred_activation"; }
                else if (mode == 7) { mode_name = "coded_fixed_fields_deferred_activation"; }
                else if (mode == 8) { mode_name = "fixed13_aligned_word_always2"; }
                else { mode_name = "fixed13_aligned_word_conditional_cross"; }
            } else {
                if (mode == 2) { mode_name = "coded"; }
                else if (mode == 3) { mode_name = "coded_fixed_fields"; }
                else if (mode == 4) { mode_name = "raw_deferred_activation"; }
                else if (mode == 5) { mode_name = "coded_fixed_fields_deferred_activation"; }
                else if (mode == 6) { mode_name = "fixed13_aligned_word_always2"; }
                else if (mode == 7) { mode_name = "fixed13_aligned_word_conditional_cross"; }
                else { mode_name = "laneql16B_fixed13_wordreadmode2"; }
            }
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
        check(cudaFree(device_laneql_qs[r]), "free");
        check(cudaFree(device_raw_metadata[r]), "free");
        check(cudaFree(device_d_code[r]), "free");
        check(cudaFree(device_dmin_code[r]), "free");
        check(cudaFree(device_fixed_d_code[r]), "free");
        check(cudaFree(device_fixed_dmin_code[r]), "free");
        check(cudaFree(device_fixed_d_word_code[r]), "free");
        check(cudaFree(device_fixed_dmin_word_code[r]), "free");
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
    if (unpack_laneql_cpu(packed) != raw) { throw std::runtime_error("CPU laneQL v2 full Q4_K payload bit mismatch"); }
    const uint64_t blocks = uint64_t(rows) * (k / 256);
    if (packed.sign_bits != 0) { throw std::runtime_error("fixed fields require signless half scales"); }
    if ((packed.d_exp_bits != 2 && packed.d_exp_bits != 3) || packed.dmin_exp_bits != 3 ||
        packed.fixed_d_word_code.size() != packed.fixed_d_code.size() + 4 ||
        packed.fixed_dmin_word_code.size() != packed.fixed_dmin_code.size() + 4) {
        throw std::runtime_error("aligned fixed13 stream framing mismatch");
    }
    for (uint64_t b = 0; b < blocks; ++b) {
        const uint8_t* src = raw.data() + b * 144;
        const uint16_t d = decode_half_field_fixed(packed.fixed_d_code, b, packed.d_exp_min, packed.d_exp_bits);
        const uint16_t dmin = decode_half_field_fixed(packed.fixed_dmin_code, b, packed.dmin_exp_min, packed.dmin_exp_bits);
        const uint16_t source_d = uint16_t(src[0]) | (uint16_t(src[1]) << 8);
        const uint16_t source_dmin = uint16_t(src[2]) | (uint16_t(src[3]) << 8);
        if (d != source_d || dmin != source_dmin) { throw std::runtime_error("CPU fixed-width half stream mismatch"); }
        const uint32_t dcode = read_bits_host(packed.fixed_d_code.data(), b * (10 + packed.d_exp_bits), 10 + packed.d_exp_bits);
        const uint32_t dmincode = read_bits_host(packed.fixed_dmin_code.data(), b * 13, 13);
        if ((packed.d_exp_bits == 3 &&
             (read_half13_word_host(packed.fixed_d_word_code, uint32_t(b), false) != dcode ||
              read_half13_word_host(packed.fixed_d_word_code, uint32_t(b), true) != dcode)) ||
            read_half13_word_host(packed.fixed_dmin_word_code, uint32_t(b), false) != dmincode ||
            read_half13_word_host(packed.fixed_dmin_word_code, uint32_t(b), true) != dmincode) {
            throw std::runtime_error("CPU aligned-word fixed13 plane mismatch");
        }
    }
    const uint64_t raw_bytes = raw.size();
    const uint64_t bytes = packed.qs.size() + packed.raw_metadata.size() + packed.d_code.size() +
                           packed.dmin_code.size() + 64;
    const uint64_t laneql_bytes = packed.laneql_qs.size() + packed.raw_metadata.size() +
        packed.fixed_d_word_code.size() + packed.fixed_dmin_word_code.size() + 64;
    std::cout << "{\"matrix\":\"" << label << "\",\"N\":" << rows << ",\"K\":" << k
              << ",\"blocks\":" << blocks << ",\"d_exp_min\":" << unsigned(packed.d_exp_min)
              << ",\"d_exp_bits\":" << unsigned(packed.d_exp_bits) << ",\"dmin_exp_min\":" << unsigned(packed.dmin_exp_min)
              << ",\"dmin_exp_bits\":" << unsigned(packed.dmin_exp_bits) << ",\"sign_bits\":" << unsigned(packed.sign_bits)
              << ",\"half_field_bits_max\":" << unsigned(packed.field_bits) << ",\"coded\":" << (packed.coded ? "true" : "false")
              << ",\"cpu_fixed_width_halves_exact\":true"
              << ",\"cpu_aligned_word_fixed13_exact\":true"
              << ",\"word_decoder_planes\":\"" << (packed.d_exp_bits == 3 ? "d+dmin" : "dmin_only") << "\""
              << ",\"word_decoder_tail_padding_bytes\":" << (packed.d_exp_bits == 3 ? 8 : 4)
              << ",\"fixed_d_code_bits\":" << (10 + packed.d_exp_bits)
              << ",\"fixed_dmin_code_bits\":" << (10 + packed.dmin_exp_bits)
              << ",\"cpu_all_payload_bytes_exact\":true,\"baseline_bytes\":" << raw_bytes
              << ",\"laneql_down_cpu_full_roundtrip_exact\":true"
              << ",\"laneql_down_format_version\":2"
              << ",\"laneql_down_layout\":\"per_lane_16B_first16_low_last16_high\""
              << ",\"candidate_bytes_including_headers_and_alignment\":" << bytes
              << ",\"net_savings_bytes\":" << (raw_bytes > bytes ? raw_bytes - bytes : 0)
              << ",\"candidate_larger_bytes\":" << (bytes > raw_bytes ? bytes - raw_bytes : 0)
              << ",\"fixed13_word_candidate_bytes_including_tail_bytes\":" << (bytes + (packed.d_exp_bits == 3 ? 8 : 4))
              << ",\"laneql_down_candidate_bytes_including_word_tails_and_headers\":" << laneql_bytes
              << ",\"fixed13_word_net_savings_bytes\":" << (raw_bytes > bytes + (packed.d_exp_bits == 3 ? 8 : 4) ? raw_bytes - bytes - (packed.d_exp_bits == 3 ? 8 : 4) : 0)
              << "}\n";
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

static void run_cpu_edge_checks() {
    std::array<bool, 32> phases{};
    for (const uint32_t count : {1u, 2u, 3u, 31u, 32u, 33u, 127u, 128u, 129u, 8192u}) {
        const size_t payload_bytes = (size_t(count) * 13 + 7) / 8;
        const size_t padded_bytes = ((payload_bytes + 15) & ~size_t(15)) + 4;
        std::vector<uint8_t> code(padded_bytes, 0);
        for (uint32_t block = 0; block < count; ++block) {
            const uint32_t value = (block * 7919u + count * 17u) & 0x1FFFu;
            write_bits(code, uint64_t(block) * 13, 13, value);
            phases[(block * 13u) & 31u] = true;
        }
        for (uint32_t block = 0; block < count; ++block) {
            const uint32_t expected = read_bits_host(code.data(), uint64_t(block) * 13, 13);
            const uint32_t byte_value = read_half13_byte_host(code, block);
            const uint32_t word_value = read_half13_word_host(code, block, false);
            const uint32_t conditional_value = read_half13_word_host(code, block, true);
            if (byte_value != expected || word_value != expected || conditional_value != expected) {
                throw std::runtime_error("fixed13 byte/aligned-word CPU mapping mismatch");
            }
        }
    }
    if (!std::all_of(phases.begin(), phases.end(), [](bool value) { return value; })) {
        throw std::runtime_error("fixed13 CPU phase coverage incomplete");
    }
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
    for (const unsigned d_exp_bits : {2u, 3u}) {
        constexpr int blocks = 131;
        std::vector<uint8_t> fixed(size_t(blocks) * 144, 0);
        for (int b = 0; b < blocks; ++b) {
            const uint16_t d = uint16_t(((b % (1 << d_exp_bits)) << 10) | ((b * 23) & 1023));
            const uint16_t dmin = uint16_t(((2 + b % 8) << 10) | ((b * 31) & 1023));
            uint8_t* dst = fixed.data() + size_t(b) * 144;
            dst[0] = uint8_t(d); dst[1] = uint8_t(d >> 8); dst[2] = uint8_t(dmin); dst[3] = uint8_t(dmin >> 8);
        }
        const PackedScales fixed_packed = pack_scales(fixed, blocks, 256);
        if (!fixed_packed.coded || fixed_packed.sign_bits != 0 || fixed_packed.d_exp_bits != d_exp_bits ||
            fixed_packed.d_exp_min != 0 || fixed_packed.dmin_exp_bits != 3 || fixed_packed.dmin_exp_min != 2) {
            throw std::runtime_error("fixed-width fixture framing mismatch");
        }
        for (uint64_t b = 0; b < blocks; ++b) {
            const uint8_t* src = fixed.data() + b * 144;
            const uint16_t d = decode_half_field_fixed(fixed_packed.fixed_d_code, b, fixed_packed.d_exp_min, fixed_packed.d_exp_bits);
            const uint16_t dmin = decode_half_field_fixed(fixed_packed.fixed_dmin_code, b, fixed_packed.dmin_exp_min, fixed_packed.dmin_exp_bits);
            if (d != (uint16_t(src[0]) | (uint16_t(src[1]) << 8)) ||
                dmin != (uint16_t(src[2]) | (uint16_t(src[3]) << 8))) {
                throw std::runtime_error("fixed-width fixture reconstruction mismatch");
            }
        }
    }
}

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--cpu-selftest") {
        run_cpu_edge_checks();
        std::cout << "{\"cpu_fixed13_all_32_phases_exact\":true,\"cpu_fixed13_tail_and_word_padding_exact\":true,\"word_decoder_tail_padding_bytes\":8}\n";
        return 0;
    }
    if ((argc != 7 && argc != 8) || (std::string(argv[6]) != "scan" && std::string(argv[6]) != "bench")) {
        std::cerr << "model offset N K label scan|bench [activation]\n";
        return 2;
    }
    const int rows = std::stoi(argv[3]);
    const int k = std::stoi(argv[4]);
    if (k != Q4_HALFWORD_K || rows % 8 != 0) { throw std::runtime_error("requires configured K and complete eight-row tiles"); }
    run_cpu_edge_checks();
    std::vector<uint8_t> raw(size_t(rows) * (k / 256) * 144);
    std::ifstream file(argv[1], std::ios::binary);
    file.seekg(std::stoull(argv[2]));
    file.read(reinterpret_cast<char*>(raw.data()), raw.size());
    if (size_t(file.gcount()) != raw.size()) { throw std::runtime_error("truncated model tensor"); }
    PackedScales packed = pack_scales(raw, rows, k);
    const std::vector<uint8_t> restored = unpack_cpu(packed);
    if (restored != raw) { throw std::runtime_error("CPU full Q4_K payload bit mismatch"); }
    if (unpack_laneql_cpu(packed) != raw) { throw std::runtime_error("CPU laneQL v2 full Q4_K payload bit mismatch"); }
    if (std::string(argv[6]) == "scan") {
        if (argc == 8) { throw std::runtime_error("activation is bench-only"); }
        run_scan(raw, packed, rows, k, argv[5]);
        return 0;
    }
    if (argc == 8) {
        const Activation activation = load_activation(argv[7]);
        if (activation.t == 4 && activation.fused == 0) { measure<4>(raw, packed, argv[5], &activation); }
        else { throw std::runtime_error("laneQL down probe requires captured ordinary T4 activation"); }
        return 0;
    }
    measure<4>(raw, packed, argv[5]);
    measure<8>(raw, packed, argv[5]);
    return 0;
}
