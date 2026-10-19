#include "ops/linear/gguf/ggml_bridge_vec.cuh"
#include "ops/linear/gguf/q6_fields.cuh"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

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

struct Activation {
    int k = 0;
    int t = 0;
    std::vector<int8_t> qs;
    std::vector<half2> ds;
};

struct Payloads {
    int rows = 0;
    int k = 0;
    std::vector<uint8_t> raw_planar;
    std::vector<uint8_t> transposed;
    std::vector<uint8_t> compressed_planar;
    std::vector<uint8_t> d_bits;
    std::array<uint64_t, 32> d_exponents{};
    uint64_t d_negative = 0;
    uint8_t d_exp_base = 0;
    uint8_t d_exp_max = 0;
    uint8_t d_exp_bits = 0;
    uint8_t d_sign_variable = 0;
    uint8_t d_sign_value = 0;
    uint8_t d_width = 0;
};

static std::array<uint32_t, 8> format_header(const Payloads& payloads, int rows, int k) {
    const uint32_t flags = uint32_t(payloads.d_exp_base) | (uint32_t(payloads.d_exp_bits) << 5) |
                           (uint32_t(payloads.d_sign_variable) << 10) | (uint32_t(payloads.d_sign_value) << 11);
    return {0x31503651u, 1u, uint32_t(GGML_TYPE_Q6_K), uint32_t(rows), uint32_t(k), 832u,
            uint32_t(payloads.d_width), flags};
}

static bool valid_format_header(const std::array<uint32_t, 8>& header, const std::array<uint32_t, 8>& expected,
                               int rows, int k,
                               size_t planar_bytes, size_t d_bytes) {
    if (header != expected) { return false; }
    if (rows <= 0 || k <= 0 || k % 1024 != 0 || header[0] != 0x31503651u || header[1] != 1u ||
        header[2] != uint32_t(GGML_TYPE_Q6_K) || header[3] != uint32_t(rows) || header[4] != uint32_t(k) ||
        header[5] != 832u || header[6] < 10u || header[6] > 16u || (header[7] & ~0xfffu) != 0) { return false; }
    const uint32_t base = header[7] & 31u;
    const uint32_t exp_bits = (header[7] >> 5) & 31u;
    const uint32_t sign_variable = (header[7] >> 10) & 1u;
    const uint32_t sign_value = (header[7] >> 11) & 1u;
    if (exp_bits > 5u || (sign_variable && sign_value) || header[6] != 10u + exp_bits + sign_variable ||
        base + ((1u << exp_bits) - 1u) > 31u) { return false; }
    const uint64_t blocks = uint64_t(rows) * uint64_t(k / 256);
    const uint64_t expected_planar = uint64_t(rows) * uint64_t(k / 1024) * 832u;
    const uint64_t expected_d = (blocks * header[6] + 7u) / 8u;
    return expected_planar == planar_bytes && expected_d == d_bytes;
}

static void validate_format_header(const Payloads& payloads, int rows, int k) {
    const auto header = format_header(payloads, rows, k);
    if (!valid_format_header(header, header, rows, k, payloads.compressed_planar.size(), payloads.d_bits.size())) {
        throw std::runtime_error("valid compressed format header rejected");
    }
    for (size_t field = 0; field < header.size(); ++field) {
        auto damaged = header;
        damaged[field] ^= 1u;
        if (valid_format_header(damaged, header, rows, k, payloads.compressed_planar.size(), payloads.d_bits.size())) {
            throw std::runtime_error("corrupt compressed format header accepted");
        }
    }
    if (valid_format_header(header, header, rows, k + 1024, payloads.compressed_planar.size(), payloads.d_bits.size()) ||
        valid_format_header(header, header, rows + 2, k, payloads.compressed_planar.size(), payloads.d_bits.size()) ||
        valid_format_header(header, header, rows, k, payloads.compressed_planar.size() + 1, payloads.d_bits.size()) ||
        valid_format_header(header, header, rows, k, payloads.compressed_planar.size(), payloads.d_bits.size() + 1)) {
        throw std::runtime_error("corrupt compressed format descriptor accepted");
    }
}

static uint32_t load_u32(const uint8_t* p) {
    uint32_t value;
    std::memcpy(&value, p, sizeof(value));
    return value;
}

static void store_u32(uint8_t* p, uint32_t value) {
    std::memcpy(p, &value, sizeof(value));
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

static uint32_t read_bits_u32_host(const uint8_t* data, uint32_t block, uint32_t width) {
    const uint32_t bit = block * width;
    const uint32_t word_index = bit >> 5;
    const uint32_t shift = bit & 31u;
    uint32_t lo = 0;
    std::memcpy(&lo, data + word_index * 4, sizeof(lo));
    uint32_t value = lo >> shift;
    if (shift + width > 32) {
        uint32_t hi = 0;
        std::memcpy(&hi, data + (word_index + 1) * 4, sizeof(hi));
        value = (lo >> shift) | (hi << (32 - shift));
    }
    return value & ((1u << width) - 1u);
}

static size_t ql_planar_offset(int slot, int h, int pair, int word) {
    return size_t(slot) * 128 + size_t(h * 2 + pair) * 32 + size_t(word) * 4;
}

static size_t qh_planar_offset(int slot, int h, int word) {
    return 512 + size_t(slot) * 64 + size_t(h) * 32 + size_t(word) * 4;
}

static size_t scale_planar_offset(int slot, int index) {
    return 768 + size_t(slot) * 16 + size_t(index);
}

static size_t d_planar_offset(int slot) {
    return 832 + size_t(slot) * 2;
}

static size_t ql_transposed_offset(int slot, int h, int pair, int word) {
    const int group = h * 2 + pair;
    return size_t((word * 4 + group) * 16 + slot * 4);
}

static size_t qh_transposed_offset(int slot, int h, int word) {
    return 512 + size_t((word * 2 + h) * 16 + slot * 4);
}

static size_t scale_transposed_offset(int slot, int index) {
    return 768 + size_t(index * 4 + slot);
}

static size_t d_transposed_offset(int slot) {
    return 832 + size_t(slot) * 2;
}

static Payloads transform_payload(const std::vector<uint8_t>& raw, int rows, int k) {
    const int blocks_per_row = k / 256;
    const size_t row_bytes = size_t(blocks_per_row) * 210;
    if (k % 1024 != 0 || blocks_per_row % 4 != 0 || raw.size() != size_t(rows) * row_bytes) {
        throw std::runtime_error("Q6_K tensor requires complete four-block groups");
    }
    Payloads out;
    out.rows = rows;
    out.k = k;
    const int groups_per_row = blocks_per_row / 4;
    out.raw_planar.resize(raw.size());
    out.transposed.resize(raw.size());
    out.compressed_planar.resize(size_t(rows) * groups_per_row * 832);
    for (int row = 0; row < rows; ++row) {
        const size_t row_src = size_t(row) * row_bytes;
        const size_t row_dst = size_t(row) * row_bytes;
        for (int group = 0; group < groups_per_row; ++group) {
            const size_t tile_src = row_src + size_t(group) * 840;
            const size_t tile_dst = row_dst + size_t(group) * 840;
            for (int slot = 0; slot < 4; ++slot) {
                const uint8_t* block = raw.data() + tile_src + size_t(slot) * 210;
                for (int h = 0; h < 2; ++h) {
                    for (int pair = 0; pair < 2; ++pair) {
                        for (int word = 0; word < 8; ++word) {
                            const size_t source = size_t(64 * h + 32 * pair + 4 * word);
                            const uint32_t value = load_u32(block + source);
                            store_u32(out.raw_planar.data() + tile_dst + ql_planar_offset(slot, h, pair, word), value);
                            store_u32(out.transposed.data() + tile_dst + ql_transposed_offset(slot, h, pair, word), value);
                        }
                    }
                    for (int word = 0; word < 8; ++word) {
                        const uint32_t value = load_u32(block + 128 + 32 * h + 4 * word);
                        store_u32(out.raw_planar.data() + tile_dst + qh_planar_offset(slot, h, word), value);
                        store_u32(out.transposed.data() + tile_dst + qh_transposed_offset(slot, h, word), value);
                    }
                }
                for (int index = 0; index < 16; ++index) {
                    const uint8_t value = block[192 + index];
                    out.raw_planar[tile_dst + scale_planar_offset(slot, index)] = value;
                    out.transposed[tile_dst + scale_transposed_offset(slot, index)] = value;
                }
                std::memcpy(out.raw_planar.data() + tile_dst + d_planar_offset(slot), block + 208, 2);
                std::memcpy(out.transposed.data() + tile_dst + d_transposed_offset(slot), block + 208, 2);
            }
            std::memcpy(out.compressed_planar.data() + row_dst / 840 * 832 + size_t(group) * 832,
                        out.raw_planar.data() + tile_dst, 832);
        }
    }
    const uint64_t blocks = uint64_t(rows) * blocks_per_row;
    uint8_t exp_min = 31, exp_max = 0;
    uint64_t negative = 0;
    for (uint64_t block = 0; block < blocks; ++block) {
        uint16_t bits;
        std::memcpy(&bits, raw.data() + block * 210 + 208, sizeof(bits));
        const uint8_t exponent = uint8_t((bits >> 10) & 31u);
        out.d_exponents[exponent]++;
        exp_min = std::min(exp_min, exponent);
        exp_max = std::max(exp_max, exponent);
        negative += (bits & 0x8000u) != 0;
    }
    out.d_negative = negative;
    out.d_exp_base = exp_min;
    out.d_exp_max = exp_max;
    uint32_t exp_cardinality = uint32_t(exp_max - exp_min + 1);
    uint8_t exp_bits = 0;
    while ((uint32_t(1) << exp_bits) < exp_cardinality) { ++exp_bits; }
    out.d_exp_bits = exp_bits;
    out.d_sign_variable = negative != 0 && negative != blocks;
    out.d_sign_value = negative == blocks;
    out.d_width = uint8_t(10 + exp_bits + out.d_sign_variable);
    out.d_bits.resize(size_t((blocks * out.d_width + 7) / 8), 0);
    for (uint64_t block = 0; block < blocks; ++block) {
        uint16_t bits;
        std::memcpy(&bits, raw.data() + block * 210 + 208, sizeof(bits));
        const uint32_t exponent = ((bits >> 10) & 31u) - exp_min;
        const uint32_t sign = (bits >> 15) & 1u;
        uint32_t code = bits & 0x03ffu;
        code |= exponent << 10;
        if (out.d_sign_variable) { code |= sign << (10 + exp_bits); }
        write_bits(out.d_bits, block * out.d_width, out.d_width, code);
    }
    return out;
}

static std::vector<uint8_t> restore_payload(const uint8_t* payload, int rows, int k, bool transposed) {
    const int blocks_per_row = k / 256;
    const size_t row_bytes = size_t(blocks_per_row) * 210;
    const int groups_per_row = blocks_per_row / 4;
    std::vector<uint8_t> restored(size_t(rows) * row_bytes);
    for (int row = 0; row < rows; ++row) {
        for (int group = 0; group < groups_per_row; ++group) {
            const size_t tile = size_t(row) * row_bytes + size_t(group) * 840;
            for (int slot = 0; slot < 4; ++slot) {
                uint8_t* block = restored.data() + tile + size_t(slot) * 210;
                for (int h = 0; h < 2; ++h) {
                    for (int pair = 0; pair < 2; ++pair) {
                        for (int word = 0; word < 8; ++word) {
                            const size_t src = transposed ? ql_transposed_offset(slot, h, pair, word) :
                                                           ql_planar_offset(slot, h, pair, word);
                            store_u32(block + size_t(64 * h + 32 * pair + 4 * word), load_u32(payload + tile + src));
                        }
                    }
                    for (int word = 0; word < 8; ++word) {
                        const size_t src = transposed ? qh_transposed_offset(slot, h, word) :
                                                       qh_planar_offset(slot, h, word);
                        store_u32(block + 128 + 32 * h + 4 * word, load_u32(payload + tile + src));
                    }
                }
                for (int index = 0; index < 16; ++index) {
                    const size_t src = transposed ? scale_transposed_offset(slot, index) :
                                                   scale_planar_offset(slot, index);
                    block[192 + index] = payload[tile + src];
                }
                const size_t dsrc = transposed ? d_transposed_offset(slot) : d_planar_offset(slot);
                std::memcpy(block + 208, payload + tile + dsrc, 2);
            }
        }
    }
    return restored;
}

static uint16_t unpack_d(const Payloads& payloads, uint64_t block) {
    const uint32_t code = read_bits_host(payloads.d_bits.data(), block * payloads.d_width, payloads.d_width);
    const uint32_t exponent = uint32_t(payloads.d_exp_base) + ((code >> 10) & ((1u << payloads.d_exp_bits) - 1u));
    const uint32_t sign = payloads.d_sign_variable ? ((code >> (10 + payloads.d_exp_bits)) & 1u) : payloads.d_sign_value;
    return uint16_t((code & 0x03ffu) | (exponent << 10) | (sign << 15));
}

static std::vector<uint8_t> restore_compressed_planar(const Payloads& payloads, int rows, int k) {
    const int blocks_per_row = k / 256;
    const size_t row_bytes = size_t(blocks_per_row) * 210;
    const size_t plane_row_bytes = size_t(blocks_per_row / 4) * 832;
    std::vector<uint8_t> restored(size_t(rows) * row_bytes);
    for (uint64_t block = 0; block < uint64_t(rows) * blocks_per_row; ++block) {
        const int row = int(block / blocks_per_row);
        const int local_block = int(block % blocks_per_row);
        const int tile = local_block / 4;
        const int slot = local_block % 4;
        const uint8_t* plane = payloads.compressed_planar.data() + size_t(row) * plane_row_bytes + size_t(tile) * 832;
        uint8_t* dst = restored.data() + block * 210;
        for (int h = 0; h < 2; ++h) {
            for (int pair = 0; pair < 2; ++pair) {
                for (int word = 0; word < 8; ++word) {
                    store_u32(dst + 64 * h + 32 * pair + 4 * word,
                              load_u32(plane + size_t(slot) * 128 + size_t(h * 2 + pair) * 32 + size_t(word) * 4));
                }
            }
            for (int word = 0; word < 8; ++word) {
                store_u32(dst + 128 + 32 * h + 4 * word,
                          load_u32(plane + 512 + size_t(slot) * 64 + size_t(h) * 32 + size_t(word) * 4));
            }
        }
        for (int index = 0; index < 16; ++index) {
            dst[192 + index] = plane[768 + size_t(slot) * 16 + index];
        }
        const uint16_t d = unpack_d(payloads, block);
        std::memcpy(dst + 208, &d, sizeof(d));
    }
    return restored;
}

static Activation make_activation(int k, int t) {
    Activation activation;
    activation.k = k;
    activation.t = t;
    activation.qs.resize(size_t(k) * t);
    activation.ds.resize(size_t(k / 32) * t);
    for (size_t i = 0; i < activation.qs.size(); ++i) {
        activation.qs[i] = int8_t((i * 37 + i / 7 * 11) % 255 - 127);
    }
    for (size_t i = 0; i < activation.ds.size(); ++i) {
        int sum = 0;
        for (int j = 0; j < 32; ++j) { sum += activation.qs[i * 32 + j]; }
        activation.ds[i] = __floats2half2_rn(0.0625f, sum * 0.0625f);
    }
    return activation;
}

static Activation load_activation(const std::string& path, int expected_k, int expected_t) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { throw std::runtime_error("cannot open activation snapshot"); }
    std::array<uint32_t, 8> header{};
    file.read(reinterpret_cast<char*>(header.data()), sizeof(header));
    const uint32_t expected_bytes = uint32_t(expected_k * expected_t + (expected_k / 32) * expected_t * 4);
    if (!file || header[0] != 0x31534156u || header[1] != 1 || header[2] != uint32_t(GGML_TYPE_Q6_K) ||
        header[3] != uint32_t(expected_k) || header[4] != uint32_t(expected_t) || header[5] != 0 ||
        header[6] != expected_bytes || header[7] != 0) {
        throw std::runtime_error("activation snapshot header mismatch");
    }
    Activation activation;
    activation.k = expected_k;
    activation.t = expected_t;
    activation.qs.resize(size_t(expected_k) * expected_t);
    activation.ds.resize(size_t(expected_k / 32) * expected_t);
    file.read(reinterpret_cast<char*>(activation.qs.data()), std::streamsize(activation.qs.size()));
    file.read(reinterpret_cast<char*>(activation.ds.data()), std::streamsize(activation.ds.size() * sizeof(half2)));
    if (!file || file.peek() != std::char_traits<char>::eof()) { throw std::runtime_error("activation snapshot payload size mismatch"); }
    return activation;
}

__device__ __forceinline__ uint32_t load_word(const uint8_t* p) {
    return *reinterpret_cast<const uint32_t*>(p);
}

__device__ __forceinline__ uint32_t read_bits_device(const uint8_t* data, uint32_t at, uint32_t width) {
    const uint32_t byte = at >> 3;
    const uint32_t shift = at & 7;
    uint32_t value = uint32_t(data[byte]) | (uint32_t(data[byte + 1]) << 8);
    if (shift + width > 16) { value |= uint32_t(data[byte + 2]) << 16; }
    return (value >> shift) & ((1u << width) - 1u);
}

__device__ __forceinline__ int decode_q6_word(uint32_t lo, uint32_t hi, int q) {
    const uint32_t code = ((lo >> (4 * (q >> 1))) & 0x0F0F0F0Fu) |
                          (((hi >> (2 * q)) << 4) & 0x30303030u);
    return int(code ^ 0xE0E0E0E0u ^ ((code & 0x20202020u) * 6u));
}

template <int Width>
__device__ __forceinline__ uint32_t read_bits_device_u32(const uint8_t* data, uint32_t block) {
    const uint32_t bit = block * Width;
    const uint32_t word_index = bit >> 5;
    const uint32_t shift = bit & 31u;
    const uint32_t lo = load_word(data + word_index * 4);
    uint32_t value = lo >> shift;
    if (shift + Width > 32) {
        const uint32_t hi = load_word(data + (word_index + 1) * 4);
        value = __funnelshift_r(lo, hi, shift);
    }
    return value & ((1u << Width) - 1u);
}

template <int T, bool Transposed, bool CompressedD = false, int DWidth = 0, int DExpBits = 0,
          bool VectorLoads = false>
__launch_bounds__(128) __global__ void rearranged_kernel(VecArgs p, const uint8_t* d_bits,
                                                        uint8_t d_width, uint8_t d_exp_base,
                                                        uint8_t d_exp_bits, uint8_t d_sign_variable,
                                                        uint8_t d_sign_value, uint32_t d_block_origin) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int k = p.k;
    const int slices = k / 32;
    const int blocks_per_row = k / 256;
    const int tile_bytes = CompressedD ? 832 : 840;
    const int row_bytes = (blocks_per_row / 4) * tile_bytes;
    for (int row0 = (blockIdx.x * 4 + warp) * 2; row0 < p.rows;
         row0 += gridDim.x * 8) {
        float acc[2][T];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
#pragma unroll
            for (int j = 0; j < T; ++j) { acc[r][j] = 0.0f; }
        }
        for (int it = 0; lane + 32 * it < slices; ++it) {
            int a[T][8];
            float d[T], sum[T];
#pragma unroll
            for (int j = 0; j < T; ++j) {
                const float2 ds = __half22float2(p.ds[int64_t(j) * slices + lane + 32 * it]);
                d[j] = ds.x;
                sum[j] = ds.y;
                const int4* q = reinterpret_cast<const int4*>(p.qs + int64_t(j) * k + 32 * (lane + 32 * it));
                const int4 v0 = q[0], v1 = q[1];
                a[j][0] = v0.x; a[j][1] = v0.y; a[j][2] = v0.z; a[j][3] = v0.w;
                a[j][4] = v1.x; a[j][5] = v1.y; a[j][6] = v1.z; a[j][7] = v1.w;
            }
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                const int row = row0 + r;
                if (row < p.rows) {
                    const int slot = lane / 8;
                    const int tile_index = it;
                    const uint8_t* tile = p.weight + int64_t(row) * row_bytes + int64_t(tile_index) * tile_bytes;
                    Slice sl;
                    if constexpr (VectorLoads && CompressedD) {
                        static_assert(!Transposed);
                        const int h = (lane % 8) / 4;
                        const int q = lane % 4;
                        const int group = h * 2 + (q & 1);
                        const size_t lo_offset = size_t(slot * 128 + group * 32);
                        const size_t hi_offset = size_t(512 + slot * 64 + h * 32);
                        {
                            const uint4 lo = *reinterpret_cast<const uint4*>(tile + lo_offset);
                            const uint4 hi = *reinterpret_cast<const uint4*>(tile + hi_offset);
                            sl.w[0] = decode_q6_word(lo.x, hi.x, q);
                            sl.w[1] = decode_q6_word(lo.y, hi.y, q);
                            sl.w[2] = decode_q6_word(lo.z, hi.z, q);
                            sl.w[3] = decode_q6_word(lo.w, hi.w, q);
                        }
                        {
                            const uint4 lo = *reinterpret_cast<const uint4*>(tile + lo_offset + 16);
                            const uint4 hi = *reinterpret_cast<const uint4*>(tile + hi_offset + 16);
                            sl.w[4] = decode_q6_word(lo.x, hi.x, q);
                            sl.w[5] = decode_q6_word(lo.y, hi.y, q);
                            sl.w[6] = decode_q6_word(lo.z, hi.z, q);
                            sl.w[7] = decode_q6_word(lo.w, hi.w, q);
                        }
                    } else {
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const int h = (lane % 8) / 4;
                        const int q = lane % 4;
                        const int group = h * 2 + (q & 1);
                        const size_t lo_offset = Transposed ? size_t((i * 4 + group) * 16 + slot * 4) :
                                                              size_t(slot * 128 + group * 32 + i * 4);
                        const size_t hi_offset = Transposed ? size_t(512 + (i * 2 + h) * 16 + slot * 4) :
                                                              size_t(512 + slot * 64 + h * 32 + i * 4);
                        const uint32_t lo = load_word(tile + lo_offset);
                        const uint32_t hi = load_word(tile + hi_offset);
                        const uint32_t code = ((lo >> (4 * (q >> 1))) & 0x0F0F0F0Fu) |
                                              (((hi >> (2 * q)) << 4) & 0x30303030u);
                        sl.w[i] = int(code ^ 0xE0E0E0E0u ^ ((code & 0x20202020u) * 6u));
                    }
                    }
                    const int h = (lane % 8) / 4;
                    const int q = lane % 4;
                    const size_t scale_offset = Transposed ? size_t(768 + (h * 8 + 2 * q) * 4 + slot) :
                                                              size_t(768 + slot * 16 + h * 8 + 2 * q);
                    const uint8_t* scale_ptr = tile + scale_offset;
                    const int8_t s0 = *reinterpret_cast<const int8_t*>(scale_ptr);
                    const int8_t s1 = *reinterpret_cast<const int8_t*>(scale_ptr + (Transposed ? 4 : 1));
                    float block_d;
                    if constexpr (CompressedD) {
                        const uint32_t block_index = d_block_origin + uint32_t(row) * uint32_t(blocks_per_row) + uint32_t(tile_index * 4 + slot);
                        uint32_t exponent;
                        uint32_t sign;
                        uint32_t mantissa;
                        if constexpr (DWidth != 0) {
                            const uint32_t code = read_bits_device_u32<DWidth>(d_bits, block_index);
                            exponent = (code >> 10) & ((1u << DExpBits) - 1u);
                            sign = (code >> (10 + DExpBits)) & 1u;
                            mantissa = code & 0x03ffu;
                        } else {
                            const uint32_t code = read_bits_device(d_bits, block_index * uint32_t(d_width), d_width);
                            const uint32_t exp_mask = (1u << d_exp_bits) - 1u;
                            exponent = uint32_t(d_exp_base) + ((code >> 10) & exp_mask);
                            sign = d_sign_variable ? ((code >> (10 + d_exp_bits)) & 1u) : d_sign_value;
                            mantissa = code & 0x03ffu;
                        }
                        const uint16_t half_bits = uint16_t(mantissa | (exponent << 10) | (sign << 15));
                        block_d = __half2float(__ushort_as_half(half_bits));
                    } else {
                        block_d = half_at(tile + 832 + slot * 2);
                    }
                    sl.f[0] = block_d * float(s0);
                    sl.f[1] = block_d * float(s1);
#pragma unroll
                    for (int j = 0; j < T; ++j) { acc[r][j] += Decoder<GGML_TYPE_Q6_K>::dot(sl, a[j], d[j], sum[j]); }
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
                if (lane == 0 && row0 + r < p.rows) { vec_store(p.out, row0 + r, j, acc[r][j]); }
            }
        }
    }
}

template <int T, bool Transposed, bool CompressedD = false, int DWidth = 0, int DExpBits = 0,
          bool VectorLoads = false>
static void launch_rearranged(VecArgs args, const uint8_t* d_bits,
                              uint8_t d_width, uint8_t d_exp_base,
                              uint8_t d_exp_bits, uint8_t d_sign_variable, uint8_t d_sign_value,
                              cudaStream_t stream, uint32_t d_block_origin = 0) {
    int active = 0;
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, rearranged_kernel<T, Transposed, CompressedD, DWidth, DExpBits, VectorLoads>, 128, 0), "rearranged occupancy");
    const int groups = (args.rows + 7) / 8;
    const int grid = std::min(groups, std::max(active, 1) * device_facts().sm_count);
    rearranged_kernel<T, Transposed, CompressedD, DWidth, DExpBits, VectorLoads><<<grid, 128, 0, stream>>>(
        args, d_bits, d_width, d_exp_base, d_exp_bits, d_sign_variable, d_sign_value, d_block_origin);
    check(cudaGetLastError(), "rearranged launch");
}

template <int T, bool FastWidth12>
static void launch_q6_fields_direct(VecArgs args, const uint8_t* d_bits,
                                    uint8_t d_width, uint8_t d_exp_base,
                                    uint8_t d_exp_bits, uint8_t d_sign_variable, uint8_t d_sign_value,
                                    cudaStream_t stream, uint32_t row_begin = 0) {
    int active = 0;
    const int groups = (args.rows + 7) / 8;
    if constexpr (FastWidth12) {
        check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,
            ninfer::ops::gguf::detail::q6_fields::kernel<T, false, true, true>, 128, 0), "q6_fields direct occupancy");
        const int grid = std::min(groups, std::max(active, 1) * device_facts().sm_count);
        ninfer::ops::gguf::detail::q6_fields::kernel<T, false, true, true><<<grid, 128, 0, stream>>>(
            args, d_bits, d_width, d_exp_base, d_exp_bits, d_sign_variable, d_sign_value, row_begin);
    } else {
        check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,
            ninfer::ops::gguf::detail::q6_fields::kernel<T, false, true, false>, 128, 0), "q6_fields direct occupancy");
        const int grid = std::min(groups, std::max(active, 1) * device_facts().sm_count);
        ninfer::ops::gguf::detail::q6_fields::kernel<T, false, true, false><<<grid, 128, 0, stream>>>(
            args, d_bits, d_width, d_exp_base, d_exp_bits, d_sign_variable, d_sign_value, row_begin);
    }
    check(cudaGetLastError(), "q6_fields direct launch");
}

template <int T>
static void launch_q6_fields_runtime(VecArgs args, const uint8_t* d_bits,
                                     uint8_t d_width, uint8_t d_exp_base,
                                     uint8_t d_exp_bits, uint8_t d_sign_variable, uint8_t d_sign_value,
                                     cudaStream_t stream, uint32_t row_begin = 0) {
    ninfer::ops::gguf::detail::q6_fields::launch<T, false, true>(args, d_bits, d_width, d_exp_base,
        d_exp_bits, d_sign_variable, d_sign_value, row_begin, stream);
    check(cudaGetLastError(), "q6_fields runtime launch");
}

struct KernelResource {
    int registers = 0;
    int shared_bytes = 0;
    int local_bytes = 0;
    int active_blocks = 0;
    int active_warps = 0;
    float occupancy = 0.0f;
};

static constexpr size_t kGuardElements = 16;
static constexpr uint32_t kF32Guard = 0x7fc12345u;
static constexpr uint16_t kBf16Guard = 0x7fc1u;
static constexpr uint8_t kByteGuard = 0xa5u;

static void write_f32_guard(float* data, size_t count) {
    uint32_t guard[kGuardElements];
    for (size_t i = 0; i < kGuardElements; ++i) { guard[i] = kF32Guard; }
    check(cudaMemcpy(reinterpret_cast<uint8_t*>(data) + count * sizeof(float), guard, sizeof(guard), cudaMemcpyHostToDevice), "f32 guard write");
}

static void check_f32_guard(const float* data, size_t count) {
    uint32_t guard[kGuardElements];
    check(cudaMemcpy(guard, reinterpret_cast<const uint8_t*>(data) + count * sizeof(float), sizeof(guard), cudaMemcpyDeviceToHost), "f32 guard read");
    for (uint32_t value : guard) { if (value != kF32Guard) { throw std::runtime_error("f32 output canary changed"); } }
}

static void write_bf16_guard(__nv_bfloat16* data, size_t count) {
    uint16_t guard[kGuardElements];
    for (size_t i = 0; i < kGuardElements; ++i) { guard[i] = kBf16Guard; }
    check(cudaMemcpy(reinterpret_cast<uint8_t*>(data) + count * sizeof(__nv_bfloat16), guard, sizeof(guard), cudaMemcpyHostToDevice), "bf16 guard write");
}

static void check_bf16_guard(const __nv_bfloat16* data, size_t count) {
    uint16_t guard[kGuardElements];
    check(cudaMemcpy(guard, reinterpret_cast<const uint8_t*>(data) + count * sizeof(__nv_bfloat16), sizeof(guard), cudaMemcpyDeviceToHost), "bf16 guard read");
    for (uint16_t value : guard) { if (value != kBf16Guard) { throw std::runtime_error("bf16 output canary changed"); } }
}

static void write_byte_guard(uint8_t* data, size_t count) {
    uint8_t guard[kGuardElements];
    for (size_t i = 0; i < kGuardElements; ++i) { guard[i] = kByteGuard; }
    check(cudaMemcpy(data + count, guard, sizeof(guard), cudaMemcpyHostToDevice), "byte guard write");
}

static void check_byte_guard(const uint8_t* data, size_t count) {
    uint8_t guard[kGuardElements];
    check(cudaMemcpy(guard, data + count, sizeof(guard), cudaMemcpyDeviceToHost), "byte guard read");
    for (uint8_t value : guard) { if (value != kByteGuard) { throw std::runtime_error("payload reconstruction canary changed"); } }
}

template <typename Kernel>
static KernelResource kernel_resource(Kernel kernel) {
    cudaFuncAttributes attributes{};
    check(cudaFuncGetAttributes(&attributes, kernel), "kernel attributes");
    int active = 0;
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, kernel, 128, 0), "kernel occupancy");
    cudaDeviceProp properties{};
    check(cudaGetDeviceProperties(&properties, 0), "device properties");
    KernelResource out;
    out.registers = attributes.numRegs;
    out.shared_bytes = int(attributes.sharedSizeBytes);
    out.local_bytes = int(attributes.localSizeBytes);
    out.active_blocks = active;
    out.active_warps = active * 4;
    out.occupancy = float(out.active_warps) / float(properties.maxThreadsPerMultiProcessor / 32);
    return out;
}

static void print_resource(const char* name, const KernelResource& resource) {
    std::cout << "\"" << name << "\":{\"registers_per_thread\":" << resource.registers
              << ",\"static_shared_bytes\":" << resource.shared_bytes
              << ",\"local_bytes_per_thread\":" << resource.local_bytes
              << ",\"active_blocks_per_sm\":" << resource.active_blocks
              << ",\"active_warps_per_sm\":" << resource.active_warps
              << ",\"warp_occupancy\":" << resource.occupancy << '}';
}

__device__ __forceinline__ uint16_t decode_d_device(const uint8_t* d_bits, uint32_t block,
                                                    uint8_t width, uint8_t exp_base,
                                                    uint8_t exp_bits, uint8_t sign_variable,
                                                    uint8_t sign_value) {
    const uint32_t code = read_bits_device(d_bits, block * uint32_t(width), width);
    const uint32_t exponent = uint32_t(exp_base) + ((code >> 10) & ((1u << exp_bits) - 1u));
    const uint32_t sign = sign_variable ? ((code >> (10 + exp_bits)) & 1u) : sign_value;
    return uint16_t((code & 0x03ffu) | (exponent << 10) | (sign << 15));
}

__global__ void reconstruct_compressed_kernel(int rows, int k, const uint8_t* planar,
                                               const uint8_t* d_bits, uint8_t d_width,
                                               uint8_t d_exp_base, uint8_t d_exp_bits,
                                               uint8_t d_sign_variable, uint8_t d_sign_value,
                                               uint8_t* raw) {
    const uint64_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t blocks = uint64_t(rows) * uint64_t(k / 256);
    const uint64_t byte_count = blocks * 210;
    if (index >= byte_count) { return; }
    const uint64_t block = index / 210;
    const uint32_t field = uint32_t(index % 210);
    const uint32_t blocks_per_row = uint32_t(k / 256);
    const uint32_t row = uint32_t(block / blocks_per_row);
    const uint32_t local = uint32_t(block % blocks_per_row);
    const uint32_t tile = local / 4;
    const uint32_t slot = local % 4;
    const uint32_t tile_count = blocks_per_row / 4;
    const uint8_t* tile_data = planar + uint64_t(row) * tile_count * 832 + uint64_t(tile) * 832;
    uint8_t value;
    if (field < 128) {
        const uint32_t h = field / 64;
        const uint32_t pair = (field % 64) / 32;
        const uint32_t word = (field % 32) / 4;
        const uint32_t byte = field & 3;
        value = tile_data[slot * 128 + (h * 2 + pair) * 32 + word * 4 + byte];
    } else if (field < 192) {
        const uint32_t index = field - 128;
        const uint32_t h = index / 32;
        const uint32_t word = (index % 32) / 4;
        const uint32_t byte = index & 3;
        value = tile_data[512 + slot * 64 + h * 32 + word * 4 + byte];
    } else if (field < 208) {
        value = tile_data[768 + slot * 16 + field - 192];
    } else {
        const uint16_t bits = decode_d_device(d_bits, uint32_t(block), d_width, d_exp_base,
                                              d_exp_bits, d_sign_variable, d_sign_value);
        value = uint8_t(bits >> (8 * (field - 208)));
    }
    raw[index] = value;
}

__global__ void compare_payload_kernel(const uint8_t* a, const uint8_t* b, uint64_t count, uint32_t* mismatch) {
    const uint64_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count && a[index] != b[index]) { atomicExch(mismatch, 1u); }
}

template <int T>
static void verify_subview_graph(const Payloads& payloads, int n, int k, uint8_t* compressed_planar,
                                 uint8_t* d_bits, const int8_t* qs, const half2* ds,
                                 const std::vector<float>& full_f32, const std::vector<uint16_t>& full_bf16,
                                 bool fast_decoder, cudaStream_t stream, bool vector_load = false) {
    const int view_origin_rows = n / 2;
    const int first_row_in_view = 1;
    const int first_global_row = view_origin_rows + first_row_in_view;
    constexpr int view_rows = 2;
    if (first_global_row + view_rows > n) { throw std::runtime_error("invalid GDN subview test range"); }
    const int blocks_per_row = k / 256;
    const int row_bytes = (blocks_per_row / 4) * 832;
    float* view_f32 = nullptr;
    __nv_bfloat16* view_bf16 = nullptr;
    check(cudaMalloc(&view_f32, (size_t(view_rows) * T + kGuardElements) * sizeof(float)), "subview f32 alloc");
    check(cudaMalloc(&view_bf16, (size_t(view_rows) * T + kGuardElements) * sizeof(__nv_bfloat16)), "subview bf16 alloc");
    write_f32_guard(view_f32, size_t(view_rows) * T);
    write_bf16_guard(view_bf16, size_t(view_rows) * T);
    VecArgs view{compressed_planar + int64_t(first_global_row) * row_bytes, row_bytes, view_rows, k, qs, ds,
                 {nullptr, view_f32, view_rows, ninfer::ops::gguf::Epilogue::Store, nullptr, 0}};
    const uint32_t d_block_origin = uint32_t(first_global_row) * uint32_t(blocks_per_row);
    auto launch = [&] {
        if (!fast_decoder) {
            launch_q6_fields_direct<T, false>(view, d_bits, payloads.d_width, payloads.d_exp_base,
                payloads.d_exp_bits, payloads.d_sign_variable, payloads.d_sign_value, stream, uint32_t(first_global_row));
        } else if (payloads.d_width == 12 && payloads.d_exp_bits == 1 && payloads.d_exp_base == 0 && payloads.d_sign_variable) {
            launch_q6_fields_direct<T, true>(view, d_bits, payloads.d_width, payloads.d_exp_base,
                payloads.d_exp_bits, payloads.d_sign_variable, payloads.d_sign_value, stream, uint32_t(first_global_row));
        } else if (vector_load) {
            launch_rearranged<T, false, true, 0, 0, true>(view, d_bits, payloads.d_width, payloads.d_exp_base,
                                                          payloads.d_exp_bits, payloads.d_sign_variable,
                                                          payloads.d_sign_value, stream, d_block_origin);
        } else if (payloads.d_width == 12 && payloads.d_exp_bits == 1) {
            launch_rearranged<T, false, true, 12, 1>(view, d_bits, payloads.d_width, payloads.d_exp_base,
                                                       payloads.d_exp_bits, payloads.d_sign_variable,
                                                       payloads.d_sign_value, stream, d_block_origin);
        } else if (payloads.d_width == 13 && payloads.d_exp_bits == 2) {
            launch_rearranged<T, false, true, 13, 2>(view, d_bits, payloads.d_width, payloads.d_exp_base,
                                                       payloads.d_exp_bits, payloads.d_sign_variable,
                                                       payloads.d_sign_value, stream, d_block_origin);
        } else { throw std::runtime_error("unsupported GDN subview d format"); }
    };
    auto graph_run = [&](bool bf16) {
        view.out.f32 = bf16 ? nullptr : view_f32;
        view.out.bf16 = bf16 ? view_bf16 : nullptr;
        cudaGraph_t graph;
        cudaGraphExec_t executable;
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "subview capture begin");
        launch();
        check(cudaStreamEndCapture(stream, &graph), "subview capture end");
        check(cudaGraphInstantiate(&executable, graph, 0), "subview graph instantiate");
        check(cudaGraphDestroy(graph), "subview graph destroy");
        for (int repeat = 0; repeat < 3; ++repeat) { check(cudaGraphLaunch(executable, stream), "subview graph replay"); }
        check(cudaStreamSynchronize(stream), "subview graph sync");
        check(cudaGraphExecDestroy(executable), "subview graph executable destroy");
    };
    graph_run(false);
    graph_run(true);
    check_f32_guard(view_f32, size_t(view_rows) * T);
    check_bf16_guard(view_bf16, size_t(view_rows) * T);
    std::vector<float> result_f32(size_t(view_rows) * T);
    std::vector<uint16_t> result_bf16(size_t(view_rows) * T);
    check(cudaMemcpy(result_f32.data(), view_f32, result_f32.size() * sizeof(float), cudaMemcpyDeviceToHost), "subview f32 copy");
    check(cudaMemcpy(result_bf16.data(), view_bf16, result_bf16.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost), "subview bf16 copy");
    for (int column = 0; column < T; ++column) {
        for (int local_row = 0; local_row < view_rows; ++local_row) {
            const size_t full_index = size_t(column) * n + first_global_row + local_row;
            const size_t view_index = size_t(column) * view_rows + local_row;
            if (std::memcmp(&full_f32[full_index], &result_f32[view_index], sizeof(float)) != 0 ||
                full_bf16[full_index] != result_bf16[view_index]) {
                throw std::runtime_error("GDN subview graph output mismatch");
            }
        }
    }
    check(cudaFree(view_f32), "subview f32 free");
    check(cudaFree(view_bf16), "subview bf16 free");
}

template <int T>
static void run_bench(const std::vector<uint8_t>& raw, const Payloads& payloads, int n, int k,
                      const std::string& activation_path, const std::string& label) {
    if (k != 5120 || payloads.d_width != 12 || payloads.d_exp_base != 0 ||
        payloads.d_exp_bits != 1 || !payloads.d_sign_variable) {
        throw std::runtime_error("head Q6 width-12 probe requires width12/base0/expbits1/signvar");
    }
    Activation activation = activation_path.empty() ? make_activation(k, T) : load_activation(activation_path, k, T);
    if (activation.k != k || activation.t != T) { throw std::runtime_error("activation shape mismatch"); }
    uint8_t* device_weights[3]{};
    uint8_t* device_d_bits = nullptr;
    uint8_t* device_reconstructed = nullptr;
    uint32_t* device_mismatch = nullptr;
    int8_t* device_qs = nullptr;
    half2* device_ds = nullptr;
    float* output = nullptr;
    auto alloc = [](auto** ptr, size_t bytes) { check(cudaMalloc(ptr, bytes), "alloc"); };
    alloc(&device_weights[0], raw.size());
    check(cudaMemcpy(device_weights[0], raw.data(), raw.size(), cudaMemcpyHostToDevice), "raw upload");
    alloc(&device_weights[1], payloads.raw_planar.size());
    check(cudaMemcpy(device_weights[1], payloads.raw_planar.data(), payloads.raw_planar.size(), cudaMemcpyHostToDevice), "planar upload");
    alloc(&device_weights[2], payloads.compressed_planar.size());
    check(cudaMemcpy(device_weights[2], payloads.compressed_planar.data(), payloads.compressed_planar.size(), cudaMemcpyHostToDevice), "compressed planar upload");
    alloc(&device_d_bits, payloads.d_bits.size());
    check(cudaMemcpy(device_d_bits, payloads.d_bits.data(), payloads.d_bits.size(), cudaMemcpyHostToDevice), "compressed d upload");
    alloc(&device_reconstructed, raw.size() + kGuardElements);
    write_byte_guard(device_reconstructed, raw.size());
    alloc(&device_mismatch, sizeof(uint32_t));
    check(cudaMemset(device_mismatch, 0, sizeof(uint32_t)), "oracle flag clear");
    reconstruct_compressed_kernel<<<(raw.size() + 255) / 256, 256>>>(
        n, k, device_weights[2], device_d_bits, payloads.d_width, payloads.d_exp_base,
        payloads.d_exp_bits, payloads.d_sign_variable, payloads.d_sign_value, device_reconstructed);
    check(cudaGetLastError(), "reconstruction launch");
    compare_payload_kernel<<<(raw.size() + 255) / 256, 256>>>(
        device_weights[0], device_reconstructed, raw.size(), device_mismatch);
    check(cudaGetLastError(), "reconstruction compare launch");
    uint32_t mismatch = 0;
    check(cudaMemcpy(&mismatch, device_mismatch, sizeof(mismatch), cudaMemcpyDeviceToHost), "oracle flag copy");
    if (mismatch) { throw std::runtime_error("compressed format GPU full-payload reconstruction mismatch"); }
    check_byte_guard(device_reconstructed, raw.size());
    check(cudaFree(device_mismatch), "free oracle flag");
    check(cudaFree(device_reconstructed), "free oracle payload");
    device_mismatch = nullptr;
    device_reconstructed = nullptr;
    alloc(&device_qs, activation.qs.size());
    check(cudaMemcpy(device_qs, activation.qs.data(), activation.qs.size(), cudaMemcpyHostToDevice), "activation q upload");
    alloc(&device_ds, activation.ds.size() * sizeof(half2));
    check(cudaMemcpy(device_ds, activation.ds.data(), activation.ds.size() * sizeof(half2), cudaMemcpyHostToDevice), "activation ds upload");
    const size_t output_elements = size_t(n) * T;
    alloc(&output, (output_elements + kGuardElements) * sizeof(float));
    write_f32_guard(output, output_elements);
    VecArgs args{device_weights[0], k / 256 * 210, n, k, device_qs, device_ds,
                 {nullptr, output, n, ninfer::ops::gguf::Epilogue::Store, nullptr, 0}};
    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
    auto dynamic_baseline = [&] {
        args.weight = device_weights[0];
        vec::launch<GGML_TYPE_Q6_K, T, false, 0>(args, stream);
    };
    auto q6_fields_runtime = [&] {
        args.weight = device_weights[2];
        launch_q6_fields_runtime<T>(args, device_d_bits, payloads.d_width, payloads.d_exp_base,
            payloads.d_exp_bits, payloads.d_sign_variable, payloads.d_sign_value, stream);
    };
    auto q6_fields_dynamic = [&] {
        args.weight = device_weights[2];
        launch_q6_fields_direct<T, false>(args, device_d_bits, payloads.d_width, payloads.d_exp_base,
            payloads.d_exp_bits, payloads.d_sign_variable, payloads.d_sign_value, stream);
    };
    auto q6_fields_width12 = [&] {
        args.weight = device_weights[2];
        if (k != 5120 || payloads.d_width != 12 || payloads.d_exp_base != 0 ||
            payloads.d_exp_bits != 1 || !payloads.d_sign_variable) {
            throw std::runtime_error("head width-12 specialization metadata mismatch");
        }
        launch_q6_fields_direct<T, true>(args, device_d_bits, payloads.d_width, payloads.d_exp_base,
            payloads.d_exp_bits, payloads.d_sign_variable, payloads.d_sign_value, stream);
    };
    const std::array<const char*, 4> names{"raw_dynamic_baseline", "q6_fields_runtime_dispatch", "q6_fields_dynamic_compressed", "q6_fields_forced_fastwidth12"};
    const std::array<std::function<void()>, 4> modes{dynamic_baseline, q6_fields_runtime, q6_fields_dynamic, q6_fields_width12};
    std::vector<float> reference(size_t(n) * T), actual(reference.size());
    for (int mode = 0; mode < 4; ++mode) {
        modes[mode]();
        check(cudaStreamSynchronize(stream), "output sync");
        check_f32_guard(output, output_elements);
        check(cudaMemcpy(actual.data(), output, actual.size() * sizeof(float), cudaMemcpyDeviceToHost), "output copy");
        if (mode == 0) { reference = actual; }
        else if (std::memcmp(reference.data(), actual.data(), actual.size() * sizeof(float)) != 0) {
            throw std::runtime_error(std::string(names[mode]) + " FP32 bit mismatch");
        }
    }
    __nv_bfloat16* bfout;
    alloc(&bfout, (output_elements + kGuardElements) * sizeof(__nv_bfloat16));
    write_bf16_guard(bfout, output_elements);
    args.out.f32 = nullptr;
    args.out.bf16 = bfout;
    std::vector<uint16_t> bf_reference(size_t(n) * T), bf_actual(bf_reference.size());
    for (int mode = 0; mode < 4; ++mode) {
        modes[mode]();
        check(cudaStreamSynchronize(stream), "bf16 sync");
        check_bf16_guard(bfout, output_elements);
        check(cudaMemcpy(bf_actual.data(), bfout, bf_actual.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost), "bf16 copy");
        if (mode == 0) { bf_reference = bf_actual; }
        else if (bf_reference != bf_actual) { throw std::runtime_error(std::string(names[mode]) + " BF16 bit mismatch"); }
    }
    args.out.bf16 = nullptr;
    args.out.f32 = output;
    verify_subview_graph<T>(payloads, n, k, device_weights[2], device_d_bits, device_qs, device_ds,
                            reference, bf_reference, false, stream);
    verify_subview_graph<T>(payloads, n, k, device_weights[2], device_d_bits, device_qs, device_ds,
                            reference, bf_reference, true, stream);
    cudaGraphExec_t graphs[4];
    for (int mode = 0; mode < 4; ++mode) {
        cudaGraph_t graph;
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "capture begin");
        modes[mode]();
        check(cudaStreamEndCapture(stream, &graph), "capture end");
        check(cudaGraphInstantiate(&graphs[mode], graph, 0), "graph instantiate");
        check(cudaGraphDestroy(graph), "graph destroy");
    }
    for (int i = 0; i < 5; ++i) {
        for (int mode = 0; mode < 4; ++mode) { check(cudaGraphLaunch(graphs[mode], stream), "warmup"); }
    }
    check(cudaStreamSynchronize(stream), "warmup sync");
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start), "start event");
    check(cudaEventCreate(&stop), "stop event");
    KernelResource dynamic_resource = kernel_resource(kernel<GGML_TYPE_Q6_K, T, false, 0>);
    KernelResource q6_fields_dynamic_resource = kernel_resource(ninfer::ops::gguf::detail::q6_fields::kernel<T, false, true, false>);
    KernelResource q6_fields_width12_resource = kernel_resource(ninfer::ops::gguf::detail::q6_fields::kernel<T, false, true, true>);
    KernelResource q6_fields_runtime_resource = q6_fields_dynamic_resource;
    const auto header = format_header(payloads, n, k);
    std::cout << "{\"matrix\":\"" << label << "\",\"N\":" << n << ",\"K\":" << k << ",\"T\":" << T
              << ",\"baseline_bytes\":" << raw.size() << ",\"raw_planar_bytes_including_header\":" << payloads.raw_planar.size() + 32
              << ",\"compressed_planar_dwindow_bytes_including_headers\":" << payloads.compressed_planar.size() + payloads.d_bits.size() + 32
              << ",\"compressed_d_width_bits\":" << unsigned(payloads.d_width)
              << ",\"format_header_u32\":[";
    for (size_t i = 0; i < header.size(); ++i) { if (i) { std::cout << ','; } std::cout << header[i]; }
    std::cout << "]"
              << ",\"planar_field_offsets\":{\"ql\":0,\"qh\":512,\"scales\":768,\"tile_bytes\":832}"
              << ",\"d_plane_offset\":" << payloads.compressed_planar.size()
              << ",\"gpu_full_payload_reconstruction_exact\":true"
              << ",\"head_subview_graph_f32_bf16_exact_q6_fields_dynamic_and_width12\":true"
              << ",\"subview_parent_row_origin\":" << n / 2
              << ",\"subview_first_row_in_view\":1,\"subview_rows\":2"
              << ",\"fp32_output_bit_exact\":true,\"bf16_output_bit_exact\":true,\"activation_snapshot\":\"" << (activation_path.empty() ? "synthetic" : "provided")
              << "\",\"resources\":{";
    print_resource("raw_dynamic_baseline", dynamic_resource);
    std::cout << ',';
    print_resource("q6_fields_runtime_dispatch", q6_fields_runtime_resource);
    std::cout << ',';
    print_resource("q6_fields_dynamic_compressed", q6_fields_dynamic_resource);
    std::cout << ',';
    print_resource("q6_fields_forced_fastwidth12", q6_fields_width12_resource);
    std::cout << "},\"samples_ms\":[";
    int sample = 0;
    for (int round = 0; round < 5; ++round) {
        for (int order = 0; order < 4; ++order) {
            const int mode = (round + order) % 4;
            check(cudaEventRecord(start, stream), "start record");
            for (int repeat = 0; repeat < 20; ++repeat) { check(cudaGraphLaunch(graphs[mode], stream), "graph launch"); }
            check(cudaEventRecord(stop, stream), "stop record");
            check(cudaEventSynchronize(stop), "stop sync");
            float elapsed = 0.0f;
            check(cudaEventElapsedTime(&elapsed, start, stop), "event elapsed");
            if (sample++) { std::cout << ','; }
            std::cout << "{\"mode\":\"" << names[mode] << "\",\"ms\":" << elapsed / 20 << '}';
        }
    }
    check_f32_guard(output, output_elements);
    check_bf16_guard(bfout, output_elements);
    std::cout << "]}\n";
    for (auto graph : graphs) { check(cudaGraphExecDestroy(graph), "graph exec destroy"); }
    check(cudaEventDestroy(start), "event destroy");
    check(cudaEventDestroy(stop), "event destroy");
    check(cudaFree(bfout), "free");
    check(cudaFree(output), "free");
    check(cudaFree(device_ds), "free");
    check(cudaFree(device_qs), "free");
    check(cudaFree(device_d_bits), "free");
    for (auto* weight : device_weights) { check(cudaFree(weight), "free"); }
    check(cudaStreamDestroy(stream), "stream destroy");
}

static void scan(const std::vector<uint8_t>& raw, const Payloads& payloads, int rows, int k, const char* label) {
    if (restore_payload(payloads.raw_planar.data(), rows, k, false) != raw) {
        throw std::runtime_error("raw planar CPU full payload mismatch");
    }
    if (restore_payload(payloads.transposed.data(), rows, k, true) != raw) {
        throw std::runtime_error("transposed CPU full payload mismatch");
    }
    if (restore_compressed_planar(payloads, rows, k) != raw) {
        throw std::runtime_error("compressed d CPU full payload mismatch");
    }
    const uint64_t blocks = uint64_t(rows) * uint64_t(k / 256);
    for (uint32_t block = 0; block < blocks; ++block) {
        if (read_bits_u32_host(payloads.d_bits.data(), block, payloads.d_width) !=
            read_bits_host(payloads.d_bits.data(), uint64_t(block) * payloads.d_width, payloads.d_width)) {
            throw std::runtime_error("u32 d bitstream extraction mismatch");
        }
    }
    const uint64_t compressed_bytes = payloads.compressed_planar.size() + payloads.d_bits.size() + 32;
    const auto header = format_header(payloads, rows, k);
    std::cout << "{\"matrix\":\"" << label << "\",\"N\":" << rows << ",\"K\":" << k
              << ",\"blocks_per_row\":" << k / 256 << ",\"raw_bytes\":" << raw.size()
              << ",\"raw_planar_bytes_including_header\":" << payloads.raw_planar.size() + 32
              << ",\"transposed_bytes_including_header\":" << payloads.transposed.size() + 32
              << ",\"compressed_planar_dwindow_bytes_including_headers\":" << compressed_bytes
              << ",\"compressed_planar_dwindow_net_savings_bytes\":" << (raw.size() > compressed_bytes ? raw.size() - compressed_bytes : 0)
              << ",\"compressed_planar_dwindow_larger_bytes\":" << (compressed_bytes > raw.size() ? compressed_bytes - raw.size() : 0)
              << ",\"d_blocks\":" << blocks << ",\"d_negative_blocks\":" << payloads.d_negative
              << ",\"d_exp_min\":" << unsigned(payloads.d_exp_base) << ",\"d_exp_max\":" << unsigned(payloads.d_exp_max)
              << ",\"d_exp_bits\":" << unsigned(payloads.d_exp_bits) << ",\"d_sign_variable\":" << unsigned(payloads.d_sign_variable)
              << ",\"d_code_width_bits\":" << unsigned(payloads.d_width) << ",\"d_exp_histogram\":[";
    for (size_t i = 0; i < payloads.d_exponents.size(); ++i) { if (i) { std::cout << ','; } std::cout << payloads.d_exponents[i]; }
    std::cout << "]"
              << ",\"d_compressed_bytes\":" << payloads.d_bits.size()
              << ",\"format_header_u32\":[";
    for (size_t i = 0; i < header.size(); ++i) { if (i) { std::cout << ','; } std::cout << header[i]; }
    std::cout << "]"
              << ",\"planar_field_offsets\":{\"ql\":0,\"qh\":512,\"scales\":768,\"tile_bytes\":832}"
              << ",\"d_plane_offset\":" << payloads.compressed_planar.size()
              << ",\"d_window_format_cpu_full_roundtrip_exact\":true"
              << ",\"format_header_corruption_fixtures_rejected\":true"
              << ",\"raw_planar_cpu_full_roundtrip_exact\":true,\"transposed_cpu_full_roundtrip_exact\":true,\"compressed_d_cpu_full_roundtrip_exact\":true}\n";
}

static void check_decoder_mapping() {
    std::vector<uint8_t> sample(2 * 840);
    for (size_t i = 0; i < sample.size(); ++i) { sample[i] = uint8_t(i * 73 + i / 7 * 29); }
    const Payloads payloads = transform_payload(sample, 2, 1024);
    for (int row = 0; row < 2; ++row) {
        const size_t tile_offset = size_t(row) * 840;
        for (int slot = 0; slot < 4; ++slot) {
            const uint8_t* block = sample.data() + tile_offset + size_t(slot) * 210;
            for (int u = 0; u < 8; ++u) {
                const int h = u / 4;
                const int q = u % 4;
                const int group = h * 2 + (q & 1);
                for (int word = 0; word < 8; ++word) {
                    const uint32_t low_source = load_u32(block + 64 * h + 32 * (q & 1) + 4 * word);
                    const uint32_t high_source = load_u32(block + 128 + 32 * h + 4 * word);
                    const uint32_t low_planar = load_u32(payloads.raw_planar.data() + tile_offset +
                                                          ql_planar_offset(slot, h, q & 1, word));
                    const uint32_t high_planar = load_u32(payloads.raw_planar.data() + tile_offset +
                                                           qh_planar_offset(slot, h, word));
                    const uint32_t low_transposed = load_u32(payloads.transposed.data() + tile_offset +
                                                             ql_transposed_offset(slot, h, q & 1, word));
                    const uint32_t high_transposed = load_u32(payloads.transposed.data() + tile_offset +
                                                              qh_transposed_offset(slot, h, word));
                    const uint32_t source_code = ((low_source >> (4 * (q >> 1))) & 0x0F0F0F0Fu) |
                                                 (((high_source >> (2 * q)) << 4) & 0x30303030u);
                    const uint32_t planar_code = ((low_planar >> (4 * (q >> 1))) & 0x0F0F0F0Fu) |
                                                 (((high_planar >> (2 * q)) << 4) & 0x30303030u);
                    const uint32_t transposed_code = ((low_transposed >> (4 * (q >> 1))) & 0x0F0F0F0Fu) |
                                                     (((high_transposed >> (2 * q)) << 4) & 0x30303030u);
                    if (source_code != planar_code || source_code != transposed_code) {
                        throw std::runtime_error("Q6_K decoder lane mapping mismatch");
                    }
                }
                const int scale_index = h * 8 + 2 * q;
                const size_t raw_scale = tile_offset + size_t(slot) * 210 + 192 + size_t(scale_index);
                if (payloads.raw_planar[tile_offset + scale_planar_offset(slot, scale_index)] != sample[raw_scale] ||
                    payloads.transposed[tile_offset + scale_transposed_offset(slot, scale_index)] != sample[raw_scale] ||
                    payloads.raw_planar[tile_offset + scale_planar_offset(slot, scale_index + 1)] != sample[raw_scale + 1] ||
                    payloads.transposed[tile_offset + scale_transposed_offset(slot, scale_index + 1)] != sample[raw_scale + 1]) {
                    throw std::runtime_error("Q6_K scale lane mapping mismatch");
                }
            }
            if (std::memcmp(payloads.raw_planar.data() + tile_offset + d_planar_offset(slot), block + 208, 2) != 0 ||
                std::memcmp(payloads.transposed.data() + tile_offset + d_transposed_offset(slot), block + 208, 2) != 0) {
                throw std::runtime_error("Q6_K block scale mapping mismatch");
            }
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 7 || argc > 9 || (std::string(argv[6]) != "scan" && std::string(argv[6]) != "bench")) {
        std::cerr << "model offset N K label scan|bench [activation-file [T]]\n";
        return 2;
    }
    const int rows = std::stoi(argv[3]);
    const int k = std::stoi(argv[4]);
    const uint64_t bytes = uint64_t(rows) * uint64_t(k / 256) * 210;
    if (k != 5120 || rows % 2 != 0 || bytes > size_t(-1)) { throw std::runtime_error("unsupported Q6_K matrix shape"); }
    check_decoder_mapping();
    std::vector<uint8_t> raw(static_cast<size_t>(bytes));
    std::ifstream file(argv[1], std::ios::binary);
    file.seekg(std::stoull(argv[2]));
    file.read(reinterpret_cast<char*>(raw.data()), std::streamsize(raw.size()));
    if (size_t(file.gcount()) != raw.size()) { throw std::runtime_error("truncated model tensor"); }
    Payloads payloads = transform_payload(raw, rows, k);
    validate_format_header(payloads, rows, k);
    if (std::string(argv[6]) == "scan") {
        scan(raw, payloads, rows, k, argv[5]);
        return 0;
    }
    const std::string activation = argc == 8 || argc == 9 ? argv[7] : std::string();
    if (argc == 9) {
        const int t = std::stoi(argv[8]);
        if (t == 1) { run_bench<1>(raw, payloads, rows, k, activation, argv[5]); }
        else if (t == 2) { run_bench<2>(raw, payloads, rows, k, activation, argv[5]); }
        else if (t == 3) { run_bench<3>(raw, payloads, rows, k, activation, argv[5]); }
        else if (t == 4) { run_bench<4>(raw, payloads, rows, k, activation, argv[5]); }
        else if (t == 5) { run_bench<5>(raw, payloads, rows, k, activation, argv[5]); }
        else if (t == 6) { run_bench<6>(raw, payloads, rows, k, activation, argv[5]); }
        else if (t == 7) { run_bench<7>(raw, payloads, rows, k, activation, argv[5]); }
        else if (t == 8) { run_bench<8>(raw, payloads, rows, k, activation, argv[5]); }
        else { throw std::runtime_error("T must be between 1 and 8"); }
    } else if (!activation.empty()) {
        throw std::runtime_error("activation snapshot requires an explicit T");
    } else {
        run_bench<1>(raw, payloads, rows, k, activation, argv[5]);
        run_bench<2>(raw, payloads, rows, k, activation, argv[5]);
        run_bench<3>(raw, payloads, rows, k, activation, argv[5]);
        run_bench<4>(raw, payloads, rows, k, activation, argv[5]);
        run_bench<5>(raw, payloads, rows, k, activation, argv[5]);
        run_bench<6>(raw, payloads, rows, k, activation, argv[5]);
        run_bench<7>(raw, payloads, rows, k, activation, argv[5]);
        run_bench<8>(raw, payloads, rows, k, activation, argv[5]);
    }
    return 0;
}
