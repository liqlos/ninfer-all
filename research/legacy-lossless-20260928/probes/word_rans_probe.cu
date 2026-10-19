#define NINFER_CODEC_LIBRARY_ONLY
#include "rans_probe.cpp"
#include "ops/linear/gguf/ggml_bridge_vec.cuh"
#include <cmath>
#include <limits>

namespace ninfer::ops::gguf::detail {
const DeviceFacts& device_facts() {
    static DeviceFacts f = [] {
        cudaDeviceProp p;
        check(cudaGetDeviceProperties(&p, 0), "device");
        return DeviceFacts{p.major * 100 + p.minor * 10, p.multiProcessorCount, p.sharedMemPerBlockOptin};
    }();
    return f;
}
}

using namespace ninfer::ops::gguf::detail;
using namespace ninfer::ops::gguf::detail::vec;

constexpr uint32_t wr_lower = 1u << 16;
constexpr unsigned wr_precision = 10;
constexpr unsigned wr_total = 1u << wr_precision;
constexpr int wr_threads = 128;
constexpr int wr_rows = 32;
constexpr int wr_states = 128;
constexpr int wr_k = 5120;
constexpr int wr_blocks_per_row = wr_k / 256;
constexpr int wr_row_bytes = wr_blocks_per_row * 144;
constexpr int wr_tile_bytes = wr_rows * wr_row_bytes;
constexpr int wr_stripe_blocks_per_row = 4;
constexpr int wr_stripe_row_bytes = wr_stripe_blocks_per_row * 144;
constexpr int wr_stripe_bytes = wr_rows * wr_stripe_row_bytes;
constexpr int wr_tile_blocks = wr_rows * wr_blocks_per_row;
constexpr int wr_meta_bytes = wr_tile_blocks * 16;
constexpr int wr_state_offset = wr_meta_bytes + 4;
constexpr int wr_offsets_offset = wr_state_offset + wr_states * 4;
constexpr int wr_stream_offset = wr_offsets_offset + (wr_states + 1) * 4;

struct WordTable {
    std::array<uint16_t, 256> freq{}, cum{};
    std::array<uint8_t, wr_total> symbol{};
    std::array<uint32_t, wr_total> decode{};
    std::array<uint32_t, 256> encode{};
    explicit WordTable(const std::vector<uint8_t>& input) {
        std::array<uint64_t, 256> counts{};
        uint64_t count_total = 0;
        for (uint8_t x : input) { ++counts[x]; ++count_total; }
        if (!count_total) throw std::runtime_error("empty table input");
        unsigned nonzero = 0;
        for (unsigned x = 0; x < 256; ++x) if (counts[x]) { freq[x] = 1; ++nonzero; }
        unsigned remaining = wr_total - nonzero;
        std::array<uint64_t, 256> remainder{};
        unsigned assigned = 0;
        for (unsigned x = 0; x < 256; ++x) if (counts[x]) {
            uint64_t scaled = counts[x] * remaining;
            unsigned add = static_cast<unsigned>(scaled / count_total);
            freq[x] = static_cast<uint16_t>(freq[x] + add);
            remainder[x] = scaled % count_total;
            assigned += add;
        }
        while (assigned < remaining) {
            unsigned best = 0;
            for (unsigned x = 1; x < 256; ++x) if (remainder[x] > remainder[best]) best = x;
            ++freq[best];
            remainder[best] = 0;
            ++assigned;
        }
        unsigned at = 0;
        for (unsigned x = 0; x < 256; ++x) {
            cum[x] = static_cast<uint16_t>(at);
            if (!freq[x]) continue;
            encode[x] = uint32_t(freq[x]) | (uint32_t(cum[x]) << 16);
            for (unsigned j = 0; j < freq[x]; ++j) {
                symbol[at] = static_cast<uint8_t>(x);
                decode[at] = x | (uint32_t(freq[x]) << 8) | (uint32_t(cum[x]) << 19);
                ++at;
            }
        }
        if (at != wr_total) throw std::runtime_error("word table normalization");
    }
};

struct WordStream {
    uint32_t state = wr_lower;
    std::vector<uint16_t> words;
};

WordStream word_encode(const uint8_t* input, size_t n, const WordTable& t) {
    uint32_t state = wr_lower;
    std::vector<uint16_t> emitted;
    emitted.reserve(n / 4 + 8);
    for (size_t i = n; i; --i) {
        const unsigned x = input[i - 1];
        const uint32_t f = t.freq[x], c = t.cum[x];
        if (!f) throw std::runtime_error("word symbol absent from table");
        const uint64_t threshold = (uint64_t(wr_lower >> wr_precision) << 16) * f;
        while (uint64_t(state) >= threshold) {
            emitted.push_back(static_cast<uint16_t>(state));
            state >>= 16;
        }
        state = ((state / f) << wr_precision) + (state % f) + c;
    }
    std::reverse(emitted.begin(), emitted.end());
    return {state, std::move(emitted)};
}

std::vector<uint8_t> word_decode(uint32_t state, const uint16_t* words, size_t word_count, size_t n, const WordTable& t) {
    if (state < wr_lower) throw std::runtime_error("word state range");
    std::vector<uint8_t> result(n);
    size_t cursor = 0;
    for (uint8_t& x : result) {
        const uint32_t slot = state & (wr_total - 1);
        const uint32_t entry = t.decode[slot];
        x = static_cast<uint8_t>(entry);
        const uint32_t f = (entry >> 8) & 2047;
        const uint32_t c = entry >> 19;
        state = f * (state >> wr_precision) + slot - c;
        while (state < wr_lower) {
            if (cursor == word_count) throw std::runtime_error("word stream truncated");
            state = (state << 16) | words[cursor++];
        }
    }
    if (state != wr_lower || cursor != word_count) throw std::runtime_error("word stream boundary");
    return result;
}

struct WordPacked {
    std::vector<uint64_t> index;
    std::vector<uint8_t> bytes;
    std::array<uint32_t, wr_total> lut{};
    std::array<uint16_t, 256> frequencies{};
    uint64_t raw_tiles = 0;
    uint64_t stream_words = 0;
    uint64_t coded_tiles = 0;
};

std::vector<uint8_t> byte_alphabet(const std::vector<uint8_t>& raw) {
    std::vector<uint8_t> symbols;
    symbols.reserve(raw.size() / 144 * 128);
    for (size_t b = 0; b < raw.size() / 144; ++b)
        symbols.insert(symbols.end(), raw.begin() + b * 144 + 16, raw.begin() + b * 144 + 144);
    return symbols;
}

WordPacked word_pack(const std::vector<uint8_t>& raw, int n, int k) {
    if (k != wr_k || n % wr_rows) throw std::runtime_error("wordwise layout requires K5120 and 32-row tiles");
    WordTable t(byte_alphabet(raw));
    WordPacked packed;
    packed.frequencies = t.freq;
    packed.lut = t.decode;
    for (int row0 = 0; row0 < n; row0 += wr_rows) {
        std::vector<uint8_t> tile(wr_meta_bytes);
        for (int row = 0; row < wr_rows; ++row) for (int block = 0; block < wr_blocks_per_row; ++block) {
            const size_t src = (size_t(row0 + row) * wr_blocks_per_row + block) * 144;
            const size_t dst = (size_t(row) * wr_blocks_per_row + block) * 16;
            std::memcpy(tile.data() + dst, raw.data() + src, 16);
        }
        std::array<std::vector<uint8_t>, wr_states> source{};
        for (int it = 0; it < wr_k / 1024; ++it)
            for (int row = 0; row < wr_rows; ++row)
                for (int localblock = 0; localblock < 4; ++localblock) {
                    const size_t block = size_t(row0 + row) * wr_blocks_per_row + it * 4 + localblock;
                    for (int byte = 0; byte < 128; ++byte)
                        source[byte].push_back(raw[block * 144 + 16 + byte]);
                }
        std::array<WordStream, wr_states> streams;
        uint32_t common_words = std::numeric_limits<uint32_t>::max();
        for (int s = 0; s < wr_states; ++s) {
            streams[s] = word_encode(source[s].data(), source[s].size(), t);
            auto decoded = word_decode(streams[s].state, streams[s].words.data(), streams[s].words.size(), source[s].size(), t);
            if (decoded != source[s]) throw std::runtime_error("wordwise CPU stream mismatch");
            common_words = std::min<uint32_t>(common_words, static_cast<uint32_t>(streams[s].words.size()));
            packed.stream_words += streams[s].words.size();
        }
        tile.resize(wr_stream_offset);
        std::memcpy(tile.data() + wr_meta_bytes, &common_words, 4);
        for (int s = 0; s < wr_states; ++s)
            std::memcpy(tile.data() + wr_state_offset + s * 4, &streams[s].state, 4);
        for (uint32_t w = 0; w < common_words; ++w)
            for (int s = 0; s < wr_states; ++s) {
                const uint16_t word = streams[s].words[w];
                tile.push_back(static_cast<uint8_t>(word));
                tile.push_back(static_cast<uint8_t>(word >> 8));
            }
        const size_t tails_start = tile.size();
        std::array<uint32_t, wr_states + 1> offsets{};
        for (int s = 0; s < wr_states; ++s) {
            offsets[s] = static_cast<uint32_t>(tile.size() - tails_start);
            for (size_t w = common_words; w < streams[s].words.size(); ++w) {
                const uint16_t word = streams[s].words[w];
                tile.push_back(static_cast<uint8_t>(word));
                tile.push_back(static_cast<uint8_t>(word >> 8));
            }
        }
        offsets[wr_states] = static_cast<uint32_t>(tile.size() - tails_start);
        std::memcpy(tile.data() + wr_offsets_offset, offsets.data(), sizeof(offsets));
        tile.resize((tile.size() + 15) & ~size_t(15));
        if (tile.size() >= wr_tile_bytes) {
            packed.index.push_back(uint64_t(packed.bytes.size()) | (1ull << 63));
            packed.bytes.insert(packed.bytes.end(), raw.begin() + size_t(row0) * wr_row_bytes, raw.begin() + size_t(row0 + wr_rows) * wr_row_bytes);
            ++packed.raw_tiles;
        } else {
            packed.index.push_back(packed.bytes.size());
            packed.bytes.insert(packed.bytes.end(), tile.begin(), tile.end());
            ++packed.coded_tiles;
        }
    }
    packed.index.push_back(packed.bytes.size());
    return packed;
}

void word_verify_cpu(const WordPacked& packed, const std::vector<uint8_t>& raw, int n, int k) {
    WordTable t(byte_alphabet(raw));
    if (packed.frequencies != t.freq || packed.index.size() != size_t(n / wr_rows + 1)) throw std::runtime_error("word artifact table/index mismatch");
    std::vector<uint8_t> output(raw.size());
    for (int tile_id = 0; tile_id < n / wr_rows; ++tile_id) {
        const uint64_t at = packed.index[tile_id] & ~(1ull << 63);
        const uint64_t end = packed.index[tile_id + 1] & ~(1ull << 63);
        if (at > end || end > packed.bytes.size()) throw std::runtime_error("word tile index range");
        uint8_t* dst = output.data() + size_t(tile_id) * wr_tile_bytes;
        const uint8_t* src = packed.bytes.data() + at;
        if (packed.index[tile_id] >> 63) {
            if (end - at != wr_tile_bytes) throw std::runtime_error("word raw fallback size");
            std::memcpy(dst, src, wr_tile_bytes);
            continue;
        }
        if (end - at < wr_stream_offset) throw std::runtime_error("word tile header truncation");
        const uint32_t common = *reinterpret_cast<const uint32_t*>(src + wr_meta_bytes);
        const uint64_t common_end = uint64_t(wr_stream_offset) + uint64_t(common) * wr_states * 2;
        if (common_end > end - at) throw std::runtime_error("word common stream range");
        for (size_t block = 0; block < wr_tile_blocks; ++block) std::memcpy(dst + block * 144, src + block * 16, 16);
        std::array<uint32_t, wr_states + 1> offsets{};
        std::memcpy(offsets.data(), src + wr_offsets_offset, sizeof(offsets));
        const uint8_t* tail = src + common_end;
        const size_t tail_bytes = end - at - common_end;
        for (int s = 0; s < wr_states; ++s) {
            if (offsets[s] > offsets[s + 1] || offsets[s + 1] > tail_bytes || (offsets[s] & 1) || (offsets[s + 1] & 1)) throw std::runtime_error("word tail offsets");
            uint32_t state;
            std::memcpy(&state, src + wr_state_offset + s * 4, 4);
            std::vector<uint16_t> words;
            words.reserve(common + (offsets[s + 1] - offsets[s]) / 2);
            for (uint32_t w = 0; w < common; ++w) {
                const size_t pos = wr_stream_offset + (size_t(w) * wr_states + s) * 2;
                words.push_back(uint16_t(src[pos]) | (uint16_t(src[pos + 1]) << 8));
            }
            for (uint32_t pos = offsets[s]; pos < offsets[s + 1]; pos += 2)
                words.push_back(uint16_t(tail[pos]) | (uint16_t(tail[pos + 1]) << 8));
            auto bytes = word_decode(state, words.data(), words.size(), wr_k / 1024 * wr_rows * 4, t);
            size_t index = 0;
            for (int it = 0; it < wr_k / 1024; ++it)
                for (int row = 0; row < wr_rows; ++row)
                    for (int localblock = 0; localblock < 4; ++localblock) {
                        dst[size_t(row) * wr_row_bytes + size_t(it * 4 + localblock) * 144 + 16 + s] = bytes[index++];
                    }
        }
    }
    if (output != raw) {
        auto mismatch = std::mismatch(output.begin(), output.end(), raw.begin());
        throw std::runtime_error("wordwise CPU full GGUF reconstruction mismatch at " + std::to_string(mismatch.first - output.begin()) + " expected=" + std::to_string(*mismatch.second) + " decoded=" + std::to_string(*mismatch.first));
    }
}

void word_save(const char* path, const WordPacked& p, uint32_t n, uint32_t k) {
    std::ofstream f(path, std::ios::binary);
    const char magic[8] = {'N','L','W','R','A','N',1,0};
    uint64_t count = p.index.size(), bytes = p.bytes.size();
    f.write(magic, 8); f.write(reinterpret_cast<const char*>(&n), 4); f.write(reinterpret_cast<const char*>(&k), 4);
    f.write(reinterpret_cast<const char*>(&count), 8); f.write(reinterpret_cast<const char*>(&bytes), 8);
    f.write(reinterpret_cast<const char*>(p.frequencies.data()), sizeof(p.frequencies));
    f.write(reinterpret_cast<const char*>(p.index.data()), count * 8);
    f.write(reinterpret_cast<const char*>(p.bytes.data()), bytes);
    if (!f) throw std::runtime_error("word artifact write");
}

WordPacked word_load(const char* path, uint32_t n, uint32_t k) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("word artifact open");
    const auto file_size = f.tellg(); f.seekg(0);
    char magic[8], expected[8] = {'N','L','W','R','A','N',1,0};
    uint32_t sn, sk; uint64_t count, bytes;
    f.read(magic, 8); f.read(reinterpret_cast<char*>(&sn), 4); f.read(reinterpret_cast<char*>(&sk), 4);
    f.read(reinterpret_cast<char*>(&count), 8); f.read(reinterpret_cast<char*>(&bytes), 8);
    if (!f || std::memcmp(magic, expected, 8) || sn != n || sk != k || n % wr_rows || k != wr_k || count != uint64_t(n / wr_rows + 1) || 32 + 512 + count * 8 + bytes != uint64_t(file_size)) throw std::runtime_error("word artifact version/shape/size");
    WordPacked p;
    f.read(reinterpret_cast<char*>(p.frequencies.data()), sizeof(p.frequencies));
    p.index.resize(count); p.bytes.resize(bytes);
    f.read(reinterpret_cast<char*>(p.index.data()), count * 8); f.read(reinterpret_cast<char*>(p.bytes.data()), bytes);
    if (!f || (p.index.front() & ~(1ull << 63)) || p.index.back() != bytes) throw std::runtime_error("word artifact framing");
    uint32_t sum = 0; unsigned at = 0;
    for (unsigned x = 0; x < 256; ++x) {
        const uint32_t freq = p.frequencies[x];
        if (freq > wr_total || at + freq > wr_total) throw std::runtime_error("word artifact frequency");
        if (!freq) continue;
        for (unsigned j = 0; j < freq; ++j) p.lut[at + j] = x | (freq << 8) | (at << 19);
        at += freq; sum += freq;
    }
    if (sum != wr_total || at != wr_total) throw std::runtime_error("word artifact frequency sum");
    for (size_t i = 0; i + 1 < count; ++i) {
        uint64_t a = p.index[i] & ~(1ull << 63), b = p.index[i + 1] & ~(1ull << 63);
        if (a > b || b > bytes || ((p.index[i] >> 63) && b - a != wr_tile_bytes)) throw std::runtime_error("word artifact tile range");
        p.raw_tiles += p.index[i] >> 63;
        p.coded_tiles += !(p.index[i] >> 63);
    }
    return p;
}

__device__ void word_unpack_stripe(const uint8_t* payload, const uint64_t* index, const uint32_t* lut, int tile, int iteration, uint8_t* shared, bool force_raw, uint32_t& state, uint32_t& word_index, uint32_t common_words, const uint32_t* offsets) {
    const uint64_t encoded = index[tile];
    const bool raw = (encoded >> 63) || force_raw;
    const uint8_t* src = force_raw ? payload + size_t(tile) * wr_tile_bytes : payload + (encoded & ~(1ull << 63));
    if (raw) {
        for (int j = threadIdx.x; j < wr_stripe_bytes; j += wr_threads) {
            const int row = j / wr_stripe_row_bytes;
            const int offset = j % wr_stripe_row_bytes;
            shared[j] = src[size_t(row) * wr_row_bytes + size_t(iteration * wr_stripe_blocks_per_row * 144) + offset];
        }
    } else {
        for (int j = threadIdx.x; j < wr_rows * wr_stripe_blocks_per_row * 4; j += wr_threads) {
            const int block_in_stripe = j / 4;
            const int row = block_in_stripe / wr_stripe_blocks_per_row;
            const int localblock = block_in_stripe % wr_stripe_blocks_per_row;
            const size_t block = size_t(row) * wr_blocks_per_row + iteration * wr_stripe_blocks_per_row + localblock;
            reinterpret_cast<uint32_t*>(shared + size_t(block_in_stripe) * 144)[j % 4] = reinterpret_cast<const uint32_t*>(src + block * 16)[j % 4];
        }
        const uint8_t* common = src + wr_stream_offset;
        const uint8_t* tails = common + size_t(common_words) * wr_states * 2;
        const uint32_t tail_start = offsets[threadIdx.x];
        for (int row = 0; row < wr_rows; ++row)
            for (int localblock = 0; localblock < wr_stripe_blocks_per_row; ++localblock) {
                    const uint32_t slot = state & (wr_total - 1);
                    const uint32_t entry = lut[slot];
                    const uint8_t value = static_cast<uint8_t>(entry);
                    const uint32_t freq = (entry >> 8) & 2047;
                    const uint32_t cum = entry >> 19;
                    state = freq * (state >> wr_precision) + slot - cum;
                    while (state < wr_lower) {
                        uint16_t word;
                        if (word_index < common_words) {
                            const uint8_t* ptr = common + (size_t(word_index) * wr_states + threadIdx.x) * 2;
                            word = uint16_t(ptr[0]) | (uint16_t(ptr[1]) << 8);
                        } else {
                            const uint8_t* ptr = tails + tail_start + size_t(word_index - common_words) * 2;
                            word = uint16_t(ptr[0]) | (uint16_t(ptr[1]) << 8);
                        }
                        state = (state << 16) | word;
                        ++word_index;
                    }
                    const size_t offset = size_t(row * wr_stripe_blocks_per_row + localblock) * 144 + 16 + threadIdx.x;
                    shared[offset] = value;
                }
    }
}

template<int T, bool Verify = false, bool DecodeOnly = false>
__global__ __launch_bounds__(128) void word_fused(VecArgs p, const uint64_t* index, const uint32_t* lut, uint8_t* debug = nullptr, bool force_raw = false) {
    extern __shared__ __align__(16) uint8_t memory[];
    uint8_t* shared = memory;
    uint32_t* shared_lut = reinterpret_cast<uint32_t*>(memory + wr_stripe_bytes);
    for (int i = threadIdx.x; i < wr_total; i += wr_threads) shared_lut[i] = lut[i];
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int tile = blockIdx.x; tile < p.rows / wr_rows; tile += gridDim.x) {
        const uint8_t* src = force_raw ? p.weight + size_t(tile) * wr_tile_bytes : p.weight + (index[tile] & ~(1ull << 63));
        const bool raw = force_raw || (index[tile] >> 63);
        const uint32_t common_words = raw ? 0 : *reinterpret_cast<const uint32_t*>(src + wr_meta_bytes);
        const uint32_t* states = raw ? nullptr : reinterpret_cast<const uint32_t*>(src + wr_state_offset);
        const uint32_t* offsets = raw ? nullptr : reinterpret_cast<const uint32_t*>(src + wr_offsets_offset);
        uint32_t state = raw ? wr_lower : states[threadIdx.x];
        uint32_t word_index = 0;
        float acc[8][T];
        if constexpr (!Verify && !DecodeOnly) for (int r = 0; r < 8; ++r) for (int j = 0; j < T; ++j) acc[r][j] = 0;
#pragma unroll
        for (int it = 0; it < wr_k / 1024; ++it) {
            word_unpack_stripe(p.weight, index, shared_lut, tile, it, shared, force_raw, state, word_index, common_words, offsets);
            __syncthreads();
            if constexpr (Verify) {
                for (int j = threadIdx.x; j < wr_stripe_bytes; j += wr_threads) {
                    const int row = j / wr_stripe_row_bytes;
                    const int offset = j % wr_stripe_row_bytes;
                    debug[size_t(tile) * wr_tile_bytes + size_t(row) * wr_row_bytes + size_t(it * wr_stripe_blocks_per_row * 144) + offset] = shared[j];
                }
            } else if constexpr (!DecodeOnly) {
                const uint8_t* rowp[8];
                for (int r = 0; r < 8; ++r) rowp[r] = shared + (warp * 8 + r) * wr_stripe_row_bytes + (lane / 8) * 144;
                int a[T][8]; float d[T], sum[T];
#pragma unroll
                for (int j = 0; j < T; ++j) {
                    float2 ds = __half22float2(p.ds[int64_t(j) * (wr_k / 32) + lane + 32 * it]);
                    d[j] = ds.x; sum[j] = ds.y;
                    const int4* q = reinterpret_cast<const int4*>(p.qs + int64_t(j) * wr_k + 32 * (lane + 32 * it));
                    const int4 v0 = q[0], v1 = q[1];
                    a[j][0] = v0.x; a[j][1] = v0.y; a[j][2] = v0.z; a[j][3] = v0.w;
                    a[j][4] = v1.x; a[j][5] = v1.y; a[j][6] = v1.z; a[j][7] = v1.w;
                }
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    Slice sl;
                    Decoder<GGML_TYPE_Q4_K>::decode(rowp[r], lane % 8, nullptr, sl);
#pragma unroll
                    for (int j = 0; j < T; ++j) acc[r][j] += Decoder<GGML_TYPE_Q4_K>::dot(sl, a[j], d[j], sum[j]);
                }
            }
            __syncthreads();
        }
        if constexpr (Verify) {
            if (!raw) {
                const uint32_t tail_end = offsets[threadIdx.x + 1];
                const uint32_t tail_start = offsets[threadIdx.x];
                if (state != wr_lower || word_index != common_words + (tail_end - tail_start) / 2) asm("trap;");
            }
        } else if constexpr (!DecodeOnly) {
#pragma unroll
            for (int r = 0; r < 8; ++r) for (int j = 0; j < T; ++j) {
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) acc[r][j] += __shfl_xor_sync(0xffffffffu, acc[r][j], o);
                if (lane == 0) vec_store(p.out, tile * wr_rows + warp * 8 + r, j, acc[r][j]);
            }
        }
        __syncthreads();
    }
}

template<int T>
void word_measure(const std::vector<uint8_t>& raw, const WordPacked& packed, int n, const std::string& label) {
    uint8_t *weights[3], *compressed[3], *debug;
    uint64_t* index;
    uint32_t* lut;
    int8_t* x;
    half2* ds;
    float* y;
    auto alloc = [](auto** p, size_t bytes) { check(cudaMalloc(p, bytes), "alloc"); };
    for (int r = 0; r < 3; ++r) {
        alloc(&weights[r], raw.size()); check(cudaMemcpy(weights[r], raw.data(), raw.size(), cudaMemcpyHostToDevice), "raw upload");
        alloc(&compressed[r], packed.bytes.size()); check(cudaMemcpy(compressed[r], packed.bytes.data(), packed.bytes.size(), cudaMemcpyHostToDevice), "compressed upload");
    }
    alloc(&index, packed.index.size() * 8); check(cudaMemcpy(index, packed.index.data(), packed.index.size() * 8, cudaMemcpyHostToDevice), "index");
    alloc(&lut, packed.lut.size() * 4); check(cudaMemcpy(lut, packed.lut.data(), packed.lut.size() * 4, cudaMemcpyHostToDevice), "lut");
    std::vector<int8_t> hostx(wr_k * T);
    std::vector<half2> hostds(wr_k / 32 * T);
    for (size_t i = 0; i < hostx.size(); ++i) hostx[i] = (i * 37 + i / 7 * 11) % 255 - 127;
    for (size_t i = 0; i < hostds.size(); ++i) {
        int sum = 0; for (int j = 0; j < 32; ++j) sum += hostx[i * 32 + j];
        hostds[i] = __floats2half2_rn(0.0625f, sum * 0.0625f);
    }
    alloc(&x, hostx.size()); check(cudaMemcpy(x, hostx.data(), hostx.size(), cudaMemcpyHostToDevice), "x");
    alloc(&ds, hostds.size() * 4); check(cudaMemcpy(ds, hostds.data(), hostds.size() * 4, cudaMemcpyHostToDevice), "ds");
    alloc(&y, size_t(n) * T * 4);
    VecArgs p{weights[0], wr_row_bytes, n, wr_k, x, ds, {nullptr, y, n, ninfer::ops::gguf::Epilogue::Store, nullptr, 0}};
    cudaStream_t stream; check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
    const int shared_bytes = wr_stripe_bytes + wr_total * 4;
    check(cudaFuncSetAttribute(word_fused<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes), "set dynamic shared memory");
    check(cudaFuncSetAttribute(word_fused<T,true>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes), "set verify dynamic shared memory");
    check(cudaFuncSetAttribute(word_fused<T,false,true>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes), "set decode dynamic shared memory");
    int resident;
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, word_fused<T>, wr_threads, shared_bytes), "occupancy");
    if (resident <= 0) throw std::runtime_error("wordwise kernel has zero occupancy");
    int grid = std::min(n / wr_rows, resident * device_facts().sm_count);
    auto launch = [&](int mode, int r) {
        p.weight = (mode == 2 || mode == 3) ? compressed[r] : weights[r];
        if (mode == 0) vec::launch<GGML_TYPE_Q4_K,T,false,wr_k>(p, stream);
        else if (mode == 1) word_fused<T><<<grid, wr_threads, shared_bytes, stream>>>(p, index, lut, nullptr, true);
        else if (mode == 2) word_fused<T,false,true><<<grid, wr_threads, shared_bytes, stream>>>(p, index, lut);
        else word_fused<T><<<grid, wr_threads, shared_bytes, stream>>>(p, index, lut);
        check(cudaGetLastError(), "launch");
    };
    if constexpr (T == 1) {
        alloc(&debug, raw.size()); p.weight = compressed[0];
        word_fused<T,true><<<grid, wr_threads, shared_bytes, stream>>>(p, index, lut, debug);
        check(cudaGetLastError(), "verify launch");
        check(cudaStreamSynchronize(stream), "verify sync");
        std::vector<uint8_t> back(raw.size()); check(cudaMemcpy(back.data(), debug, raw.size(), cudaMemcpyDeviceToHost), "verify copy");
        if (back != raw) {
            std::cerr << "wordwise GPU mismatch offsets";
            size_t reported = 0;
            for (size_t i = 0; i < raw.size() && reported < 10; ++i) if (back[i] != raw[i]) {
                std::cerr << " [" << i << ":" << unsigned(raw[i]) << "->" << unsigned(back[i]) << "]";
                ++reported;
            }
            std::cerr << '\n';
            throw std::runtime_error("wordwise GPU full GGUF reconstruction mismatch");
        }
        check(cudaFree(debug), "debug free");
    }
    std::vector<float> base(size_t(n) * T), test(base.size());
    launch(0, 0); check(cudaStreamSynchronize(stream), "baseline sync"); check(cudaMemcpy(base.data(), y, base.size() * 4, cudaMemcpyDeviceToHost), "baseline copy");
    launch(3, 0); check(cudaStreamSynchronize(stream), "compressed sync"); check(cudaMemcpy(test.data(), y, test.size() * 4, cudaMemcpyDeviceToHost), "compressed copy");
    if (std::memcmp(base.data(), test.data(), base.size() * 4)) throw std::runtime_error("wordwise operator FP32 bit mismatch");
    double oracle_max_abs = 0, oracle_max_normalized = 0;
    for (int row = 0; row < std::min(n, 16); ++row) {
        double sum = 0, l1 = 0;
        for (int block = 0; block < wr_k / 256; ++block) {
            const uint8_t* b = raw.data() + size_t(row) * wr_row_bytes + block * 144;
            const double d0 = __half2float(*reinterpret_cast<const half*>(b));
            const double dmin = __half2float(*reinterpret_cast<const half*>(b + 2));
            const uint8_t* scales = b + 4;
            for (int group = 0; group < 8; ++group) {
                const int scale = group < 4 ? scales[group] & 63 : (scales[group + 4] & 15) | ((scales[group - 4] >> 6) << 4);
                const int minimum = group < 4 ? scales[group + 4] & 63 : (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4);
                for (int j = 0; j < 32; ++j) {
                    const int code = (b[16 + 32 * (group / 2) + j] >> (4 * (group & 1))) & 15;
                    const double weight = d0 * scale * code - dmin * minimum;
                    const double term = weight * double(hostx[block * 256 + group * 32 + j]) * 0.0625;
                    sum += term;
                    l1 += std::abs(term);
                }
            }
        }
        const double error = std::abs(double(base[row]) - sum);
        oracle_max_abs = std::max(oracle_max_abs, error);
        oracle_max_normalized = std::max(oracle_max_normalized, error / (1 + std::abs(sum)));
        if (error > 1e-5 * (1 + l1)) throw std::runtime_error("wordwise independent FP64 oracle mismatch");
    }
    __nv_bfloat16* bfout; alloc(&bfout, size_t(n) * T * 2);
    std::vector<uint16_t> bfa(size_t(n) * T), bfb(bfa.size());
    p.out.bf16 = bfout; p.out.f32 = nullptr;
    launch(0,0); check(cudaStreamSynchronize(stream), "bf16 sync"); check(cudaMemcpy(bfa.data(), bfout, bfa.size() * 2, cudaMemcpyDeviceToHost), "bf16 copy");
    launch(3,0); check(cudaStreamSynchronize(stream), "bf16 sync"); check(cudaMemcpy(bfb.data(), bfout, bfb.size() * 2, cudaMemcpyDeviceToHost), "bf16 copy");
    if (bfa != bfb) throw std::runtime_error("wordwise operator BF16 bit mismatch");
    check(cudaFree(bfout), "bf16 free"); p.out.bf16 = nullptr; p.out.f32 = y;
    cudaGraphExec_t graphs[4];
    for (int mode = 0; mode < 4; ++mode) {
        cudaGraph_t graph; check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "capture");
        for (int r = 0; r < 3; ++r) launch(mode, r);
        check(cudaStreamEndCapture(stream, &graph), "capture end"); check(cudaGraphInstantiate(&graphs[mode], graph, 0), "instantiate"); check(cudaGraphDestroy(graph), "destroy");
    }
    for (int j = 0; j < 4; ++j) for (auto g : graphs) check(cudaGraphLaunch(g, stream), "warmup");
    check(cudaStreamSynchronize(stream), "warmup sync");
    cudaEvent_t start, stop; check(cudaEventCreate(&start), "event"); check(cudaEventCreate(&stop), "event");
    const uint64_t resident_bytes = packed.bytes.size() + packed.index.size() * 8 + packed.lut.size() * 4;
    std::cout << "{\"matrix\":\"" << label << "\",\"T\":" << T << ",\"N\":" << n << ",\"K\":" << wr_k << ",\"baseline_bytes\":" << raw.size()
              << ",\"resident_bytes_with_index_lut\":" << resident_bytes << ",\"raw_tiles\":" << packed.raw_tiles << ",\"coded_tiles\":" << packed.coded_tiles
              << ",\"shared_bytes_per_cta\":" << shared_bytes << ",\"lookup_table_bytes\":" << wr_total * 4
              << ",\"states_per_tile\":" << wr_states << ",\"fp32_output_bit_exact\":true,\"gpu_all_payload_bytes_exact\":" << (T == 1 ? "true" : "null")
              << ",\"bf16_output_bit_exact\":true,\"fp64_oracle_rows\":" << std::min(n, 16) << ",\"oracle_max_abs_error\":" << oracle_max_abs
              << ",\"oracle_max_normalized_error\":" << oracle_max_normalized << ",\"rotations\":3,\"graphs\":true,\"samples_ms\":[";
    bool first_sample = true;
    for (int round = 0; round < 5; ++round) for (int step = 0; step < 4; ++step) {
        int mode = (round + step) % 4; check(cudaEventRecord(start, stream), "start");
        for (int rep = 0; rep < 10; ++rep) check(cudaGraphLaunch(graphs[mode], stream), "replay");
        check(cudaEventRecord(stop, stream), "stop"); check(cudaEventSynchronize(stop), "wait"); float ms; check(cudaEventElapsedTime(&ms, start, stop), "time");
        if (!first_sample) std::cout << ',';
        first_sample = false;
        const char* mode_name[] = {"raw", "raw_staging", "decode_only", "word_rans_fused"};
        std::cout << "{\"mode\":\"" << mode_name[mode] << "\",\"ms\":" << ms / 30 << '}';
    }
    std::cout << "]}\n";
    for (auto g : graphs) check(cudaGraphExecDestroy(g), "graph free");
    check(cudaEventDestroy(start), "event free"); check(cudaEventDestroy(stop), "event free");
    for (int r = 0; r < 3; ++r) { check(cudaFree(weights[r]), "free"); check(cudaFree(compressed[r]), "free"); }
    check(cudaFree(index), "free"); check(cudaFree(lut), "free"); check(cudaFree(x), "free"); check(cudaFree(ds), "free"); check(cudaFree(y), "free"); check(cudaStreamDestroy(stream), "free");
}

int main(int argc, char** argv) {
    if (argc != 6 && argc != 8) { std::cerr << "model offset N K label [pack|bench artifact.nlwran]\n"; return 2; }
    int n = std::stoi(argv[3]), k = std::stoi(argv[4]);
    if (argc == 8 && std::string(argv[6]) != "pack" && std::string(argv[6]) != "bench") throw std::runtime_error("mode must be pack or bench");
    if (k != wr_k || n % wr_rows) throw std::runtime_error("prototype supports K5120 and complete 32-row tiles");
    std::vector<uint8_t> raw(size_t(n) * wr_row_bytes);
    std::ifstream file(argv[1], std::ios::binary); file.seekg(std::stoull(argv[2])); file.read(reinterpret_cast<char*>(raw.data()), raw.size());
    if (size_t(file.gcount()) != raw.size()) throw std::runtime_error("raw input");
    WordPacked packed;
    if (argc == 8 && std::string(argv[6]) == "bench") packed = word_load(argv[7], n, k);
    else {
        packed = word_pack(raw, n, k);
        if (argc == 8) {
            const uint64_t stream_words = packed.stream_words;
            word_save(argv[7], packed, n, k);
            packed = word_load(argv[7], n, k);
            packed.stream_words = stream_words;
        }
    }
    word_verify_cpu(packed, raw, n, k);
    if (argc == 8 && std::string(argv[6]) == "pack") {
        std::cout << "{\"layout\":\"gguf_q4_k_32row_wordrans_v1\",\"cpu_all_bytes_exact\":true,\"N\":" << n << ",\"K\":" << k << ",\"raw_bytes\":" << raw.size()
                  << ",\"artifact_file_bytes\":" << 32 + 512 + packed.index.size() * 8 + packed.bytes.size()
                  << ",\"resident_bytes\":" << packed.bytes.size() + packed.index.size() * 8 + packed.lut.size() * 4
                  << ",\"coded_tiles\":" << packed.coded_tiles << ",\"raw_tiles\":" << packed.raw_tiles << ",\"states\":" << wr_states
                  << ",\"renorm_word_bits\":16,\"precision\":" << wr_precision << ",\"table_bytes\":" << wr_total * 4
                  << ",\"total_stream_words\":" << packed.stream_words << "}\n";
        return 0;
    }
    word_measure<1>(raw, packed, n, argv[5]); word_measure<2>(raw, packed, n, argv[5]); word_measure<3>(raw, packed, n, argv[5]);
    word_measure<4>(raw, packed, n, argv[5]); word_measure<8>(raw, packed, n, argv[5]);
}
