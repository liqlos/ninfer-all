#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

constexpr uint32_t lower = 1u << 23;
constexpr unsigned precision = 12;
constexpr unsigned total = 1u << precision;

struct Table {
    std::array<uint16_t, 16> freq{}, cum{};
    std::array<uint8_t, total> symbol{};
    explicit Table(const std::vector<uint8_t>& symbols) {
        std::array<uint64_t, 16> counts{};
        for (auto x : symbols) ++counts[x];
        for (int x = 0; x < 16; ++x)
            freq[x] = std::max<uint16_t>(1, uint16_t(counts[x] * total / symbols.size()));
        int sum = std::accumulate(freq.begin(), freq.end(), 0);
        while (sum != int(total)) {
            int x = 0;
            for (int i = 1; i < 16; ++i)
                if ((sum < int(total) ? double(counts[i]) / freq[i] : double(freq[i]) / std::max<uint64_t>(counts[i], 1)) >
                    (sum < int(total) ? double(counts[x]) / freq[x] : double(freq[x]) / std::max<uint64_t>(counts[x], 1))) x = i;
            if (sum < int(total)) { ++freq[x]; ++sum; }
            else if (freq[x] > 1) { --freq[x]; --sum; }
            else throw std::runtime_error("frequency normalization");
        }
        int at = 0;
        for (int x = 0; x < 16; ++x) {
            cum[x] = at;
            for (int j = 0; j < freq[x]; ++j) symbol[at++] = x;
        }
    }
};

std::vector<uint8_t> encode(const uint8_t* a, size_t n, const Table& t) {
    uint32_t state = lower;
    std::vector<uint8_t> bytes;
    for (size_t i = n; i > 0; --i) {
        const auto x = a[i - 1];
        const uint32_t f = t.freq[x];
        const uint32_t threshold = ((lower >> precision) << 8) * f;
        while (state >= threshold) { bytes.push_back(state & 255); state >>= 8; }
        state = ((state / f) << precision) + (state % f) + t.cum[x];
    }
    for (int j = 3; j >= 0; --j) bytes.push_back((state >> (8 * j)) & 255);
    std::reverse(bytes.begin(), bytes.end());
    return bytes;
}

std::vector<uint8_t> decode(const std::vector<uint8_t>& a, size_t n, const Table& t) {
    if (a.size() < 4) throw std::runtime_error("truncated state");
    uint32_t state = uint32_t(a[0]) | (uint32_t(a[1]) << 8) | (uint32_t(a[2]) << 16) | (uint32_t(a[3]) << 24);
    size_t at = 4;
    std::vector<uint8_t> out(n);
    for (auto& x : out) {
        uint32_t c = state & (total - 1);
        x = t.symbol[c];
        state = t.freq[x] * (state >> precision) + c - t.cum[x];
        while (state < lower) {
            if (at == a.size()) throw std::runtime_error("truncated stream");
            state = (state << 8) | a[at++];
        }
    }
    if (state != lower || at != a.size()) throw std::runtime_error("stream boundary");
    return out;
}

#ifndef NINFER_CODEC_LIBRARY_ONLY
int main(int argc, char** argv) {
    if (argc != 6) { std::cerr << "model offset nblocks label output_prefix\n"; return 2; }
    std::vector<uint8_t> alph;
    for (int r = 0; r < 127; ++r) for (int i = 0; i < 16; ++i) alph.push_back(i);
    Table test(alph);
    for (size_t n : {size_t(0), size_t(1), size_t(15), size_t(16), size_t(31), size_t(256), alph.size()}) {
        auto bytes = encode(alph.data(), n, test);
        auto back = decode(bytes, n, test);
        if (!std::equal(back.begin(), back.end(), alph.begin())) throw std::runtime_error("alphabet test");
    }
    size_t blocks = std::stoull(argv[3]);
    std::vector<uint8_t> raw(blocks * 144), symbols(blocks * 256);
    std::ifstream input(argv[1], std::ios::binary);
    input.seekg(std::stoull(argv[2]));
    input.read(reinterpret_cast<char*>(raw.data()), raw.size());
    if (size_t(input.gcount()) != raw.size()) throw std::runtime_error("model read");
    for (size_t b = 0; b < blocks; ++b) for (int j = 0; j < 128; ++j) {
        uint8_t x = raw[b * 144 + 16 + j];
        symbols[b * 256 + 2 * j] = x & 15;
        symbols[b * 256 + 2 * j + 1] = x >> 4;
    }
    Table table(symbols);
    std::ofstream csv(std::string(argv[5]) + ".csv");
    csv << "label,blocks_per_tile,streams,raw_bytes,stored_bytes,ratio,raw_tiles,total_tiles,encode_decode_ms,exact\n";
    for (int tile : {1, 4, 16, 64, 256}) for (int streams : {1, 8, 32}) {
        auto begin = std::chrono::steady_clock::now();
        uint64_t stored = 64 + 32, raw_tiles = 0, tiles = 0;
        for (size_t b = 0; b < blocks; b += tile) {
            size_t count = std::min<size_t>(tile, blocks - b);
            size_t coded = count * 16 + streams * 4;
            for (int lane = 0; lane < streams; ++lane) {
                size_t start = count * 256 * lane / streams;
                size_t end = count * 256 * (lane + 1) / streams;
                const uint8_t* ptr = symbols.data() + b * 256 + start;
                auto bytes = encode(ptr, end - start, table);
                auto back = decode(bytes, end - start, table);
                if (!std::equal(back.begin(), back.end(), ptr)) throw std::runtime_error("roundtrip");
                coded += bytes.size();
            }
            coded = (coded + 15) / 16 * 16;
            size_t original = count * 144;
            if (coded >= original) { coded = original; ++raw_tiles; }
            stored += coded + 8;
            ++tiles;
        }
        auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
        csv << argv[4] << ',' << tile << ',' << streams << ',' << raw.size() << ',' << stored << ',' << double(stored) / raw.size() << ',' << raw_tiles << ',' << tiles << ',' << ms << ",true\n";
        std::cout << argv[4] << " tile=" << tile << " streams=" << streams << " ratio=" << double(stored) / raw.size() << " exact=1 ms=" << ms << '\n';
    }
    const uint32_t cap = 2048;
    const uint32_t count = (symbols.size() + cap - 1) / cap;
    std::vector<uint32_t> offsets;
    std::vector<uint8_t> payload;
    for (uint32_t stream = 0; stream < count; ++stream) {
        size_t at = size_t(stream) * cap;
        size_t n = std::min<size_t>(cap, symbols.size() - at);
        auto bytes = encode(symbols.data() + at, n, table);
        auto back = decode(bytes, n, table);
        if (!std::equal(back.begin(), back.end(), symbols.data() + at)) throw std::runtime_error("serialized stream");
        offsets.push_back(payload.size());
        payload.insert(payload.end(), bytes.begin(), bytes.end());
    }
    offsets.push_back(payload.size());
    std::ofstream artifact(std::string(argv[5]) + ".nlrans", std::ios::binary);
    const char magic[8] = {'N','L','R','A','N','S',0,1};
    uint64_t n = symbols.size();
    artifact.write(magic, 8);
    artifact.write(reinterpret_cast<const char*>(&n), 8);
    artifact.write(reinterpret_cast<const char*>(&cap), 4);
    artifact.write(reinterpret_cast<const char*>(&count), 4);
    artifact.write(reinterpret_cast<const char*>(table.freq.data()), 32);
    artifact.write(reinterpret_cast<const char*>(offsets.data()), offsets.size() * 4);
    artifact.write(reinterpret_cast<const char*>(payload.data()), payload.size());
    for (size_t b = 0; b < blocks; ++b) artifact.write(reinterpret_cast<const char*>(raw.data() + b * 144), 16);
    artifact.close();
    std::ifstream saved(std::string(argv[5]) + ".nlrans", std::ios::binary | std::ios::ate);
    std::vector<uint8_t> file(saved.tellg());
    saved.seekg(0); saved.read(reinterpret_cast<char*>(file.data()), file.size());
    if (!saved || std::memcmp(file.data(), magic, 8) || std::memcmp(file.data() + 24, table.freq.data(), 32))
        throw std::runtime_error("serialized metadata");
    size_t base = 56 + offsets.size() * 4;
    for (uint32_t stream = 0; stream < count; ++stream) {
        uint32_t lo, hi;
        std::memcpy(&lo, file.data() + 56 + stream * 4, 4);
        std::memcpy(&hi, file.data() + 60 + stream * 4, 4);
        std::vector<uint8_t> bytes(file.begin() + base + lo, file.begin() + base + hi);
        size_t start = size_t(stream) * cap;
        auto back = decode(bytes, std::min<size_t>(cap, symbols.size() - start), table);
        if (!std::equal(back.begin(), back.end(), symbols.data() + start)) throw std::runtime_error("file code mismatch");
    }
    for (size_t b = 0; b < blocks; ++b)
        if (std::memcmp(file.data() + base + payload.size() + b * 16, raw.data() + b * 144, 16))
            throw std::runtime_error("file scale/min mismatch");
    std::ofstream packed(std::string(argv[5]) + ".raw-low", std::ios::binary);
    for (size_t b = 0; b < blocks; ++b) packed.write(reinterpret_cast<const char*>(raw.data() + b * 144 + 16), 128);
    std::cout << "artifact " << argv[5] << ".nlrans codebytes=" << payload.size() << " offsets=" << offsets.size() * 4 << " scales=" << blocks * 16 << '\n';
}
#endif
