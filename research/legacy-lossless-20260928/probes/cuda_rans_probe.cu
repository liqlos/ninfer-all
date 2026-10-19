#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

void check(cudaError_t x) { if (x != cudaSuccess) throw std::runtime_error(cudaGetErrorString(x)); }

__global__ void raw_codes(const uint8_t* packed, uint64_t symbols, uint32_t cap, uint32_t* out) {
    uint64_t id = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    uint64_t first = id * cap;
    if (first >= symbols) return;
    uint32_t sum = 0;
    uint64_t end = min(symbols, first + cap);
    for (uint64_t i = first; i < end; i += 2) {
        uint32_t q = packed[i / 2];
        sum += (q & 15) + (q >> 4);
    }
    out[id] = sum;
}

template<bool Verify = false>
__global__ void ans_codes(const uint8_t* payload, const uint32_t* offsets, const uint32_t* lut,
                          uint64_t symbols, uint32_t cap, uint32_t* out, uint8_t* decoded = nullptr) {
    uint64_t id = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    uint64_t first = id * cap;
    if (first >= symbols) return;
    const uint8_t* p = payload + offsets[id];
    uint32_t state = uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
    p += 4;
    uint32_t sum = 0;
    uint64_t end = min(symbols, first + cap);
    for (uint64_t i = first; i < end; ++i) {
        uint32_t c = state & 4095;
        uint32_t v = lut[c];
        sum += v & 15;
        if constexpr (Verify) decoded[i] = v & 15;
        state = ((v >> 4) & 4095) * (state >> 12) + c - (v >> 16);
        while (state < (1u << 23)) state = (state << 8) | *p++;
    }
    out[id] = sum;
}

std::vector<uint8_t> read(const char* path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("input open");
    std::vector<uint8_t> b(f.tellg());
    f.seekg(0); f.read(reinterpret_cast<char*>(b.data()), b.size());
    if (!f) throw std::runtime_error("input read");
    return b;
}

template<class T> T word(const uint8_t* p) { T x; std::memcpy(&x, p, sizeof(x)); return x; }

int main(int argc, char** argv) {
    if (argc != 3) { std::cerr << "artifact.nlrans tensor.raw-low\n"; return 2; }
    auto a = read(argv[1]), raw = read(argv[2]);
    const uint8_t magic[8] = {'N','L','R','A','N','S',0,1};
    if (a.size() < 56 || std::memcmp(a.data(), magic, 8)) throw std::runtime_error("artifact version");
    uint64_t symbols = word<uint64_t>(a.data() + 8);
    uint32_t cap = word<uint32_t>(a.data() + 16), streams = word<uint32_t>(a.data() + 20);
    if (cap != 2048 || raw.size() * 2 != symbols || streams != (symbols + cap - 1) / cap)
        throw std::runtime_error("artifact geometry");
    size_t payload_start = 56 + (size_t(streams) + 1) * 4;
    if (a.size() < payload_start) throw std::runtime_error("artifact index");
    uint32_t payload_bytes = word<uint32_t>(a.data() + 56 + streams * 4);
    if (a.size() != payload_start + payload_bytes + symbols / 16) throw std::runtime_error("artifact size");
    std::vector<uint32_t> lut(4096);
    unsigned cumulative = 0;
    for (int x = 0; x < 16; ++x) {
        unsigned freq = word<uint16_t>(a.data() + 24 + x * 2);
        if (!freq || cumulative + freq > 4096) throw std::runtime_error("table");
        for (unsigned j = 0; j < freq; ++j) lut[cumulative + j] = x | (freq << 4) | (cumulative << 16);
        cumulative += freq;
    }
    if (cumulative != 4096) throw std::runtime_error("table sum");
    for (uint32_t s = 0; s < streams; ++s) {
        uint32_t lo = word<uint32_t>(a.data() + 56 + s * 4), hi = word<uint32_t>(a.data() + 60 + s * 4);
        if (lo > hi || hi > payload_bytes || hi - lo < 4) throw std::runtime_error("stream offset");
    }
    cudaDeviceProp props; check(cudaGetDeviceProperties(&props, 0));
    if (props.major != 8 || props.minor != 6) throw std::runtime_error("requires sm_86");
    uint32_t *dlut, *out;
    check(cudaMalloc(&dlut, lut.size() * 4)); check(cudaMemcpy(dlut, lut.data(), lut.size() * 4, cudaMemcpyHostToDevice));
    check(cudaMalloc(&out, streams * 4));
    uint8_t *draw[3], *dcomp[3];
    uint32_t *doff[3];
    for (int r = 0; r < 3; ++r) {
        check(cudaMalloc(&draw[r], raw.size())); check(cudaMemcpy(draw[r], raw.data(), raw.size(), cudaMemcpyHostToDevice));
        check(cudaMalloc(&dcomp[r], payload_bytes)); check(cudaMemcpy(dcomp[r], a.data() + payload_start, payload_bytes, cudaMemcpyHostToDevice));
        check(cudaMalloc(&doff[r], (streams + 1) * 4)); check(cudaMemcpy(doff[r], a.data() + 56, (streams + 1) * 4, cudaMemcpyHostToDevice));
    }
    int grid = (streams + 127) / 128;
    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    auto launch = [&](bool compressed, int r) {
        if (compressed) ans_codes<<<grid, 128, 0, stream>>>(dcomp[r], doff[r], dlut, symbols, cap, out);
        else raw_codes<<<grid, 128, 0, stream>>>(draw[r], symbols, cap, out);
        check(cudaGetLastError());
    };
    std::vector<uint32_t> reference(streams), got(streams);
    launch(false, 0); check(cudaStreamSynchronize(stream)); check(cudaMemcpy(reference.data(), out, streams * 4, cudaMemcpyDeviceToHost));
    launch(true, 0); check(cudaStreamSynchronize(stream)); check(cudaMemcpy(got.data(), out, streams * 4, cudaMemcpyDeviceToHost));
    if (got != reference) throw std::runtime_error("GPU checksum mismatch");
    uint8_t* debug;
    check(cudaMalloc(&debug, symbols));
    ans_codes<true><<<grid, 128, 0, stream>>>(dcomp[0], doff[0], dlut, symbols, cap, out, debug);
    check(cudaGetLastError());
    check(cudaStreamSynchronize(stream));
    std::vector<uint8_t> exact(symbols);
    check(cudaMemcpy(exact.data(), debug, symbols, cudaMemcpyDeviceToHost));
    for (uint64_t i = 0; i < symbols; ++i)
        if (exact[i] != ((raw[i / 2] >> (4 * (i & 1))) & 15)) throw std::runtime_error("GPU bit mismatch");
    check(cudaFree(debug));
    cudaGraphExec_t graphs[2];
    for (int mode = 0; mode < 2; ++mode) {
        cudaGraph_t graph;
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        for (int r = 0; r < 3; ++r) launch(mode, r);
        check(cudaStreamEndCapture(stream, &graph));
        check(cudaGraphInstantiate(&graphs[mode], graph, 0)); check(cudaGraphDestroy(graph));
    }
    for (int warm = 0; warm < 4; ++warm) for (auto graph : graphs) check(cudaGraphLaunch(graph, stream));
    check(cudaDeviceSynchronize());
    cudaEvent_t start, stop; check(cudaEventCreate(&start)); check(cudaEventCreate(&stop));
    std::cout << "{\"device\":\"" << props.name << "\",\"scope\":\"code_plane_decode_to_register_checksum_only\",\"raw_code_bytes\":" << raw.size()
              << ",\"compressed_code_bytes\":" << payload_bytes << ",\"index_bytes\":" << (streams + 1) * 4
              << ",\"lut_bytes\":16384,\"scale_bytes_untouched\":" << symbols / 16
              << ",\"rotations\":3,\"gpu_checksum_equal\":true,\"gpu_all_codes_bit_exact\":true,\"debug_buffer_only_during_correctness_bytes\":" << symbols
              << ",\"graphs\":true,\"samples_ms\":[";
    for (int round = 0; round < 5; ++round) for (int order = 0; order < 2; ++order) {
        int mode = (round + order) % 2;
        check(cudaEventRecord(start, stream));
        for (int rep = 0; rep < 10; ++rep) check(cudaGraphLaunch(graphs[mode], stream));
        check(cudaEventRecord(stop, stream)); check(cudaEventSynchronize(stop));
        float ms; check(cudaEventElapsedTime(&ms, start, stop));
        if (round || order) std::cout << ',';
        std::cout << "{\"mode\":\"" << (mode ? "ans" : "raw") << "\",\"ms\":" << ms / 30 << '}';
    }
    std::cout << "]}\n";
    for (auto graph : graphs) check(cudaGraphExecDestroy(graph));
    check(cudaEventDestroy(start)); check(cudaEventDestroy(stop));
    for (int r = 0; r < 3; ++r) { check(cudaFree(draw[r])); check(cudaFree(dcomp[r])); check(cudaFree(doff[r])); }
    check(cudaFree(dlut)); check(cudaFree(out));
    check(cudaStreamDestroy(stream));
}
