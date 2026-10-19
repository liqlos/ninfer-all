#define NINFER_CODEC_LIBRARY_ONLY
#include "rans_probe.cpp"
#include "ops/linear/gguf/ggml_bridge_vec.cuh"
#include <cmath>

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

struct Packed {
    std::vector<uint64_t> index;
    std::vector<uint8_t> bytes;
    std::vector<uint32_t> lut;
    std::array<uint16_t, 16> frequencies;
    uint64_t raw_tiles = 0;
};

Packed pack(const std::vector<uint8_t>& raw, int n, int k) {
    const int row_bytes = k / 256 * 144;
    std::vector<uint8_t> all;
    all.reserve(size_t(n) * k);
    for (size_t b = 0; b < raw.size() / 144; ++b) for (int j = 0; j < 128; ++j) {
        all.push_back(raw[b * 144 + 16 + j] & 15);
        all.push_back(raw[b * 144 + 16 + j] >> 4);
    }
    Table t(all);
    Packed p;
    p.frequencies = t.freq;
    p.lut.resize(total);
    for (unsigned c = 0; c < total; ++c) {
        unsigned s = t.symbol[c];
        p.lut[c] = s | (uint32_t(t.freq[s]) << 4) | (uint32_t(t.cum[s]) << 16);
    }
    for (int row = 0; row < n; row += 8) {
        if (row + 8 > n) throw std::runtime_error("prototype requires whole 8-row tiles");
        const size_t blocks = size_t(8) * k / 256;
        const size_t lowbytes = blocks * 128;
        std::vector<uint8_t> tile(blocks * 16 + 4 + 32 * 4 + 33 * 4);
        for (size_t b = 0; b < blocks; ++b)
            std::memcpy(tile.data() + b * 16, raw.data() + size_t(row) * row_bytes + b * 144, 16);
        std::array<uint32_t, 33> offsets{};
        std::array<uint32_t, 32> states{};
        std::array<std::vector<uint8_t>, 32> streams;
        uint32_t common_words = ~0u;
        for (int lane = 0; lane < 32; ++lane) {
            std::vector<uint8_t> symbols;
            for (size_t byte = lane; byte < lowbytes; byte += 32) {
                uint8_t q = raw[size_t(row) * row_bytes + (byte / 128) * 144 + 16 + byte % 128];
                symbols.push_back(q & 15); symbols.push_back(q >> 4);
            }
            auto encoded = encode(symbols.data(), symbols.size(), t);
            auto back = decode(encoded, symbols.size(), t);
            if (back != symbols) throw std::runtime_error("CPU codes");
            std::memcpy(&states[lane], encoded.data(), 4);
            streams[lane].assign(encoded.begin() + 4, encoded.end());
            streams[lane].resize((streams[lane].size() + 3) / 4 * 4);
            common_words = std::min<uint32_t>(common_words, streams[lane].size() / 4);
        }
        std::memcpy(tile.data() + blocks * 16, &common_words, 4);
        std::memcpy(tile.data() + blocks * 16 + 4, states.data(), 32 * 4);
        for (uint32_t word = 0; word < common_words; ++word) for (int lane = 0; lane < 32; ++lane)
            tile.insert(tile.end(), streams[lane].begin() + word * 4, streams[lane].begin() + word * 4 + 4);
        size_t tail_start = tile.size();
        for (int lane = 0; lane < 32; ++lane) {
            offsets[lane] = tile.size() - tail_start;
            tile.insert(tile.end(), streams[lane].begin() + common_words * 4, streams[lane].end());
        }
        offsets[32] = tile.size() - tail_start;
        std::memcpy(tile.data() + blocks * 16 + 4 + 32 * 4, offsets.data(), 33 * 4);
        tile.resize((tile.size() + 15) / 16 * 16);
        if (tile.size() >= size_t(8) * row_bytes) {
            p.index.push_back(uint64_t(p.bytes.size()) | (1ull << 63));
            p.bytes.insert(p.bytes.end(), raw.begin() + size_t(row) * row_bytes, raw.begin() + size_t(row + 8) * row_bytes);
            ++p.raw_tiles;
        } else {
            p.index.push_back(p.bytes.size());
            p.bytes.insert(p.bytes.end(), tile.begin(), tile.end());
        }
    }
    p.index.push_back(p.bytes.size());
    return p;
}

void verify_cpu(const Packed& p, const std::vector<uint8_t>& raw, int n, int k) {
    std::vector<uint8_t> symbols;
    for (int x = 0; x < 16; ++x) for (unsigned j = 0; j < p.frequencies[x]; ++j) symbols.push_back(x);
    Table table(symbols);
    if (table.freq != p.frequencies) throw std::runtime_error("stored frequency compatibility");
    std::vector<uint8_t> back(raw.size());
    const size_t blocks = 8 * k / 256, row_bytes = k / 256 * 144;
    for (int tile = 0; tile < n / 8; ++tile) {
        uint64_t at = p.index[tile], end = p.index[tile+1] & ~(1ull<<63);
        at &= ~(1ull<<63);
        if (at > end || end > p.bytes.size()) throw std::runtime_error("artifact index range");
        uint8_t* dst = back.data() + size_t(tile) * row_bytes * 8;
        const uint8_t* src = p.bytes.data() + at;
        if (p.index[tile] >> 63) {
            if (end-at != row_bytes*8) throw std::runtime_error("raw tile bytes");
            std::memcpy(dst,src,row_bytes*8);
            continue;
        }
        if (end-at < blocks*16+264) throw std::runtime_error("compressed tile header");
        for (size_t b = 0; b < blocks; ++b) std::memcpy(dst+b*144,src+b*16,16);
        uint32_t common;
        std::memcpy(&common,src+blocks*16,4);
        size_t common_start = blocks*16+264;
        size_t tail_start = common_start+size_t(common)*32*4;
        if (tail_start > end-at) throw std::runtime_error("common stream boundary");
        for (int lane = 0; lane < 32; ++lane) {
            uint32_t state,lo,hi;
            std::memcpy(&state,src+blocks*16+4+lane*4,4);
            std::memcpy(&lo,src+blocks*16+132+lane*4,4);
            std::memcpy(&hi,src+blocks*16+136+lane*4,4);
            if (lo > hi || tail_start+hi > end-at || lo%4 || hi%4) throw std::runtime_error("tail range");
            std::vector<uint8_t> stream(4);
            std::memcpy(stream.data(),&state,4);
            for (uint32_t word = 0; word < common; ++word) {
                const uint8_t* a = src+common_start+(size_t(word)*32+lane)*4;
                stream.insert(stream.end(),a,a+4);
            }
            stream.insert(stream.end(),src+tail_start+lo,src+tail_start+hi);
            bool valid = false;
            std::vector<uint8_t> decoded;
            for (int trim = 0; trim < 4 && !valid; ++trim) {
                auto copy = stream;
                if (size_t(trim) > copy.size()-4) continue;
                copy.resize(copy.size()-trim);
                try { decoded = decode(copy,blocks*8,table); valid = true; } catch (const std::runtime_error&) {}
            }
            if (!valid) throw std::runtime_error("CPU stream termination");
            for (size_t j = 0; j < decoded.size()/2; ++j) {
                size_t byte = j*32+lane;
                dst[(byte/128)*144+16+byte%128] = decoded[2*j] | (decoded[2*j+1]<<4);
            }
        }
    }
    if (back != raw) throw std::runtime_error("CPU full GGUF bit mismatch");
}

void save(const char* path, const Packed& p, uint32_t n, uint32_t k) {
    std::ofstream f(path,std::ios::binary);
    const char magic[8] = {'N','L','G','Q','4',0,1,0};
    uint64_t count=p.index.size(), bytes=p.bytes.size();
    f.write(magic,8); f.write(reinterpret_cast<char*>(&n),4); f.write(reinterpret_cast<char*>(&k),4);
    f.write(reinterpret_cast<char*>(&count),8); f.write(reinterpret_cast<char*>(&bytes),8);
    f.write(reinterpret_cast<const char*>(p.frequencies.data()),32);
    f.write(reinterpret_cast<const char*>(p.index.data()),count*8);
    f.write(reinterpret_cast<const char*>(p.bytes.data()),bytes);
    if (!f) throw std::runtime_error("artifact write");
}

Packed load(const char* path, uint32_t n, uint32_t k) {
    std::ifstream f(path,std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("artifact open");
    auto filebytes=f.tellg(); f.seekg(0);
    char magic[8],expected[8]={'N','L','G','Q','4',0,1,0};
    uint32_t storedn,storedk; uint64_t count,bytes;
    f.read(magic,8); f.read(reinterpret_cast<char*>(&storedn),4); f.read(reinterpret_cast<char*>(&storedk),4);
    f.read(reinterpret_cast<char*>(&count),8); f.read(reinterpret_cast<char*>(&bytes),8);
    if (!f || std::memcmp(magic,expected,8) || storedn!=n || storedk!=k || count!=n/8+1 || bytes>uint64_t(filebytes) || 64+count*8+bytes!=uint64_t(filebytes))
        throw std::runtime_error("experimental layout/version/shape compatibility");
    Packed p;
    f.read(reinterpret_cast<char*>(p.frequencies.data()),32);
    p.index.resize(count); p.bytes.resize(bytes);
    f.read(reinterpret_cast<char*>(p.index.data()),count*8); f.read(reinterpret_cast<char*>(p.bytes.data()),bytes);
    if (!f || (p.index.front() & ~(1ull<<63)) || p.index.back()!=bytes) throw std::runtime_error("artifact framing");
    p.lut.resize(4096); unsigned c=0;
    for (int x=0;x<16;++x) {
        unsigned freq=p.frequencies[x];
        if (!freq || c+freq>4096) throw std::runtime_error("artifact table");
        for (unsigned j=0;j<freq;++j) p.lut[c+j]=x|(freq<<4)|(c<<16);
        c+=freq;
    }
    if (c!=4096) throw std::runtime_error("artifact frequency sum");
    for (size_t i=0;i+1<count;++i) p.raw_tiles+=p.index[i]>>63;
    return p;
}

template<int K, bool Verify>
__device__ void unpack_tile(const uint8_t* payload, const uint64_t* index, const uint32_t* lut,
                            int tile, uint8_t* shared) {
    constexpr int blocks = 8 * K / 256;
    constexpr int bytes = blocks * 144;
    uint64_t at = index[tile];
    const bool raw = at >> 63;
    const uint8_t* src = payload + (at & ~(1ull << 63));
    if (raw) {
        for (int j = threadIdx.x; j < bytes; j += 128) shared[j] = src[j];
    } else {
        for (int j = threadIdx.x; j < blocks * 4; j += 128) {
            reinterpret_cast<uint32_t*>(shared + (j / 4) * 144)[j % 4] = reinterpret_cast<const uint32_t*>(src)[j];
        }
        if (threadIdx.x < 32) {
            uint32_t common_words = *reinterpret_cast<const uint32_t*>(src + blocks * 16);
            const uint32_t* states = reinterpret_cast<const uint32_t*>(src + blocks * 16 + 4);
            const uint32_t* offsets = states + 32;
            const uint32_t* words = offsets + 33;
            const uint32_t* tail = words + common_words * 32 + offsets[threadIdx.x] / 4;
            uint32_t state = states[threadIdx.x];
            uint32_t read_bytes = 0, cached_word = 0;
            for (int j = threadIdx.x; j < blocks * 128; j += 32) {
                unsigned byte = 0;
                for (int q = 0; q < 2; ++q) {
                    const uint32_t c = state & 4095;
                    const uint32_t entry = lut[c];
                    byte |= (entry & 15) << (q * 4);
                    state = ((entry >> 4) & 4095) * (state >> 12) + c - (entry >> 16);
                    while (state < lower) {
                        uint32_t word = read_bytes / 4;
                        if ((read_bytes & 3) == 0) cached_word = word < common_words ? words[word * 32 + threadIdx.x] : tail[word - common_words];
                        state = (state << 8) | (cached_word & 255);
                        cached_word >>= 8;
                        ++read_bytes;
                    }
                }
                shared[(j / 128) * 144 + 16 + j % 128] = byte;
            }
        }
    }
    __syncthreads();
}

template<int K, int T, bool Verify = false>
__global__ __launch_bounds__(128) void fused(VecArgs p, const uint64_t* index, const uint32_t* lut, uint8_t* debug = nullptr) {
    constexpr int row_bytes = K / 256 * 144;
    __shared__ __align__(16) uint8_t shared[8 * row_bytes];
    __shared__ uint32_t shared_lut[4096];
    for (int i = threadIdx.x; i < 4096; i += 128) shared_lut[i] = lut[i];
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int tile = blockIdx.x; tile < p.rows / 8; tile += gridDim.x) {
        unpack_tile<K, Verify>(p.weight, index, shared_lut, tile, shared);
        if constexpr (Verify) {
            for (int j = threadIdx.x; j < 8 * row_bytes; j += 128)
                debug[size_t(tile) * 8 * row_bytes + j] = shared[j];
        } else {
            const uint8_t* rowp[2];
            for (int r = 0; r < 2; ++r) rowp[r] = shared + (warp * 2 + r) * row_bytes + (lane / 8) * 144;
            float acc[2][T];
            for (int r = 0; r < 2; ++r) for (int j = 0; j < T; ++j) acc[r][j] = 0;
#pragma unroll
            for (int it = 0; it < K / 1024; ++it) {
                int a[T][8]; float d[T], sum[T];
#pragma unroll
                for (int j = 0; j < T; ++j) {
                    float2 ds = __half22float2(p.ds[int64_t(j) * (K / 32) + lane + 32 * it]);
                    d[j] = ds.x; sum[j] = ds.y;
                    const int4* q = reinterpret_cast<const int4*>(p.qs + int64_t(j) * K + 32 * (lane + 32 * it));
                    const int4 v0 = q[0], v1 = q[1];
                    a[j][0] = v0.x; a[j][1] = v0.y; a[j][2] = v0.z; a[j][3] = v0.w;
                    a[j][4] = v1.x; a[j][5] = v1.y; a[j][6] = v1.z; a[j][7] = v1.w;
                }
#pragma unroll
                for (int r = 0; r < 2; ++r) {
                    Slice sl;
                    Decoder<GGML_TYPE_Q4_K>::decode(rowp[r] + 4 * it * 144, lane % 8, nullptr, sl);
#pragma unroll
                    for (int j = 0; j < T; ++j) acc[r][j] += Decoder<GGML_TYPE_Q4_K>::dot(sl, a[j], d[j], sum[j]);
                }
            }
#pragma unroll
            for (int r = 0; r < 2; ++r) for (int j = 0; j < T; ++j) {
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) acc[r][j] += __shfl_xor_sync(0xffffffffu, acc[r][j], o);
                if (lane == 0) vec_store(p.out, tile * 8 + warp * 2 + r, j, acc[r][j]);
            }
        }
        __syncthreads();
    }
}

template<int T>
void measure(const std::vector<uint8_t>& raw, const Packed& packed, int n, const std::string& label) {
    constexpr int k = 5120;
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
    std::vector<int8_t> hostx(k * T);
    std::vector<half2> hostds(k / 32 * T);
    for (size_t i = 0; i < hostx.size(); ++i) hostx[i] = (i * 37 + i / 7 * 11) % 255 - 127;
    for (size_t i = 0; i < hostds.size(); ++i) {
        int sum = 0; for (int j = 0; j < 32; ++j) sum += hostx[i * 32 + j];
        hostds[i] = __floats2half2_rn(0.0625f, sum * 0.0625f);
    }
    alloc(&x, hostx.size()); check(cudaMemcpy(x, hostx.data(), hostx.size(), cudaMemcpyHostToDevice), "x");
    alloc(&ds, hostds.size() * 4); check(cudaMemcpy(ds, hostds.data(), hostds.size() * 4, cudaMemcpyHostToDevice), "ds");
    alloc(&y, n * T * 4);
    VecArgs p{weights[0], k / 256 * 144, n, k, x, ds, {nullptr, y, n, ninfer::ops::gguf::Epilogue::Store, nullptr, 0}};
    cudaStream_t stream; check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
    int resident;
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, fused<k,T>, 128, 0), "occupancy");
    int grid = std::min(n / 8, resident * device_facts().sm_count);
    auto launch = [&](int mode, int r) {
        p.weight = mode ? compressed[r] : weights[r];
        if (mode) fused<k,T><<<grid,128,0,stream>>>(p, index, lut);
        else vec::launch<GGML_TYPE_Q4_K,T,false,k>(p, stream);
        check(cudaGetLastError(), "launch");
    };
    if constexpr (T == 1) {
        alloc(&debug, raw.size());
        p.weight = compressed[0];
        fused<k,T,true><<<grid,128,0,stream>>>(p, index, lut, debug);
        check(cudaStreamSynchronize(stream), "verify sync");
        std::vector<uint8_t> back(raw.size()); check(cudaMemcpy(back.data(), debug, raw.size(), cudaMemcpyDeviceToHost), "verify copy");
        if (back != raw) throw std::runtime_error("GPU full GGUF payload mismatch");
        check(cudaFree(debug), "debug free");
    }
    std::vector<float> base(size_t(n) * T), test(base.size());
    launch(0,0); check(cudaStreamSynchronize(stream), "baseline sync"); check(cudaMemcpy(base.data(), y, base.size()*4, cudaMemcpyDeviceToHost), "baseline copy");
    launch(1,0); check(cudaStreamSynchronize(stream), "compressed sync"); check(cudaMemcpy(test.data(), y, test.size()*4, cudaMemcpyDeviceToHost), "compressed copy");
    if (std::memcmp(base.data(), test.data(), base.size() * 4)) throw std::runtime_error("operator FP32 bit mismatch");
    __nv_bfloat16* bfout;
    alloc(&bfout,size_t(n)*T*2);
    std::vector<uint16_t> bfa(size_t(n)*T),bfb(bfa.size());
    p.out.bf16=bfout; p.out.f32=nullptr;
    launch(0,0); check(cudaStreamSynchronize(stream),"bf16 sync"); check(cudaMemcpy(bfa.data(),bfout,bfa.size()*2,cudaMemcpyDeviceToHost),"bf16 copy");
    launch(1,0); check(cudaStreamSynchronize(stream),"bf16 sync"); check(cudaMemcpy(bfb.data(),bfout,bfb.size()*2,cudaMemcpyDeviceToHost),"bf16 copy");
    if (bfa!=bfb) throw std::runtime_error("operator BF16 bit mismatch");
    check(cudaFree(bfout),"bf16 free"); p.out.bf16=nullptr; p.out.f32=y;
    double oracle_max_abs=0,oracle_max_normalized=0;
    for (int row=0;row<std::min(n,16);++row) for (int col=0;col<T;++col) {
        double sum=0,l1=0;
        for (int block=0;block<k/256;++block) {
            const uint8_t* b=raw.data()+size_t(row)*(k/256*144)+block*144;
            double d=__half2float(*reinterpret_cast<const half*>(b));
            double m0=__half2float(*reinterpret_cast<const half*>(b+2));
            const uint8_t* s=b+4;
            for (int group=0;group<8;++group) {
                int sc=group<4 ? s[group]&63 : (s[group+4]&15)|((s[group-4]>>6)<<4);
                int mn=group<4 ? s[group+4]&63 : (s[group+4]>>4)|((s[group]>>6)<<4);
                for (int j=0;j<32;++j) {
                    int code=(b[16+32*(group/2)+j]>>(4*(group&1)))&15;
                    double w=d*sc*code-m0*mn;
                    double term=w*double(hostx[col*k+block*256+group*32+j])*0.0625;
                    sum+=term; l1+=std::abs(term);
                }
            }
        }
        double err=std::abs(double(base[col*n+row])-sum);
        oracle_max_abs=std::max(oracle_max_abs,err);
        oracle_max_normalized=std::max(oracle_max_normalized,err/(1+std::abs(sum)));
        if (err>1e-5*(1+l1)) throw std::runtime_error("independent FP64 oracle");
    }
    cudaGraphExec_t graphs[2];
    for (int mode = 0; mode < 2; ++mode) {
        cudaGraph_t graph; check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal), "capture");
        for (int r = 0; r < 3; ++r) launch(mode,r);
        check(cudaStreamEndCapture(stream,&graph), "capture end"); check(cudaGraphInstantiate(&graphs[mode],graph,0), "instantiate");
        check(cudaGraphDestroy(graph), "destroy");
    }
    for (int j = 0; j < 4; ++j) for (auto g : graphs) check(cudaGraphLaunch(g,stream), "warmup");
    check(cudaStreamSynchronize(stream), "warmup sync");
    cudaEvent_t start,stop; check(cudaEventCreate(&start), "event"); check(cudaEventCreate(&stop), "event");
    std::cout << "{\"matrix\":\"" << label << "\",\"T\":" << T << ",\"N\":" << n << ",\"K\":" << k << ",\"baseline_bytes\":" << raw.size()
              << ",\"compressed_bytes_with_index_and_lut\":" << packed.bytes.size()+packed.index.size()*8+packed.lut.size()*4
              << ",\"raw_tiles\":" << packed.raw_tiles << ",\"shared_bytes_per_cta\":" << 8 * k / 256 * 144
              << ",\"lut_shared_bytes_per_cta\":16384,\"stream_layout\":\"warp_interleaved_words_compact_tail\""
              << ",\"resident_ctas_per_sm\":" << resident << ",\"fp32_output_bit_exact\":true,\"gpu_all_payload_bytes_exact\":" << (T==1?"true":"null")
              << ",\"bf16_output_bit_exact\":true,\"fp64_oracle_rows\":16,\"oracle_max_abs_error\":" << oracle_max_abs << ",\"oracle_max_normalized_error\":" << oracle_max_normalized
              << ",\"rotations\":3,\"graphs\":true,\"samples_ms\":[";
    for (int round = 0; round < 5; ++round) for (int order = 0; order < 2; ++order) {
        int mode = (round+order)%2; check(cudaEventRecord(start,stream), "start");
        for (int rep = 0; rep < 10; ++rep) check(cudaGraphLaunch(graphs[mode],stream), "replay");
        check(cudaEventRecord(stop,stream), "stop"); check(cudaEventSynchronize(stop), "wait"); float ms;
        check(cudaEventElapsedTime(&ms,start,stop), "time");
        if (round || order) std::cout << ',';
        std::cout << "{\"mode\":\"" << (mode?"ans":"baseline") << "\",\"ms\":" << ms/30 << '}';
    }
    std::cout << "]}\n";
    for (auto g : graphs) check(cudaGraphExecDestroy(g), "graph free");
    check(cudaEventDestroy(start), "event free"); check(cudaEventDestroy(stop), "event free");
    for (int r = 0; r < 3; ++r) { check(cudaFree(weights[r]),"free"); check(cudaFree(compressed[r]),"free"); }
    check(cudaFree(index),"free"); check(cudaFree(lut),"free"); check(cudaFree(x),"free"); check(cudaFree(ds),"free"); check(cudaFree(y),"free"); check(cudaStreamDestroy(stream),"free");
}

int main(int argc, char** argv) {
    if (argc != 6 && argc != 8) { std::cerr << "model offset N K label [pack|bench artifact.nlgq4]\n"; return 2; }
    int n = std::stoi(argv[3]), k = std::stoi(argv[4]);
    if (argc==8 && std::string(argv[6])!="pack" && std::string(argv[6])!="bench") throw std::runtime_error("mode must be pack or bench");
    if (k != 5120 || n % 8) throw std::runtime_error("prototype supports K5120, whole 8-row tiles only");
    std::vector<uint8_t> raw(size_t(n) * (k / 256) * 144);
    std::ifstream file(argv[1],std::ios::binary); file.seekg(std::stoull(argv[2])); file.read(reinterpret_cast<char*>(raw.data()),raw.size());
    if (size_t(file.gcount()) != raw.size()) throw std::runtime_error("raw input");
    Packed p;
    if (argc==8 && std::string(argv[6])=="bench") p=load(argv[7],n,k);
    else {
        p=pack(raw,n,k);
        if (argc==8) { save(argv[7],p,n,k); p=load(argv[7],n,k); }
    }
    verify_cpu(p,raw,n,k);
    if (argc==8 && std::string(argv[6])=="pack") {
        std::cout << "{\"layout\":\"gguf_q4_k_cta8_warp_rans_v1\",\"cpu_all_bytes_exact\":true,\"N\":" << n << ",\"K\":" << k << ",\"raw_bytes\":" << raw.size()
                  << ",\"artifact_file_bytes\":" << 64+p.index.size()*8+p.bytes.size() << ",\"resident_bytes\":" << p.index.size()*8+p.bytes.size()+p.lut.size()*4 << "}\n";
        return 0;
    }
    measure<1>(raw,p,n,argv[5]); measure<2>(raw,p,n,argv[5]); measure<3>(raw,p,n,argv[5]); measure<4>(raw,p,n,argv[5]); measure<8>(raw,p,n,argv[5]);
}
