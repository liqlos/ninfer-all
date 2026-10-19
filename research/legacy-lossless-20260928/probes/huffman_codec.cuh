#define NINFER_CODEC_LIBRARY_ONLY
#include "rans_probe.cpp"
#include "ops/linear/gguf/ggml_bridge_vec.cuh"
#include <queue>
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
    std::array<uint16_t,16> frequencies{};
    uint64_t raw_tiles=0;
};

struct Huff {
    std::array<unsigned,16> len{},code{};
    std::vector<uint32_t> lut;
    explicit Huff(const std::array<uint16_t,16>& lengths) {
        for(int i=0;i<16;++i) len[i]=lengths[i];
        unsigned at=0,prior=0,kraft=0;
        for(unsigned l=1;l<=10;++l) for(int s=0;s<16;++s) if(len[s]==l) {
            at<<=l-prior; prior=l;
            unsigned rev=0;
            for(unsigned j=0;j<l;++j) rev|=((at>>(l-1-j))&1)<<j;
            code[s]=rev; ++at; kraft+=1u<<(10-l);
        }
        if(kraft!=1024) throw std::runtime_error("invalid Huffman lengths");
        for(auto l:len) if(l<1 || l>10) throw std::runtime_error("Huffman length range");
        lut.resize(1024);
        for(unsigned b=0;b<1024;++b) {
            int s0=-1,s1=-1;
            for(int s=0;s<16;++s) if((b&((1u<<len[s])-1))==code[s]) {s0=s;break;}
            if(s0<0) throw std::runtime_error("Huffman prefix");
            for(int s=0;s<16;++s) if(len[s]+len[s0]<=10 && ((b>>len[s0])&((1u<<len[s])-1))==code[s]) {s1=s;break;}
            lut[b]=s0 | (s1<0?0:s1<<4) | ((len[s0]+(s1<0?0:len[s1]))<<8) | ((s1<0?1:2)<<16) | (len[s0]<<20);
        }
    }
};

std::array<uint16_t,16> huff_lengths(const std::vector<uint8_t>& raw) {
    std::array<uint64_t,31> counts{};
    std::array<int,31> parent{}; parent.fill(-1);
    for(size_t b=0;b<raw.size()/144;++b) for(int j=16;j<144;++j) {
        ++counts[raw[b*144+j]&15]; ++counts[raw[b*144+j]>>4];
    }
    using P=std::pair<uint64_t,int>;
    std::priority_queue<P,std::vector<P>,std::greater<P>> q;
    for(int s=0;s<16;++s) q.push({std::max<uint64_t>(1,counts[s]),s});
    for(int node=16;node<31;++node) {
        auto a=q.top();q.pop();auto b=q.top();q.pop();
        parent[a.second]=parent[b.second]=node;
        q.push({a.first+b.first,node});
    }
    std::array<uint16_t,16> lengths{};
    for(int s=0;s<16;++s) for(int p=s;parent[p]>=0;p=parent[p]) ++lengths[s];
    return lengths;
}

uint32_t hpeek(const uint8_t* p,unsigned bit) {
    unsigned a=bit>>3;
    return (unsigned(p[a])|(unsigned(p[a+1])<<8)|(unsigned(p[a+2])<<16))>>(bit&7);
}

Packed pack(const std::vector<uint8_t>& raw,int n,int k) {
    Packed p; p.frequencies=huff_lengths(raw); Huff h(p.frequencies);p.lut=h.lut;
    unsigned blocks=8*k/256;
    size_t tilebytes=size_t(blocks)*144;
    for(int tile=0;tile<n/8;++tile) {
        const uint8_t* r=raw.data()+tile*tilebytes;
        std::vector<uint8_t> data(blocks*16+129*2);
        for(unsigned b=0;b<blocks;++b) std::memcpy(data.data()+b*16,r+b*144,16);
        std::array<uint16_t,129> offsets{};
        for(unsigned lane=0;lane<128;++lane) {
            size_t off=data.size()-blocks*16-258;
            if(off>65535) throw std::runtime_error("Huffman stream offset");
            offsets[lane]=off;
            std::vector<uint8_t> stream;
            unsigned bit=0;
            for(unsigned b=0;b<blocks;++b) {
                uint8_t x=r[b*144+16+lane];
                for(unsigned sh:{0u,4u}) {
                    unsigned s=(x>>sh)&15;
                    for(unsigned j=0;j<h.len[s];++j,++bit) {
                        if(bit/8==stream.size()) stream.push_back(0);
                        stream[bit/8]|=((h.code[s]>>j)&1)<<(bit&7);
                    }
                }
            }
            data.insert(data.end(),stream.begin(),stream.end());
        }
        offsets[128]=data.size()-blocks*16-258;
        std::memcpy(data.data()+blocks*16,offsets.data(),258);
        data.resize((data.size()+3+15)/16*16);
        if(data.size()>=tilebytes) {
            p.index.push_back(p.bytes.size()|(1ull<<63));
            p.bytes.insert(p.bytes.end(),r,r+tilebytes);++p.raw_tiles;
        } else { p.index.push_back(p.bytes.size());p.bytes.insert(p.bytes.end(),data.begin(),data.end()); }
    }
    p.index.push_back(p.bytes.size());return p;
}

void verify_cpu(const Packed& p,const std::vector<uint8_t>& raw,int n,int k) {
    Huff h(p.frequencies);size_t blocks=8*k/256,tilebytes=blocks*144;
    if(p.index.size()!=size_t(n/8+1) || p.index.back()!=p.bytes.size() || (p.index[0]&~(1ull<<63))) throw std::runtime_error("Huffman index shape");
    std::vector<uint8_t> out(raw.size());
    for(int tile=0;tile<n/8;++tile) {
        uint64_t at=p.index[tile]&~(1ull<<63),end=p.index[tile+1]&~(1ull<<63);
        if(at>end || end>p.bytes.size()) throw std::runtime_error("Huffman index range");
        const uint8_t* src=p.bytes.data()+at;uint8_t* dst=out.data()+tile*tilebytes;
        if(p.index[tile]>>63) {if(end-at!=tilebytes) throw std::runtime_error("raw length");std::memcpy(dst,src,tilebytes);continue;}
        if(end-at<blocks*16+258+3) throw std::runtime_error("Huffman header");
        for(size_t b=0;b<blocks;++b) std::memcpy(dst+b*144,src+b*16,16);
        for(int lane=0;lane<128;++lane) {
            uint16_t lo,hi;std::memcpy(&lo,src+blocks*16+lane*2,2);std::memcpy(&hi,src+blocks*16+(lane+1)*2,2);
            if(lo>hi || size_t(hi)+blocks*16+258+3>end-at) throw std::runtime_error("Huffman stream range");
            const uint8_t* stream=src+blocks*16+258+lo;unsigned bit=0;
            for(size_t b=0;b<blocks;++b) {
                unsigned byte=0;
                for(unsigned sh:{0u,4u}) {
                    unsigned bits=hpeek(stream,bit);int symbol=-1;
                    for(int s=0;s<16;++s) if((bits&((1u<<h.len[s])-1))==h.code[s]) {symbol=s;break;}
                    if(symbol<0 || bit+h.len[symbol]>unsigned(hi-lo)*8) throw std::runtime_error("Huffman truncated");
                    bit+=h.len[symbol];byte|=symbol<<sh;
                }
                dst[b*144+16+lane]=byte;
            }
            if((bit+7)/8!=unsigned(hi-lo)) throw std::runtime_error("Huffman stream termination");
            if(bit%8 && (stream[bit/8]>>(bit%8))) throw std::runtime_error("Huffman padding");
        }
    }
    if(out!=raw) throw std::runtime_error("Huffman CPU bytes mismatch");
}

void save(const char* path,const Packed& p,uint32_t n,uint32_t k) {
    std::ofstream f(path,std::ios::binary);const char magic[8]={'N','L','H','Q','4',0,1,0};
    uint64_t count=p.index.size(),bytes=p.bytes.size();
    f.write(magic,8);f.write(reinterpret_cast<char*>(&n),4);f.write(reinterpret_cast<char*>(&k),4);
    f.write(reinterpret_cast<char*>(&count),8);f.write(reinterpret_cast<char*>(&bytes),8);
    f.write(reinterpret_cast<const char*>(p.frequencies.data()),32);
    f.write(reinterpret_cast<const char*>(p.index.data()),count*8);f.write(reinterpret_cast<const char*>(p.bytes.data()),bytes);
    if(!f) throw std::runtime_error("Huffman artifact write");
}

Packed load(const char* path,uint32_t n,uint32_t k) {
    std::ifstream f(path,std::ios::binary|std::ios::ate);if(!f) throw std::runtime_error("Huffman open");
    uint64_t size=f.tellg();f.seekg(0);char magic[8],expected[8]={'N','L','H','Q','4',0,1,0};uint32_t nn,kk;uint64_t count,bytes;
    f.read(magic,8);f.read(reinterpret_cast<char*>(&nn),4);f.read(reinterpret_cast<char*>(&kk),4);f.read(reinterpret_cast<char*>(&count),8);f.read(reinterpret_cast<char*>(&bytes),8);
    if(!f || std::memcmp(magic,expected,8) || nn!=n || kk!=k || count!=uint64_t(n/8+1) || bytes>size || 64+8*count+bytes!=size) throw std::runtime_error("Huffman version shape size");
    Packed p;f.read(reinterpret_cast<char*>(p.frequencies.data()),32);Huff h(p.frequencies);p.lut=h.lut;
    p.index.resize(count);p.bytes.resize(bytes);f.read(reinterpret_cast<char*>(p.index.data()),count*8);f.read(reinterpret_cast<char*>(p.bytes.data()),bytes);
    if(!f) throw std::runtime_error("Huffman artifact truncated");
    for(size_t i=0;i+1<count;++i)p.raw_tiles+=p.index[i]>>63;return p;
}

template<int K,bool Verify>
__device__ void unpack_tile(const uint8_t* payload,const uint64_t* index,const uint32_t* lut,int tile,uint8_t* shared) {
    constexpr int blocks=8*K/256,bytes=blocks*144;
    uint64_t at=index[tile];const uint8_t* src=payload+(at&~(1ull<<63));
    if(at>>63) {for(int j=threadIdx.x;j<bytes;j+=128)shared[j]=src[j];}
    else {
        for(int j=threadIdx.x;j<blocks*4;j+=128)reinterpret_cast<uint32_t*>(shared+(j/4)*144)[j%4]=reinterpret_cast<const uint32_t*>(src)[j];
        const uint16_t* offsets=reinterpret_cast<const uint16_t*>(src+blocks*16);
        const uint8_t* stream=src+blocks*16+258+offsets[threadIdx.x];unsigned bit=0;
        for(int b=0;b<blocks;++b) {
            unsigned a=bit>>3;
            unsigned bits=(unsigned(stream[a])|(unsigned(stream[a+1])<<8)|(unsigned(stream[a+2])<<16))>>(bit&7);
            uint32_t entry=lut[bits&1023];unsigned byte=entry&255;bit+=(entry>>8)&255;
            if(((entry>>16)&3)==1) {
                a=bit>>3;bits=(unsigned(stream[a])|(unsigned(stream[a+1])<<8)|(unsigned(stream[a+2])<<16))>>(bit&7);
                entry=lut[bits&1023];byte|=(entry&15)<<4;
                bit+=(entry>>20)&15;
            }
            shared[b*144+16+threadIdx.x]=byte;
        }
    }
    __syncthreads();
}
