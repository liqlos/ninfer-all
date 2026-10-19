#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <fstream>
#include <vector>
#include <algorithm>
#include <stdexcept>
#define NINFER_GGUF_Q6_COOPERATIVE_DECODE 1
#include "src/ops/linear/gguf/q6_center_probe_off.cuh"
#define NINFER_GGUF_Q6_CENTER_BIAS_ADD 1
#include "src/ops/linear/gguf/q6_center_probe_on.cuh"
using namespace ninfer::ops::gguf::detail;
static void ck(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
template<int T,bool On> static void launch(VecArgs a,int grid,cudaStream_t st) {
    if constexpr(On) on_probe::vec::kernel<GGML_TYPE_Q6_K,T,false,0><<<grid,128,0,st>>>(a);
    else off_probe::vec::kernel<GGML_TYPE_Q6_K,T,false,0><<<grid,128,0,st>>>(a);
}
static double ref_at(const std::vector<std::uint8_t>& w,int row,int k,int j,const std::vector<std::int8_t>& q,const std::vector<half2>& ds) {
    double total=0;
    for(int i=0;i<k;++i) {
        const auto* b=w.data()+std::int64_t(row)*(k/256)*210+(i/256)*210;
        int n=i%256,u=n/32,h=u/4,t=u%4,z=n%32;
        int v=((b[64*h+32*(t&1)+z]>>(4*(t/2)))&15)|(((b[128+32*h+z]>>(2*t))&3)<<4);
        half bh;std::memcpy(&bh,b+208,2);
        const int scale=std::int8_t(b[192+8*h+2*t+z/16]);
        float2 d=__half22float2(ds[j*(k/32)+i/32]);
        total+=double(__half2float(bh))*scale*(v-32)*double(d.x)*int(q[j*k+i]);
    }
    return total;
}
template<int T> static void run(const char* file,int k,int rows,const char* tag) {
    std::ifstream in(file,std::ios::binary|std::ios::ate);if(!in)throw std::runtime_error("fixture missing");
    auto size=in.tellg();if(size!=std::streamoff(std::int64_t(rows)*(k/256)*210))throw std::runtime_error("fixture bytes differ");
    std::vector<std::uint8_t>w(static_cast<std::size_t>(size));in.seekg(0);in.read(reinterpret_cast<char*>(w.data()),size);if(!in)throw std::runtime_error("fixture read failed");
    std::vector<std::int8_t>q(T*k);std::vector<half2>ds(T*(k/32));
    for(int j=0;j<T;++j)for(int i=0;i<k;++i)q[j*k+i]=std::int8_t(((i*37+j*13+17)%255)-127);
    for(int j=0;j<T;++j)for(int i=0;i<k/32;++i)ds[j*(k/32)+i]=__floats2half2_rn(0.003f+0.0001f*((i+j)%29),0.0f);
    std::uint8_t*dw;std::int8_t*dq;half2*dd;__nv_bfloat16*da;__nv_bfloat16*db;
    ck(cudaMalloc(&dw,w.size()));ck(cudaMalloc(&dq,q.size()));ck(cudaMalloc(&dd,ds.size()*sizeof(half2)));ck(cudaMalloc(&da,(std::size_t(rows)*T+32)*2));ck(cudaMalloc(&db,(std::size_t(rows)*T+32)*2));
    ck(cudaMemcpy(dw,w.data(),w.size(),cudaMemcpyHostToDevice));ck(cudaMemcpy(dq,q.data(),q.size(),cudaMemcpyHostToDevice));ck(cudaMemcpy(dd,ds.data(),ds.size()*sizeof(half2),cudaMemcpyHostToDevice));
    ck(cudaMemset(da,0xa5,(std::size_t(rows)*T+32)*2));ck(cudaMemset(db,0xa5,(std::size_t(rows)*T+32)*2));
    cudaFuncAttributes fa,fb;ck(cudaFuncGetAttributes(&fa,off_probe::vec::kernel<GGML_TYPE_Q6_K,T,false,0>));ck(cudaFuncGetAttributes(&fb,on_probe::vec::kernel<GGML_TYPE_Q6_K,T,false,0>));
    int resident;ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,off_probe::vec::kernel<GGML_TYPE_Q6_K,T,false,0>,128,0));
    cudaDeviceProp prop;ck(cudaGetDeviceProperties(&prop,0));if(prop.major!=8||prop.minor!=6)throw std::runtime_error("SM86 required");
    const int grid=std::min((rows+7)/8,resident*prop.multiProcessorCount);
    int on_resident;ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&on_resident,on_probe::vec::kernel<GGML_TYPE_Q6_K,T,false,0>,128,0));
    const int on_grid=std::min((rows+7)/8,on_resident*prop.multiProcessorCount);
    std::printf("{\"kind\":\"native_grid\",\"off_grid\":%d,\"on_grid\":%d,\"off_resident\":%d,\"on_resident\":%d}\n",grid,on_grid,resident,on_resident);
    VecArgs a{dw,std::int64_t(k/256)*210,rows,k,dq,dd,{da,nullptr,rows,ninfer::ops::gguf::Epilogue::Store,nullptr,0}};VecArgs b=a;b.out.bf16=db;
    launch<T,false>(a,grid,0);launch<T,true>(b,on_grid,0);ck(cudaGetLastError());ck(cudaDeviceSynchronize());
    std::vector<std::uint16_t>ha(std::size_t(rows)*T+32),hb(ha.size());ck(cudaMemcpy(ha.data(),da,ha.size()*2,cudaMemcpyDeviceToHost));ck(cudaMemcpy(hb.data(),db,hb.size()*2,cudaMemcpyDeviceToHost));
    if(ha!=hb)throw std::runtime_error("exact OFF/ON BF16 mismatch");for(std::size_t i=std::size_t(rows)*T;i<ha.size();++i)if(ha[i]!=0xa5a5)throw std::runtime_error("output guard overwritten");
    double maxerr=0;for(int r:{0,rows/2,rows-1})for(int j=0;j<T;++j){const double expect=ref_at(w,r,k,j,q,ds);__nv_bfloat16 actual;std::memcpy(&actual,&ha[j*rows+r],2);double got=__bfloat162float(actual);double err=std::abs(got-expect);maxerr=std::max(maxerr,err);if(!std::isfinite(got)||err>0.008*std::abs(expect)+0.003)throw std::runtime_error("independent FP64 oracle failure");}
    for(int i=0;i<8;++i){launch<T,false>(a,grid,0);launch<T,true>(b,on_grid,0);}ck(cudaDeviceSynchronize());
    cudaEvent_t start,end;ck(cudaEventCreate(&start));ck(cudaEventCreate(&end));
    std::printf("{\"kind\":\"identity\",\"shape\":\"%s\",\"rows\":%d,\"k\":%d,\"t\":%d,\"grid\":%d,\"off_regs\":%d,\"on_regs\":%d,\"off_local\":%zu,\"on_local\":%zu,\"exact_bf16\":true,\"oracle_max_abs\":%.9g}\n",tag,rows,k,T,grid,fa.numRegs,fb.numRegs,fa.localSizeBytes,fb.localSizeBytes,maxerr);
    for(int block=0;block<20;++block)for(int pos=0;pos<4;++pos){bool on=(block%2==0)?(pos==1||pos==2):(pos==0||pos==3);ck(cudaEventRecord(start));for(int i=0;i<12;++i){if(on)launch<T,true>(b,on_grid,0);else launch<T,false>(a,grid,0);}ck(cudaEventRecord(end));ck(cudaEventSynchronize(end));float ms;ck(cudaEventElapsedTime(&ms,start,end));std::printf("{\"kind\":\"sample\",\"shape\":\"%s\",\"block\":%d,\"position\":%d,\"on\":%s,\"repetitions\":12,\"ms_per_call\":%.9g}\n",tag,block,pos,on?"true":"false",ms/12);}
    ck(cudaMemcpy(hb.data(),db,hb.size()*2,cudaMemcpyDeviceToHost));if(ha!=hb)throw std::runtime_error("post-timing parity/guard mismatch");
    ck(cudaEventDestroy(start));ck(cudaEventDestroy(end));ck(cudaFree(dw));ck(cudaFree(dq));ck(cudaFree(dd));ck(cudaFree(da));ck(cudaFree(db));
}
int main(){try{
    for(std::uint32_t i=0;i<(1u<<24);++i){std::uint32_t v=(i&63u)|(((i>>6)&63u)<<8)|(((i>>12)&63u)<<16)|(((i>>18)&63u)<<24);std::uint32_t got=(v+0x60606060u)^0x80808080u;for(int z=0;z<4;++z)if(((got>>(8*z))&255u)!=std::uint8_t(int((v>>(8*z))&255u)-32))throw std::runtime_error("exhaustive centering failure");}
    run<1>("/fixtures/q6-head-full-k5120.bin",5120,248320,"head_t1");
    std::puts("{\"kind\":\"complete\",\"status\":\"PASS\",\"scope\":\"exploratory operator only; warm resident inputs; synthetic activation; real Q6 weights\"}");return 0;
}catch(const std::exception&e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
