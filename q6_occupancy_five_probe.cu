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
#define NINFER_GGUF_Q4_COOPERATIVE_DECODE 1
#include "src/ops/linear/gguf/occupancy_five_off.cuh"
#define NINFER_GGUF_Q6_T4_OCCUPANCY_FIVE 1
#include "src/ops/linear/gguf/occupancy_five_on.cuh"
using namespace ninfer::ops::gguf::detail;
static void ck(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
template<int T,bool On,ggml_type Type,bool Fused> static void launch(VecArgs a,int grid,cudaStream_t st) {
    if constexpr(On) on_probe::vec::kernel<Type,T,Fused,(Type==GGML_TYPE_Q4_K?5120:0)><<<grid,128,0,st>>>(a);
    else off_probe::vec::kernel<Type,T,Fused,(Type==GGML_TYPE_Q4_K?5120:0)><<<grid,128,0,st>>>(a);
}
template<ggml_type Type> static double ref_at(const std::vector<std::uint8_t>& w,int row,int k,int j,const std::vector<std::int8_t>& q,const std::vector<half2>& ds) {
    double total=0;
    for(int i=0;i<k;++i) {
        constexpr int block_bytes=Type==GGML_TYPE_Q4_K?144:210;
        const auto* b=w.data()+std::int64_t(row)*(k/256)*block_bytes+(i/256)*block_bytes;
        int n=i%256,u=n/32,h=u/4,t=u%4,z=n%32;
        float2 d=__half22float2(ds[j*(k/32)+i/32]);
        if constexpr(Type==GGML_TYPE_Q6_K) {
            int v=((b[64*h+32*(t&1)+z]>>(4*(t/2)))&15)|(((b[128+32*h+z]>>(2*t))&3)<<4);
            half bh;std::memcpy(&bh,b+208,2);
            const int scale=std::int8_t(b[192+8*h+2*t+z/16]);
            total+=double(__half2float(bh))*scale*(v-32)*double(d.x)*int(q[j*k+i]);
        } else {
            half dh,mh;std::memcpy(&dh,b,2);std::memcpy(&mh,b+2,2);
            const auto* sc=b+4;int scale,minimum;
            if(u<4){scale=sc[u]&63;minimum=sc[u+4]&63;}
            else{scale=(sc[u+4]&15)|((sc[u-4]>>6)<<4);minimum=(sc[u+4]>>4)|((sc[u]>>6)<<4);}
            int v=(b[16+32*(u/2)+z]>>(4*(u&1)))&15;
            total+=double(__half2float(dh))*scale*v*double(d.x)*int(q[j*k+i]);
            if(z==0)total-=double(__half2float(mh))*minimum*double(d.y);
        }
    }
    return total;
}
template<int T,ggml_type Type=GGML_TYPE_Q6_K,bool Fused=false> static void run(const char* file,int k,int rows,const char* tag) {
    std::ifstream in(file,std::ios::binary|std::ios::ate);if(!in)throw std::runtime_error("fixture missing");
    auto size=in.tellg();if(size!=std::streamoff(std::int64_t(rows)*(Fused?2:1)*(k/256)*(Type==GGML_TYPE_Q4_K?144:210)))throw std::runtime_error("fixture bytes differ");
    std::vector<std::uint8_t>w(static_cast<std::size_t>(size));in.seekg(0);in.read(reinterpret_cast<char*>(w.data()),size);if(!in)throw std::runtime_error("fixture read failed");
    std::vector<std::int8_t>q(T*k);std::vector<half2>ds(T*(k/32));
    for(int j=0;j<T;++j)for(int i=0;i<k;++i)q[j*k+i]=std::int8_t(((i*37+j*13+17)%255)-127);
    for(int j=0;j<T;++j)for(int i=0;i<k/32;++i){float d=__half2float(__float2half(0.003f+0.0001f*((i+j)%29)));int sum=0;for(int z=0;z<32;++z)sum+=q[j*k+32*i+z];ds[j*(k/32)+i]=__floats2half2_rn(d,d*sum);}
    std::uint8_t*dw;std::int8_t*dq;half2*dd;__nv_bfloat16*da;__nv_bfloat16*db;
    ck(cudaMalloc(&dw,w.size()));ck(cudaMalloc(&dq,q.size()));ck(cudaMalloc(&dd,ds.size()*sizeof(half2)));ck(cudaMalloc(&da,(std::size_t(rows)*T+32)*2));ck(cudaMalloc(&db,(std::size_t(rows)*T+32)*2));
    ck(cudaMemcpy(dw,w.data(),w.size(),cudaMemcpyHostToDevice));ck(cudaMemcpy(dq,q.data(),q.size(),cudaMemcpyHostToDevice));ck(cudaMemcpy(dd,ds.data(),ds.size()*sizeof(half2),cudaMemcpyHostToDevice));
    ck(cudaMemset(da,0xa5,(std::size_t(rows)*T+32)*2));ck(cudaMemset(db,0xa5,(std::size_t(rows)*T+32)*2));
    cudaFuncAttributes fa,fb;ck(cudaFuncGetAttributes(&fa,off_probe::vec::kernel<Type,T,Fused,(Type==GGML_TYPE_Q4_K?5120:0)>));ck(cudaFuncGetAttributes(&fb,on_probe::vec::kernel<Type,T,Fused,(Type==GGML_TYPE_Q4_K?5120:0)>));
    int resident;ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,off_probe::vec::kernel<Type,T,Fused,(Type==GGML_TYPE_Q4_K?5120:0)>,128,0));
    cudaDeviceProp prop;ck(cudaGetDeviceProperties(&prop,0));if(prop.major!=8||prop.minor!=6)throw std::runtime_error("SM86 required");
    const int grid=std::min((rows+(Fused?4:8)-1)/(Fused?4:8),resident*prop.multiProcessorCount);
    int on_resident;ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&on_resident,on_probe::vec::kernel<Type,T,Fused,(Type==GGML_TYPE_Q4_K?5120:0)>,128,0));
    const int on_grid=std::min((rows+(Fused?4:8)-1)/(Fused?4:8),on_resident*prop.multiProcessorCount);
    std::printf("{\"kind\":\"native_grid\",\"shape\":\"%s\",\"off_grid\":%d,\"on_grid\":%d}\n",tag,grid,on_grid);
    VecArgs a{dw,std::int64_t(k/256)*(Type==GGML_TYPE_Q4_K?144:210),rows,k,dq,dd,{da,nullptr,rows,ninfer::ops::gguf::Epilogue::Store,nullptr,0}};VecArgs b=a;b.out.bf16=db;
    launch<T,false,Type,Fused>(a,grid,0);launch<T,true,Type,Fused>(b,on_grid,0);ck(cudaGetLastError());ck(cudaDeviceSynchronize());
    std::vector<std::uint16_t>ha(std::size_t(rows)*T+32),hb(ha.size());ck(cudaMemcpy(ha.data(),da,ha.size()*2,cudaMemcpyDeviceToHost));ck(cudaMemcpy(hb.data(),db,hb.size()*2,cudaMemcpyDeviceToHost));
    if(ha!=hb)throw std::runtime_error("exact OFF/ON BF16 mismatch");for(std::size_t i=std::size_t(rows)*T;i<ha.size();++i)if(ha[i]!=0xa5a5)throw std::runtime_error("output guard overwritten");
    double maxerr=0;for(int r:{0,rows/2,rows-1})for(int j=0;j<T;++j){double expect=ref_at<Type>(w,r,k,j,q,ds);if constexpr(Fused){double up=ref_at<Type>(w,rows+r,k,j,q,ds);expect=expect/(1.0+std::exp(-expect))*up;}__nv_bfloat16 actual;std::memcpy(&actual,&ha[j*rows+r],2);double got=__bfloat162float(actual);double err=std::abs(got-expect);maxerr=std::max(maxerr,err);if(!std::isfinite(got)||err>0.008*std::abs(expect)+0.003)throw std::runtime_error("independent FP64 oracle failure");}
    for(int i=0;i<8;++i){launch<T,false,Type,Fused>(a,grid,0);launch<T,true,Type,Fused>(b,on_grid,0);}ck(cudaDeviceSynchronize());
    cudaEvent_t start,end;ck(cudaEventCreate(&start));ck(cudaEventCreate(&end));
    std::printf("{\"kind\":\"identity\",\"shape\":\"%s\",\"rows\":%d,\"k\":%d,\"t\":%d,\"grid\":%d,\"off_regs\":%d,\"on_regs\":%d,\"off_local\":%zu,\"on_local\":%zu,\"exact_bf16\":true,\"oracle_max_abs\":%.9g}\n",tag,rows,k,T,grid,fa.numRegs,fb.numRegs,fa.localSizeBytes,fb.localSizeBytes,maxerr);
    for(int block=0;block<20;++block)for(int pos=0;pos<4;++pos){bool on=(block%2==0)?(pos==1||pos==2):(pos==0||pos==3);ck(cudaEventRecord(start));for(int i=0;i<12;++i){if(on)launch<T,true,Type,Fused>(b,on_grid,0);else launch<T,false,Type,Fused>(a,grid,0);}ck(cudaEventRecord(end));ck(cudaEventSynchronize(end));float ms;ck(cudaEventElapsedTime(&ms,start,end));std::printf("{\"kind\":\"sample\",\"shape\":\"%s\",\"block\":%d,\"position\":%d,\"on\":%s,\"repetitions\":12,\"ms_per_call\":%.9g}\n",tag,block,pos,on?"true":"false",ms/12);}
    ck(cudaMemcpy(hb.data(),db,hb.size()*2,cudaMemcpyDeviceToHost));if(ha!=hb)throw std::runtime_error("post-timing parity/guard mismatch");
    ck(cudaEventDestroy(start));ck(cudaEventDestroy(end));ck(cudaFree(dw));ck(cudaFree(dq));ck(cudaFree(dd));ck(cudaFree(da));ck(cudaFree(db));
}
int main(){try{




    run<4,GGML_TYPE_Q6_K>("/extra/q6-full-k17408.bin",17408,5120,"q6_down_t4");
    run<4,GGML_TYPE_Q6_K>("/fixtures/q6-head-full-k5120.bin",5120,248320,"q6_head_t4");
    std::puts("{\"kind\":\"complete\",\"status\":\"PASS\",\"scope\":\"exploratory operator only; warm resident inputs; synthetic activation; real Q4/Q6 weights\"}");return 0;
}catch(const std::exception&e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
