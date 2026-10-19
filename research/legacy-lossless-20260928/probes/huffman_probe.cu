#include "huffman_codec.cuh"
template<int K, int T, bool Verify = false>
__global__ __launch_bounds__(128) void fused(VecArgs p, const uint64_t* index, const uint32_t* lut, uint8_t* debug = nullptr) {
    constexpr int row_bytes = K / 256 * 144;
    __shared__ __align__(16) uint8_t shared[8 * row_bytes];
    __shared__ uint32_t shared_lut[1024];
    for (int i = threadIdx.x; i < 1024; i += 128) shared_lut[i] = lut[i];
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
              << ",\"lut_shared_bytes_per_cta\":4096,\"stream_layout\":\"all_warps_multisymbol_huffman\""
              << ",\"resident_ctas_per_sm\":" << resident << ",\"fp32_output_bit_exact\":true,\"gpu_all_payload_bytes_exact\":" << (T==1?"true":"null")
              << ",\"bf16_output_bit_exact\":true,\"fp64_oracle_rows\":16,\"oracle_max_abs_error\":" << oracle_max_abs << ",\"oracle_max_normalized_error\":" << oracle_max_normalized
              << ",\"rotations\":3,\"graphs\":true,\"samples_ms\":[";
    for (int round = 0; round < 5; ++round) for (int order = 0; order < 2; ++order) {
        int mode = (round+order)%2; check(cudaEventRecord(start,stream), "start");
        for (int rep = 0; rep < 10; ++rep) check(cudaGraphLaunch(graphs[mode],stream), "replay");
        check(cudaEventRecord(stop,stream), "stop"); check(cudaEventSynchronize(stop), "wait"); float ms;
        check(cudaEventElapsedTime(&ms,start,stop), "time");
        if (round || order) std::cout << ',';
        std::cout << "{\"mode\":\"" << (mode?"huffman":"baseline") << "\",\"ms\":" << ms/30 << '}';
    }
    std::cout << "]}\n";
    uint64_t* raw_index;
    std::vector<uint64_t> ri(n/8+1);
    for(int i=0;i<n/8;++i) ri[i]=(uint64_t(i)*8*k/256*144)|(1ull<<63);
    ri.back()=raw.size();
    alloc(&raw_index,ri.size()*8);check(cudaMemcpy(raw_index,ri.data(),ri.size()*8,cudaMemcpyHostToDevice),"raw staging index");
    alloc(&debug,raw.size());
    cudaGraphExec_t diagnostics[2];
    for(int mode=0;mode<2;++mode) {
        cudaGraph_t graph;check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal),"diagnostic capture");
        for(int r=0;r<3;++r) {
            p.weight=mode?compressed[r]:weights[r];
            if(mode) fused<k,T,true><<<grid,128,0,stream>>>(p,index,lut,debug);
            else fused<k,T><<<grid,128,0,stream>>>(p,raw_index,lut);
        }
        check(cudaStreamEndCapture(stream,&graph),"diagnostic capture end");
        check(cudaGraphInstantiate(&diagnostics[mode],graph,0),"diagnostic instantiate");check(cudaGraphDestroy(graph),"diagnostic destroy");
    }
    p.weight=weights[0];fused<k,T><<<grid,128,0,stream>>>(p,raw_index,lut);
    check(cudaStreamSynchronize(stream),"raw staged check");check(cudaMemcpy(test.data(),y,test.size()*4,cudaMemcpyDeviceToHost),"raw staged copy");
    if(std::memcmp(base.data(),test.data(),base.size()*4)) throw std::runtime_error("raw staging FP32 mismatch");
    for(int round=0;round<5;++round) for(int mode=0;mode<2;++mode) {
        check(cudaGraphLaunch(diagnostics[mode],stream),"diagnostic warmup");
        check(cudaEventRecord(start,stream),"diagnostic start");
        for(int rep=0;rep<10;++rep)check(cudaGraphLaunch(diagnostics[mode],stream),"diagnostic replay");
        check(cudaEventRecord(stop,stream),"diagnostic stop");check(cudaEventSynchronize(stop),"diagnostic wait");float ms;
        check(cudaEventElapsedTime(&ms,start,stop),"diagnostic elapsed");
        std::cout<<"{\"matrix\":\""<<label<<"\",\"T\":"<<T<<",\"diagnostic\":\""<<(mode?"decode_stage_write_no_compute":"raw_stage_compute")<<"\",\"round\":"<<round<<",\"ms\":"<<ms/30<<"}\n";
    }
    for(auto g:diagnostics)check(cudaGraphExecDestroy(g),"diagnostic graph free");
    check(cudaFree(raw_index),"raw index free");check(cudaFree(debug),"diagnostic debug free");
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
        std::cout << "{\"layout\":\"gguf_q4_k_cta8_huffman_v1\",\"cpu_all_bytes_exact\":true,\"N\":" << n << ",\"K\":" << k << ",\"raw_bytes\":" << raw.size()
                  << ",\"artifact_file_bytes\":" << 64+p.index.size()*8+p.bytes.size() << ",\"resident_bytes\":" << p.index.size()*8+p.bytes.size()+p.lut.size()*4 << "}\n";
        return 0;
    }
    measure<1>(raw,p,n,argv[5]); measure<2>(raw,p,n,argv[5]); measure<3>(raw,p,n,argv[5]); measure<4>(raw,p,n,argv[5]); measure<8>(raw,p,n,argv[5]);
}
