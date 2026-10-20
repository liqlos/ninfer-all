set -eu
cd /src
mkdir -p /build/q4-k-stream
flags="-O3 -std=c++17 -arch=sm_86 --extended-lambda -lineinfo -Xptxas=-v -Isrc -Iinclude -Ithird_party/ggml-quants/include -Ithird_party/ggml-quants/src"
nvcc $flags --frandom-seed=q4-k-stream-on -c /lab/source/tools/q4_k_stream_probe/main.cu -o /build/q4-k-stream/on.o > /lab/results/q4-k-stream/compile-on.log 2>&1
nvcc $flags --frandom-seed=q4-k-stream-off -c tools/ffn_pair_probe/baseline.cu -o /build/q4-k-stream/off.o > /lab/results/q4-k-stream/compile-off.log 2>&1
nvcc -arch=sm_86 /build/q4-k-stream/on.o /build/q4-k-stream/off.o -o /build/q4-k-stream/probe -lcuda
