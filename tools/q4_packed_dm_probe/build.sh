set -eu
cd /lab/source
mkdir -p /build/q4-packed-dm
flags="-O3 -std=c++17 -arch=sm_86 --extended-lambda -Xptxas=-v -Isrc -Iinclude -Ithird_party/ggml-quants/include -Ithird_party/ggml-quants/src"
nvcc $flags -c tools/q4_packed_dm_probe/main.cu -o /build/q4-packed-dm/on.o > /lab/results/q4-packed-dm/compile-on.log 2>&1
nvcc $flags -c tools/ffn_pair_probe/baseline.cu -o /build/q4-packed-dm/off.o > /lab/results/q4-packed-dm/compile-off.log 2>&1
nvcc -arch=sm_86 /build/q4-packed-dm/on.o /build/q4-packed-dm/off.o -o /lab/artifacts/experimental/q4-packed-dm-probe -lcuda
