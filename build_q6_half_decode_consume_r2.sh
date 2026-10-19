set -eu
cd /src
nvcc -O3 -std=c++17 -arch=sm_86 -Xptxas=-v -Iinclude -Ithird_party/ggml-quants/include -Ithird_party/ggml-quants/src q6_half_decode_consume_r2_probe.cu -o q6-half-decode-consume-r2-probe -lcuda > half-decode-consume-r2-compile.stdout.log 2> half-decode-consume-r2-compile.stderr.log
