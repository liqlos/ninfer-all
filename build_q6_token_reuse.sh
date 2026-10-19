set -eu
cd /src
nvcc -O3 -std=c++17 -arch=sm_86 -Xptxas=-v -Iinclude -Ithird_party/ggml-quants/include -Ithird_party/ggml-quants/src q6_token_reuse_probe.cu -o q6-token-reuse-probe -lcuda > token-reuse-compile.stdout.log 2> token-reuse-compile.stderr.log
