set -eu
cd /src
nvcc -O3 -std=c++17 -arch=sm_86 -Xptxas=-v -Iinclude -Ithird_party/ggml-quants/include -Ithird_party/ggml-quants/src q6_head_bound_probe.cu -o q6-head-bound-probe -lcuda > head-bound-compile.stdout.log 2> head-bound-compile.stderr.log
