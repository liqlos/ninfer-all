set -eu
cd /src
nvcc -O3 -std=c++17 -arch=sm_86 -Xptxas=-v -Iinclude -Ithird_party/ggml-quants/include -Ithird_party/ggml-quants/src q6_head_occupancy_five_probe.cu -o q6-head-occupancy-five-probe -lcuda > head-occupancy-five-compile.stdout.log 2> head-occupancy-five-compile.stderr.log
