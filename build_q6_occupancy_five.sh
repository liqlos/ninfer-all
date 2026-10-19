set -eu
cd /src
nvcc -O3 -std=c++17 -arch=sm_86 -Xptxas=-v -Iinclude -Ithird_party/ggml-quants/include -Ithird_party/ggml-quants/src q6_occupancy_five_probe.cu -o q6-occupancy-five-probe -lcuda > occupancy-five-compile.stdout.log 2> occupancy-five-compile.stderr.log
