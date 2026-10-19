set -eu
cd /src
nvcc -O3 -std=c++17 -arch=sm_86 -Xptxas=-v -Iinclude -Ithird_party/ggml-quants/include -Ithird_party/ggml-quants/src dp4a_narrow_native_grid_probe.cu -o dp4a-narrow-native-grid-probe -lcuda > narrow_compile.stdout.log 2> narrow_compile.stderr.log
