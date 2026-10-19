set -eu
cd /src
nvcc -O3 -std=c++17 -arch=sm_86 -Isrc -Iinclude -Ithird_party/ggml-quants/include -Ithird_party/ggml-quants/src -c tools/q6_mmq_y_ldmatrix_probe/main.cu -o tools/q6_mmq_y_ldmatrix_probe/main.o > tools/q6_mmq_y_ldmatrix_probe/compile_main.stdout.log 2> tools/q6_mmq_y_ldmatrix_probe/compile_main.stderr.log
nvcc -arch=sm_86 tools/q6_mmq_y_ldmatrix_probe/main.o tools/q6_mmq_y_ldmatrix_probe/variant_off.o tools/q6_mmq_y_ldmatrix_probe/variant_on.o -o q6-mmq-y-ldmatrix-probe -lcuda > tools/q6_mmq_y_ldmatrix_probe/link.stdout.log 2> tools/q6_mmq_y_ldmatrix_probe/link.stderr.log
