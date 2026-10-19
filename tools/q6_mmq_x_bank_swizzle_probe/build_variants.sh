set -eu
cd /src
nvcc -O3 -std=c++17 -arch=sm_86 -Xptxas=-v -Isrc -Iinclude -Ithird_party/ggml-quants/include -Ithird_party/ggml-quants/src -c tools/q6_mmq_x_bank_swizzle_probe/variant_on.cu -o tools/q6_mmq_x_bank_swizzle_probe/variant_on.o > tools/q6_mmq_x_bank_swizzle_probe/compile_on.stdout.log 2> tools/q6_mmq_x_bank_swizzle_probe/compile_on.stderr.log
