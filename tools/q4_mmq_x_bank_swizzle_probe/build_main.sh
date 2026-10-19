set -eu
cd /src
nvcc -O3 -std=c++17 -arch=sm_86 -Isrc -Iinclude -Ithird_party/ggml-quants/include -Ithird_party/ggml-quants/src -c tools/q4_mmq_x_bank_swizzle_probe/main.cu -o tools/q4_mmq_x_bank_swizzle_probe/main.o > tools/q4_mmq_x_bank_swizzle_probe/compile_main.stdout.log 2> tools/q4_mmq_x_bank_swizzle_probe/compile_main.stderr.log
nvcc -arch=sm_86 tools/q4_mmq_x_bank_swizzle_probe/main.o tools/q4_mmq_x_bank_swizzle_probe/variant_off.o tools/q4_mmq_x_bank_swizzle_probe/variant_on.o -o q4-mmq-x-bank-swizzle-probe -lcuda > tools/q4_mmq_x_bank_swizzle_probe/link.stdout.log 2> tools/q4_mmq_x_bank_swizzle_probe/link.stderr.log
