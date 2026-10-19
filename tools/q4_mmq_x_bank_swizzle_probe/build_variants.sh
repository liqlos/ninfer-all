set -eu
cd /src
for v in off on; do
nvcc -O3 -std=c++17 -arch=sm_86 -Xptxas=-v -Isrc -Iinclude -Ithird_party/ggml-quants/include -Ithird_party/ggml-quants/src -c tools/q4_mmq_x_bank_swizzle_probe/variant_${v}.cu -o tools/q4_mmq_x_bank_swizzle_probe/variant_${v}.o > tools/q4_mmq_x_bank_swizzle_probe/compile_${v}.stdout.log 2> tools/q4_mmq_x_bank_swizzle_probe/compile_${v}.stderr.log
done
