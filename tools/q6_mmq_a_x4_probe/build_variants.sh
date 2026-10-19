set -eu
cd /src
for v in off on; do
nvcc -O3 -std=c++17 -arch=sm_86 -Xptxas=-v -Isrc -Iinclude -Ithird_party/ggml-quants/include -Ithird_party/ggml-quants/src -c tools/q6_mmq_a_x4_probe/variant_${v}.cu -o tools/q6_mmq_a_x4_probe/variant_${v}.o > tools/q6_mmq_a_x4_probe/compile_${v}.stdout.log 2> tools/q6_mmq_a_x4_probe/compile_${v}.stderr.log
done