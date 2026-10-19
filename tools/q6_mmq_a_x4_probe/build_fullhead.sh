set -eu
cd /src
nvcc -O3 -std=c++17 -arch=sm_86 -Isrc -Iinclude -Ithird_party/ggml-quants/include -Ithird_party/ggml-quants/src -c tools/q6_mmq_a_x4_probe/main_fullhead.cu -o tools/q6_mmq_a_x4_probe/main_fullhead.o > tools/q6_mmq_a_x4_probe/compile_fullhead.stdout.log 2> tools/q6_mmq_a_x4_probe/compile_fullhead.stderr.log
nvcc -arch=sm_86 tools/q6_mmq_a_x4_probe/main_fullhead.o tools/q6_mmq_a_x4_probe/variant_off.o tools/q6_mmq_a_x4_probe/variant_on.o -o q6-mmq-a-x4-fullhead-probe -lcuda > tools/q6_mmq_a_x4_probe/link_fullhead.stdout.log 2> tools/q6_mmq_a_x4_probe/link_fullhead.stderr.log
