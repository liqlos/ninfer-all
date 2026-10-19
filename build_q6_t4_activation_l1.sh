set -eu
cd /src
nvcc -O3 -std=c++17 -arch=sm_86 -Xptxas=-v -Iinclude -Ithird_party/ggml-quants/include -Ithird_party/ggml-quants/src q6_t4_activation_l1_probe.cu -o q6-t4-activation-l1-probe -lcuda > activation-l1-compile.stdout.log 2> activation-l1-compile.stderr.log
