set -eu
cd /src
printf '%s  %s\n' a32d451568dde06b834b8b34dfadbb2434ea0ee53c6a68b590cd06d1b946ddc5 /build/q4-k-stream/off.o | sha256sum -c -
mkdir -p /build/q4-cta-order /lab/results/q4-cta-order
python3 /lab/source/tools/q4_cta_order_probe/generate.py --source-root /src --output-root /build/q4-cta-order/generated --self-test
flags="-O3 -std=c++17 -arch=sm_86 --extended-lambda -lineinfo -Xptxas=-v -I/build/q4-cta-order/generated -Isrc -Iinclude -Ithird_party/ggml-quants/include -Ithird_party/ggml-quants/src -Ithird_party/ggml-quants/src/ggml-cuda"
nvcc $flags --frandom-seed=q4-cta-order-i4 -c /lab/source/tools/q4_cta_order_probe/main.cu -o /build/q4-cta-order/on.o > /lab/results/q4-cta-order/compile-on.log 2>&1
nvcc -arch=sm_86 /build/q4-cta-order/on.o /build/q4-k-stream/off.o -o /build/q4-cta-order/probe -lcuda
