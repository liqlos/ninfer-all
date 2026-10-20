set -eu
cd /src
mkdir -p /build/q6-raw32
python3 /lab/source/tools/q6_raw32_probe/generate.py --source-root /src --self-test > /build/q6-raw32/generated.json
flags="-O3 -std=c++17 -arch=sm_86 --extended-lambda -lineinfo -Xptxas=-v -Isrc -Iinclude -Ithird_party/ggml-quants/include -Ithird_party/ggml-quants/src -Ithird_party/ggml-quants/src/ggml-cuda"
nvcc $flags --frandom-seed=q6-raw32-off -c /lab/source/tools/q6_raw32_probe/variant_off.cu -o /build/q6-raw32/off.o > /build/q6-raw32/compile-off.log 2>&1
nvcc -I/build/q6-raw32/generated $flags --frandom-seed=q6-raw32-on -c /lab/source/tools/q6_raw32_probe/variant_on.cu -o /build/q6-raw32/on.o > /build/q6-raw32/compile-on.log 2>&1
nvcc -I/build/q6-raw32/generated $flags --frandom-seed=q6-raw32-main -c /lab/source/tools/q6_raw32_probe/main.cu -o /build/q6-raw32/main.o > /build/q6-raw32/compile-main.log 2>&1
nvcc -arch=sm_86 /build/q6-raw32/main.o /build/q6-raw32/off.o /build/q6-raw32/on.o -o /build/q6-raw32/probe -lcuda > /build/q6-raw32/link.log 2>&1
