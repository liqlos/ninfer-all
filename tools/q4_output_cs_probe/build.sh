set -eu
cd /src
printf '%s  %s\n' a32d451568dde06b834b8b34dfadbb2434ea0ee53c6a68b590cd06d1b946ddc5 /build/q4-k-stream/off.o | sha256sum -c -
mkdir -p /build/q4-output-cs/keep /lab/results/q4-output-cs
python3 /lab/source/tools/q4_output_cs_probe/generate.py --source-root /src --output-root /build/q4-output-cs/generated --self-test
flags="-O3 -std=c++17 -arch=sm_86 --extended-lambda -lineinfo -Xptxas=-v -I/build/q4-output-cs/generated -Isrc -Iinclude -Ithird_party/ggml-quants/include -Ithird_party/ggml-quants/src -Ithird_party/ggml-quants/src/ggml-cuda"
nvcc $flags -DNINFER_GGUF_Q4_MMQ_OUTPUT_CS=1 --frandom-seed=q4-output-cs --keep --keep-dir /build/q4-output-cs/keep -c /lab/source/tools/q4_output_cs_probe/main.cu -o /build/q4-output-cs/on.o > /lab/results/q4-output-cs/compile-on.log 2>&1
nvcc -arch=sm_86 /build/q4-output-cs/on.o /build/q4-k-stream/off.o -o /build/q4-output-cs/probe -lcuda
cuobjdump --dump-resource-usage /build/q4-output-cs/on.o > /lab/results/q4-output-cs/resources-on.txt
cuobjdump --dump-sass /build/q4-output-cs/on.o > /lab/results/q4-output-cs/sass-on.txt
cuobjdump --dump-resource-usage /build/q4-k-stream/off.o > /lab/results/q4-output-cs/resources-off.txt
cuobjdump --dump-sass /build/q4-k-stream/off.o > /lab/results/q4-output-cs/sass-off.txt
