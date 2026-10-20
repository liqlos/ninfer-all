set -eu
cd /src
printf '%s  %s\n' a32d451568dde06b834b8b34dfadbb2434ea0ee53c6a68b590cd06d1b946ddc5 /build/q4-k-stream/off.o | sha256sum -c -
mkdir -p /build/q4-compact-a/keep /lab/results/q4-compact-a
python3 /lab/source/tools/q4_compact_a_probe/generate.py --source-root /src --output-root /build/q4-compact-a/generated --self-test
flags="-O3 -std=c++17 -arch=sm_86 --extended-lambda -lineinfo -Xptxas=-v -I/build/q4-compact-a/generated -Isrc -Iinclude -Ithird_party/ggml-quants/include -Ithird_party/ggml-quants/src -Ithird_party/ggml-quants/src/ggml-cuda"
nvcc $flags -DNINFER_GGUF_Q4_MMQ_COMPACT_A=1 --frandom-seed=q4-compact-a --keep --keep-dir /build/q4-compact-a/keep -c /lab/source/tools/q4_compact_a_probe/main.cu -o /build/q4-compact-a/on.o > /lab/results/q4-compact-a/compile-on.log 2>&1
nvcc -arch=sm_86 /build/q4-compact-a/on.o /build/q4-k-stream/off.o -o /build/q4-compact-a/probe -lcuda
cuobjdump --dump-resource-usage /build/q4-compact-a/on.o > /lab/results/q4-compact-a/resources-on.txt
cuobjdump --dump-sass /build/q4-compact-a/on.o > /lab/results/q4-compact-a/sass-on.txt
cuobjdump --dump-resource-usage /build/q4-k-stream/off.o > /lab/results/q4-compact-a/resources-off.txt
cuobjdump --dump-sass /build/q4-k-stream/off.o > /lab/results/q4-compact-a/sass-off.txt
