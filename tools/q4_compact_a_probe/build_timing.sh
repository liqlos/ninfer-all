set -eu
test -z "${NVCC_PREPEND_FLAGS-}${NVCC_APPEND_FLAGS-}"
cd /src
probe=/lab/source/tools/q4_compact_a_probe
out=/build/q4-compact-a
printf '%s  %s\n' a32d451568dde06b834b8b34dfadbb2434ea0ee53c6a68b590cd06d1b946ddc5 /build/q4-k-stream/off.o | sha256sum -c -
mkdir -p "$out/keep-r2" "$out/audit-r2"
python3 "$probe/generate_timing.py" --source-root /src --probe-root "$probe" --output-root "$out/timing-r2" > "$out/timing-generation-r2.json"
flags="-O3 -std=c++17 -arch=sm_86 --extended-lambda -lineinfo -Xptxas=-v -I/build/q4-compact-a/generated -Isrc -Iinclude -Ithird_party/ggml-quants/include -Ithird_party/ggml-quants/src -Ithird_party/ggml-quants/src/ggml-cuda"
nvcc $flags -DNINFER_GGUF_Q4_MMQ_COMPACT_A=1 --frandom-seed=q4-compact-a --keep --keep-dir "$out/keep-r2" -c "$out/timing-r2/main.cu" -o "$out/on-r2.o" > "$out/compile-r2.log" 2>&1
nvcc -arch=sm_86 "$out/on-r2.o" /build/q4-k-stream/off.o -o "$out/probe-r2" -lcuda
cuobjdump --dump-resource-usage "$out/on-r2.o" > "$out/resources-r2.txt"
cuobjdump --dump-sass "$out/on-r2.o" > "$out/sass-r2.txt"
cd "$out/audit-r2"
cuobjdump -xelf all "$out/probe-r2"
