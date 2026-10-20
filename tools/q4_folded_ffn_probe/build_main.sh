set -eu
test -z "${NVCC_PREPEND_FLAGS-}${NVCC_APPEND_FLAGS-}"
cd /src
probe=/lab/source/tools/q4_folded_ffn_probe
out=/build/q4-folded-ffn
printf '%s  %s\n' a32d451568dde06b834b8b34dfadbb2434ea0ee53c6a68b590cd06d1b946ddc5 /build/q4-k-stream/off.o | sha256sum -c -
mkdir -p "$out/host" "$out/audit"
python3 "$probe/generate_host.py" --source-root /src --probe-root "$probe" --output-root "$out/host" > "$out/timing-generation.json"
flags="-O3 -std=c++17 -arch=sm_86 --extended-lambda -lineinfo -Xptxas=-v -I/lab/source/tools/q4_folded_ffn_probe -Isrc -Iinclude -Ithird_party/ggml-quants/include -Ithird_party/ggml-quants/src -Ithird_party/ggml-quants/src/ggml-cuda"
nvcc $flags --frandom-seed=q4-folded-ffn-host -c "$out/host/main.cu" -o "$out/host.o" > "$out/compile-host.log" 2>&1
nvcc -arch=sm_86 "$out/host.o" "$out/variant-on.o" /build/q4-k-stream/off.o -o "$out/probe" -lcuda
cuobjdump --dump-resource-usage "$out/probe" > "$out/resources-linked.txt"
cd "$out/audit"
cuobjdump -xelf all "$out/probe"
