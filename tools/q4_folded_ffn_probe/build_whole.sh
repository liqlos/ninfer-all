set -eu
test -z "${NVCC_PREPEND_FLAGS-}${NVCC_APPEND_FLAGS-}"
cd /src
probe=/lab/source/tools/q4_folded_ffn_probe
out=/build/q4-folded-ffn/whole
printf '%s  %s\n' a32d451568dde06b834b8b34dfadbb2434ea0ee53c6a68b590cd06d1b946ddc5 /build/q4-k-stream/off.o | sha256sum -c -
test -f /build/q4-folded-ffn/pack/variant-on.o
test -f /build/q4-folded-ffn/pack/generation.json
mkdir -p "$out/keep-down" "$out/keep-main"
python3 -B "$probe/generate_whole.py" --source-root /src --output-root "$out/generated" --self-test > "$out/generation.json"
sha256sum /build/q4-folded-ffn/pack/variant-on.o /build/q4-k-stream/off.o > "$out/linked-input-sha256.txt"
flags="-O3 -std=c++17 -arch=sm_86 --extended-lambda -lineinfo -Xptxas=-v -I/lab/source/tools/q4_folded_ffn_probe -I/build/q4-folded-ffn/whole/generated/support -Isrc -Iinclude -Ithird_party/ggml-quants/include -Ithird_party/ggml-quants/src -Ithird_party/ggml-quants/src/ggml-cuda"
nvcc $flags --frandom-seed=q4-folded-whole-down --objdir-as-tempdir --keep --keep-dir "$out/keep-down" -c "$probe/down.cu" -o "$out/down.o" > "$out/compile-down.log" 2>&1
nvcc $flags --frandom-seed=q4-folded-whole-main --objdir-as-tempdir --keep --keep-dir "$out/keep-main" -c "$probe/whole_main.cu" -o "$out/main.o" > "$out/compile-main.log" 2>&1
nvcc -arch=sm_86 "$out/main.o" "$out/down.o" /build/q4-folded-ffn/pack/variant-on.o /build/q4-k-stream/off.o -o "$out/probe" -lcuda
cuobjdump --dump-resource-usage "$out/probe" > "$out/resources-linked.txt"
cuobjdump --dump-sass "$out/down.o" > "$out/sass-down.txt"
nm -C "$out/down.o" > "$out/symbols-down.txt"
sha256sum "$out/probe" "$out/main.o" "$out/down.o" > "$out/output-sha256.txt"
