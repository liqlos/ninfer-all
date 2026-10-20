set -eu
test -z "${NVCC_PREPEND_FLAGS-}${NVCC_APPEND_FLAGS-}"
cd /src
probe=/lab/source/tools/q6_word_group_probe
objects=/build/q6-word-group
out="$objects/host"
mkdir -p "$out"
nvcc --version > "$out/toolchain.txt"
grep -Eq 'V13[.]1[.]115([[:space:]]|$)' "$out/toolchain.txt"
python3 -B "$probe/generate_host.py" --source-root /src --probe-root "$probe" \
    --objects-root "$objects" --output-root "$out/generated" > "$out/generation.json"
nm -u "$objects/vector.o" > "$out/vector-undefined.txt"
nm -u "$objects/mmq.o" > "$out/mmq-undefined.txt"
grep -Eq ' U __cudaLaunchKernel$' "$out/vector-undefined.txt"
grep -Eq ' U __cudaLaunchKernel$' "$out/mmq-undefined.txt"
flags="-O3 -std=c++17 -arch=sm_86 --extended-lambda -lineinfo -Xptxas=-v --frandom-seed=q6-word-group-host -DNINFER_GGUF_Q6_COOPERATIVE_DECODE=1 -DNINFER_Q6_WORD_GROUP_PROBE=1"
nvcc $flags -I"$probe" -I"$out/generated" -I"$objects/generated" \
    -Isrc -Iinclude -Ithird_party/ggml-quants/include -Ithird_party/ggml-quants/src \
    -Ithird_party/ggml-quants/src/ggml-cuda --generate-dependencies-with-compile \
    --dependency-output "$out/main.d" -c "$probe/main.cu" -o "$out/main.o" \
    > "$out/compile-main.log" 2>&1
nvcc -arch=sm_86 "$out/main.o" "$objects/vector.o" "$objects/mmq.o" \
    -Xlinker=--wrap=__cudaLaunchKernel -lcuda -o "$out/probe" > "$out/link.log" 2>&1
cuobjdump --dump-resource-usage "$out/probe" > "$out/resources-linked.txt"
nm -C "$out/probe" > "$out/symbols-linked.txt"
grep -q '__wrap___cudaLaunchKernel' "$out/symbols-linked.txt"
sha256sum "$out/probe" "$out/main.o" "$objects/vector.o" "$objects/mmq.o" \
    "$out/main.d" "$out/generation.json" "$out/toolchain.txt" > "$out/output-sha256.txt"
