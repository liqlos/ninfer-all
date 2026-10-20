set -eu
test -z "${NVCC_PREPEND_FLAGS-}${NVCC_APPEND_FLAGS-}"
cd /src
out=/build/q6-full-warp
probe=/lab/source/tools/q6_full_warp_probe
mkdir -p "$out"
python3 "$probe/generate.py" --source-root /src --output-root "$out/generated" --self-test > "$out/generated.json"
flags="-O3 -std=c++17 -arch=sm_86 --extended-lambda -lineinfo -Xptxas=-v --frandom-seed=q6-full-warp-probe -DNINFER_GGUF_Q6_COOPERATIVE_DECODE=1 -I/build/q6-full-warp/generated -Isrc -Iinclude -Ithird_party/ggml-quants/include -Ithird_party/ggml-quants/src -Ithird_party/ggml-quants/src/ggml-cuda"
printf '%s\n' "$flags" > "$out/common-flags.txt"
nvcc --version > "$out/toolchain.txt"
for variant in off on; do
    nvcc $flags --generate-dependencies-with-compile --dependency-output "$out/$variant.d" -c "$probe/variant_$variant.cu" -o "$out/$variant.o" > "$out/compile-$variant.log" 2>&1
    cuobjdump --dump-resource-usage "$out/$variant.o" > "$out/resources-$variant.txt"
    cuobjdump --dump-sass "$out/$variant.o" > "$out/sass-$variant.txt"
done
python3 "$probe/sass_gate.py" --build-root "$out"
