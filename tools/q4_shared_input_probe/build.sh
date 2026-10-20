set -eu
test -z "${NVCC_PREPEND_FLAGS-}${NVCC_APPEND_FLAGS-}"
cd /src
out=/build/q4-shared-input
probe=/lab/source/tools/q4_shared_input_probe
test ! -e "$out/generated/source_receipt.json"
mkdir -p "$out"
python3 -B "$probe/generate.py" --source-root /src --output-root "$out/generated" > "$out/generation.json"
flags="-O3 -std=c++17 -arch=sm_86 --extended-lambda -lineinfo -Xptxas=-v --frandom-seed=q4-shared-input-probe --objdir-as-tempdir -DNINFER_GGUF_Q4_COOPERATIVE_DECODE=1 -DNINFER_GGUF_Q6_COOPERATIVE_DECODE=1 -DNINFER_GGUF_MMQ_STORE_FUSION=1 -I$out/generated -I$probe -Isrc -Iinclude -Ithird_party/ggml-quants/include -Ithird_party/ggml-quants/src -Ithird_party/ggml-quants/src/ggml-cuda"
printf '%s\n' "$flags" > "$out/common-flags.txt"
nvcc --version > "$out/toolchain.txt"
for variant in native off on; do
    mkdir -p "$out/keep-$variant"
    printf 'nvcc %s --cubin --keep --keep-dir %s --generate-dependencies-with-compile --dependency-output %s %s -o %s\n' "$flags" "$out/keep-$variant" "$out/$variant.d" "$probe/variant_$variant.cu" "$out/$variant.cubin" > "$out/command-$variant.txt"
    nvcc $flags --cubin --keep --keep-dir "$out/keep-$variant" --generate-dependencies-with-compile --dependency-output "$out/$variant.d" "$probe/variant_$variant.cu" -o "$out/$variant.cubin" > "$out/compile-$variant.log" 2>&1
    cuobjdump --dump-resource-usage "$out/$variant.cubin" > "$out/resources-$variant.txt"
    cuobjdump --dump-sass "$out/$variant.cubin" > "$out/sass-$variant.txt"
done
python3 -B "$probe/sass_gate.py" --build-root "$out" > "$out/gate.json"
