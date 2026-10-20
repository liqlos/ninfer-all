set -eu
test -z "${NVCC_PREPEND_FLAGS-}${NVCC_APPEND_FLAGS-}"
cd /src
probe=/lab/source/tools/q4_folded_ffn_probe
out=/build/q4-folded-ffn/pack
mkdir -p "$out/keep-on" "$out/keep-off"
python3 -c 'import sys; assert sys.version_info[:2] >= (3, 11), sys.version'
python3 "$probe/generate_pack.py" --source-root /src --output-root "$out/generated" --self-test > "$out/generation.json"
flags="-O3 -std=c++17 -arch=sm_86 --extended-lambda -lineinfo -Xptxas=-v -I/build/q4-folded-ffn/pack/generated -Isrc -Iinclude -Ithird_party/ggml-quants/include -Ithird_party/ggml-quants/src -Ithird_party/ggml-quants/src/ggml-cuda"
nvcc $flags -DNINFER_GGUF_FFN_FOLDED16_PACK=1 --frandom-seed=q4-folded-ffn-pack --objdir-as-tempdir --keep --keep-dir "$out/keep-on" -c "$probe/variant_pack.cu" -o "$out/variant-on.o" > "$out/compile-on.log" 2>&1
nvcc $flags -DNINFER_GGUF_FFN_FOLDED16_PACK=0 --frandom-seed=q4-folded-ffn-pack --objdir-as-tempdir --keep --keep-dir "$out/keep-off" -c "$probe/variant_pack.cu" -o "$out/variant-off.o" > "$out/compile-off.log" 2>&1
cuobjdump --dump-resource-usage "$out/variant-on.o" > "$out/resources-on.txt"
cuobjdump --dump-sass "$out/variant-on.o" > "$out/sass-on.txt"
nm -C "$out/variant-off.o" > "$out/symbols-disabled-variant.txt"
