set -eu
test -z "${NVCC_PREPEND_FLAGS-}${NVCC_APPEND_FLAGS-}"
cd /src
probe=/lab/source/tools/attention_qk_x4_probe
out=/build/attention-qk-x4
mkdir -p "$out/keep-off" "$out/keep-on"
nvcc --version > "$out/toolchain.txt"
grep -F 'V13.1.115' "$out/toolchain.txt" > /dev/null
python3 -c 'import sys; assert sys.version_info[:2] >= (3, 11), sys.version'
python3 "$probe/generate.py" --source-root /src --output-root "$out/generated" --self-test > "$out/generation.json"
flags="-O3 -std=c++20 -arch=sm_86 --extended-lambda -lineinfo -Xptxas=-v --frandom-seed=attention-qk-x4 -I/build/attention-qk-x4/generated -Isrc -Iinclude"
printf '%s\n' "$flags" > "$out/common-flags.txt"
for variant in off on; do
    enabled=0
    if test "$variant" = on; then enabled=1; fi
    nvcc $flags -DNINFER_PROMPT_QK_B_X4=$enabled --objdir-as-tempdir --keep --keep-dir "$out/keep-$variant" --generate-dependencies-with-compile --dependency-output "$out/$variant.d" -c "$probe/variant.cu" -o "$out/$variant.o" > "$out/compile-$variant.log" 2>&1
    grep -F "$out/generated/attention_qk_x4_generated.cuh" "$out/$variant.d" > /dev/null
    if grep -F '/prompt_i8_fast.cuh' "$out/$variant.d" > /dev/null; then exit 1; fi
    cuobjdump --dump-resource-usage "$out/$variant.o" > "$out/resources-$variant.txt"
    cuobjdump --dump-sass "$out/$variant.o" > "$out/sass-$variant.txt"
    cuobjdump --dump-ptx "$out/$variant.o" > "$out/ptx-$variant.txt"
    nm -C "$out/$variant.o" > "$out/symbols-$variant.txt"
done
