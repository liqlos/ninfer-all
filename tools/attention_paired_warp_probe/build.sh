set -eu
test "$#" -eq 2
test -z "${NVCC_PREPEND_FLAGS-}${NVCC_APPEND_FLAGS-}"
source_root=$(cd "$1" && pwd)
out=$2
probe=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$out/keep-off" "$out/keep-on"
out=$(cd "$out" && pwd)
cd "$source_root"
nvcc --version > "$out/toolchain.txt"
grep -F 'V13.1.115' "$out/toolchain.txt" > /dev/null
python3 -B -c 'import sys; assert sys.version_info[:2] >= (3, 11), sys.version'
python3 -B "$probe/generate.py" --source-root "$source_root" --output-root "$out/generated" --self-test > "$out/generation.json"
flags=(-O3 -std=c++20 -arch=sm_86 --extended-lambda -lineinfo -Xptxas=-v --frandom-seed=attention-paired-warp "-I$out/generated" "-I$source_root/src" "-I$source_root/include")
printf '%s\n' "${flags[@]}" > "$out/common-flags.txt"
for variant in off on; do
    enabled=0
    generated=attention_native_generated.cuh
    if test "$variant" = on; then
        enabled=1
        generated=attention_paired_generated.cuh
    fi
    nvcc "${flags[@]}" -DNINFER_PROMPT_PAIRED_WARP=$enabled --objdir-as-tempdir --keep --keep-dir "$out/keep-$variant" --generate-dependencies-with-compile --dependency-output "$out/$variant.d" -c "$probe/variant.cu" -o "$out/$variant.o" > "$out/compile-$variant.log" 2>&1
    grep -F "$out/generated/$generated" "$out/$variant.d" > /dev/null
    if grep -F '/prompt_i8_fast.cuh' "$out/$variant.d" > /dev/null; then exit 1; fi
    cuobjdump --dump-resource-usage "$out/$variant.o" > "$out/resources-$variant.txt"
    cuobjdump --dump-sass "$out/$variant.o" > "$out/sass-$variant.txt"
    cuobjdump --dump-ptx "$out/$variant.o" > "$out/ptx-$variant.txt"
done
python3 -B "$probe/analyze.py" --source-root "$source_root" --build-root "$out" > "$out/gate.json"
