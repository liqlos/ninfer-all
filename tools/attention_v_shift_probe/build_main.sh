set -eu
test -z "${NVCC_PREPEND_FLAGS-}${NVCC_APPEND_FLAGS-}"
cd /src
probe=/lab/source/tools/attention_v_shift_probe
objects=/build/attention-v-shift
out="$objects/host"
mkdir -p "$out"
nvcc --version > "$out/toolchain.txt"
grep -Eq 'V13[.]1[.]115([[:space:]]|$)' "$out/toolchain.txt"
cmp "$out/toolchain.txt" "$objects/toolchain.txt"
python3 -B "$probe/generate_host.py" --source-root /src --probe-root "$probe" \
    --objects-root "$objects" --output-root "$out/generated" > "$out/generation.json"
for variant in off on; do
    nm -u "$objects/$variant.o" > "$out/$variant-undefined.txt"
    grep -Eq ' U __cudaLaunchKernel$' "$out/$variant-undefined.txt"
done
flags="-O3 -std=c++20 -arch=sm_86 --extended-lambda -lineinfo -Xptxas=-v --frandom-seed=attention-v-shift-host -Xcompiler=-ffp-contract=off"
printf '%s\n' "$flags" > "$out/host-flags.txt"
nvcc $flags -I"$probe" -I"$out/generated" -Isrc -Iinclude \
    --generate-dependencies-with-compile --dependency-output "$out/main.d" \
    -c "$probe/main.cu" -o "$out/main.o" > "$out/compile-main.log" 2>&1
if grep -Eq 'prompt_i8_fast[.]cuh|attention_v_shift_generated[.]cuh|ops/op_tester[.]h|causal_cache[.]cpp' "$out/main.d"; then
    exit 1
fi
nvcc -arch=sm_86 "$out/main.o" "$objects/off.o" "$objects/on.o" \
    -Xlinker=--wrap=__cudaLaunchKernel -lcuda -o "$out/probe" > "$out/link.log" 2>&1
cuobjdump --dump-resource-usage "$out/probe" > "$out/resources-linked.txt"
nm -C "$out/probe" > "$out/symbols-linked.txt"
grep -q '__wrap___cudaLaunchKernel' "$out/symbols-linked.txt"
python3 -B "$probe/generate_host.py" --source-root /src --probe-root "$probe" \
    --objects-root "$objects" --check-only > "$out/post-link-validation.json"
sha256sum "$out/probe" "$out/main.o" "$objects/off.o" "$objects/on.o" \
    "$out/main.d" "$out/generation.json" "$out/toolchain.txt" "$out/host-flags.txt" \
    > "$out/output-sha256.txt"
