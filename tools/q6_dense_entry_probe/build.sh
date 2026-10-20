set -eu
cd /src
mkdir -p /build/q6-dense-entry
python3 /lab/source/tools/q6_dense_entry_probe/generate.py --source-root /src --output-root /build/q6-dense-entry/generated --self-test > /build/q6-dense-entry/generated.json
flags="-O3 -std=c++17 -arch=sm_86 --extended-lambda -lineinfo -Xptxas=-v -Isrc -Iinclude -Ithird_party/ggml-quants/include -Ithird_party/ggml-quants/src -Ithird_party/ggml-quants/src/ggml-cuda"
nvcc $flags --frandom-seed=q6-dense-entry-off -c /lab/source/tools/q6_dense_entry_probe/variant_off.cu -o /build/q6-dense-entry/off.o > /build/q6-dense-entry/compile-off.log 2>&1
nvcc -I/build/q6-dense-entry/generated $flags --frandom-seed=q6-dense-entry-on -c /lab/source/tools/q6_dense_entry_probe/variant_on.cu -o /build/q6-dense-entry/on.o > /build/q6-dense-entry/compile-on.log 2>&1
nvcc -I/build/q6-dense-entry/generated/support $flags --frandom-seed=q6-dense-entry-main -c /lab/source/tools/q6_dense_entry_probe/main.cu -o /build/q6-dense-entry/main.o > /build/q6-dense-entry/compile-main.log 2>&1
nvcc -arch=sm_86 /build/q6-dense-entry/main.o /build/q6-dense-entry/off.o /build/q6-dense-entry/on.o -o /build/q6-dense-entry/probe -lcuda > /build/q6-dense-entry/link.log 2>&1
