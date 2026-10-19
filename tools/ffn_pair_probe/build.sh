set -eu
cd "${NINFER_FFN_SOURCE_ROOT:-/lab/source}"
build_dir=${NINFER_FFN_PAIR_BUILD_DIR:-/build/ffn-pair/probe}
binary=${NINFER_FFN_PAIR_BINARY:-/lab/artifacts/experimental/ffn-pair-probe}
pair_object="$build_dir/pair.o"
mkdir -p "$build_dir" "$(dirname "$binary")"
nvcc -O3 -std=c++17 -arch=sm_86 --extended-lambda -Xptxas=-v -DNINFER_GGUF_FFN_PAIR_MMQ_FUSION=1 ${NINFER_FFN_PAIR_EXTRA_FLAGS:-} -Isrc -Iinclude -Ithird_party/ggml-quants/include -Ithird_party/ggml-quants/src -c src/ops/linear/gguf/ggml_mmq_ffn_pair.cu -o "$pair_object" > "$build_dir/pair.stdout.log" 2> "$build_dir/pair.stderr.log"
nvcc -O3 -std=c++17 -arch=sm_86 -Isrc -Iinclude -Ithird_party/ggml-quants/include -Ithird_party/ggml-quants/src -c tools/ffn_pair_probe/main.cu -o "$build_dir/main.o" > "$build_dir/main.stdout.log" 2> "$build_dir/main.stderr.log"
nvcc -O3 -std=c++17 -arch=sm_86 -Isrc -Iinclude -Ithird_party/ggml-quants/include -Ithird_party/ggml-quants/src -c tools/ffn_pair_probe/baseline.cu -o "$build_dir/baseline.o" > "$build_dir/baseline.stdout.log" 2> "$build_dir/baseline.stderr.log"
nvcc -arch=sm_86 "$build_dir/main.o" "$build_dir/baseline.o" "$pair_object" -o "$binary" -lcuda > "$build_dir/link.stdout.log" 2> "$build_dir/link.stderr.log"
