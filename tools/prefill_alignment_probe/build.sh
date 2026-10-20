set -euo pipefail
probe_source_root="${1:?source root required}"
probe_build_root="${2:?existing native build root required}"
probe_output_root="${3:?output directory required}"
probe_cuda_root="${4:-/usr/local/cuda}"
probe_source_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p -- "$probe_output_root"
c++ -O3 -DNDEBUG -std=gnu++20 -DNINFER_SM8X_COMPAT=1 -DUTF8PROC_STATIC -DSPDLOG_COMPILED_LIB \
    -Werror=return-type -Werror=implicit-fallthrough \
    -I"$probe_source_root/include" -I"$probe_source_root/src" \
    -isystem "$probe_source_root/third_party" \
    -isystem "$probe_source_root/third_party/spdlog/include" -isystem "$probe_cuda_root/include" \
    -c "$probe_source_directory/main.cpp" -o "$probe_output_root/main.o"
c++ -O3 -DNDEBUG "$probe_output_root/main.o" \
    -Wl,--start-group \
    "$probe_build_root/src/runtime/libninfer_engine.a" \
    "$probe_build_root/src/runtime/libninfer_runtime_support.a" \
    "$probe_build_root/src/models/libninfer_model_runtime.a" \
    "$probe_build_root/src/models/libninfer_model_loading.a" \
    "$probe_build_root/src/artifact/libninfer_artifact.a" \
    "$probe_build_root/src/calibration/libninfer_calibration.a" \
    "$probe_build_root/src/ops/libninfer_ops.a" \
    "$probe_build_root/src/ops/libninfer_ggml_quants.a" \
    "$probe_build_root/src/core/libninfer_core.a" \
    "$probe_build_root/src/text/libninfer_text.a" \
    "$probe_build_root/src/media/libninfer_media_decode.a" \
    "$probe_build_root/third_party/llama-jinja/libninfer_jinja.a" \
    "$probe_build_root/third_party/xgrammar/libninfer_xgrammar.a" \
    "$probe_build_root/third_party/spdlog/libninfer_third_party_spdlog.a" \
    -Wl,--end-group \
    -L"$probe_cuda_root/targets/x86_64-linux/lib" \
    -L"$probe_cuda_root/targets/x86_64-linux/lib/stubs" \
    -Wl,-rpath,"$probe_cuda_root/targets/x86_64-linux/lib" \
    -lcuda -lcudart -lcublas -lcublasLt -lculibos -lcudadevrt -lcudart_static \
    -lavformat -lavcodec -lavutil -lswscale -lpthread -ldl -lrt \
    -o "$probe_output_root/prefill_alignment_probe"
