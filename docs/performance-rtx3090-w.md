# RTX 3090 GGUF kernels

Three opt-in SM86 paths preserve the existing weight and arithmetic formats:

| CMake option | Change |
|---|---|
| `NINFER_GGUF_Q4_COOPERATIVE_DECODE` | Share packed Q4_K loads between warp lanes. |
| `NINFER_GGUF_Q6_COOPERATIVE_DECODE` | Share packed Q6_K loads between warp lanes. |
| `NINFER_GGUF_MMQ_STORE_FUSION` | Store BF16 directly from MMQ, avoiding a temporary FP32 plane and conversion pass. |

All default to `OFF` and require CUDA architecture 86 when enabled. Cooperative decoding handles
up to eight activation columns. For native MMQ-compatible shapes, fusion requires Q4_K/Q6_K,
more than eight activation columns, rows divisible by 128, BF16 store/accumulate, no input gathers,
and no stream-K fixup. Other cases retain the original path.
Activation quantization stays Q8_1; accumulation and residual addition stay FP32 with BF16 rounding on store.

This branch isolates the kernels from the historical B0 import, tokenizer and HTTP/API fixes.
It remains based on upstream `d8225026`; it has not been ported to current upstream master.
Current-tree build, runtime and performance qualification remain pending.

## Build and check

The earlier qualification used CUDA 13.1.115 / Ubuntu 24.04. The narrowed branch is not yet built.
To build and check it after installing the project dependencies:

```sh
cmake -S . -B build-w -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DNINFER_BUILD_APPS=ON -DBUILD_TESTING=ON \
  -DNINFER_GGUF_Q4_COOPERATIVE_DECODE=ON \
  -DNINFER_GGUF_Q6_COOPERATIVE_DECODE=ON \
  -DNINFER_GGUF_MMQ_STORE_FUSION=ON
cmake --build build-w --parallel 4 --target ninfer-serve ninfer_tests
ctest --test-dir build-w -R '^ninfer_linear_gguf(_mmq_store)?_test$' --output-on-failure
```

The [archived GPU qualification](../benchmarks/rtx3090-w/provenance/focused-port-qualification.json)
used the previous focused port with the B0 import/API prerequisites. It passed:
independent FP64 GGUF checks; 96 bitwise BF16 store/accumulate cases at K=512/5120 with padding
guards and unsupported-route rejection; seven API checks; four complete archived outputs and token counts.
The kernel and test files are unchanged from that port; the complete current branch has not
been requalified after removing those prerequisites. A CUDA skip is not a pass. To replay the
archived requests against a compatible server:

```sh
python benchmarks/rtx3090-w/check-outputs.py --url http://127.0.0.1:18042/v1/chat/completions
```

The focused port has no separate timing estimate. The [5.407% latency result](../benchmarks/rtx3090-w/README.md)
belongs to the archived W source; other contexts, models, quants and GPUs require separate measurements.
