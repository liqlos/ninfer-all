# NInfer ggml block-quant sources

These files originate from [llama.cpp](https://github.com/ggml-org/llama.cpp),
commit [`a25c9865fe03c954c93fd755b5d79ae86ba99750`](https://github.com/ggml-org/llama.cpp/commit/a25c9865fe03c954c93fd755b5d79ae86ba99750),
`ggml/include/` and `ggml/src/`, under the MIT license; see [LICENSE](LICENSE). NInfer locally
modifies `ggml-cuda/mmq.cuh` to optionally write native Q4_K/Q6_K MMQ results directly to BF16
outputs, including BF16 residual accumulation. The remaining vendored files are unchanged.

NInfer compiles ggml's integer tensor-core matrix multiplication (`ggml-cuda/mmq.cuh`) and its
dequantizers through the bridge in `src/ops/linear/gguf/`, as a separate archive, and keeps only the
headers that code includes. Products of up to eight tokens use NInfer's own kernel
(`ggml_bridge_vec.cuh`), which reads the grids and block structs from `ggml-common.h`.
