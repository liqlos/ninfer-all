# Independent codec check: real Q4_K and Q6_K tiles

Scope: CPU only, read-only mmap of the installed NINFER artifact. No GPU workload, runtime change, source changes to the main experiment, or compressed checkpoint was produced. The native-library probe was run in WSL against the installed `libzstd.so.1` and `liblz4.so.1`; each candidate block was immediately decompressed and byte/symbol-checked.

## Artifact and sampling

The source artifact is the active Q4_K_M proposal reported by `results/analysis.json`, 17,399,607,296 bytes. The script reads the NINFER header and directory, derives the aligned tensor payload start as 249,856 bytes, and mmaps the model. It sampled 4 disjoint regions per tensor, 512 GGUF blocks per region, and split each region into 32-block independent tiles. That is 64 exact 4,608-byte Q4_K tiles or 6,720-byte Q6_K tiles per tensor. Tested tensors: `weight/000000` (Q6_K layer 0 GDN Q/K/V), `weight/000001` (Q4_K layer 0 MLP gate/up), `weight/000060` (Q4_K layer 32 GDN Q/K/V/z), and `weight/000746` (Q4_K layer 63 attention/output). The JSON retains each tensor role and all codec measurements.

## Measured 32-block tiles

Ratios include an 8-byte tile record, raw fallback, and actual compressed bytes. Values above 1.0 mean that every tile fell back to raw because compression did not repay its framing cost.

| Tensor | Format | zstd 1 / 3 / 9 | LZ4 | zlib 1 raw | zlib 1 byte-shuffle | Exact tile Huffman | Best Rice/Golomb |
|---|---:|---:|---:|---:|---:|---:|---:|
| `weight/000000` | Q6_K | 1.0012 | 1.0012 | 1.0012 | 1.0012 | 2.34% saved | 10.3% larger than low-nibble plane |
| `weight/000001` | Q4_K | 1.0017 | 1.0017 | 0.9926 | 0.9894 | 1.77% saved | 12.0% larger |
| `weight/000060` | Q4_K | 1.0013 | 1.0017 | 0.9902 | 0.9882 | 2.10% saved | 12.1% larger |
| `weight/000746` | Q4_K | 1.0010 | 1.0017 | 0.9888 | 0.9882 | 2.27% saved | 11.9% larger |

Huffman uses one static canonical codebook per tensor (16 bytes for Q4_K and 64 bytes for Q6_K), independent per-tile streams, 4-byte length per tile, 16-byte payload alignment, unchanged scale/min/float metadata, and a raw fallback. All 64 tiles per tensor round-tripped exactly. Q4_K packing was reconstructed byte-for-byte; Q6_K's 6-bit code values were reconstructed into both original `ql` and `qh` planes byte-for-byte. Scale/min fields were copied unchanged and never entropy-coded. This is a format-feasibility experiment, not a production packer or GPU decode benchmark.

The most likely near-term codec candidate remains ANS, not these generic methods. The sibling exact ANS results already in `results/ans-weight-*.csv` cover the full tensors: for 256-block tiles, measured savings span 2.22–3.45% depending on tensor and 1/8/32 ANS substreams; for 64-block tiles they span 0.14–3.26%. Four-block tiles expand by 0.8–1.4%; one-block tiles expand by 5.56%. Larger tiles repay framing and improve entropy efficiency, but decouple from a 32-lane slice and increase each decode unit's work. The measured code-plane entropy for Q4_K was 3.84–3.89 bits/symbol, so there is little headroom for any code-only entropy codec to beat a roughly 2–3% total tensor reduction once uncompressed scales and tile metadata are included.

## What the results say about the other families

- **Zstd/LZ4 and byte shuffle:** independent 32-block tiles contain too little repeated byte context. They correctly raw-fallback. At larger chunk sizes these general codecs may find more correlation, but that would reduce random access and require larger independent decode units; they do not currently justify a fused GEMV path.
- **Huffman:** exact and somewhat useful for storage, but variable-length code lengths and symbol-by-symbol dependency make lane-parallel decode awkward. At these Q4/Q6 distributions the table/framing/alignment cost consumes much of the small entropy gain.
- **Rice/Golomb:** direct unsigned nibble symbols are a poor match; measured optimal `k` still needs about 4.48 bits per symbol on Q4 and 4.41 on the Q6 low-nibble plane before headers, versus 4 raw bits. Signed remapping or delta coding could be tested separately, but it must first show structure in the actual quant layout.
- **Bitmap/top-k escape:** the existing `analyze.py` candidate maps the seven most common Q4 symbols and escapes the rest; its escape fraction leaves no useful fixed-rate saving. A bitmap format is promising for fields with a heavily skewed bit distribution, but that property is not established for these packed quant codes. ZipServ's TCA triple bitmap is specifically for skewed BF16 exponents, not an INT4 code codec.
- **FSE/tANS/rANS:** these are still the only tested family with measured tile-local gains beyond Huffman. A shared codebook per tensor and independently indexed tile streams is a plausible storage design; best CPU samples use 64 or 256 GGUF blocks per tile. The later fused Q4_K GEMV probe did execute the vector dot path with decode fused into its CTA. It preserved exact weights and outputs, but saved only 1.76% including index/LUT and took about 3.92 ms versus 0.057 ms for the raw baseline at T=1 (roughly 69× slower); T=2/3/4/8 showed similarly large regressions. That rules out this particular 32-stream rANS schedule for inference speed, while not ruling out a fundamentally different parallel entropy decoder.

## Research implications for sm_86 / one-request latency

[ZipServ (ASPLOS 2026)](https://arxiv.org/abs/2603.17435) is the relevant Chinese research reference for fixed-rate bitmap decode fused into a GEMM. Its paper attributes the redundancy to BF16 exponent distributions and explains why ordinary Huffman/ANS streams diverge under SIMT. The public repo states Ampere or newer and Apache-2.0, so the RTX 3090 architecture floor is compatible. However, the authors report RTX 4090, L40S, and RTX 5090 evaluation, not RTX 3090; the published bitmap and `LoadBF16FragWithTripleBitmap_Optimized` reconstruct BF16 sign/exponent/mantissa and are not reusable as-is for Q4_K/Q6_K. Its larger GEMM throughput figures do not imply single-token SIMT GEMV speedup.

[Approaching Shannon Bound with Lossless LLM Weight Compression (ISCA 2026)](https://arxiv.org/abs/2606.15789) directly evaluates INT4 and group-quantized formats and describes a shared per-layer codebook with independently indexed ANS tile substreams. This supports the current codec direction. But the paper's end-to-end gain is primarily increased feasible batch size under fixed GPU memory, with throughput/TPOT metrics; it is not direct evidence for lower C=1 token latency. It does not identify this NINFER GGUF Q4_K/Q6_K layout or RTX 3090 as the target configuration.

NVIDIA's [nvCOMP Device API](https://docs.nvidia.com/cuda/nvcomp/device_api.html) exposes warp-group device decompression for ANS, Zstd, LZ4, deflate and other algorithms, but labels the API experimental and requires group/shared-memory/global-scratch buffers. A call that materializes decompressed output still adds a write/read path; fitting it into the current row-split SIMT GEMV would require explicit decode-to-register/shared-memory integration and measurement on the installed nvCOMP/CUDA stack. Availability of that API alone is not evidence of a cheap fused kernel.

[DietGPU](https://github.com/facebookresearch/dietgpu) is archived since June 2026 and MIT-licensed. Its README describes byte-oriented rANS tested on V100/A100, says kernel fusion was planned, and explicitly says byte statistics miss higher-level structure. It is useful as a licensed algorithm/code reference, not an already-fused NInfer implementation or sm_86 benchmark. The ZipServ repository is Apache-2.0. No external code was copied into the experiment.

## Recommendation

Do not spend further CUDA integration effort on the tested zstd, LZ4, Huffman, Rice/Golomb, bitmap, or 32-stream rANS variants for this artifact. The fused GEMV already supplies the relevant kernel evidence: exact Q4_K payload/output with a 1.76% packed-size reduction, but around 69× T=1 latency regression. The observed cost is consistent with only one warp performing serial ANS state transitions while the other CTA lanes wait, followed by shared-memory staging; the compressed bytes saved are far too few to repay that work. Treat this as a negative finding about this rANS schedule and standalone projection path, not a universal result for all lossless codecs or fused operators. Another GPU trial is justified only for a substantially different decoder with independent symbol/bitplane work across all lanes and a credible path to beat the raw vector kernel; otherwise stop at the modest resident-memory saving measured here.

Reproduce the CPU codec probe from Windows PowerShell:

```powershell
wsl.exe -- python3 /mnt/c/AI/tuning/ninfer-lossless-20260928/alternatives-native.py
```

The full measurements and library version numbers are in [alternatives-codecs.json](alternatives-codecs.json); the rerunnable CPU script is [alternatives-native.py](alternatives-native.py).
