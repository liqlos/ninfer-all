# W v6 benchmark: RTX 3090

Archived candidate W v6 versus baseline B0: **5.4070% lower request latency**,
95% Student-t interval **5.0038–5.8102%**, from 20 A/B pairs (40 server runs, 480 timed requests).
Ten pairs were baseline-first and ten candidate-first. W enabled cooperative Q4/Q6 decoding
and BF16 MMQ store fusion. SwiGLU fusion was OFF. This measures the combined artifact.

## Method and data

- RTX 3090 24 GiB / SM86; CUDA 13.1.115; Docker on WSL Ubuntu 24.04 / Windows.
- Qwen3.8-27B Q4_K_M; K8/V8, C1, capacity 65,536, MTP3, prefill chunks 1,024.
- RU/EN/code/JSON prompts: actual 4,084–4,089 input tokens, 256 output tokens, three repeats, 1,024-token warmup.
- Latency includes TTFT and streamed generation over loopback HTTP; excludes startup, tokenization and warmup.
- Each category uses its median of three repeats; paired percentage reductions receive equal category weight.
  Four pairs form each block; five block means determine the estimate and t interval (df=4).
- Request/output/usage equality gates passed.
  See the [schedule](provenance/schedule.json), [preregistration](provenance/preregistration.json),
  [analysis](provenance/analysis.json) and [independent raw audit](provenance/independent-raw-audit.json).

Run `python reproduce.py` to recompute the result from captured records. The 40 gzip JSONL files
preserve the original requests, outputs and timings byte-for-byte. Full inference reproduction also
requires compatible weights/conversion; the pinned model SHA-256 is in the preregistration.

## Measured source

[historical-w-source.patch.gz](historical-w-source.patch.gz) is the exact B0-to-W diff, including
inactive experiments omitted from the focused port. From this repository root:

```sh
git worktree add --detach ../ninfer-historical-w 873d5e73d693793d1d64c5798c04295a7384ceb4
gzip -dc benchmarks/rtx3090-w/historical-w-source.patch.gz | git -C ../ninfer-historical-w apply --index
git -C ../ninfer-historical-w write-tree
```

The tree must be `6dd348539c859625ea4b0d47a3855bc5bbf61b77`. Use the preregistration's build options.
The [manifest](manifest.json) records source IDs, byte counts and hashes.

## Scope

The result applies to these artifacts and this 4K/256 workload. Capacity 65,536 does not establish
long-context speed. The interval treats five block means as exchangeable; chronology does not
prove independence, and one pair spans a pause/reboot. Sensitivity checks remain in the raw audit.
B0 lacks an option receipt, so individual flags' contributions cannot be inferred.
The [focused port](../../docs/performance-rtx3090-w.md) passed GPU correctness checks but has no separate timing estimate.

Historical records and license notices are preserved byte-for-byte. Weights, binaries, private
configuration and host identities are excluded. NInfer is Apache-2.0; vendored ggml-quants is MIT
(see `licenses/`). Related work: [ggml CUDA](https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-cuda)
and [Marlin](https://github.com/IST-DASLab/marlin).
