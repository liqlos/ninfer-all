# Head input census

This is a source-only diagnostic extension of `mtp_proposal_census`. It is default OFF under
`NINFER_LAB_HEAD_INPUT_CENSUS=0`. It has not been compiled, linked, or run on a GPU. It is not a
head implementation, sampler change, timing experiment, or qualified arithmetic certificate.

## Native seam and lifetime

`gguf_linear.cpp` calls the public `gguf::vector_product` in another translation unit.
`ggml_bridge.cu` converts its `activation` argument into `VecArgs.qs` and
`VecArgs.ds = activation + align16(K*T)`. For K5120 and T1/T4, that offset is exactly K*T.
The wrapper therefore observes the actual native packed operand without reading guessed CUDA
launch arguments or rebuilding Q8 from BF16. `compile.py --native-preflight` must first prove
the exact external symbol, declaration, PLT32 relocation, native definition, non-LTO objects,
and unchanged native dependencies. It refuses to infer success from source alone.

The wrapper requires Q6_K, K5120, N248320, row stride4200, T4 target or T1 proposal, BF16 Store,
column stride248320, the exact frame output pointer, one stream, and one shared head weight
pointer. A target head followed by three proposal heads must be intercepted in every captured
body. Pointer equality establishes shared live storage, not the fixture's content hash.
Fixture/artifact identity must be supplied by the runtime protocol before using row norms.

Each snapshot kernel is enqueued immediately after the original vector_product returns.
It reads inputs and output on the same stream before subsequent workspace reuse or logits
overwrite. The original target argmax, proposal argmax, acceptance, and all forwards still run.
Graph construction records four snapshot nodes. Replay runs those nodes with device-side
epoch/round control; no host callback is inserted in the graph. GPU memory and pinned mirrors
are allocated outside stream capture. The old compact observer supplies unchanged round joins,
full logical-vocabulary top2 checks, native acceptance checks, and count-update checks.

After replay, only the newly written sample regions are copied to pinned host memory.
The host reads them after the existing DeviceContext synchronize. No explicit extra synchronize, event
wait, model forward, head evaluation, or request is introduced. Nodes become no-ops after the
bounded window. Large D2H copies stop after round17. Tiny compact-state copies retain the old
observer behavior. Native SequenceState is read only on its Program thread before replay;
frontier, anchor, ingress drafts, monotonic ledger prefixes, and deltas are bound to device
records. A request/result identity and final OFF/ON output equivalence remain runtime gates.

## Causal window and representation

Record i contains Q0/Q1/Q2 produced after acceptance in round i+1 and P0/P1/P2/P3 from round
i+2. Round1's incoming prefill proposals are omitted. All seven actual head vectors and all
seven physical BF16 outputs are retained, including physically computed but logically invalid
tail columns. `extent` determines which columns are semantically valid; no padded output is
used as an actual proposal or target decision. Mathematically finite paid padding vectors can
be evaluated separately as auxiliary anchors, but are not counted as valid proposals.

One sample contains a32-byte binding header,5120 signed quant bytes,160 native half2 words,
and248320 raw BF16 outputs. The first half of each half2 is the native scale; its second half
is preserved even though the Q6 dot does not use that sum. The effective vector is exactly
`x[k] = int8(qs[k]) * fp16(ds[k/32].x)`, a dyadic number. Hidden BF16 tensors are not substitutes.

Penalty counts have **248077 logical entries**, not248320 physical entries. Their snapshot is
taken before the first saved P acceptance, after its Q anchors were produced. Subsequent bases
are reconstructed by incrementing only each native licensed span. P column j overlays the
current drafts[0:j]. The counterfactual adjusted Q at step j uses the same base and its own
proposal prefix. Native Q policy remains raw argmax. The current observer deliberately closes
scope to temperature0,top_k1,presence1.5,frequency0,no grammar,C1,FULL fixedK3,single SM86.

The device payload and pinned mirror are each57,264,704 bytes (54.612MiB), plus the existing
compact State. The binary has a48-byte header,992,308 count bytes and16*7*502432 sample bytes:
57,264,740 bytes. The D2H total is the sample bytes plus one padded count header, not the whole
54.6MiB on every round. Snapshot reads and writes each add approximately one payload's size;
four extra kernel launches per graph replay are diagnostic overhead, never head speedup. The
unchanged compact observer also retains its six reduction/control launches and their reads.
ON requires an existing absolute `NINFER_HEAD_INPUT_CENSUS_DIR`; output uses exclusive create.

## Why keep full logits

Seven Q8 vectors alone are only39.375KiB/linked round (630KiB over16) and reveal residual and
coefficient inflation. They cannot establish row pruning: the same vector geometry permits
both easy and hard head weight sets, margins, and penalty states. A mean row norm is not a
certificate for any row. A bad average cosine is not a valid class-wide rejection theorem.

A negative oracle-threshold screen can use three full Q BF16 arrays plus four native kth
threshold keys and packed inputs: roughly24MiB over16 including initial counts. Such a capture
must add and qualify a top-k reduction, and cannot replay arbitrary causal seed scores or
validate every P envelope. The present54.6MiB option is the smallest straightforward raw-copy
capture that supports all of those checks without implementing another reduction. The old
top2 observer alone does not supply the required arrays. This is a chosen simplicity tradeoff,
not a claim that54MiB is an information-theoretic minimum.

## Arithmetic and CPU gates

For source Q6 vector arithmetic let F_i(x) be the native FP32 value before BF16 output, and
R_i an independently certified outward norm of the exact decoded Q6 row. The source operation
DAG gives `|F_i(x)-w_i*x| <= gamma13 R_i ||x||`, with FP32 gamma13=13u/(1-13u).
This still requires a separately bound native instruction/flags proof; output agreement does
not prove the envelope. This observer cannot recover F from a BF16 output.

Each finite BF16 anchor b yields its pure closed rounding cell [L(b),U(b)]. The FP32 arithmetic
envelope is added separately in the formula below. Cells are asymmetric at powers of two. Closed endpoints conservatively
cover ties, including signed zero. For `r=x-X alpha`, a valid algebraic upper is

`sum(alpha_m>=0 ? alpha_m*U_m : alpha_m*L_m)
 + R_i*(||r|| + gamma13*(||x|| + sum |alpha_m| ||x_m||))`.

All construction/evaluation rounding must be enclosed. A residual obtained from an approximate
least-squares alpha is still usable only after enclosing the actual `x-X alpha`; a small SVD
residual report is not itself outward. Apply native BF16 rounding monotonically to the bound,
then the exact native presence/prefix processor and lower-global-ID tie ordering. Full P raw
argmax is also an observable path today and must be included or explicitly retained. Scoring
and full-logprob consumers retain a full head path.

`proof.py` validates capture shape/epoch/role/frontier/weight identity, full logical Q/P top2
against the compact native observer, exact current processed score bits, native acceptance,
licensed count deltas, and effective Q8 reconstruction. It reports all P/Q pairwise drift norms
and drift divided by the current P norm, plus clearly labeled oracle top1 score bits. It reports least-squares geometry for
G1, G5 contiguous1024, G20 contiguous256, and G32 actual lane slices. These NumPy calculations
are expressly nonrigorous diagnostics. They neither establish a row certificate nor fabricate
FP32 subgroup anchor values, which the current external seam cannot capture.

After this gate, bind the original head fixture and compute outward per-row norms with one
paid CPU pass. Evaluate bounds on held-out linked records, both oracle thresholds and causal
thresholds. A captured P winner/kth score is allowed only as an optimistic negative filter.
Actual thresholds must come from kth lower intervals or native scores of IDs selected before
seeing P. Do not seed with the target winner. A preselected seed beating draft[j] proves a
rejection no later than j and can make columns to its right irrelevant; otherwise bonus P3
still needs a certificate. Count union of retained rows across required T4 columns and actual
tiles/transactions, not average surviving columns. Charge seed rows, all fallback reads, cache
write/read, norms, bounds, compaction, launches, and three unchanged MTP head passes.

For a stronger negative test of continuous norm+slab bounds, use a feasible exact anchor vector a
inside the interval slabs and Gram G=X^T X. With dependent anchors require a in range(G), use
the Moore-Penrose inverse G+, and set alpha=G+X^T x and r=x-X alpha. The continuous-ball support is
`alpha^T a + sqrt(R_i^2-a^T G+ a)*||r||` for the orthogonal projection coefficients alpha.
Feasibility, rank and evaluation must themselves be rigorously enclosed. A negative witness
uses the lower outward endpoint of this support, never its upper endpoint. This lower bounds
the best upper bound from that continuous relaxation; survivors under an oracle threshold can
reject that relaxation. It does not reject methods using discrete Q6 codes or additional row structure. Full logits
alone do not qualify native rounding, an alternative trajectory, or whole-request speedup.

## Explicit next gates

`python -B compile.py` and `python -B proof.py` perform source/format checks only.
`compile.py --native-preflight` is read-only against the existing Linux native build.
`compile.py --compile-only` compiles observer objects, prints separate unexecuted link recipes,
and never links or launches a service. Actual compilation, frozen-link qualification, fresh
OFF/ON runtime/output binding, capture analysis and any further integration belong to the
parent experiment owner. Unsupported requests retain native forwarding and inactive observers.
