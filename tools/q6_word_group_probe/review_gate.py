import argparse
import collections
import json
import pathlib
import re
import sys

sys.dont_write_bytecode = True

import sass_gate as original


RAW_INSTRUCTION = re.compile(
    r"/\*(?P<pc>[0-9a-fA-F]+)\*/\s*(?P<pred>@!?(?:UP|P)(?:T|\d+)\s+)?"
    r"(?P<op>[A-Z][A-Z0-9_.]*)\s*(?P<args>[^;]*);")
REG = re.compile(r"(?:UR|R)\d+")
EXTRA_FLOAT = original.FLOAT_OPS + ("HADD", "HMUL", "HFMA")
MATRIX_WORK = EXTRA_FLOAT + original.UNPACK_OPS + ("IMMA", "LDG", "LDS", "STS", "CALL")
PARSER_FIX = (
    "Only vector_loop is overridden. A runtime K backedge must enclose the native "
    "16*T DP4A operations, no butterfly reduction, and immediately precede its BSYNC "
    "exit. Out-of-line decoder/reduction helper rejoins are not runtime K backedges. "
    "All instructions in the selected interval remain unchanged. Original run, "
    "pins, dependencies, spill/resource checks and mechanism comparisons are retained."
)


def require(condition, message):
    original.require(condition, message)


def operands(args):
    return [item.strip().removesuffix(".reuse") for item in args.split(",")]


def records(text):
    result = {}
    anchors = list(re.finditer(r"^\s*Function\s*:\s*(\S+)\s*$", text, re.MULTILINE))
    for index, anchor in enumerate(anchors):
        end = anchors[index + 1].start() if index + 1 < len(anchors) else len(text)
        result[anchor.group(1)] = [
            (int(m["pc"], 16), (m["pred"] or "").strip(), m["op"], m["args"].strip())
            for m in RAW_INSTRUCTION.finditer(text[anchor.end():end])
        ]
    for name, items in original.functions(text).items():
        require([(pc, op, args) for pc, _, op, args in result[name]] == items,
                "predicate-preserving parser changed original instruction stream")
    return result


def vector_loop(items, columns):
    by_pc = {pc: (op, args) for pc, op, args in items}
    candidates = []
    for end, op, args in items:
        target = re.fullmatch(r"0x([0-9a-fA-F]+)", args)
        if op != "BRA" or target is None:
            continue
        begin = int(target.group(1), 16)
        if begin >= end:
            continue
        region = [item for item in items if begin <= item[0] <= end]
        counts = original.histogram(region)
        if (counts.get("IDP.4A.S8.S8", 0) == 16 * columns
                and counts.get("SHFL.BFLY", 0) == 0
                and by_pc.get(end + 16, ("", ""))[0] == "BSYNC"):
            candidates.append((begin, end, region))
    require(len(candidates) == 1, "review parser cannot identify one bounded runtime K loop")
    return candidates[0]


def written_registers(op, args):
    fields = operands(args)
    if not fields or op.startswith(("ST", "BRA", "CALL", "RET", "BSSY", "BSYNC",
                                    "WARPSYNC", "BAR", "EXIT", "ISETP", "UISETP", "PLOP")):
        return []
    first = fields[1] if op.startswith("SHFL") else fields[0]
    if REG.fullmatch(first) is None:
        return []
    width = 1
    if op.startswith(("LDG", "LDS", "ULDC")):
        width = 4 if ".128" in op else 2 if ".64" in op else 1
    if ".WIDE" in op or op == "CS2R":
        width = 2
    prefix = "UR" if first.startswith("UR") else "R"
    number = int(first[len(prefix):])
    return [prefix + str(number + offset) for offset in range(width)]


def integer_prefix(items, lane, k):
    state = {}

    def value(token):
        token = token.removesuffix(".reuse")
        if token in ("RZ", "URZ"):
            return 0
        if token == "c[0x0][0x174]":
            return k
        if REG.fullmatch(token):
            return state.get(token)
        try:
            return int(token, 0)
        except ValueError:
            return None

    for _, predicate, op, args in items:
        fields = operands(args)
        destinations = written_registers(op, args)
        result = None
        if op == "S2R" and fields[1] == "SR_TID.X":
            result = lane
        elif op in ("MOV", "UMOV", "ULDC"):
            result = value(fields[1])
        elif op == "IMAD.MOV.U32":
            result = value(fields[3])
        elif op in ("SHF.R.S32.HI", "USHF.R.S32.HI", "SHF.R.U32.HI"):
            low, shift, high = (value(x) for x in fields[1:])
            if low == 0 and shift is not None and high is not None and 0 <= shift < 32:
                result = high >> shift
        elif op in ("LEA.HI", "ULEA.HI"):
            a, b, c, shift = (value(x) for x in fields[1:])
            if a == c == 0 and shift == 5 and b == k and k >= 0:
                result = b
        elif op == "LOP3.LUT" and fields[4:6] == ["0xc0", "!PT"]:
            a, b, c = (value(x) for x in fields[1:4])
            if a is not None and b is not None and c == 0:
                result = a & b
        elif op == "IADD3":
            values = [value(x) for x in fields[-3:]]
            if all(x is not None for x in values):
                result = sum(values)
        for destination in destinations:
            state[destination] = None
        if destinations and not predicate:
            state[destinations[0]] = result
    return state


def k_iterations(raw, loop):
    begin, end, _ = loop
    inside = [x for x in raw if begin <= x[0] <= end]
    exit_pc = end + 16
    increments = []
    for pc, _, op, args in inside:
        f = operands(args)
        if op == "IADD3" and len(f) == 4 and f[0] == f[1] and f[2:] == ["0x20", "RZ"]:
            increments.append((pc, f[0]))
    matches = []
    for increment_pc, counter in increments:
        for pc, _, op, args in inside:
            f = operands(args)
            if (op == "ISETP.GE.AND" and increment_pc < pc and len(f) == 5
                    and f[1] == f[4] == "PT" and f[2] == counter and REG.fullmatch(f[3])):
                matches.append((increment_pc, counter, pc, f[0], f[3]))
    require(len(matches) == 1, "runtime K induction and compare are ambiguous")
    increment_pc, counter, compare_pc, predicate, limit = matches[0]
    require(sum(counter in written_registers(op, args) for _, _, op, args in inside) == 1,
            "runtime K induction has additional writes")
    require(not any(limit in written_registers(op, args) for _, _, op, args in inside),
            "runtime K bound is modified")
    require(not any(pc > compare_pc and op.startswith(("ISETP", "PLOP"))
                    and operands(args)[0] == predicate for pc, _, op, args in inside),
            "runtime K predicate was overwritten")
    backedge = inside[-1]
    calls = [x for x in inside if x[2].startswith("CALL")]
    if backedge[1] == "@!" + predicate:
        require(not calls, "predicated K backedge also contains a call")
        exit_kind = "predicated_backedge"
    else:
        require(backedge[1] == "" and len(calls) == 1,
                "K exit is neither a predicated backedge nor a single exit call")
        call_pc, call_predicate, call_op, call_args = calls[0]
        require(call_pc == end - 16 and call_predicate == "@" + predicate
                and call_op == "CALL.REL.NOINC" and int(call_args, 16) == exit_pc,
                "K call is not the adjacent BSYNC exit")
        exit_kind = "predicated_call_to_adjacent_BSYNC"
    prefix = [x for x in raw if x[0] < begin]
    states = [integer_prefix(prefix, lane, 5120) for lane in range(128)]
    require(all(state.get(counter) == lane % 32 and state.get(limit) == 160
                for lane, state in enumerate(states)),
            "SASS prefix does not establish lane counter and K/32 bound for K=5120")
    iterations = [len(range(state[counter], state[limit], 32)) for state in states]
    require(set(iterations) == {5}, "K=5120 no longer has five native iterations")
    return {"counter": counter, "bound": limit, "increment_pc": hex(increment_pc),
            "compare_pc": hex(compare_pc), "increment": 32, "K": 5120,
            "slices": 160, "lanes_checked": 128, "iterations_each_lane": 5,
            "exit_kind": exit_kind, "exit_pc": hex(exit_pc)}


def vector_fp_shapes(items, columns):
    state = {}
    roots = []

    def get(token):
        token = token.removesuffix(".reuse")
        if token in ("RZ", "-RZ"):
            return ("zero", token)
        if REG.fullmatch(token):
            return state.get(token, ("live", token))
        return ("opaque",)

    for _, op, args in items:
        f = operands(args)
        dst = written_registers(op, args)
        expr = None
        if op == "IDP.4A.S8.S8":
            expr = (op, ("packed_code",), ("packed_activation",), get(f[3]))
        elif op == "I2F.S16":
            expr = (op, ("scale_s16",))
        elif op == "HADD2.F32":
            require(f[1] == "-RZ" and f[2].endswith(".H0_H0"),
                    "unexpected vector half-to-float lowering")
            expr = (op, ("literal_half",))
        elif op == "I2FP.F32.S32":
            expr = (op, get(f[1]))
        elif op in ("FMUL", "FFMA", "FADD"):
            source = [get(x) for x in f[1:]]
            if op == "FFMA" and f[0] == f[3] and source[-1] == ("live", f[0]):
                source[-1] = ("loop_accumulator",)
                roots.append((op, *source))
            expr = (op, *source)
        elif op in ("MOV", "IMAD.MOV.U32"):
            expr = get(f[1] if op == "MOV" else f[3])
        for reg in dst:
            state[reg] = ("opaque",)
        if dst and expr is not None:
            state[dst[0]] = expr
    require(len(roots) == 2 * columns, "vector FP accumulator roots are incomplete")

    def dp_depth(node):
        if node == ("zero", "RZ"):
            return 0
        require(node[0] == "IDP.4A.S8.S8", "unexpected DP4A accumulation dependency")
        return 1 + dp_depth(node[3])

    def inspect(node):
        if node[0] == "I2FP.F32.S32":
            require(dp_depth(node[1]) == 4, "vector signed-dot chain is not four DP4As")
        for child in node[1:]:
            if isinstance(child, tuple):
                inspect(child)

    for root in roots:
        inspect(root)
    return collections.Counter(json.dumps(root, separators=(",", ":")) for root in roots)


def reduction(raw, loop_end, columns):
    total = 10 * columns
    state = {}
    primary = []
    complete = {}
    seen = collections.Counter()
    first_pc = last_pc = None
    for pc, predicate, op, args in raw:
        if pc <= loop_end:
            continue
        f = operands(args)
        if op == "SHFL.BFLY":
            require(not predicate, "primary reduction butterfly is predicated")
            require(len(primary) < total, "reduction primary span contains too many butterflies")
            require(f[0] == "PT" and f[4] == "0x1f", "reduction shuffle width changed")
            source = state.get(f[2], ("leaf", f[2], (), ()))
            require(source[0] == "leaf", "reduction shuffle consumes a pending shuffle")
            offset = int(f[3], 16)
            require(offset == 16 >> len(source[2]), "reduction butterfly stage order changed")
            state[f[1]] = ("shuffle", source, offset)
            primary.append(pc)
            first_pc = pc if first_pc is None else first_pc
        elif op == "FADD" and primary:
            require(not predicate, "primary reduction FADD is predicated")
            a = state.get(f[1], ("leaf", f[1], (), ()))
            b = state.get(f[2], ("leaf", f[2], (), ()))
            if a[0] == "shuffle" and a[1] == b:
                previous, offset, direction = b, a[2], "shuffle_then_value"
            elif b[0] == "shuffle" and b[1] == a:
                previous, offset, direction = a, b[2], "value_then_shuffle"
            else:
                raise ValueError("reduction FADD does not combine an exact value and its butterfly")
            updated = ("leaf", previous[1], previous[2] + (offset,), previous[3] + (direction,))
            state[f[0]] = updated
            seen[previous[1]] += 1
            if len(updated[2]) == 5:
                complete[previous[1]] = updated
            last_pc = pc
            if sum(seen.values()) == total:
                break
        elif primary:
            for reg in written_registers(op, args):
                require(reg not in state, "non-reduction operation overwrites a live reduction register")
    require(len(primary) == total and len(complete) == 2 * columns
            and set(seen.values()) == {5}, "reduction does not contain complete five-stage trees")
    helpers = []
    for index, (pc, _, op, _) in enumerate(raw):
        if op != "SHFL.BFLY" or pc in primary:
            continue
        require(index > 0 and index + 1 < len(raw)
                and raw[index - 1][2] == "WARPSYNC"
                and raw[index + 1][2].startswith("RET"),
                "unclassified butterfly outside primary reduction")
        helpers.append(pc)
    require(len(helpers) == 1, "expected one out-of-line butterfly helper")
    signatures = collections.Counter(json.dumps((v[2], v[3])) for v in complete.values())
    return {"primary_span": [hex(first_pc), hex(last_pc)], "primary_butterflies": total,
            "out_of_line_helper_butterflies": [hex(pc) for pc in helpers],
            "complete_trees": len(complete), "stages": [16, 8, 4, 2, 1],
            "ordered_tree_signatures": dict(signatures)}


def selected(root):
    manifest = json.loads((root / "build_manifest.json").read_text(encoding="utf-8"))
    result = {}
    for unit in manifest["units"]:
        sass = (root / unit["sass"]).read_text(encoding="utf-8")
        raw = records(sass)
        resource_text = (root / unit["resources"]).read_text(encoding="utf-8")
        for name, items in original.functions(sass).items():
            match = original.VECTOR.search(name) if unit["name"] == "vector" else original.MATRIX.search(name)
            if match is None:
                continue
            mode = int(match.group(1)) if unit["name"] == "vector" else tuple(map(int, match.groups()))
            key = unit["name"], mode, "q6_word_group_" in name
            require(key not in result, "duplicate review specialization")
            result[key] = items, raw[name], original.resources(resource_text, name)
    require(len(result) == 10, "review requires all ten selected kernels")
    return result


def original_comparison(function, *args):
    try:
        return {"status": "PASS", "result": function(*args)}
    except ValueError as error:
        return {"status": "FAIL", "reason": str(error)}


def review_vector(native, candidate, columns):
    ni, nraw, nr = native
    ci, craw, cr = candidate
    nl = vector_loop(ni, columns)
    cl = vector_loop(ci, columns)
    require(not any(predicate and op.startswith(EXTRA_FLOAT + ("IDP",))
                    for raw, loop in ((nraw, nl), (craw, cl))
                    for pc, predicate, op, _ in raw if loop[0] <= pc <= loop[1]),
            "vector arithmetic is predicated")
    nk, ck = k_iterations(nraw, nl), k_iterations(craw, cl)
    nfp, cfp = vector_fp_shapes(nl[2], columns), vector_fp_shapes(cl[2], columns)
    require(nfp == cfp, "vector ordered FP dependency shapes changed")
    nreduce, creduce = reduction(nraw, nl[1], columns), reduction(craw, cl[1], columns)
    require(nreduce["ordered_tree_signatures"] == creduce["ordered_tree_signatures"],
            "ordered butterfly/FADD dependency trees changed")
    nh, ch = original.histogram(nl[2]), original.histogram(cl[2])
    require(nh.get("SHFL.IDX", 0) == 32 and original.histogram(ci).get("SHFL.IDX", 0) == 0,
            "cooperative decode shuffles were not removed")
    require(not any(op.startswith(("VOTE", "BRA.DIV")) for _, op, _ in cl[2]),
            "candidate K loop contains cooperative decode control flow")
    require(original.histogram(nl[2], EXTRA_FLOAT) == original.histogram(cl[2], EXTRA_FLOAT),
            "vector FP/conversion totals changed")
    credits = original.memory_width(cl[2], "LDG", 64) + 2 * (
        original.memory_width(cl[2], "LDG", 128) - original.memory_width(nl[2], "LDG", 128))
    require(credits >= 4, "wide vector code loads are incomplete")
    require(65536 // original.allocation_proxy(cr["REG"], 128)
            >= 65536 // original.allocation_proxy(nr["REG"], 128),
            "vector register allocation proxy regressed")
    return {
        "columns": columns,
        "original_comparison": original_comparison(original.compare_vector, ni, ci, nr, cr, columns),
        "registers": {"native": nr["REG"], "candidate": cr["REG"]},
        "K_loop": {"native": [hex(nl[0]), hex(nl[1])], "candidate": [hex(cl[0]), hex(cl[1])]},
        "K_induction": {"native": nk, "candidate": ck},
        "decode_SHFL_IDX": {"native_loop": nh.get("SHFL.IDX", 0), "candidate_whole_function": 0},
        "FP_counts": dict(original.histogram(cl[2], EXTRA_FLOAT)),
        "ordered_FP_dependency_shapes_equal": True,
        "FP_shape_scope": (
            "Ordered FP operations and four-DP4A integer chains; code, activation, scale and d "
            "leaves are opaque. This does not prove operand addresses or output association."
        ),
        "reduction": {"native": nreduce, "candidate": creduce, "ordered_tree_signatures_equal": True},
        "reduction_scope": (
            "Primary inlined paths and ordered FADD operands; the one out-of-line WARPSYNC "
            "helper is classified structurally. Helper call argument equivalence is not proved."
        ),
        "wide_code_load_credits": credits,
        "numerical_outputs": "not_qualified",
    }


def matrix_work_windows(items):
    barriers = [index for index, (_, op, _) in enumerate(items) if op.startswith("BAR.SYNC")]
    require(len(barriers) == 13, "MMQ barrier topology differs from the reviewed build")
    windows = []
    for left, right in ((3, 6), (9, 12)):
        part = items[barriers[left] + 1:barriers[right]]
        require(original.histogram(part).get("IMMA.16816.S8.S8", 0) == 256,
                "MMQ complete work window has unexpected tensor-core count")
        windows.append({"range": [hex(part[0][0]), hex(part[-1][0])],
                        "counts": dict(original.histogram(part, MATRIX_WORK))})
    return windows


def review_matrix(native, candidate, mode):
    ni, _, nr = native
    ci, _, cr = candidate
    nfp = original.histogram(ni, EXTRA_FLOAT + ("IMMA",))
    cfp = original.histogram(ci, EXTRA_FLOAT + ("IMMA",))
    require(nfp == cfp, "MMQ whole-function FP or tensor-core totals changed")
    nw, cw = matrix_work_windows(ni), matrix_work_windows(ci)
    require([x["counts"] for x in nw] == [x["counts"] for x in cw],
            "MMQ complete work windows contain different FP/unpack/memory totals")
    nload, cload = (original.memory_width(x, "LDG", 64) for x in (ni, ci))
    nstore, cstore = (original.memory_width(x, "STS", 128) for x in (ni, ci))
    require(cload > nload and cstore > nstore, "wide MMQ loads/stores are incomplete")
    require(nr["SHARED"] == cr["SHARED"]
            and original.allocation_proxy(cr["REG"], 256) <= 65536,
            "MMQ shared extent or one-CTA register feasibility changed")
    return {
        "mode": mode,
        "original_comparison": original_comparison(original.compare_matrix, ni, ci, nr, cr, mode),
        "registers": {"native": nr["REG"], "candidate": cr["REG"]},
        "dynamic_shared_bytes": 57856,
        "LDG64": {"native": nload, "candidate": cload},
        "STS128": {"native": nstore, "candidate": cstore},
        "whole_function_FP_and_IMMA_counts_equal": True,
        "whole_function_counts": dict(cfp),
        "complete_work_windows": {"native": nw, "candidate": cw},
        "work_window_scope": (
            "Diagnostic union of each three consecutive arithmetic barrier intervals, including "
            "intervals with no IMMA. Equal totals do not prove arithmetic order or shared-memory "
            "happens-before equivalence and do not replace the original strict comparison."
        ),
        "operand_aware_FP_order": "not_proved_by_this_review",
        "numerical_outputs": "not_qualified",
    }


def source_reference(path, needles, first=None, following=None):
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    begin = text.index(first) if first is not None else 0
    end = text.index(following, begin) if following is not None else len(text)
    anchors = []
    for needle in needles:
        lines = [text.count("\n", 0, match.start()) + 1
                 for match in re.finditer(re.escape(needle), text)
                 if begin <= match.start() < end]
        require(lines, "source-preservation anchor missing: " + needle)
        anchors.append({"literal": needle, "lines": lines})
    return {"path": str(path), "sha256": original.sha(path), "anchors": anchors}


def source_preservation(source_root, probe_root, receipt):
    vector = source_root / original.generate.VECTOR
    mmq = source_root / original.generate.MMQ
    dot = source_root / "third_party/ggml-quants/src/ggml-cuda/mmq-vec-dot.cuh"
    vector_first = "template <ggml_type type, int T, bool Fused, int KS, bool Prefetch2 = true,"
    vector_following = "template <ggml_type type, int T, bool Fused, int KS>\nvoid launch_baseline("
    vector_math = [
        "acc[r][j] += D::dot(sl, a[j], d[j], sum[j]);",
        "for (int it = 0; lane + 32 * it < slices; ++it) { slice(it); }",
        "acc[r][j] += __shfl_xor_sync(0xFFFFFFFFu, acc[r][j], o);",
    ]
    mmq_math = [
        "vec_dot    = ggml_cuda_mmq_get_vec_dot<type, J, fallback, prec_src1>();",
        "for (int kb0 = kb0_start; kb0 < kb0_stop; kb0 += blocks_per_iter)",
        "vec_dot(tile_x, tile_y, sum, 0);",
        "vec_dot(tile_x, tile_y, sum, MMQ_TILE_NE_K);",
    ]
    return {
        "source_root": str(source_root),
        "generator": source_reference(probe_root / "generate.py", [
            "if restored != self.original:",
            "q6_word_group_decode(operand, row_block[r] + std::uint64_t(32 / kPerBlock) * it, sub, sl);",
            "q6_word_group_load_tiles<type, J, fallback>(operand, tile_x, offset_x + kb0, ",
        ]),
        "reversible_regions": {
            name: {key: value for key, value in transform.items() if key != "substitutions"}
            for name, transform in receipt["transforms"].items()
        },
        "native_vector_K_dot_and_reduction": source_reference(
            vector, vector_math, vector_first, vector_following),
        "generated_vector_K_dot_and_reduction": source_reference(
            probe_root / "generated/q6_word_group_vector.cuh", vector_math),
        "native_Q6_vector_dot": source_reference(vector, [
            "return d * (s.f[0] * float(dot4(s.w, a, 0)) + s.f[1] * float(dot4(s.w + 4, a + 4, 0)));",
        ], "struct Decoder<GGML_TYPE_Q6_K>", "struct BaselineDecoder<GGML_TYPE_Q6_K>"),
        "native_MMQ_K_and_dot_calls": source_reference(mmq, mmq_math),
        "generated_MMQ_K_and_dot_calls": source_reference(
            probe_root / "generated/q6_word_group_mmq.cuh", mmq_math),
        "native_MMQ_Q6_dispatch": source_reference(mmq, [
            "ggml_cuda_mmq_vec_dot_q6_K_q8_1_mma<type, J, fallback>,",
        ]),
        "native_MMQ_Q6_mma_arithmetic": source_reference(dot, [
            "for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 8)",
            "for (int j0 = 0; j0 < J; j0 += ntx*tile_C::J)",
            "tmp[n][l] += (C[0].x[l]*scA[n][l/2][k01/4 + 0] + C[1].x[l]*scA[n][l/2][k01/4 + 1])*dB[l%2];",
            "sum[(j0/tile_C::J + n)*tile_C::ne + l] += tmp[n][l]*dA[n][l/2];",
        ], "void ggml_cuda_mmq_vec_dot_q6_K_q8_1_mma(",
            "void ggml_cuda_mmq_vec_dot_fp4_fp4_mma("),
        "scope": (
            "The pinned generator reverses every listed signature/address/loader substitution to "
            "the exact normalized native function. Vector D::dot, runtime K traversal and reduction, "
            "and MMQ vec_dot dispatch, arithmetic, K traversal and epilogues are not substituted. "
            "Compiled dependencies are separately verified by original.run. This source-preservation "
            "proof does not replace numerical output qualification."
        ),
    }


def run(root):
    result = {
        "schema": 1,
        "review_sha256": original.sha(pathlib.Path(__file__).resolve()),
        "original_gate_sha256": original.sha(pathlib.Path(original.__file__).resolve()),
        "build_manifest_sha256": original.sha(root / "build_manifest.json"),
        "parser_fix": PARSER_FIX,
        "mutated_existing_sources": False,
        "compiled": False,
        "gpu_run": False,
        "performance_claim": False,
    }
    manifest = json.loads((root / "build_manifest.json").read_text(encoding="utf-8"))
    probe_root = pathlib.Path(manifest["probe_root"])
    generated, receipt = original.generate.prepare(pathlib.Path(manifest["source_root"]))
    for name, content in generated.items():
        require((probe_root / "generated" / name).read_bytes() == content.encode(),
                "live generated header differs from pinned generator: " + name)
    receipt_path = probe_root / "generated" / "source_receipt.json"
    require(json.loads(receipt_path.read_text(encoding="utf-8")) == receipt,
            "live source receipt differs from pinned generator metadata")
    result["additional_live_generated_validation"] = {
        "headers": len(generated), "source_receipt_json_equal": True,
        "source_receipt_sha256": original.sha(receipt_path),
    }
    result["source_preservation"] = source_preservation(
        pathlib.Path(manifest["source_root"]), probe_root, receipt)
    result["original_run_additional_harness_limitation"] = (
        "The frozen manifest uses unit name mmq; original.run later looks up selected[\"matrix\",...]. "
        "Its earlier vector comparisons fail before this lookup. This review calls each original "
        "MMQ comparison independently and does not alter original.run."
    )
    previous = original.vector_loop
    original.vector_loop = vector_loop
    try:
        try:
            original_result = original.run(root)
            result["original_run"] = {"status": "PASS", "result": original_result}
        except (ValueError, KeyError, OSError) as error:
            frames = []
            traceback = error.__traceback__
            while traceback is not None:
                frames.append(traceback.tb_frame.f_code.co_name)
                traceback = traceback.tb_next
            result["original_run"] = {"status": "FAIL", "reason": str(error)}
            require("compare_vector" in frames or "compare_matrix" in frames,
                    "original manifest/dependency/resource validation did not complete: " + str(error))
        result["original_pin_dependency_artifact_and_resource_checks_completed"] = True
        kernels = selected(root)
        result["vector"] = [
            review_vector(kernels["vector", columns, False], kernels["vector", columns, True], columns)
            for columns in (1, 4)
        ]
        result["matrix"] = [
            review_matrix(kernels["mmq", mode, False], kernels["mmq", mode, True], label)
            for mode, label in (((0, 0), "PlainFP32"), ((1, 0), "StoreBF16"), ((1, 1), "AccumulateBF16"))
        ]
    finally:
        original.vector_loop = previous
    result["status"] = (
        "PASS_ORIGINAL_COMPILE_RESOURCE_AND_MECHANISM_SCREEN"
        if result["original_run"]["status"] == "PASS"
        else "PARTIAL_STATIC_REVIEW_ORIGINAL_MECHANISM_GATE_BLOCKED"
    )
    result["next_gate"] = (
        "Parent reviews the raw resource/mechanism evidence and the unresolved operand-aware limits "
        "to decide admission to a standalone correctness harness. Per-BAR histograms are not numerical "
        "semantics. Full real-fixture qualification and numerical GPU screening remain unrun."
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-root", required=True, type=pathlib.Path)
    args = parser.parse_args()
    try:
        result = run(args.build_root.resolve())
    except (ValueError, KeyError, OSError, TypeError, IndexError) as error:
        print(json.dumps({"status": "FAIL_STATIC_REVIEW", "reason": str(error),
                          "review_sha256": original.sha(pathlib.Path(__file__).resolve()),
                          "gpu_run": False, "compiled": False}, indent=2))
        raise SystemExit(1)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["original_run"]["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
