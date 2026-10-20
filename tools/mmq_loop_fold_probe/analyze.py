import argparse
import collections
import json
import pathlib
import re
import shlex
import sys

import generate
from generate import require


INSTRUCTION = re.compile(r"/\*(?P<pc>[0-9a-fA-F]+)\*/\s*(?P<pred>@!?(?:UP|P)(?:T|\d+)\s+)?"
                         r"(?P<op>[A-Z][A-Z0-9_.]*)\s*(?P<args>[^;]*);")
KERNEL = re.compile(r"mul_mat_qIL9ggml_type12ELi128ELb0EL9ggml_prec30ELb([01])ELb([01])ELb0ELb0EE")
MODES = {(0, 0): "plain", (1, 0): "store", (1, 1): "accumulate"}
MATH = ("IMMA", "HMMA", "DMMA", "FADD", "FMUL", "FFMA", "F2F", "I2F", "HADD", "HMUL", "HFMA", "MUFU")
MANUAL = [
    "Map the folded parent loop to non-fixup full tiles; the native kernel retains a separate fixup body.",
    "Trace initialization, compare, increment and backedge operands: exactly two CTA-uniform phases, 0 then 1.",
    "Trace global Y source and shared X k00 source: native half0 then half1; no register-array dynamic indexing.",
    "Check full A/DM preload before J-major/K32 math and operand-aware ordered FP32 accumulator DAG.",
    "Check all four dynamic publish/retire barriers and plain/Store/Accumulate final conversion boundary.",
]


def functions(text):
    anchors = list(re.finditer(r"^\s*Function\s*:\s*(\S+)\s*$", text, re.MULTILINE))
    result = {}
    for index, match in enumerate(anchors):
        end = anchors[index + 1].start() if index + 1 < len(anchors) else len(text)
        name = match.group(1)
        require(name not in result, "duplicate SASS function image")
        body = text[match.end():end]
        items = [(int(m["pc"], 16), m["op"], m["args"].strip(), (m["pred"] or "").strip())
                 for m in INSTRUCTION.finditer(body)]
        require(items and len({item[0] for item in items}) == len(items), "missing or duplicate SASS PCs")
        addressed = re.findall(r"/\*[0-9a-fA-F]{4,}\*/\s+(?!/\*)[^\n]*;", body)
        require(len(addressed) == len(items), "SASS instruction parser missed an addressed instruction")
        require(all(b[0] - a[0] == 16 for a, b in zip(items, items[1:])), "noncontiguous SM86 instruction PCs")
        result[name] = items
    require(result, "no SASS functions")
    return result


def hist(items, prefixes=None):
    return collections.Counter(op for _, op, _, _ in items if prefixes is None or op.startswith(prefixes))


def count(items, prefix):
    return sum(hist(items, (prefix,)).values())


def branch_target(op, operands):
    if op not in ("BRA", "CALL.REL.NOINC"):
        return None
    match = re.fullmatch(r"(?:(?:UP|P)(?:T|\d+),\s*)?0x([0-9a-fA-F]+)", operands)
    require(match, "unknown direct control-transfer operands")
    return int(match.group(1), 16)


def operand_predicate(operands):
    match = re.match(r"((?:UP|P)\d+),", operands)
    return match.group(1) if match else ""


def loops(items):
    result = []
    pcs = {item[0] for item in items}
    for pc, op, operands, predicate in items:
        begin = branch_target(op, operands)
        if begin is None:
            continue
        if begin >= pc:
            continue
        require(begin in pcs, "backedge target is missing")
        body = [item for item in items if begin <= item[0] <= pc]
        result.append({"begin": begin, "end": pc, "predicate": predicate,
                       "operand_predicate": operand_predicate(operands), "backedge_op": op, "items": body})
    return result


def contains(outer, inner):
    return outer["begin"] <= inner["begin"] and outer["end"] > inner["end"]


def compact(loop):
    items = loop["items"]
    return {"begin": hex(loop["begin"]), "backedge": hex(loop["end"]),
            "bytes": loop["end"] - loop["begin"] + 16, "instruction_slots": len(items),
            "backedge_predicate": loop["predicate"], "backedge_op": loop["backedge_op"],
            "backedge_operand_predicate": loop["operand_predicate"],
            "static_math": dict(hist(items, MATH)),
            "static_barriers": count(items, "BAR.SYNC")}


def native_hot_loops(items):
    all_loops = loops(items)
    result = [loop for loop in all_loops if count(loop["items"], "IMMA") == 128 and
              count(loop["items"], "BAR.SYNC") == 4 and
              not any(contains(loop, inner) and count(inner["items"], "IMMA") for inner in all_loops)]
    require(result, "cannot identify native K256 loop bodies")
    for loop in result:
        require(not any(pred for _, op, _, pred in loop["items"] if op.startswith("BAR.SYNC")),
                "predicated native CTA barrier")
    return result


def compare_hot(off, on):
    controls = native_hot_loops(off)
    all_loops = loops(on)
    phases = [loop for loop in all_loops if count(loop["items"], "IMMA") == 64 and
              count(loop["items"], "BAR.SYNC") == 2 and
              not any(contains(loop, inner) and count(inner["items"], "IMMA") for inner in all_loops)]
    require(len(phases) == 1, "no unique retained runtime phase loop; unroll or unknown lowering")
    phase = phases[0]
    parents = [loop for loop in all_loops if contains(loop, phase) and
               count(loop["items"], "IMMA") == 64 and count(loop["items"], "BAR.SYNC") == 2]
    require(parents, "folded phase has no K256 parent backedge")
    parent = min(parents, key=lambda loop: loop["end"] - loop["begin"])
    allowed_backedges = {(phase["end"], phase["begin"]), (parent["end"], parent["begin"])}
    exits = []
    for pc, op, operands, predicate in parent["items"]:
        require(not op.startswith(("RET", "EXIT", "JMP", "JMX", "BRX", "BRA.DIV", "VOTE")) and
                (not op.startswith("CALL") or op == "CALL.REL.NOINC"),
                "unexpected control flow or cooperation in hot body")
        target = branch_target(op, operands)
        if target is not None and (pc, target) not in allowed_backedges:
            active = phase if phase["begin"] <= pc <= phase["end"] else parent
            last_barrier = max(v[0] for v in active["items"] if v[1].startswith("BAR.SYNC"))
            require((predicate or operand_predicate(operands)) and pc > last_barrier and target > active["end"],
                    "unexpected hot-body branch requires a new parser/control-flow design")
            exits.append({"pc": hex(pc), "op": op, "target": hex(target), "predicate": predicate,
                          "loop": "phase" if active is phase else "K256"})
        if op.startswith(MATH + ("BAR.SYNC",)):
            require(not predicate, "predicated hot arithmetic or barrier")
    mma_ops = hist(parent["items"], ("IMMA",))
    for name, loop in (("phase", phase), ("K256", parent)):
        require(loop["predicate"] or loop["operand_predicate"] or any(v["loop"] == name for v in exits),
                "loop termination predicate missing")
    require(all(op == "IMMA.16832.S8.S8" for op in mma_ops), "unexpected Q4 MMA shape or signedness")
    phase_math = hist(phase["items"], MATH)
    parent_math = hist(parent["items"], MATH)
    weighted = parent_math + phase_math
    for loop in controls:
        require(weighted == hist(loop["items"], MATH), "conditional dynamic math opcode counts differ")
        require(compact(parent)["bytes"] < compact(loop)["bytes"], "hot K256 instruction footprint did not shrink")
    require(count(parent["items"], "BAR.SYNC") + count(phase["items"], "BAR.SYNC") == 4,
            "conditional dynamic barrier count differs")
    other_math_loops = [loop for loop in all_loops if count(loop["items"], "IMMA") == 128 and
                       count(loop["items"], "BAR.SYNC") == 4 and
                       not any(contains(loop, inner) and count(inner["items"], "IMMA") for inner in all_loops)]
    require(len(other_math_loops) == len(controls) - 1, "native fixup loop count changed")
    return {"native_K256_bodies": [compact(loop) for loop in controls],
            "candidate_K256_body": compact(parent), "candidate_K128_phase": compact(phase),
            "hot_bytes_ratio_vs_smallest_native": compact(parent)["bytes"] / min(compact(v)["bytes"] for v in controls),
            "conditional_dynamic_math_per_K256": dict(weighted),
            "conditional_dynamic_barriers_per_K256": 4, "dynamic_count_assumption": "phase executes exactly twice",
            "loop_exit_control_transfers": exits,
            "trip_count_proven_by_parser": False, "operand_equivalence_proven_by_parser": False}


def resources(text, name):
    found = re.findall(r"^\s*Function\s+" + re.escape(name) + r":\s*\n([^\n]+)", text, re.MULTILINE)
    require(len(found) == 1, "missing or ambiguous resource record")
    result = {key: int(value) for key, value in re.findall(r"([A-Z]+):(\d+)", found[0])}
    require({"REG", "STACK", "SHARED", "LOCAL"} <= result.keys(), "incomplete resource record")
    return result


def spills(text, name):
    found = re.findall(r"Function properties for\s+" + re.escape(name) +
                       r"\s*\n\s*(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", text)
    require(len(found) == 1, "missing or ambiguous ptxas record")
    return dict(zip(("stack", "stores", "loads"), map(int, found[0])))


def occupancy(resource):
    regs_per_block = ((resource["REG"] + 7) // 8) * 8 * 256
    shared = resource["SHARED"] + 57856
    if not 0 < resource["REG"] <= 255 or shared > 99 * 1024:
        return 0
    return min(16, 48 // 8, 65536 // regs_per_block, (100 * 1024) // shared)


def resource_ok(resource, spill, items):
    return bool(occupancy(resource) >= 1 and resource["STACK"] == resource["LOCAL"] == 0 and
                resource["SHARED"] == 0 and all(value == 0 for value in spill.values()) and
                not hist(items, ("LDL", "STL")) and
                not any(op.startswith("CALL") and op != "CALL.REL.NOINC" for _, op, _, _ in items))


def parse_dependencies(source, expected_target):
    require(not source.rstrip().endswith("\\"), "dangling dependency continuation")
    rule = re.sub(r"\\\r?\n", " ", source).strip()
    require("\n" not in rule and "\r" not in rule and rule.count(":") == 1, "expected one POSIX Make rule")
    target, prerequisites = rule.split(":", 1)
    require(shlex.split(target) == [str(expected_target)], "unexpected dependency target")
    paths = shlex.split(prerequisites)
    require(paths, "empty dependency prerequisites")
    return paths


def parse_unit(root, source, unit, receipt):
    paths = {key: root / unit[key] for key in ("object", "dependencies", "log", "resources", "sass", "ptx")}
    for key, path in paths.items():
        require(generate.sha(path.read_bytes()) == unit["artifact_sha256"][key], "compiled artifact changed: " + key)
    dependencies = {str((source / path).resolve()) for path in parse_dependencies(paths["dependencies"].read_text(), paths["object"])}
    require(dependencies == set(unit["dependency_sha256"]), "dependency receipt differs")
    for path, digest in unit["dependency_sha256"].items():
        require(generate.sha(pathlib.Path(path).read_bytes()) == digest, "compiled dependency changed: " + path)
    required = {str((source / name).resolve()) for name in generate.PINS}
    generated = root / "generated"
    required.add(str(generated / "variant.cu"))
    for name in ("mmq.cuh", "mmq-load-tiles.cuh", "mmq-vec-dot.cuh"):
        overlay = str(generated / "on/ggml-cuda" / name)
        native = str(source / generate.CUDA / name)
        if unit["name"] == "on":
            required.discard(native)
            required.add(overlay)
            require(native not in dependencies, "candidate bypassed overlay")
        else:
            require(overlay not in dependencies, "OFF used candidate overlay")
    require(required <= dependencies, "actual compilation missed pinned source")
    texts = {key: paths[key].read_text(encoding="utf-8") for key in ("log", "resources", "sass", "ptx")}
    require(set(re.findall(r"code for (sm_\d+)", texts["sass"])) == {"sm_86"}, "unexpected SASS architecture")
    require(set(re.findall(r"arch = (sm_\d+)", texts["resources"])) == {"sm_86"}, "unexpected resource architecture")
    require(set(re.findall(r"\.target\s+(sm_\d+)", texts["ptx"])) == {"sm_86"}, "unexpected PTX architecture")
    require(not re.search(r"\.(?:local|callprototype)\b|\b(?:ld|st)\.local\b", texts["ptx"]), "PTX local storage or call prototype")
    selected = {}
    for name, items in functions(texts["sass"]).items():
        match = KERNEL.search(name)
        require(match is not None, "unexpected emitted kernel: " + name)
        mode = MODES.get(tuple(map(int, match.groups())))
        require(mode is not None and mode not in selected, "unknown or duplicate output contract")
        resource, spill = resources(texts["resources"], name), spills(texts["log"], name)
        selected[mode] = {"symbol": name, "resources": resource, "spills": spill, "resident_CTA_model": occupancy(resource),
                          "resource_pass": resource_ok(resource, spill, items), "items": items}
    require(set(selected) == set(MODES.values()), "plain/Store/Accumulate instantiation missing")
    return selected


def run(root):
    root = root.resolve()
    manifest = json.loads((root / "build_manifest.json").read_text())
    receipt = json.loads((root / "generation.json").read_text())
    source = pathlib.Path(manifest["source_root"])
    files, current = generate.prepare(source)
    require(receipt == current, "generation no longer matches pinned sources")
    require(manifest["generation_sha256"] == generate.sha((root / "generation.json").read_bytes()), "generation receipt changed")
    for name, data in files.items():
        require((root / "generated" / name).read_bytes() == data, "generated bytes changed: " + name)
    for name, digest in manifest["probe_sha256"].items():
        require(generate.sha((pathlib.Path(__file__).parent / name).read_bytes()) == digest, "probe changed after build: " + name)
    require(generate.sha((root / "toolchain.txt").read_bytes()) == manifest["toolchain_sha256"], "toolchain receipt changed")
    require(re.search(r"\bV13\.1\.115\b", (root / "toolchain.txt").read_text()), "unexpected CUDA compiler")
    require(manifest["compile_only"] and not manifest["host_main"] and not manifest["link"] and not manifest["gpu_run"], "wrong build scope")
    require([v["name"] for v in manifest["units"]] == ["off", "on"], "unexpected build units")
    units = {}
    for unit in manifest["units"]:
        require(unit["returncode"] == 0, "compile failed")
        require("-arch=sm_86" in unit["argv"] and "-D" + generate.SWITCH + "=" + str(int(unit["name"] == "on")) in unit["argv"], "wrong compile route")
        units[unit["name"]] = parse_unit(root, source, unit, receipt)
    result = {"status": "REJECT_RESOURCE_GATE", "compile_only": True, "GPU_numerics_qualified": False,
              "performance_claim": False, "GPU_harness_allowed": False,
              "kernels": {arm: {mode: {k: v for k, v in kernel.items() if k != "items"} for mode, kernel in kernels.items()}
                          for arm, kernels in units.items()}}
    if not all(v["resource_pass"] for kernels in units.values() for v in kernels.values()):
        return result
    hot = {}
    for mode in MODES.values():
        off, on = units["off"][mode], units["on"][mode]
        require(on["resident_CTA_model"] >= off["resident_CTA_model"], "candidate loses occupancy")
        hot[mode] = compare_hot(off["items"], on["items"])
    result.update(status="PASS_COMPILE_RESOURCE_HOT_FOOTPRINT_REQUIRES_MANUAL_REVIEW", hot=hot,
                  manual_review_required=MANUAL, full_operand_aware_SASS_equivalence=False)
    return result


def self_test():
    def rejected(operation):
        try:
            operation()
        except ValueError:
            return
        raise ValueError("negative parser test was accepted")
    sample = "Function : kernel\n/*0000*/ IMMA.16832.S8.S8 R4, R8, R12, RZ;\n/*0010*/ @P0 BRA 0x0;\n"
    items = functions(sample)["kernel"]
    require(len(loops(items)) == 1, "backedge parse failed")
    rejected(lambda: functions(sample.replace("/*0010*/", "/*0020*/")))
    rejected(lambda: functions(sample.replace("@P0 BRA", "@UNKNOWN BRA")))
    rejected(lambda: functions(sample + sample))
    require(parse_dependencies("/build/a.o : /src/a.h \\\n /src/a\\ b.h\n", "/build/a.o") == ["/src/a.h", "/src/a b.h"], "Make parsing failed")
    for invalid in ("/bad.o : /src/a.h", "/build/a.o :", "/build/a.o : /src/a.h\\", "/build/a.o : : /src/a.h"):
        rejected(lambda value=invalid: parse_dependencies(value, "/build/a.o"))
    resource = resources("Function kernel:\n REG:221 STACK:0 SHARED:0 LOCAL:0\n", "kernel")
    spill = spills("Function properties for kernel\n 0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads", "kernel")
    require(resource_ok(resource, spill, items) and occupancy(resource) == 1, "native resource model failed")
    for key, value in (("REG", 256), ("STACK", 1), ("LOCAL", 1), ("SHARED", 1)):
        require(not resource_ok({**resource, key: value}, spill, items), "resource limit accepted")
    for key in spill:
        require(not resource_ok(resource, {**spill, key: 1}, items), "spill accepted")
    for op in ("LDL", "STL", "CALL"):
        require(not resource_ok(resource, spill, [(0, op, "R0", "")]), "local storage/call accepted")
    require(resource_ok(resource, spill, [(0, "CALL.REL.NOINC", "0x10", "@P0")]),
            "non-incrementing native control transfer rejected as device call")
    rejected(lambda: compare_hot(items, items))
    def synthetic(folded, extra_nops=0, wrong_math=False, use_exit=False):
        code = []
        def emit(op, args="R0", pred=""):
            code.append((len(code) * 16, op, args, pred))
        emit("I2F.F16")
        for half in range(1 if folded else 2):
            begin = len(code) * 16
            emit("BAR.SYNC", "0x0")
            for _ in range(64):
                emit("IMMA.16832.S8.S8")
            for _ in range(512 + int(wrong_math)):
                emit("FFMA")
            for _ in range(extra_nops):
                emit("NOP", "")
            emit("BAR.SYNC", "0x0")
            if folded:
                if use_exit:
                    emit("CALL.REL.NOINC", hex((len(code) + 2) * 16), "@P0")
                emit("BRA", hex(begin), "" if use_exit else "@P0")
        if use_exit:
            emit("CALL.REL.NOINC", hex((len(code) + 2) * 16), "@P1")
        emit("BRA", "0x0", "" if use_exit else "@P1")
        emit("NOP", "")
        return code
    control = synthetic(False)
    folded = synthetic(True)
    require(compare_hot(control, folded)["conditional_dynamic_math_per_K256"]["IMMA.16832.S8.S8"] == 128,
            "static versus dynamic math accounting failed")
    require(len(compare_hot(synthetic(False, use_exit=True), synthetic(True, use_exit=True))["loop_exit_control_transfers"]) == 2,
            "native non-incrementing loop exit parsing failed")
    rejected(lambda: compare_hot(control, control))
    rejected(lambda: compare_hot(control, synthetic(True, wrong_math=True)))
    rejected(lambda: compare_hot(control, synthetic(True, extra_nops=600)))
    broken = [(*v[:3], "@P0" if v[1] == "BAR.SYNC" else v[3]) for v in folded]
    rejected(lambda: compare_hot(control, broken))
    return {"status": "PASS_CPU_PARSER_RESOURCE_TESTS", "compile_or_GPU_run": False,
            "synthetic_hot_loop_tests": "fold, unroll, weighted-math mismatch, no shrink, predicated barrier, NOINC exits"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-root", type=pathlib.Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--saved-native-sass", type=pathlib.Path)
    args = parser.parse_args()
    if sum((args.self_test, bool(args.build_root), bool(args.saved_native_sass))) != 1:
        parser.error("select exactly one analyzer mode")
    if args.self_test:
        result = self_test()
    elif args.saved_native_sass:
        parsed = functions(args.saved_native_sass.read_text())
        selected = {name: [compact(loop) for loop in native_hot_loops(items)]
                    for name, items in parsed.items() if KERNEL.search(name)}
        require(selected, "no saved Q4 J128 native kernel")
        result = {"status": "PASS_SAVED_NATIVE_PARSER_ONLY", "native_loops": selected,
                  "candidate_compiled": False, "compile_or_GPU_run": False}
    else:
        result = run(args.build_root)
    print(json.dumps(result, indent=2))
    return 0 if result["status"].startswith("PASS") else 3


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
