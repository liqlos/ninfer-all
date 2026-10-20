import argparse
import collections
import json
import pathlib
import re
import shlex
import sys

import generate
from proof import require


INSTRUCTION = re.compile(
    r"/\*(?P<pc>[0-9a-fA-F]+)\*/\s*(?P<pred>@!?(?:UP|P)(?:T|\d+)\s+)?"
    r"(?P<op>[A-Z][A-Z0-9_.]*)\s*(?P<args>[^;]*);")
OFF = re.compile(r"q6_j256_off_native_mul_mat_qIL9ggml_type14ELi128ELb0EL9ggml_prec30ELb([01])ELb([01])ELb0ELb0EE")
ON = re.compile(r"q6_j256_kernelILb([01])ELb([01])EE")
MODES = {(0, 0): "plain", (1, 0): "store", (1, 1): "accumulate"}
FLOAT_OPS = ("FADD", "FMUL", "FFMA", "F2F", "I2F")


def parse_dependencies(source, expected_target):
    require(not source.rstrip().endswith("\\"), "dangling dependency continuation")
    rule = re.sub(r"\\\r?\n", " ", source).strip()
    require("\n" not in rule and "\r" not in rule, "expected one logical dependency rule")
    require(rule.count(":") == 1, "expected one POSIX dependency target separator")
    target, prerequisites = rule.split(":", 1)
    require(shlex.split(target) == [str(expected_target)], "unexpected dependency target")
    paths = shlex.split(prerequisites)
    require(paths, "empty dependency prerequisites")
    return paths


def functions(text):
    anchors = list(re.finditer(r"^\s*Function\s*:\s*(\S+)\s*$", text, re.MULTILINE))
    result = {}
    for index, match in enumerate(anchors):
        end = anchors[index + 1].start() if index + 1 < len(anchors) else len(text)
        name = match.group(1)
        require(name not in result, "duplicate SASS function image")
        result[name] = [(int(m["pc"], 16), m["op"], m["args"].strip(), (m["pred"] or "").strip())
                        for m in INSTRUCTION.finditer(text[match.end():end])]
        items = result[name]
        require(items and len({item[0] for item in items}) == len(items), "incomplete SASS instructions")
    require(result, "no SASS functions")
    return result


def histogram(items, prefixes=None):
    return collections.Counter(op for _, op, _, _ in items if prefixes is None or op.startswith(prefixes))


def resources(text, name):
    matches = list(re.finditer(r"^\s*Function\s+" + re.escape(name) + r":\s*\n([^\n]+)", text, re.MULTILINE))
    require(len(matches) == 1, "selected kernel resource missing or ambiguous")
    result = {key: int(value) for key, value in re.findall(r"([A-Z]+):(\d+)", matches[0].group(1))}
    require({"REG", "STACK", "SHARED", "LOCAL"} <= result.keys(), "incomplete resource dump")
    return result


def spills(text, name):
    matches = re.findall(r"Function properties for\s+" + re.escape(name) +
                         r"\s*\n\s*(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", text)
    require(len(matches) == 1, "selected ptxas spill record missing or ambiguous")
    return dict(zip(("stack", "stores", "loads"), map(int, matches[0])))


def resource_gate(resource, spill, items, candidate):
    dynamic = 76800 if candidate else 57856
    return bool(0 < resource["REG"] <= (128 if candidate else 255) and
                resource["STACK"] == resource["LOCAL"] == 0 and
                all(value == 0 for value in spill.values()) and
                resource["SHARED"] + dynamic <= dynamic and
                not histogram(items, ("LDL", "STL")))


def dot_regions(items):
    start = 0
    regions = []
    for index, (_, op, _, predicate) in enumerate(items):
        if not op.startswith("BAR.SYNC"):
            continue
        require(not predicate, "predicated CTA barrier")
        region = items[start:index]
        start = index + 1
        count = sum(histogram(region, ("IMMA",)).values())
        if count:
            require(count == 128, "unknown K128 MMA count")
            require(all(op.startswith("IMMA.16816.S8.S8") for op in histogram(region, ("IMMA",))),
                    "unknown MMA width or signedness")
            require(not histogram(region, ("HMMA", "DMMA", "MUFU", "HFMA", "HADD", "HMUL")),
                    "unexpected arithmetic family")
            regions.append((region[0][0], region[-1][0], dict(histogram(region, FLOAT_OPS + ("IMMA",)))))
    require(regions, "no barrier-delimited K128 dot regions")
    return regions


def candidate_loop(items):
    loops = []
    for end, op, operands, _ in items:
        match = re.fullmatch(r"0x([0-9a-fA-F]+)", operands)
        if op != "BRA" or not match:
            continue
        begin = int(match.group(1), 16)
        if begin >= end:
            continue
        region = [item for item in items if begin <= item[0] <= end]
        if sum(histogram(region, ("IMMA",)).values()) == 256:
            loops.append((begin, end, region))
    require(len(loops) == 1, "cannot identify unique closed K256 loop")
    begin, end, region = loops[0]
    require(sum(histogram(region, ("BAR.SYNC",)).values()) == 4, "closed K loop must have four CTA barriers")
    require(not histogram(region, ("CALL", "BRA.DIV", "VOTE", "SHFL", "WARPSYNC")),
            "unexpected call or warp cooperation in closed K loop")
    return {"begin": hex(begin), "backedge": hex(end), "opcodes": dict(histogram(region))}


def parse_unit(root, unit, source, receipt):
    paths = {key: root / unit[key] for key in ("object", "dependencies", "log", "resources", "sass", "ptx")}
    for key, path in paths.items():
        require(generate.sha(path.read_bytes()) == unit["artifact_sha256"][key], "compiled artifact changed: " + key)
    dependencies = {str((source / path).resolve()) for path in parse_dependencies(
        paths["dependencies"].read_text(), paths["object"])}
    require(dependencies == set(unit["dependency_sha256"]), "dependency receipt differs")
    for path, digest in unit["dependency_sha256"].items():
        require(generate.sha(pathlib.Path(path).read_bytes()) == digest, "compiled dependency changed: " + path)
    generated = root / "generated"
    required = {str((source / name).resolve()) for name in generate.PINS}
    if unit["name"] == "on":
        for name in receipt["generated_sha256"]:
            original = source / "third_party/ggml-quants/src" / name
            require(str(original.resolve()) not in dependencies, "native sibling bypassed overlay: " + str(original))
            required.discard(str(original.resolve()))
            required.add(str((generated / name).resolve()))
    else:
        require(not any(str((generated / name).resolve()) in dependencies for name in receipt["generated_sha256"]),
                "OFF must use byte-exact native headers directly")
    probe = pathlib.Path(__file__).resolve().parent
    required.update(str(probe / name) for name in ("variant.cu", "variant_impl.cuh", "probe_api.h"))
    require(required <= dependencies, "actual dependency set missed generated or pinned native headers")
    text = {key: path.read_text(encoding="utf-8") for key, path in paths.items() if key != "object"}
    require(set(re.findall(r"code for (sm_\d+)", text["sass"])) == {"sm_86"}, "SASS target is not only SM86")
    require(set(re.findall(r"arch = (sm_\d+)", text["resources"])) == {"sm_86"}, "resource target is not only SM86")
    require(set(re.findall(r"\.target\s+(sm_\d+)", text["ptx"])) == {"sm_86"}, "PTX target is not only SM86")
    pattern = ON if unit["name"] == "on" else OFF
    selected = {}
    for name, items in functions(text["sass"]).items():
        match = pattern.search(name)
        if not match:
            if "q6_j256_kernel" in name or "_native_mul_mat_qI" in name:
                raise ValueError("unknown selected kernel signature: " + name)
            continue
        mode = MODES.get(tuple(map(int, match.groups())))
        require(mode is not None and mode not in selected, "unknown or duplicate output contract")
        resource = resources(text["resources"], name)
        spill = spills(text["log"], name)
        dynamic = 76800 if unit["name"] == "on" else 57856
        resource_pass = resource_gate(resource, spill, items, unit["name"] == "on")
        selected[mode] = {"symbol": name, "resources": resource, "spills": spill,
                          "dynamic_shared_bytes": dynamic, "resource_gate": bool(resource_pass),
                          "items": items}
    require(set(selected) == set(MODES.values()), "missing plain/store/accumulate kernel")
    return selected


def run(root):
    root = root.resolve()
    manifest = json.loads((root / "build_manifest.json").read_text())
    receipt = json.loads((root / "generation.json").read_text())
    source = pathlib.Path(manifest["source_root"])
    files, current = generate.prepare(source)
    require(receipt == current, "source generation or probe pins changed since compile")
    require(manifest["generation_sha256"] == generate.sha((root / "generation.json").read_bytes()),
            "generation receipt changed")
    for name, data in files.items():
        require((root / "generated" / name).read_bytes() == data, "generated source bytes differ")
    require(re.search(r"\bV13\.1\.115\b", (root / "toolchain.txt").read_text()), "unexpected toolchain")
    require(generate.sha((root / "toolchain.txt").read_bytes()) == manifest["toolchain_sha256"],
            "toolchain receipt changed")
    require(manifest["compile_only"] and not manifest["host_main"] and not manifest["gpu_run"], "unknown build scope")
    require([unit["name"] for unit in manifest["units"]] == ["off", "on"], "unknown build units")
    units = {}
    for unit in manifest["units"]:
        require(unit["returncode"] == 0, "compile failed")
        require("-arch=sm_86" in unit["argv"] and
                "-DNINFER_Q6_J256_PROBE=" + str(int(unit["name"] == "on")) in unit["argv"],
                "compile route differs")
        units[unit["name"]] = parse_unit(root, unit, source, receipt)
    result = {"status": "REJECT_RESOURCE_GATE", "resource_gate": all(
        kernel["resource_gate"] for kernels in units.values() for kernel in kernels.values()),
        "compile_only": True, "GPU_numerics_qualified": False, "performance_claim": False,
        "kernels": {variant: {mode: {key: value for key, value in kernel.items() if key != "items"}
                              for mode, kernel in kernels.items()} for variant, kernels in units.items()}}
    if not result["resource_gate"]:
        return result
    static = {}
    for mode in MODES.values():
        off, on = units["off"][mode]["items"], units["on"][mode]["items"]
        native_regions, candidate_regions = dot_regions(off), dot_regions(on)
        require(len(candidate_regions) == 2, "candidate must have two K128 dot regions")
        native_signatures = {json.dumps(region[2], sort_keys=True) for region in native_regions}
        require(all(json.dumps(region[2], sort_keys=True) in native_signatures for region in candidate_regions),
                "K128 FP32/IMMA counts differ from native")
        require(sum(histogram(on, ("IMMA",)).values()) == 256, "candidate has extra tensor arithmetic")
        require(sum(histogram(on, ("BAR.SYNC",)).values()) == 4, "candidate has extra CTA barriers")
        static[mode] = {"K256_loop": candidate_loop(on), "K128_regions": candidate_regions,
                        "same_native_K128_math_counts": True}
    result.update(status="PASS_RESOURCE_AND_STATIC_GATE_REQUIRES_ORDER_REVIEW", static_gate=static,
                  manual_order_review_required=["two k16 MMA and signed INT32 scale combine",
                      "four ordered FP32 D4 updates then dA FMA per K128",
                      "unique metadata writers and four publish/read/retire barriers",
                      "plain/store/BF16Accumulate final conversion boundary"],
                  full_operand_aware_SASS_equivalence=False)
    return result


def self_test():
    target = "/build/q6-j256/off.o"
    require(parse_dependencies(target + " : /src/a.h \\\n /src/b.h\n", target) == ["/src/a.h", "/src/b.h"],
            "spaced-colon Make rule failed")
    require(parse_dependencies(target + ": /src/a\\ b.h\n", target) == ["/src/a b.h"], "escaped path failed")
    for invalid in (target + " : : /src/a.h", target + " : /src/a.h\\", target + " :\n", "/bad.o : /src/a.h"):
        try:
            parse_dependencies(invalid, target)
        except ValueError:
            pass
        else:
            raise ValueError("malformed dependency rule accepted")
    text = "Function : kernel\n/*0000*/ @P0 BRA 0x20;\n/*0010*/ IMMA.16816.S8.S8 R4, R8, R12, RZ;"
    require(len(functions(text)["kernel"]) == 2, "SASS parser failed")
    require(spills("ptxas info    : Function properties for kernel\n    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n", "kernel") ==
            {"stack": 0, "stores": 0, "loads": 0}, "ptxas parser failed")
    resource = resources("Function kernel:\n  REG:128 STACK:0 SHARED:0 LOCAL:0\n", "kernel")
    require(resource == {"REG": 128, "STACK": 0, "SHARED": 0, "LOCAL": 0}, "resource parser failed")
    zero_spill = {"stack": 0, "stores": 0, "loads": 0}
    require(resource_gate(resource, zero_spill, [], True), "128-register boundary rejected")
    for key, value in (("REG", 129), ("STACK", 1), ("LOCAL", 1), ("SHARED", 1)):
        require(not resource_gate({**resource, key: value}, zero_spill, [], True), "hard resource limit accepted")
    for key in zero_spill:
        require(not resource_gate(resource, {**zero_spill, key: 1}, [], True), "ptxas spill accepted")
    for opcode in ("LDL", "STL"):
        require(not resource_gate(resource, zero_spill, [(0, opcode, "R0, [R1]", "")], True), "local instruction accepted")
    require(ON.search("_ZL14q6_j256_kernelILb1ELb1EEvPKcPKiPfP13__nv_bfloat16i").groups() == ("1", "1"),
            "candidate output contract parser failed")
    require(OFF.search("_Z29q6_j256_off_native_mul_mat_qIL9ggml_type14ELi128ELb0EL9ggml_prec30ELb1ELb0ELb0ELb0EEv").groups() == ("1", "0"),
            "native output contract parser failed")
    return {"status": "PASS_PARSER_CPU_TESTS", "GPU_or_build_run": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-root", type=pathlib.Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test == bool(args.build_root):
        parser.error("select exactly one of --self-test and --build-root")
    result = self_test() if args.self_test else run(args.build_root)
    print(json.dumps(result, indent=2))
    return 0 if result["status"].startswith("PASS") else 3


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
