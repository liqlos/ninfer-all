import argparse
import collections
import json
import pathlib
import re
import shlex
import sys

from generate import HERE, NATIVE, PAIRED, digest, generate, require


def parse_dependencies(source, expected_target):
    require(not source.rstrip().endswith("\\"), "dangling dependency continuation")
    rule = re.sub(r"\\\r?\n", " ", source).strip()
    require("\n" not in rule and "\r" not in rule and rule.count(":") == 1,
            "expected one POSIX dependency rule")
    target, prerequisites = rule.split(":", 1)
    require(shlex.split(target) == [str(expected_target)], "unexpected dependency target")
    result = shlex.split(prerequisites)
    require(result, "empty dependencies")
    return result


def sass_counts(source):
    return collections.Counter(re.findall(
        r"/\*[0-9a-fA-F]+\*/\s+(?:@!?P\w+\s+)?([A-Z][A-Z0-9_.]*)(?:\s|;)", source))


def variant(build_root, source_root, name, generation):
    files = {kind: (build_root / filename).read_text(encoding="utf-8") for kind, filename in {
        "resource": "resources-" + name + ".txt", "compile": "compile-" + name + ".log",
        "sass": "sass-" + name + ".txt", "ptx": "ptx-" + name + ".txt",
        "dependencies": name + ".d"}.items()}
    matches = re.findall(
        r"\bFunction\s+(\S+):\s*\n\s+REG:(\d+)\s+STACK:(\d+)\s+SHARED:(\d+)\s+LOCAL:(\d+)",
        files["resource"])
    require(len(matches) == 1, "expected one kernel resource record: " + name)
    symbol, registers, stack, shared, local = matches[0]
    require("causal_attention_prompt_i8_paired_" + name + "_kernel" in symbol, "wrong variant")
    require("arch = sm_86" in files["resource"], "wrong cubin architecture")
    spills = re.findall(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads",
                        files["compile"])
    require(len(spills) == 1, "ambiguous ptxas stack/spill record")
    ptx_stack, stores, loads = map(int, spills[0])
    require(ptx_stack == int(stack), "ptxas/cubin stack mismatch")
    counts = sass_counts(files["sass"])
    require(any(op.startswith("IMMA") for op in counts) and
            any(op.startswith("HMMA") for op in counts), "SASS parse incomplete")
    limits = re.findall(r"\.maxntid\s+(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", files["ptx"])
    require(limits == [("512" if name == "on" else "256", "1", "1")], "PTX launch bound")
    dependency_paths = parse_dependencies(files["dependencies"], build_root / (name + ".o"))
    header = PAIRED if name == "on" else NATIVE
    require(any(path.endswith("/" + header) for path in dependency_paths), "generated dependency absent")
    require(not any(path.endswith("/prompt_i8_fast.cuh") for path in dependency_paths),
            "native sibling include trap")
    pinned_dependencies = {}
    for path in dependency_paths:
        actual = pathlib.Path(path)
        if not actual.is_absolute():
            actual = source_root / actual
        actual = actual.resolve()
        if actual.is_relative_to(HERE):
            relative = actual.relative_to(HERE).as_posix()
            expected = generation["probe_source_pins"].get(relative)
            require(expected == digest(actual.read_bytes()), "probe dependency changed: " + relative)
        elif actual.is_relative_to(build_root / "generated"):
            require(actual.name == header, "unexpected generated dependency")
        elif actual.is_relative_to(source_root):
            relative = actual.relative_to(source_root).as_posix()
            expected = generation["source_pins"].get(relative)
            require(expected == digest(actual.read_bytes()), "unpinned project dependency: " + relative)
            pinned_dependencies[relative] = expected
    dynamic = 101376 if name == "on" else 100352
    return {"symbol": symbol, "registers": int(registers), "stack_bytes": int(stack),
            "spill_store_bytes": stores, "spill_load_bytes": loads, "local_bytes": int(local),
            "static_shared_bytes": int(shared), "dynamic_shared_bytes": dynamic,
            "total_shared_bytes": dynamic + int(shared), "sass_opcode_counts": dict(sorted(counts.items())),
            "PTX_maxntid": list(map(int, limits[0])),
            "PTX_local_memory_operations": len(re.findall(r"\b(?:ld|st)\.local\b", files["ptx"])),
            "PTX_MMA_instructions": collections.Counter(re.findall(r"\bmma\.sync\.[^\s]+", files["ptx"])),
            "pinned_project_dependencies": pinned_dependencies,
            "object_sha256": digest((build_root / (name + ".o")).read_bytes()),
            "evidence_sha256": {kind: digest(value.encode("utf-8")) for kind, value in files.items()}}


def resource_gate(item):
    return (0 < item["registers"] <= 128 and
            all(item[key] == 0 for key in ("stack_bytes", "spill_store_bytes", "spill_load_bytes", "local_bytes")) and
            item["total_shared_bytes"] <= 101376 and item["PTX_maxntid"] == [512, 1, 1] and
            item["PTX_local_memory_operations"] == 0 and
            not any(count for opcode, count in item["sass_opcode_counts"].items()
                    if opcode.startswith(("LDL", "STL"))))


def self_test():
    valid = {"registers": 128, "stack_bytes": 0, "spill_store_bytes": 0,
             "spill_load_bytes": 0, "local_bytes": 0, "total_shared_bytes": 101376,
             "PTX_maxntid": [512, 1, 1], "PTX_local_memory_operations": 0,
             "sass_opcode_counts": {"IMMA.16832.S8.S8": 8, "HMMA.16816.F16": 64}}
    require(resource_gate(valid), "legal resource boundary rejected")
    for key, value in (("registers", 129), ("stack_bytes", 8), ("spill_store_bytes", 4),
                       ("spill_load_bytes", 4), ("local_bytes", 4),
                       ("total_shared_bytes", 101377), ("PTX_maxntid", [256, 1, 1]),
                       ("PTX_local_memory_operations", 1), ("sass_opcode_counts", {"LDL": 1})):
        require(not resource_gate(dict(valid, **{key: value})), "resource negative control: " + key)
    require(parse_dependencies("/b/on.o: /p/variant.cu \\\n /b/generated/header.cuh\n",
                               pathlib.PurePosixPath("/b/on.o")) == ["/p/variant.cu", "/b/generated/header.cuh"],
            "dependency continuation")
    require(sass_counts("/*0010*/ @!P0 LDL R3, [R2];\n/*0020*/ IMMA.16832.S8.S8 R8, R2, R4, R8;\n")
            == {"LDL": 1, "IMMA.16832.S8.S8": 1}, "predicated local operation parsing")
    return {"status": "PASS", "resource_rejection_negative_controls": 9,
            "compiled_artifact_evidence": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=pathlib.Path)
    parser.add_argument("--build-root", type=pathlib.Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
        return 0
    require(args.source_root is not None and args.build_root is not None, "source/build root required")
    source_root, build_root = args.source_root.resolve(), args.build_root.resolve()
    native, paired, current = generate(source_root)
    generation_bytes = (build_root / "generation.json").read_bytes()
    generation = json.loads(generation_bytes)
    for key in current:
        require(generation.get(key) == current[key], "generation differs: " + key)
    require(generation["cpu_proofs"]["status"] == "PASS", "missing CPU proof")
    require((build_root / "generated" / NATIVE).read_bytes() == native, "OFF bytes changed")
    require((build_root / "generated" / PAIRED).read_bytes() == paired, "ON bytes changed")
    require("V13.1.115" in (build_root / "toolchain.txt").read_text(), "compiler version")
    flags = (build_root / "common-flags.txt").read_text().splitlines()
    expected_flags = ["-O3", "-std=c++20", "-arch=sm_86", "--extended-lambda", "-lineinfo", "-Xptxas=-v",
                      "--frandom-seed=attention-paired-warp", "-I" + str(build_root / "generated"),
                      "-I" + str(source_root / "src"), "-I" + str(source_root / "include")]
    require(flags == expected_flags, "compile flags differ")
    off = variant(build_root, source_root, "off", generation)
    on = variant(build_root, source_root, "on", generation)
    passed = resource_gate(on)
    result = {
        "status": "RESOURCE_PASS_REQUIRES_ORDER_REVIEW" if passed else "REJECT_RESOURCES_NO_GPU",
        "actual_compile_resource_gate": passed, "off": off, "on": on,
        "source_generation_sha256": digest(generation_bytes),
        "manual_PTX_SASS_order_review_required": [
            "QK grp 0..3 and kk 0..1 IMMA/FMA order; rolled group has no local addressing",
            "fmax finite/signed-zero lowering, no reassociation of denominator FP32 adds",
            "same MUFU, half P pack, half V decode/shift, four ordered HMMA steps, FP32 promotion/FMA",
            "all lanes reach four per-page CTA barriers and carry-alias warp synchronization",
            "current K/K-scale dead before P/shift reuse and both PV readers before next same-stage copy",
            "side 1 final quad sum/reciprocal is published before both sides normalize"],
        "known_native_baseline_registers_stack_spill_store_spill_load": [255, 8, 4, 4],
        "observed_native_baseline": [off[key] for key in
                                     ("registers", "stack_bytes", "spill_store_bytes", "spill_load_bytes")],
        "GPU_harness_authorized_by_this_file": False,
        "GPU_correctness_qualified": False, "performance_claim": False,
    }
    print(json.dumps(result, indent=2))
    return 0 if passed else 3


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError, AssertionError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
