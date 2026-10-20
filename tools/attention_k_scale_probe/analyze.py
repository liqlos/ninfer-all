import argparse
import collections
import json
import pathlib
import re
import shlex
import sys

from generate import GENERATED, PINS, prepare, require, sha256


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


def sass_counts(source):
    return collections.Counter(re.findall(
        r"/\*[0-9a-fA-F]+\*/\s+(?:@!?P\w+\s+)?([A-Z][A-Z0-9_.]*)(?:\s|;)", source))


def ptx_locations(source, ranges):
    files = {int(number): path for number, path in re.findall(r'\.file\s+(\d+)\s+"([^"]+)"', source)}
    generated_ids = {number for number, path in files.items() if pathlib.PurePosixPath(path).name == GENERATED}
    require(len(generated_ids) == 1, "generated header PTX file mapping missing or ambiguous")
    location = None
    conversions = collections.Counter()
    producer = []
    for line in source.splitlines():
        loc = re.match(r"\s*\.loc\s+(\d+)\s+(\d+)\s+\d+", line)
        if loc:
            location = tuple(map(int, loc.groups()))
        if location is None or location[0] not in generated_ids:
            continue
        instruction = line.strip()
        if "cvt.f32.f16" in instruction:
            for region, (first, last) in ranges.items():
                if first <= location[1] <= last:
                    conversions[region] += 1
        first, last = ranges["producer"]
        if first <= location[1] <= last and re.search(r"\b(?:ld\.global|st\.shared|cvt\.)", instruction):
            producer.append({"generated_line": location[1], "instruction": instruction})
    return {"direct_generated_line_half_to_float_counts": dict(conversions),
            "all_half_to_float_count": len(re.findall(r"\bcvt\.f32\.f16\b", source)),
            "producer_load_convert_store_instructions": producer,
            "line_attribution_limit": "only direct .loc attribution; absent attribution is not proof of absent conversion"}


def variant(build_root, source_root, name, generation):
    files = {kind: (build_root / filename).read_text(encoding="utf-8") for kind, filename in {
        "resource": "resources-" + name + ".txt", "compile": "compile-" + name + ".log",
        "sass": "sass-" + name + ".txt", "ptx": "ptx-" + name + ".txt",
        "dependencies": name + ".d"}.items()}
    matches = re.findall(r"\bFunction\s+(\S+):\s*\n\s+REG:(\d+)\s+STACK:(\d+)\s+SHARED:(\d+)\s+LOCAL:(\d+)",
                         files["resource"])
    require(len(matches) == 1, "expected one native kernel resource record: " + name)
    symbol, registers, stack, shared, local = matches[0]
    require("causal_attention_prompt_i8_fast_k_scale_" + name + "_kernel" in symbol, "wrong kernel variant")
    require("arch = sm_86" in files["resource"], "wrong cubin architecture")
    spills = re.findall(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads",
                        files["compile"])
    require(len(spills) == 1, "ambiguous ptxas stack/spill record")
    ptx_stack, stores, loads = map(int, spills[0])
    require(ptx_stack == int(stack), "ptxas/cubin stack mismatch")
    counts = sass_counts(files["sass"])
    require(any(op.startswith("IMMA") for op in counts), "SASS instruction parse incomplete")
    dependency_paths = parse_dependencies(files["dependencies"], build_root / (name + ".o"))
    require(any(path.endswith("/" + GENERATED) for path in dependency_paths), "generated include dependency missing")
    require(not any(path.endswith("/prompt_i8_fast.cuh") for path in dependency_paths), "native sibling include trap")
    local_dependencies = {}
    probe_root = pathlib.Path(__file__).resolve().parent
    for path in dependency_paths:
        actual = pathlib.Path(path)
        if not actual.is_absolute():
            actual = source_root / actual
        actual = actual.resolve()
        if actual.is_relative_to(source_root):
            relative = actual.relative_to(source_root).as_posix()
            require(relative in PINS, "unpinned project dependency: " + relative)
            require(sha256(actual.read_bytes()) == PINS[relative], "changed project dependency: " + relative)
            local_dependencies[relative] = PINS[relative]
        elif actual.is_relative_to(probe_root):
            relative = actual.relative_to(probe_root).as_posix()
            expected = generation["probe_source_pins"].get(relative)
            require(expected == sha256(actual.read_bytes()), "changed probe dependency: " + relative)
    dynamic = 100352 if name == "off" else 101376
    return {"symbol": symbol, "registers": int(registers), "stack_bytes": int(stack),
            "spill_store_bytes": stores, "spill_load_bytes": loads, "local_bytes": int(local),
            "static_shared_bytes": int(shared), "dynamic_shared_bytes": dynamic,
            "total_shared_bytes": dynamic + int(shared), "sass_opcode_counts": dict(sorted(counts.items())),
            "PTX": ptx_locations(files["ptx"], generation["generated_line_ranges"]),
            "pinned_project_dependencies": local_dependencies,
            "object_sha256": sha256((build_root / (name + ".o")).read_bytes()),
            "evidence_sha256": {kind: sha256(value.encode("utf-8")) for kind, value in files.items()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=pathlib.Path, default=pathlib.Path("/src"))
    parser.add_argument("--build-root", type=pathlib.Path, required=True)
    parser.add_argument("--diagnostic-original-analyzer-sha256")
    parser.add_argument("--diagnostic-generation-sha256")
    args = parser.parse_args()
    root = args.build_root.resolve()
    source_root = args.source_root.resolve()
    generated, prepared = prepare(source_root)
    generation_bytes = (root / "generation.json").read_bytes()
    generation = json.loads(generation_bytes)
    generation_sha256 = sha256(generation_bytes)
    analyzer_sha256 = sha256(pathlib.Path(__file__).read_bytes())
    diagnostic = args.diagnostic_original_analyzer_sha256 is not None
    require(diagnostic == (args.diagnostic_generation_sha256 is not None),
            "diagnostic mode requires both original analyzer and generation SHA256")
    if diagnostic:
        require(generation_sha256 == args.diagnostic_generation_sha256,
                "diagnostic original generation SHA256 mismatch")
        require(generation["probe_source_pins"]["analyze.py"] == args.diagnostic_original_analyzer_sha256,
                "diagnostic original analyzer SHA256 mismatch")
    require(generation["source_pins"] == PINS, "generation source pins differ")
    require(generation["generated_sha256"] == sha256(generated), "generation digest differs")
    require((root / "generated" / GENERATED).read_bytes() == generated, "generated build bytes differ")
    require(generation["generated_line_ranges"] == prepared["generated_line_ranges"], "source location ranges differ")
    for name, digest in generation["probe_source_pins"].items():
        if diagnostic and name == "analyze.py":
            continue
        require(sha256((pathlib.Path(__file__).parent / name).read_bytes()) == digest, "probe changed after generation: " + name)
    require("V13.1.115" in (root / "toolchain.txt").read_text(), "unexpected compiler version")
    off = variant(root, source_root, "off", generation)
    on = variant(root, source_root, "on", generation)
    resource = (on["registers"] <= 255 and
                all(on[key] == 0 for key in ("stack_bytes", "spill_store_bytes", "spill_load_bytes", "local_bytes")) and
                on["total_shared_bytes"] <= 101376 and
                not any(count for opcode, count in on["sass_opcode_counts"].items() if opcode.startswith(("LDL", "STL"))))
    families = ("IMMA", "HMMA", "LDSM", "FADD", "FMUL", "FFMA", "MUFU", "HADD2", "HMUL2", "HFMA2", "SHFL")
    math_counts = lambda item: {op: count for op, count in item["sass_opcode_counts"].items() if op.startswith(families)}
    math_same = math_counts(off) == math_counts(on)
    barriers = lambda item: {op: count for op, count in item["sass_opcode_counts"].items() if op.startswith("BAR")}
    barriers_same = barriers(off) == barriers(on) and sum(barriers(on).values()) == 1
    qk_pv = all(sum(count for op, count in item["sass_opcode_counts"].items() if op.startswith(family)) == expected
                for item in (off, on) for family, expected in (("IMMA", 64), ("HMMA", 128)))
    conversions_off = off["PTX"]["direct_generated_line_half_to_float_counts"]
    conversions_on = on["PTX"]["direct_generated_line_half_to_float_counts"]
    conversion_moved = (conversions_off.get("qk", 0) > 0 and conversions_on.get("qk", 0) == 0 and
                        conversions_on.get("producer", 0) > 0)
    passed = resource and math_same and barriers_same and qk_pv and conversion_moved
    result = {"status": "PASS_STATIC_GATE_REQUIRES_ORDER_REVIEW" if passed else "REJECT_OR_UNRESOLVED_STATIC_GATE",
              "resource_gate": resource, "same_selected_SASS_math_counts": math_same,
              "same_single_CTA_barrier": barriers_same,
              "native_IMMA64_HMMA128": qk_pv, "selected_PTX_conversion_moved": conversion_moved,
              "native_baseline_stack_spill_store_spill_load_expected": [8, 4, 4],
              "native_baseline_observed": [off[key] for key in ("stack_bytes", "spill_store_bytes", "spill_load_bytes")],
              "source_generation_sha256": generation_sha256,
              "parser_provenance": {
                  "mode": "read_only_diagnostic_of_existing_objects" if diagnostic else "compile_manifest_analyzer",
                  "original_compile_analyzer_sha256": generation["probe_source_pins"]["analyze.py"],
                  "current_analyzer_sha256": analyzer_sha256,
                  "analyzer_matches_compile_manifest": analyzer_sha256 == generation["probe_source_pins"]["analyze.py"],
                  "compiled_probe_source_pins": generation["probe_source_pins"],
              },
              "off": off, "on": on,
              "manual_PTX_SASS_order_review_required": ["producer global vector loads and FP32 shared stores",
                                                        "no new CTA barrier or staged-half reads",
                                                        "same wait/publish/read/stage-reuse order"],
              "GPU_correctness_qualified": False, "performance_claim": False}
    print(json.dumps(result, indent=2))
    return 0 if passed else 3


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
