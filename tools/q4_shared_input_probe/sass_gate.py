import argparse
import collections
import hashlib
import json
import pathlib
import re
import shlex
import struct


SIGNATURE = re.compile(r"vec6kernelIL9ggml_type12ELi4ELb1ELi5120ELb1ELb0ELi2EEEv")
INSTRUCTION = re.compile(
    r"/\*(?P<pc>[0-9a-fA-F]+)\*/\s*(?P<pred>@!?(?:UP|P)(?:T|\d+)\s+)?"
    r"(?P<op>[A-Z][A-Z0-9_.]*)\s*(?P<args>[^;]*);")
MATH = ("IDP.4A", "FFMA", "FADD", "FMUL", "SHFL.BFLY", "MUFU", "F2F", "F2FP", "RRO")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def elf_text(data, symbol):
    if len(data) < 64 or data[:6] != b"\x7fELF\x02\x01":
        raise ValueError("expected a little-endian ELF64 cubin")
    header = struct.unpack_from("<16sHHIQQQIHHHHHH", data)
    if header[2] != 190:
        raise ValueError("ELF is not an NVIDIA CUDA cubin")
    offset, stride, count, names_index = header[6], header[11], header[12], header[13]
    if stride != 64 or count == 0 or names_index >= count or offset + count * stride > len(data):
        raise ValueError("invalid ELF section table")
    sections = [struct.unpack_from("<IIQQQQIIQQ", data, offset + i * stride) for i in range(count)]

    def payload(section):
        start, size = section[4], section[5]
        if start + size > len(data):
            raise ValueError("ELF section extends outside the cubin")
        return data[start:start + size]

    names = payload(sections[names_index])
    matches = []
    for section in sections:
        index = section[0]
        if index >= len(names):
            raise ValueError("invalid ELF section name")
        end = names.find(b"\0", index)
        if end < 0:
            raise ValueError("unterminated ELF section name")
        name = names[index:end].decode("ascii")
        if name == ".text." + symbol:
            if section[1] != 1 or not section[2] & 4:
                raise ValueError("selected ELF section is not executable PROGBITS")
            matches.append(payload(section))
    if len(matches) != 1 or not matches[0]:
        raise ValueError("selected kernel does not have one nonempty machine-text section")
    return matches[0]


def selected_sass(text):
    entries = list(re.finditer(r"^\s*Function\s*:\s*(\S+)\s*$", text, re.MULTILINE))
    selected = []
    for index, match in enumerate(entries):
        if SIGNATURE.search(match.group(1)):
            end = entries[index + 1].start() if index + 1 < len(entries) else len(text)
            selected.append((match.group(1), text[match.end():end]))
    if len(selected) != 1:
        raise ValueError("require one selected Q4/T4/fused/KS5120 SM86 kernel")
    symbol, body = selected[0]
    items = [{"pc": int(m.group("pc"), 16), "predicate": (m.group("pred") or "PT").strip(),
              "opcode": m.group("op"), "operands": m.group("args").strip()}
             for m in INSTRUCTION.finditer(body)]
    if not items or len({item["pc"] for item in items}) != len(items):
        raise ValueError("empty or ambiguous selected SASS")
    return symbol, items


def resources(text, symbol):
    pattern = (r"^\s*Function\s*:?[ \t]*" + re.escape(symbol) +
               r"\s*:?\s*\n(?P<body>.*?)(?=^\s*Function|\Z)")
    matches = list(re.finditer(pattern, text, re.MULTILINE | re.DOTALL))
    if len(matches) != 1:
        raise ValueError("selected resource record is absent or duplicated")
    found = {key: int(value) for key, value in re.findall(r"\b([A-Z]+):(\d+)", matches[0].group("body"))}
    if not {"REG", "STACK", "LOCAL", "SHARED"} <= set(found):
        raise ValueError("resource record is incomplete")
    return found


def ptxas_stack(text, symbol):
    pattern = (r"Function properties for\s+" + re.escape(symbol) +
               r"\s*\n\s*(\d+) bytes stack frame,\s*(\d+) bytes spill stores,\s*(\d+) bytes spill loads")
    matches = re.findall(pattern, text)
    if len(matches) != 1:
        raise ValueError("selected ptxas stack/spill record is absent or duplicated")
    return dict(zip(("stack_bytes", "spill_store_bytes", "spill_load_bytes"), map(int, matches[0])))


def resource_bound(record, stack, dynamic):
    if not 0 < record["REG"] <= 128 or any(record[k] for k in ("STACK", "LOCAL")) or any(stack.values()):
        return {"status": "FAIL_REGISTERS_STACK_LOCAL_OR_SPILLS"}
    registers_per_block = ((record["REG"] + 7) // 8 * 8) * 128
    shared_per_block = ((record["SHARED"] + dynamic + 255) // 256 * 256) + 1024
    capacities = {"registers": 65536 // registers_per_block,
                  "shared": 102400 // shared_per_block,
                  "threads": 1536 // 128, "blocks": 16}
    return {"status": "PASS_DOCUMENTED_SM86_FOUR_CTA_BOUND" if min(capacities.values()) >= 4
                      else "FAIL_FOUR_CTA_RESOURCE_BOUND",
            "register_allocation_per_block": registers_per_block,
            "static_shared_bytes": record["SHARED"], "dynamic_shared_bytes": dynamic,
            "shared_allocation_with_reserved_1KiB": shared_per_block,
            "capacities": capacities, "minimum_capacity": min(capacities.values()),
            "actual_cuda_occupancy_query_run": False,
            "actual_occupancy_required_before_GPU_harness_admission": True}


def row_loop(items):
    loops = []
    for item in items:
        if item["opcode"] not in ("BRA", "BRA.U"):
            continue
        target = re.fullmatch(r"0x([0-9a-fA-F]+)", item["operands"])
        if not target or int(target.group(1), 16) >= item["pc"]:
            continue
        begin, end = int(target.group(1), 16), item["pc"]
        body = [entry for entry in items if begin <= entry["pc"] <= end]
        if (any(entry["opcode"].startswith("IDP.4A") for entry in body) and
                any(entry["opcode"].startswith("SHFL.BFLY") for entry in body)):
            loops.append((begin, end, body))
    minimal = [loop for loop in loops if not any(
        other[0] >= loop[0] and other[1] <= loop[1] and other[:2] != loop[:2] for other in loops)]
    if len(minimal) != 1:
        raise ValueError("cannot uniquely locate native unrolled-K grid-stride row loop")
    return minimal[0]


def histogram(items, prefixes=None):
    return dict(sorted(collections.Counter(item["opcode"] for item in items
                       if prefixes is None or item["opcode"].startswith(prefixes)).items()))


def structural_review(off, on):
    result = {"status": "MANUAL_REQUIRED", "operand_aware_math_DAG_proven": False,
              "input_LDG_pointer_provenance_proven": False,
              "control_flow_equivalence_proven": False}
    try:
        ob, oe, old = row_loop(off)
        nb, ne, new = row_loop(on)
    except ValueError as error:
        result["parser_limit"] = str(error)
        return result
    old_barriers = [item for item in old if item["opcode"].startswith("BAR.")]
    new_barriers = [item for item in new if item["opcode"].startswith("BAR.")]
    if len(new_barriers) > len(old_barriers):
        result["status"] = "FAIL_NEW_CTA_BARRIER_IN_ROW_LOOP"
    shared_loads = [item for item in new if item["opcode"].startswith("LDS")]
    if not shared_loads and not result["status"].startswith("FAIL"):
        result["status"] = "FAIL_NO_SHARED_LOAD_IN_ROW_LOOP"
    old_math, new_math = histogram(old, MATH), histogram(new, MATH)
    result.update({
        "OFF_row_loop": [hex(ob), hex(oe)], "ON_row_loop": [hex(nb), hex(ne)],
        "OFF_row_math_histogram": old_math, "ON_row_math_histogram": new_math,
        "row_math_counts_equal_not_equivalence_proof": old_math == new_math,
        "OFF_row_CTA_barriers": old_barriers, "ON_row_CTA_barriers": new_barriers,
        "ON_CTA_barriers_outside_row_loop": [item for item in on
            if item["opcode"].startswith("BAR.") and not nb <= item["pc"] <= ne],
        "ON_shared_loads_inside_row_loop": shared_loads,
        "ON_global_loads_outside_row_loop": [item for item in on
            if item["opcode"].startswith("LDG") and not nb <= item["pc"] <= ne],
        "ON_global_loads_inside_row_loop": [item for item in new if item["opcode"].startswith("LDG")],
        "manual_required": [
            "Bind qs and ds kernel arguments to every ON LDG; all input LDG must be before the row loop.",
            "Confirm both code halves load via LDS.128 in the row loop and literal half2 metadata via LDS.",
            "Confirm exactly one added CTA barrier before rows, including all alternate control-flow edges.",
            "Compare DP4A operands, ordered FP32 dot/accumulate, butterfly, SwiGLU and BF16 epilogue DAGs.",
            "Confirm original weight load sequence and static five-slice KS5120 unroll.",
        ],
    })
    return result


def dependencies(path, target, source_root):
    text = path.read_text(encoding="utf-8").replace("\\\n", " ").strip()
    escaped = str(target).replace("\\", "\\\\").replace(" ", "\\ ").replace("#", "\\#")
    prefixes = (str(target) + ":", escaped + ":")
    prefix = next((value for value in prefixes if text.startswith(value)), None)
    if prefix is None:
        raise ValueError("dependency target does not match the exact compiled cubin")
    return {(source_root / name).resolve() for name in shlex.split(text[len(prefix):], posix=True)}


def run(root):
    generated = root / "generated"
    receipt = json.loads((generated / "source_receipt.json").read_text(encoding="utf-8"))
    if receipt["status"] != "PASS_PINNED_SOURCE_ONLY":
        raise ValueError("missing pinned source generation")
    source_root, probe_root = pathlib.Path(receipt["source_root"]), pathlib.Path(receipt["probe_root"])
    for directory, records in ((source_root, receipt["source_files"]),
                               (probe_root, receipt["probe_files"]),
                               (generated, receipt["generated_files"])):
        for name, digest in records.items():
            if sha((directory / name).read_bytes()) != digest:
                raise ValueError("input changed after source generation: " + str(directory / name))
    common_flags = (root / "common-flags.txt").read_text().strip()
    variants = {}
    text_sections = {}
    all_items = {}
    baseline_symbol = None
    for variant in ("native", "off", "on"):
        cubin = root / (variant + ".cubin")
        command = (root / ("command-" + variant + ".txt")).read_text().strip()
        expected = (f"nvcc {common_flags} --cubin --keep --keep-dir {root / ('keep-' + variant)} "
                    f"--generate-dependencies-with-compile --dependency-output {root / (variant + '.d')} "
                    f"{probe_root / ('variant_' + variant + '.cu')} -o {cubin}")
        if command != expected:
            raise ValueError("compile command does not use the identical common flags")
        deps = dependencies(root / (variant + ".d"), cubin, source_root)
        required = {probe_root / "probe_contract.cuh", probe_root / "instantiate.cuh",
                    probe_root / ("variant_" + variant + ".cu"),
                    generated / "q4_shared_input_flags_generated.cuh"}
        required.update(source_root / name for name in receipt["source_files"]
                        if name not in ("src/ops/linear/gguf/ggml_bridge_vec.cuh", "src/ops/common/mma.cuh"))
        selected_header = (source_root / "src/ops/linear/gguf/ggml_bridge_vec.cuh" if variant == "native"
                           else generated / "q4_shared_input_generated.cuh")
        required.add(selected_header)
        if not {path.resolve() for path in required} <= deps:
            raise ValueError("actual dependencies omitted required pinned inputs")
        native_header = (source_root / "src/ops/linear/gguf/ggml_bridge_vec.cuh").resolve()
        if variant != "native" and native_header in deps:
            raise ValueError("original vector header leaked into a generated-header translation unit")
        symbol, items = selected_sass((root / ("sass-" + variant + ".txt")).read_text())
        if baseline_symbol is None:
            baseline_symbol = symbol
        elif baseline_symbol != symbol:
            raise ValueError("kernel template signature differs between variants")
        record = resources((root / ("resources-" + variant + ".txt")).read_text(), symbol)
        stack = ptxas_stack((root / ("compile-" + variant + ".log")).read_text(), symbol)
        machine_text = elf_text(cubin.read_bytes(), symbol)
        ptx_files = list((root / ("keep-" + variant)).glob("*.ptx"))
        if len(ptx_files) != 1:
            raise ValueError("require one kept PTX file for each compile")
        ptx = ptx_files[0].read_text()
        if ".target sm_86" not in ptx:
            raise ValueError("kept PTX does not target SM86")
        dynamic = 23040 if variant == "on" else 0
        variants[variant] = {
            "cubin_sha256": sha(cubin.read_bytes()), "kernel_text_sha256": sha(machine_text),
            "kernel_text_bytes": len(machine_text), "resources": record, "ptxas": stack,
            "resource_bound": resource_bound(record, stack, dynamic),
            "PTX_sha256": sha(ptx_files[0].read_bytes()),
            "dynamic_shared_declaration_seen": bool(re.search(
                r"\.extern\s+\.shared\s+\.align\s+16\s+\.b8\s+\S*resident_q8\[\]", ptx)),
            "opcode_histogram": histogram(items),
        }
        text_sections[variant] = machine_text
        all_items[variant] = items
    identity = text_sections["native"] == text_sections["off"]
    structural = structural_review(all_items["off"], all_items["on"])
    resource_pass = all(value["resource_bound"]["status"] == "PASS_DOCUMENTED_SM86_FOUR_CTA_BOUND"
                        for value in variants.values())
    dynamic_shared_pass = (variants["on"]["dynamic_shared_declaration_seen"] and
                           not variants["native"]["dynamic_shared_declaration_seen"] and
                           not variants["off"]["dynamic_shared_declaration_seen"])
    hard_fail = (not identity or not resource_pass or not dynamic_shared_pass or
                 structural["status"].startswith("FAIL"))
    return {
        "status": "FAIL_COMPILE_GATE" if hard_fail else "COMPILE_RESOURCE_PASS_MANUAL_MECHANISM_REQUIRED",
        "selected_symbol": baseline_symbol,
        "OFF_native_machine_text_byte_identity": identity,
        "candidate_only_dynamic_shared_declaration": dynamic_shared_pass,
        "identical_common_flags": common_flags,
        "toolchain": (root / "toolchain.txt").read_text().strip(),
        "source_receipt_sha256": sha((generated / "source_receipt.json").read_bytes()),
        "analyzer_sha256": sha(pathlib.Path(__file__).read_bytes()),
        "variants": variants, "mechanism": structural,
        "ready_for_host_harness": False,
        "runtime_shape_guards_and_actual_occupancy_pending": True,
        "numerical_validation_run": False, "GPU_run": False, "performance_claim": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-root", required=True, type=pathlib.Path)
    args = parser.parse_args()
    result = run(args.build_root.resolve())
    print(json.dumps(result, indent=2))
    if result["status"] == "FAIL_COMPILE_GATE":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
