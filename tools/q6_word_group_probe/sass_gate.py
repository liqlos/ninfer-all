import argparse
import collections
import hashlib
import json
import pathlib
import re
import shlex

import generate


INSTRUCTION = re.compile(
    r"/\*(?P<pc>[0-9a-fA-F]+)\*/\s*(?:@!?(?:UP|P)(?:T|\d+)\s+)?"
    r"(?P<op>[A-Z][A-Z0-9_.]*)\s*(?P<args>[^;]*);")
VECTOR = re.compile(
    r"(?:q6_word_group_vector_kernel|vec6kernel)IL9ggml_type14ELi([14])ELb0ELi0ELb1ELb0ELi2EE")
MATRIX = re.compile(
    r"mul_mat_qIL9ggml_type14ELi128ELb0EL9ggml_prec30ELb([01])ELb([01])ELb0ELb0EE")
FLOAT_OPS = ("FADD", "FMUL", "FFMA", "F2F", "I2F")
UNPACK_OPS = ("LOP", "SHF", "PRMT", "BFE", "BFI", "VADD", "VABSDIFF", "VSET", "VSHL", "VSHR")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def functions(text):
    anchors = list(re.finditer(r"^\s*Function\s*:\s*(\S+)\s*$", text, re.MULTILINE))
    result = {}
    for index, match in enumerate(anchors):
        end = anchors[index + 1].start() if index + 1 < len(anchors) else len(text)
        name = match.group(1)
        require(name not in result, "duplicate kernel image")
        result[name] = [(int(m["pc"], 16), m["op"], m["args"].strip())
                        for m in INSTRUCTION.finditer(text[match.end():end])]
        require(result[name], "missing selected instructions")
    return result


def resources(text, name):
    match = re.search(r"^\s*Function\s+" + re.escape(name) + r":\s*\n([^\n]+)",
                      text, re.MULTILINE)
    require(match is not None, "selected kernel missing from resource dump")
    result = {key: int(value) for key, value in re.findall(r"([A-Z]+):(\d+)", match.group(1))}
    require({"REG", "STACK", "LOCAL", "SHARED"} <= result.keys(), "incomplete resource dump")
    require(result["STACK"] == result["LOCAL"] == 0, "stack or local memory is nonzero")
    require(0 < result["REG"] <= 255, "invalid register allocation")
    return result


def histogram(items, prefixes=None):
    return collections.Counter(op for _, op, _ in items
                               if prefixes is None or op.startswith(prefixes))


def memory_width(items, operation, width):
    return sum(count for op, count in histogram(items).items()
               if op.startswith(operation) and str(width) in op.split("."))


def vector_loop(items, columns):
    candidates = []
    for end, op, args in items:
        target = re.fullmatch(r"0x([0-9a-fA-F]+)", args)
        if op != "BRA" or not target:
            continue
        begin = int(target.group(1), 16)
        if begin >= end:
            continue
        region = [item for item in items if begin <= item[0] <= end]
        if histogram(region).get("IDP.4A.S8.S8", 0) == 16 * columns:
            candidates.append((begin, end, region))
    minimal = [x for x in candidates if not any(
        x[0] <= y[0] and y[1] <= x[1] and x[:2] != y[:2] for y in candidates)]
    require(len(minimal) == 1, "cannot identify exactly one native-order runtime K loop")
    return minimal[0]


def matrix_inner_regions(items):
    regions = []
    start = 0
    for index, (_, op, _) in enumerate(items):
        if not op.startswith("BAR.SYNC"):
            continue
        region = items[start:index]
        start = index + 1
        if any(code.startswith("IMMA") for _, code, _ in region):
            signature = histogram(region, UNPACK_OPS + FLOAT_OPS + ("IMMA",))
            regions.append(tuple(sorted(signature.items())))
    require(regions, "no barrier-delimited MMQ dot regions")
    return collections.Counter(regions)


def allocation_proxy(registers, threads):
    return ((registers + 7) // 8) * 8 * threads


def dependency_paths(path, source_root):
    text = path.read_text(encoding="utf-8").replace("\\\n", " ")
    fields = re.split(r":\s+", text, maxsplit=1)
    require(len(fields) == 2, "invalid compiler dependencies")
    return {(source_root / item).resolve() for item in shlex.split(fields[1])}


def compare_vector(native, candidate, nr, cr, columns):
    n0, n1, nloop = vector_loop(native, columns)
    c0, c1, cloop = vector_loop(candidate, columns)
    require(histogram(nloop).get("SHFL.IDX", 0) == 32, "native cooperative decoder anchor changed")
    require(histogram(candidate).get("SHFL.IDX", 0) == 0, "candidate retains decode shuffles")
    require(not any(op.startswith(("CALL", "BRA.DIV", "VOTE")) for _, op, _ in cloop),
            "candidate K loop retains cooperative decode helpers")
    require(histogram(nloop, FLOAT_OPS) == histogram(cloop, FLOAT_OPS),
            "vector K-loop FP or conversion counts changed")
    nb = histogram(native).get("SHFL.BFLY", 0)
    cb = histogram(candidate).get("SHFL.BFLY", 0)
    require(nb == cb == 10 * columns, "native butterfly reduction changed")
    wide_credits = memory_width(cloop, "LDG", 64) + 2 * (
        memory_width(cloop, "LDG", 128) - memory_width(nloop, "LDG", 128))
    require(wide_credits >= 4, "four QL64-equivalent code loads were not established")
    require(65536 // allocation_proxy(cr["REG"], 128) >=
            65536 // allocation_proxy(nr["REG"], 128),
            "register residency proxy regressed")
    return {"columns": columns, "native_registers": nr["REG"], "candidate_registers": cr["REG"],
            "native_K_loop": [hex(n0), hex(n1)], "candidate_K_loop": [hex(c0), hex(c1)],
            "candidate_decode_shuffles": 0, "butterfly_shuffles": cb,
            "wide_code_load_credits": wide_credits, "fp_counts_equal": True}


def compare_matrix(native, candidate, nr, cr, mode):
    require(histogram(native, ("IMMA",)) == histogram(candidate, ("IMMA",)),
            "MMQ tensor-core arithmetic changed")
    require(matrix_inner_regions(native) == matrix_inner_regions(candidate),
            "MMQ inner dot regions changed FP/unpack work; inspect SASS")
    require(memory_width(candidate, "LDG", 64) > memory_width(native, "LDG", 64),
            "MMQ QL64 loads were not established")
    require(memory_width(candidate, "STS", 128) > memory_width(native, "STS", 128),
            "MMQ code-word STS128 lowering was not established")
    require(nr["SHARED"] == cr["SHARED"], "static shared extent changed")
    require(allocation_proxy(cr["REG"], 256) <= 65536, "one CTA exceeds register capacity")
    return {"mode": mode, "native_registers": nr["REG"], "candidate_registers": cr["REG"],
            "dynamic_shared_bytes": 57856, "shared_memory_CTA_upper_bound": 1,
            "native_LDG64": memory_width(native, "LDG", 64),
            "candidate_LDG64": memory_width(candidate, "LDG", 64),
            "native_STS128": memory_width(native, "STS", 128),
            "candidate_STS128": memory_width(candidate, "STS", 128),
            "barrier_delimited_dot_region_signatures_equal": True}


def run(root):
    manifest = json.loads((root / "build_manifest.json").read_text(encoding="utf-8"))
    require(manifest["compile_only"] is True and manifest["candidate_enabled"] is True,
            "not an explicitly enabled compile-only build")
    require(len(manifest["units"]) == 2, "the gate requires exactly vector and MMQ units")
    source_root = pathlib.Path(manifest["source_root"])
    probe_root = pathlib.Path(manifest["probe_root"])
    require(sha(root / "toolchain.txt") == manifest["toolchain_sha256"], "toolchain record changed")
    require(re.search(r"\bV" + re.escape(manifest["expected_cuda_version"]) + r"\b",
                      manifest["actual_nvcc_version"]) is not None, "unexpected CUDA build")
    for name, expected in manifest["probe_sha256"].items():
        require(sha(probe_root / name) == expected, "probe changed after build: " + name)
    generated, _ = generate.prepare(source_root)
    for name, content in generated.items():
        require((root / "generated" / name).read_bytes() == content.encode(),
                "generated header or source drift")
    selected = {}
    for unit in manifest["units"]:
        require(unit["returncode"] == 0, "selected compilation failed")
        command = unit["argv"]
        require("-arch=sm_86" in command and "-DNINFER_Q6_WORD_GROUP_PROBE=1" in command and
                "-DNINFER_GGUF_Q6_COOPERATIVE_DECODE=1" in command and "-c" in command,
                "selected compiler flags are missing")
        require(not any("fast-math" in arg or "use_fast_math" in arg for arg in command),
                "fast math is outside this experiment")
        for name, expected in unit["artifact_sha256"].items():
            require(sha(root / name) == expected, "compile artifact changed: " + name)
        dependencies = dependency_paths(root / unit["dependency"], source_root)
        require({str(path) for path in dependencies} == set(unit["dependency_sha256"]),
                "dependency receipt is incomplete")
        for name, expected in unit["dependency_sha256"].items():
            require(sha(pathlib.Path(name)) == expected, "compiled dependency changed: " + name)
        expected_header = root / "generated" / (
            "q6_word_group_vector.cuh" if unit["name"] == "vector" else "q6_word_group_mmq.cuh")
        require(expected_header.resolve() in dependencies, "selected generated header was not compiled")
        log = (root / unit["log"]).read_text(encoding="utf-8")
        spills = re.findall(r"(\d+) bytes spill stores, (\d+) bytes spill loads", log)
        require(spills and all(int(a) == int(b) == 0 for a, b in spills),
                "missing spill reports or nonzero spills")
        resource_text = (root / unit["resources"]).read_text(encoding="utf-8")
        for name, items in functions((root / unit["sass"]).read_text(encoding="utf-8")).items():
            match = VECTOR.search(name) if unit["name"] == "vector" else MATRIX.search(name)
            if not match:
                continue
            candidate = "q6_word_group_" in name
            mode = int(match.group(1)) if unit["name"] == "vector" else tuple(map(int, match.groups()))
            key = unit["name"], mode, candidate
            require(key not in selected, "duplicate selected specialization")
            require(not any(op.startswith(("LDL", "STL")) for _, op, _ in items),
                    "selected kernel contains local-memory traffic")
            selected[key] = items, resources(resource_text, name)
    result = {"vector": [], "matrix": []}
    require(len(selected) == 10, "expected four vector and six matrix kernels")
    for columns in (1, 4):
        native, nr = selected["vector", columns, False]
        candidate, cr = selected["vector", columns, True]
        result["vector"].append(compare_vector(native, candidate, nr, cr, columns))
    for mode, label in (((0, 0), "PlainFP32"), ((1, 0), "StoreBF16"), ((1, 1), "AccumulateBF16")):
        native, nr = selected["matrix", mode, False]
        candidate, cr = selected["matrix", mode, True]
        result["matrix"].append(compare_matrix(native, candidate, nr, cr, label))
    result.update({
        "status": "PASS_COMPILE_RESOURCE_AND_MECHANISM_SCREEN",
        "numerical_outputs": "not_qualified", "gpu_run": False, "performance_claim": False,
        "register_residency": "8-register allocation proxy; runtime occupancy not queried",
        "hardware_reference": "https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html",
        "next_gate": "actual full-fixture roundtrip, independent numerical oracle and bounded GPU screen",
    })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-root", required=True, type=pathlib.Path)
    args = parser.parse_args()
    try:
        result = run(args.build_root.resolve())
    except (ValueError, KeyError, OSError) as error:
        print(json.dumps({"status": "FAIL_COMPILE_RESOURCE_OR_MECHANISM_GATE",
                          "reason": str(error), "gpu_run": False}, indent=2))
        raise SystemExit(1)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
