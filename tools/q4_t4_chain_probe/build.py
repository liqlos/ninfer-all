import argparse
import hashlib
import json
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import sys
import time

import generate
import symbolic


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def invoke(argv, cwd):
    options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    return subprocess.run(argv, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, check=False, **options)


def resource_review(resource_text, sass_text, compile_log):
    records = {}
    for match in re.finditer(r"^\s*Function\s+(\S+):\s*\n([^\n]+)", resource_text, re.MULTILINE):
        records[match[1]] = {k: int(v) for k, v in re.findall(r"([A-Z]+):(\d+)", match[2])}
    candidate = [n for n in records if "q4_t4_chain_kernel" in n]
    native = [n for n in records if re.search(
        r"kernelIL9ggml_type12ELi4ELb0ELi5120ELb1ELb0ELi2EE", n)]
    failures = []
    if len(candidate) != 1 or len(native) != 1:
        failures.append("expected exactly one native W kernel and one chain kernel")
    selected = {n: records[n] for n in candidate + native}
    for name, values in selected.items():
        if not {"REG", "STACK", "LOCAL", "SHARED"} <= values.keys():
            failures.append("incomplete resource report: " + name)
            continue
        if values["STACK"] or values["LOCAL"]:
            failures.append("nonzero stack/local: " + name)
        if name in candidate and (not 0 < values["REG"] <= 80 or values["SHARED"] != 2048):
            failures.append("candidate exceeds 80 registers or differs from 2 KiB terminal shared")
    spill_reports = re.findall(r"(\d+) bytes spill stores, (\d+) bytes spill loads", compile_log)
    if not spill_reports or any(int(a) or int(b) for a, b in spill_reports):
        failures.append("missing spill reports or nonzero spills")
    functions = list(re.finditer(r"^\s*Function\s*:\s*(\S+)\s*$", sass_text, re.MULTILINE))
    histograms = {}
    calls = {}
    mechanism_failures = []
    for index, match in enumerate(functions):
        name = match[1]
        if name not in selected:
            continue
        end = functions[index + 1].start() if index + 1 < len(functions) else len(sass_text)
        instructions = list(re.finditer(
            r"/\*([0-9a-fA-F]+)\*/\s*(@!?(?:UP|P)(?:T|\d+)\s+)?"
            r"([A-Z][A-Z0-9_.]*)\s*([^;]*);", sass_text[match.end():end]))
        ops = [item[3] for item in instructions]
        counts = {op: ops.count(op) for op in sorted(set(ops))}
        histograms[name] = counts
        calls[name] = []
        for position, item in enumerate(instructions):
            if not item[3].startswith("CALL"):
                continue
            target = re.search(r"0x[0-9a-fA-F]+", item[4])
            calls[name].append({
                "pc": "0x" + item[1],
                "predicate": (item[2] or "").strip(),
                "operation": item[3],
                "operands": item[4].strip(),
                "target": target[0] if target else None,
                "previous_operation": ops[position - 1] if position else None,
                "next_operation": ops[position + 1] if position + 1 < len(ops) else None,
                "classification": "REQUIRES_CONTROL_FLOW_REVIEW",
            })
        if any(op.startswith(("LDL", "STL")) for op in ops):
            failures.append("local memory instruction: " + name)
        expected = "IMMA" if name in candidate else "IDP"
        if not any(op.startswith(expected) for op in ops):
            mechanism_failures.append("missing " + expected + " mechanism: " + name)
    if set(histograms) != set(selected):
        mechanism_failures.append("selected SASS function missing")
    return {
        "status": ("RESOURCE_FAIL" if failures else
                   "MECHANISM_FAIL" if mechanism_failures else "RESOURCE_PASS_REVIEW_REQUIRED"),
        "resource_status": "FAIL" if failures else "PASS",
        "mechanism_status": "FAIL" if mechanism_failures else "REVIEW_REQUIRED",
        "failures": failures + mechanism_failures,
        "resource_failures": failures,
        "mechanism_failures": mechanism_failures,
        "resources": selected,
        "spill_reports": spill_reports,
        "whole_function_instruction_counts": histograms,
        "whole_function_calls": calls,
        "CALL_policy": "diagnostic control flow evidence, not a stack/local/spill failure",
        "required_manual_review": [
            "CALL targets and reachability in projection, reduction, and supported epilogues",
            "exact five native FP32 updates per chain and compiled FP32 operation order",
            "MMA low/high calls retain one shared set of four packed fragment loads per pair",
            "no duplicated decode or operand reload introduced by lowering",
            "terminal expression implements original XOR16/8/4/2/1 tree",
        ],
        "GPU_correctness": "NOT_RUN",
        "full_Gemm_helperpack": "NOT_RUN",
        "performance_claim": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--enable-candidate", action="store_true")
    parser.add_argument("--expected-cuda-version", default="13.1.115")
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument("--cuobjdump", default="cuobjdump")
    args = parser.parse_args()
    if not args.enable_candidate:
        parser.error("candidate defaults OFF; explicit --enable-candidate is required")
    if os.environ.get("NVCC_PREPEND_FLAGS") or os.environ.get("NVCC_APPEND_FLAGS"):
        parser.error("ambient NVCC flags must be empty")
    root, out = args.source_root.resolve(), args.output_dir.resolve()
    probe = pathlib.Path(__file__).resolve().parent
    if out == root or root in out.parents:
        parser.error("objects belong in the existing external build area")
    nvcc, dump = shutil.which(args.nvcc), shutil.which(args.cuobjdump)
    if not nvcc or not dump:
        parser.error("nvcc and cuobjdump are required")
    header, source_receipt = generate.prepare(root)
    proof = symbolic.prove(root)
    version = invoke([nvcc, "--version"], root)
    if (version.returncode or not re.fullmatch(r"\d+\.\d+\.\d+", args.expected_cuda_version) or
            not re.search(r"\bV" + re.escape(args.expected_cuda_version) + r"\b", version.stdout)):
        raise RuntimeError("nvcc differs from explicitly selected CUDA version")
    out.mkdir(parents=True, exist_ok=True)
    generated = out / "generated"
    generated.mkdir(exist_ok=True)
    (generated / "native_q4.cuh").write_bytes(header.encode())
    (generated / "source_receipt.json").write_text(
        json.dumps(source_receipt, indent=2) + "\n", encoding="utf-8")
    (out / "symbolic.json").write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
    flags = ["-O3", "-std=c++17", "-arch=sm_86", "--extended-lambda", "-lineinfo",
             "-Xptxas=-v", "--frandom-seed=q4-t4-chain-probe",
             "-DNINFER_Q4_T4_CHAIN_PROBE=1"]
    flags.extend("-D" + name + "=1" for name, enabled in generate.W_FLAGS.items() if enabled)
    for path in (generated, probe, root / "src", root / "include",
                 root / "third_party/ggml-quants/include",
                 root / "third_party/ggml-quants/src",
                 root / "third_party/ggml-quants/src/ggml-cuda"):
        flags.extend(["-I", str(path)])
    obj, dep = out / "chain.o", out / "chain.d"
    command = [nvcc, *flags, "--generate-dependencies-with-compile", "--dependency-output",
               str(dep), "-c", str(probe / "chain_probe.cu"), "-o", str(obj)]
    started = time.monotonic()
    result = invoke(command, root)
    (out / "compile.log").write_text(result.stdout, encoding="utf-8")
    manifest = {
        "compile_only": True, "host_main": False, "link": False, "GPU_run": False,
        "candidate_default": "OFF",
        "source_receipt": source_receipt,
        "toolchain": version.stdout, "expected_cuda_version": args.expected_cuda_version,
        "python": sys.version, "command": command, "compile_returncode": result.returncode,
        "compile_seconds": time.monotonic() - started,
        "probe_sha256": {p.name: sha(p) for p in sorted(probe.iterdir())
                         if p.suffix in (".py", ".cu", ".h")},
    }
    (out / "build_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if result.returncode:
        raise RuntimeError("isolated compile failed; see compile.log")
    for option, name in (("--dump-resource-usage", "resources.txt"), ("--dump-sass", "sass.txt")):
        dumped = invoke([dump, option, str(obj)], root)
        (out / name).write_text(dumped.stdout, encoding="utf-8")
        if dumped.returncode:
            raise RuntimeError(option + " failed")
    dependencies = dep.read_text().replace("\\\n", " ").split(":", 1)[1]
    paths = [(pathlib.Path(p) if pathlib.Path(p).is_absolute() else root / p).resolve()
             for p in shlex.split(dependencies)]
    manifest["compiled_dependency_sha256"] = {str(p): sha(p) for p in sorted(set(paths))}
    manifest["artifact_sha256"] = {
        name: sha(out / name) for name in ("chain.o", "chain.d", "compile.log", "resources.txt", "sass.txt")}
    review = resource_review((out / "resources.txt").read_text(),
                             (out / "sass.txt").read_text(), result.stdout)
    (out / "resource_gate.json").write_text(json.dumps(review, indent=2) + "\n", encoding="utf-8")
    (out / "build_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(review, indent=2))
    return 1 if review["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
