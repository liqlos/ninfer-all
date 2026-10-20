import argparse
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True

import generate
import proof


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def invoke(argv, cwd):
    options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    return subprocess.run(argv, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, check=False, **options)


def resource_records(text):
    records = {}
    for match in re.finditer(r"^\s*Function\s+(\S+):\s*\n([^\n]+)", text, re.MULTILINE):
        if "mul_mat_q" in match[1]:
            if match[1] in records:
                raise ValueError("duplicate resource record: " + match[1])
            records[match[1]] = {key: int(value) for key, value in
                                 re.findall(r"([A-Z]+):(\d+)", match[2])}
    return records


def sass_records(text):
    functions = list(re.finditer(r"^\s*Function\s*:\s*(\S+)\s*$", text, re.MULTILINE))
    result = {}
    for index, match in enumerate(functions):
        if "mul_mat_q" not in match[1]:
            continue
        end = functions[index + 1].start() if index + 1 < len(functions) else len(text)
        body = text[match.end():end]
        encoding = re.findall(r"/\*\s*0x([0-9a-fA-F]{16})\s*\*/", body)
        operations = re.findall(
            r"/\*[0-9a-fA-F]+\*/\s*(?:@!?(?:UP|P)(?:T|\d+)\s+)?([A-Z][A-Z0-9_.]*)\s", body)
        if match[1] in result:
            raise ValueError("duplicate SASS record: " + match[1])
        result[match[1]] = {"encoding": encoding, "operations": operations}
    return result


def review(arms):
    failures = []
    resources = {name: resource_records(arm["resources"]) for name, arm in arms.items()}
    sass = {name: sass_records(arm["sass"]) for name, arm in arms.items()}
    report = {}
    for arm in ("control", "candidate"):
        if len(resources[arm]) != 6 or set(resources[arm]) != set(sass[arm]):
            failures.append(arm + ": expected exactly six native kernel resource/SASS records")
        spills = re.findall(r"(\d+) bytes spill stores, (\d+) bytes spill loads", arms[arm]["compile"])
        if len(spills) < 6 or any(int(stores) or int(loads) for stores, loads in spills):
            failures.append(arm + ": missing spill records or nonzero spill traffic")
        report[arm] = {"resources": resources[arm], "spill_reports": spills, "sass": {}}
        for name, values in resources[arm].items():
            if not {"REG", "STACK", "LOCAL", "SHARED"} <= values.keys():
                failures.append(arm + ": incomplete resource report: " + name)
            elif (not 0 < values["REG"] <= 255 or values["STACK"] or values["LOCAL"] or
                  values["SHARED"]):
                failures.append(arm + ": invalid registers or nonzero stack/local/static shared: " + name)
            item = sass[arm].get(name, {"encoding": [], "operations": []})
            if not item["encoding"] or len(item["encoding"]) % 2:
                failures.append(arm + ": incomplete SASS encoding: " + name)
            if not any(op.startswith("IMMA") for op in item["operations"]):
                failures.append(arm + ": native IMMA mechanism missing: " + name)
            if any(op.startswith(("LDL", "STL")) for op in item["operations"]):
                failures.append(arm + ": local memory instruction: " + name)
            report[arm]["sass"][name] = {
                "encoding_sha256": generate.digest("".join(item["encoding"])),
                "encoded_instruction_count": len(item["encoding"]) // 2,
            }
    if resources["control"] != resources["candidate"]:
        failures.append("OFF/ON native resource records differ")
    if set(sass["control"]) != set(sass["candidate"]):
        failures.append("OFF/ON native SASS function sets differ")
    for name in set(sass["control"]) & set(sass["candidate"]):
        if sass["control"][name]["encoding"] != sass["candidate"][name]["encoding"]:
            failures.append("OFF/ON native SASS bytes differ: " + name)
    return {
        "status": "COMPILE_RESOURCE_PASS" if not failures else "COMPILE_RESOURCE_FAIL",
        "failures": failures, "arms": report,
        "dynamic_shared_bytes_per_CTA": 57856,
        "same_native_device_code": not failures,
        "candidate_default": "OFF", "host_main": False, "link": False,
        "GPU_correctness": "NOT_RUN", "performance_claim": False,
        "next_gate": "independent samples and carry oracle/canaries, then whole paired FFN vs strongest folded16",
    }


def dependency_paths(text, root):
    text = re.sub(r"\\\r?\n", " ", text)
    parts = re.split(r":(?=\s)", text, maxsplit=1)
    if len(parts) != 2:
        raise ValueError("unrecognized nvcc dependency target")
    tokens = re.findall(r'(?:"[^"\n]*"|\\[ \t#:]|[^\s])+', parts[1])
    paths = set()
    for token in tokens:
        token = re.sub(r"\\([ \t#:])", r"\1", token).strip('"')
        path = pathlib.Path(token)
        path = (path if path.is_absolute() else root / path).resolve()
        if not path.is_file():
            raise ValueError("compiled dependency unavailable: " + str(path))
        paths.add(path)
    if not paths:
        raise ValueError("empty nvcc dependency list")
    return sorted(paths)


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


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
    header, receipt = generate.prepare(root)
    address_proof = proof.prove(root)
    version = invoke([nvcc, "--version"], root)
    if (version.returncode or not re.fullmatch(r"\d+\.\d+\.\d+", args.expected_cuda_version) or
            not re.search(r"\bV" + re.escape(args.expected_cuda_version) + r"\b", version.stdout)):
        raise RuntimeError("nvcc differs from explicitly selected CUDA version")
    out.mkdir(parents=True, exist_ok=True)
    generated = out / "generated"
    generated.mkdir(exist_ok=True)
    (generated / "mmq_two_input_native.cuh").write_bytes(header.encode())
    save(generated / "source_receipt.json", receipt)
    save(out / "address_proof.json", address_proof)
    probe_hashes = {p.name: sha(p) for p in sorted(probe.iterdir())
                    if p.suffix in (".py", ".cu", ".h")}
    flags = ["-O3", "-std=c++17", "-arch=sm_86", "--extended-lambda", "-lineinfo",
             "-Xptxas=-v", "--frandom-seed=mmq-two-input-probe"]
    flags.extend("-D" + name + "=1" for name, enabled in generate.W_FLAGS.items() if enabled)
    for path in (generated, probe, root / "src", root / "include",
                 root / "third_party/ggml-quants/include",
                 root / "third_party/ggml-quants/src",
                 root / "third_party/ggml-quants/src/ggml-cuda"):
        flags.extend(["-I", str(path)])
    manifest = {
        "status": "BUILD_STARTED", "compile_only": True, "host_main": False,
        "link": False, "GPU_run": False, "performance_claim": False,
        "source_receipt": receipt, "toolchain": version.stdout,
        "expected_cuda_version": args.expected_cuda_version, "python": sys.version,
        "probe_sha256": probe_hashes, "arms": {},
    }
    save(out / "build_manifest.json", manifest)
    arms = {}
    for arm, enabled in (("control", 0), ("candidate", 1)):
        obj, dep = out / (arm + ".o"), out / (arm + ".d")
        command = [nvcc, *flags, "-DNINFER_LAB_MMQ_TWO_INPUT=" + str(enabled),
                   "--generate-dependencies-with-compile", "--dependency-output", str(dep),
                   "-c", str(probe / "variant.cu"), "-o", str(obj)]
        started = time.monotonic()
        result = invoke(command, root)
        (out / (arm + "-compile.log")).write_text(result.stdout, encoding="utf-8")
        manifest["arms"][arm] = {"command": command, "returncode": result.returncode,
                                 "compile_seconds": time.monotonic() - started}
        save(out / "build_manifest.json", manifest)
        if result.returncode:
            raise RuntimeError(arm + " isolated compile failed; see compile log")
        arms[arm] = {"compile": result.stdout}
        for option, kind in (("--dump-resource-usage", "resources"), ("--dump-sass", "sass")):
            dumped = invoke([dump, option, str(obj)], root)
            (out / (arm + "-" + kind + ".txt")).write_text(dumped.stdout, encoding="utf-8")
            if dumped.returncode:
                raise RuntimeError(arm + " " + option + " failed")
            arms[arm][kind] = dumped.stdout
        manifest["arms"][arm]["dependency_sha256"] = {
            str(p): sha(p) for p in dependency_paths(dep.read_text(), root)}
        manifest["arms"][arm]["object_sha256"] = sha(obj)
    generate.prepare(root)
    if probe_hashes != {p.name: sha(p) for p in sorted(probe.iterdir())
                       if p.suffix in (".py", ".cu", ".h")}:
        raise RuntimeError("probe sources changed during compile")
    gate = review(arms)
    manifest["status"] = gate["status"]
    save(out / "build_manifest.json", manifest)
    save(out / "resource_gate.json", gate)
    print(json.dumps(gate, indent=2))
    return 1 if gate["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
