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

import generate
import sass_gate


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def invoke(argv, cwd):
    options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    return subprocess.run(argv, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, check=False, **options)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=pathlib.Path)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    parser.add_argument("--enable-candidate", action="store_true")
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument("--cuobjdump", default="cuobjdump")
    parser.add_argument("--expected-cuda-version", default="13.1.115")
    args = parser.parse_args()
    if not args.enable_candidate:
        parser.error("compile-only candidate screen requires --enable-candidate")
    if os.environ.get("NVCC_PREPEND_FLAGS") or os.environ.get("NVCC_APPEND_FLAGS"):
        parser.error("ambient NVCC flags must be empty")
    source = args.source_root.resolve()
    out = args.output_dir.resolve()
    probe = pathlib.Path(__file__).resolve().parent
    if out == source or source in out.parents:
        parser.error("compiler output must use the existing external build area")
    nvcc = shutil.which(args.nvcc)
    dump = shutil.which(args.cuobjdump)
    if not nvcc or not dump:
        parser.error("nvcc and cuobjdump are prerequisites")
    files, receipt = generate.prepare(source)
    out.mkdir(parents=True, exist_ok=True)
    generated = out / "generated"
    generated.mkdir(exist_ok=True)
    for name, text in files.items():
        (generated / name).write_bytes(text.encode())
    (generated / "source_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    version = invoke([nvcc, "--version"], source)
    if (version.returncode or not re.fullmatch(r"\d+\.\d+\.\d+", args.expected_cuda_version) or
            not re.search(r"\bV" + re.escape(args.expected_cuda_version) + r"\b", version.stdout)):
        raise RuntimeError("nvcc differs from the explicitly selected CUDA build")
    (out / "toolchain.txt").write_text(version.stdout, encoding="utf-8")
    flags = ["-O3", "-std=c++17", "-arch=sm_86", "--extended-lambda", "-lineinfo",
             "-Xptxas=-v", "--frandom-seed=q6-word-group-probe",
             "-DNINFER_GGUF_Q6_COOPERATIVE_DECODE=1", "-DNINFER_Q6_WORD_GROUP_PROBE=1"]
    for path in (generated, probe, source / "src", source / "include",
                 source / "third_party/ggml-quants/include",
                 source / "third_party/ggml-quants/src",
                 source / "third_party/ggml-quants/src/ggml-cuda"):
        flags.extend(["-I", str(path)])
    manifest = {
        "source_root": str(source), "probe_root": str(probe),
        "compile_only": True, "candidate_enabled": True, "host_main": False, "link": False,
        "gpu_run": False,
        "expected_cuda_version": args.expected_cuda_version,
        "actual_nvcc_version": version.stdout,
        "toolchain_sha256": sha(out / "toolchain.txt"),
        "python_version": sys.version,
        "probe_sha256": {path.name: sha(path) for path in sorted(probe.iterdir())
                         if path.is_file() and path.suffix in (".py", ".cu", ".cuh", ".h")},
        "units": [],
    }
    manifest_path = out / "build_manifest.json"
    for name in ("vector", "mmq"):
        obj = out / f"{name}.o"
        dep = out / f"{name}.d"
        argv = [nvcc, *flags, "--generate-dependencies-with-compile",
                "--dependency-output", str(dep), "-c", str(probe / f"{name}_probe.cu"),
                "-o", str(obj)]
        start = time.monotonic()
        compiled = invoke(argv, source)
        elapsed = time.monotonic() - start
        log = out / f"compile-{name}.log"
        log.write_text(compiled.stdout, encoding="utf-8")
        unit = {"name": name, "argv": argv, "returncode": compiled.returncode,
                "compile_seconds": elapsed, "object": obj.name, "dependency": dep.name,
                "log": log.name, "resources": f"resources-{name}.txt",
                "sass": f"sass-{name}.txt", "artifact_sha256": {}}
        manifest["units"].append(unit)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        if compiled.returncode:
            raise RuntimeError(f"{name} compile failed; see {log}")
        for option, target in (("--dump-resource-usage", unit["resources"]),
                               ("--dump-sass", unit["sass"])):
            dumped = invoke([dump, option, str(obj)], source)
            (out / target).write_text(dumped.stdout, encoding="utf-8")
            if dumped.returncode:
                raise RuntimeError(f"{name} {option} failed")
        names = (obj.name, dep.name, log.name, unit["resources"], unit["sass"])
        unit["artifact_sha256"] = {item: sha(out / item) for item in names}
        unit["dependency_sha256"] = {
            str(path): sha(path) for path in sorted(sass_gate.dependency_paths(dep, source))}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    result = sass_gate.run(out)
    (out / "gate.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
