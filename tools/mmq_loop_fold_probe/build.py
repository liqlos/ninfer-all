import argparse
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time

import analyze
import generate


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
    args = parser.parse_args()
    if not args.enable_candidate:
        parser.error("source-only default; compile gate requires explicit --enable-candidate")
    if os.name != "posix":
        parser.error("use the existing Linux CUDA environment; no Windows configure or install")
    if sys.version_info[:2] != (3, 11) and sys.version_info[:3] != (3, 12, 3):
        parser.error("select Python 3.11 or the existing pinned builder Python 3.12.3")
    if os.environ.get("NVCC_PREPEND_FLAGS") or os.environ.get("NVCC_APPEND_FLAGS"):
        parser.error("ambient NVCC flags must be empty")
    source, out = args.source_root.resolve(), args.output_dir.resolve()
    probe = pathlib.Path(__file__).resolve().parent
    if out == source or source in out.parents or out == probe or probe in out.parents:
        parser.error("compiler output must use the external current build area")
    if out.exists() and any(out.iterdir()):
        parser.error("output directory must be new or empty")
    nvcc, dump = shutil.which(args.nvcc), shutil.which(args.cuobjdump)
    if not nvcc or not dump:
        parser.error("existing nvcc and cuobjdump are prerequisites")
    files, receipt = generate.prepare(source)
    analyze.self_test()
    version = invoke([nvcc, "--version"], source)
    if version.returncode or not re.search(r"\bV13\.1\.115\b", version.stdout):
        raise RuntimeError("requires the pinned existing CUDA 13.1.115 compiler")
    out.mkdir(parents=True, exist_ok=True)
    generated = out / "generated"
    for name, data in files.items():
        target = generated / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    (out / "toolchain.txt").write_text(version.stdout, encoding="utf-8")
    generation = out / "generation.json"
    generation.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    flags = ["-O3", "-std=c++17", "-arch=sm_86", "--extended-lambda", "-lineinfo", "-Xptxas=-v",
             "--objdir-as-tempdir", "--frandom-seed=mmq-loop-fold-probe"]
    for directory in (source / "src", source / "include", source / "third_party/ggml-quants/include",
                      source / "third_party/ggml-quants/src", source / generate.CUDA):
        flags.extend(["-I", str(directory)])
    manifest = {"source_root": str(source), "probe_root": str(probe), "compile_only": True,
                "host_main": False, "link": False, "gpu_run": False, "python_version": sys.version,
                "toolchain_sha256": generate.sha((out / "toolchain.txt").read_bytes()),
                "generation_sha256": generate.sha(generation.read_bytes()),
                "probe_sha256": {name: generate.sha((probe / name).read_bytes())
                                  for name in ("generate.py", "proof.py", "build.py", "analyze.py", "compile.sh")},
                "units": []}
    manifest_path = out / "build_manifest.json"
    for name, enabled in (("off", 0), ("on", 1)):
        unit = {"name": name, "object": name + ".o", "dependencies": name + ".d",
                "log": "compile-" + name + ".log", "resources": "resources-" + name + ".txt",
                "sass": "sass-" + name + ".txt", "ptx": "ptx-" + name + ".txt"}
        obj, dep = out / unit["object"], out / unit["dependencies"]
        overlay = ["-I", str(generated / "on")] if enabled else []
        argv = [nvcc, *overlay, *flags, "-D" + generate.SWITCH + "=" + str(enabled),
                "--generate-dependencies-with-compile", "--dependency-output", str(dep),
                "-c", str(generated / "variant.cu"), "-o", str(obj)]
        start = time.monotonic()
        compiled = invoke(argv, source)
        (out / unit["log"]).write_text(compiled.stdout, encoding="utf-8")
        unit.update(argv=argv, returncode=compiled.returncode, compile_seconds=time.monotonic() - start)
        manifest["units"].append(unit)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        if compiled.returncode:
            raise RuntimeError(name + " compile failed; inspect " + str(out / unit["log"]))
        for option, key in (("--dump-resource-usage", "resources"), ("--dump-sass", "sass"), ("--dump-ptx", "ptx")):
            dumped = invoke([dump, option, str(obj)], source)
            (out / unit[key]).write_text(dumped.stdout, encoding="utf-8")
            if dumped.returncode:
                raise RuntimeError(name + " " + option + " failed")
        unit["artifact_sha256"] = {key: generate.sha((out / unit[key]).read_bytes())
                                   for key in ("object", "dependencies", "log", "resources", "sass", "ptx")}
        dependencies = {(source / path).resolve() for path in analyze.parse_dependencies(dep.read_text(), obj)}
        unit["dependency_sha256"] = {str(path): generate.sha(path.read_bytes()) for path in sorted(dependencies)}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    try:
        result = analyze.run(out)
    except (OSError, ValueError, KeyError) as error:
        result = {"status": "REJECT_COMPILE_STATIC_GATE", "reason": str(error), "GPU_harness_allowed": False,
                  "GPU_numerics_qualified": False, "performance_claim": False}
    (out / "gate.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["status"].startswith("PASS") else 3


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
