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

import build as kernel_build
import generate_host


HERE = pathlib.Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def invoke(command, root, log=None):
    options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    result = subprocess.run(command, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, check=False, **options)
    if log is not None:
        log.write_text(result.stdout, encoding="utf-8")
    require(result.returncode == 0, "command failed: " + " ".join(command) + "\n" + result.stdout)
    return result.stdout


def verify_linked_native(resources, sass, gate):
    expected = gate["arms"]["control"]
    resource_count = 0
    resource_names = set()
    for match in re.finditer(r"^\s*Function\s+(\S+):\s*\n([^\n]+)", resources, re.MULTILINE):
        if "mul_mat_q" not in match[1]:
            continue
        record = {key: int(value) for key, value in re.findall(r"([A-Z]+):(\d+)", match[2])}
        require(expected["resources"].get(match[1]) == record,
                "linked native resources differ: " + match[1])
        resource_count += 1
        resource_names.add(match[1])
    functions = list(re.finditer(r"^\s*Function\s*:\s*(\S+)\s*$", sass, re.MULTILINE))
    sass_count = 0
    sass_names = set()
    for index, match in enumerate(functions):
        if "mul_mat_q" not in match[1]:
            continue
        end = functions[index + 1].start() if index + 1 < len(functions) else len(sass)
        words = re.findall(r"/\*\s*0x([0-9a-fA-F]{16})\s*\*/", sass[match.end():end])
        require(words and len(words) % 2 == 0, "linked SASS encoding incomplete")
        expected_sha = expected["sass"].get(match[1], {}).get("encoding_sha256")
        require(generate_host.sha("".join(words).encode()) == expected_sha,
                "linked native SASS differs: " + match[1])
        sass_count += 1
        sass_names.add(match[1])
    require(resource_names == sass_names == set(expected["resources"]) and len(sass_names) == 6,
            "linked binary does not contain all six reviewed native kernels")
    return {"status": "PASS", "unique_native_kernels": 6,
            "resource_records": resource_count, "sass_records": sass_count,
            "native_SASS_and_resources_unchanged": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=pathlib.Path, default=generate_host.ROOT)
    parser.add_argument("--objects-root", type=pathlib.Path)
    parser.add_argument("--compile-manifest-sha256")
    parser.add_argument("--resource-gate-sha256")
    parser.add_argument("--output-root", type=pathlib.Path)
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    root = args.source_root.resolve()
    files, metadata = generate_host.prepare(root)
    generated_sha = {name: generate_host.sha(data) for name, data in files.items()}
    if not args.build:
        print(json.dumps({"source_check_only": True, "metadata": metadata,
                          "generated_sha256": generated_sha}, sort_keys=True))
        return
    require(args.objects_root is not None, "--objects-root is required for the host build")
    require(all(value and re.fullmatch(r"[0-9a-f]{64}", value) for value in
                (args.compile_manifest_sha256, args.resource_gate_sha256)),
            "explicit compile manifest and resource gate SHA256 pins are required")
    require(not os.environ.get("NVCC_PREPEND_FLAGS") and not os.environ.get("NVCC_APPEND_FLAGS"),
            "ambient NVCC flags must be empty")
    objects = args.objects_root.resolve()
    manifest_path, gate_path = objects / "build_manifest.json", objects / "resource_gate.json"
    require(sha(manifest_path) == args.compile_manifest_sha256, "compile manifest differs")
    require(sha(gate_path) == args.resource_gate_sha256, "resource gate differs")
    manifest, gate = json.loads(manifest_path.read_bytes()), json.loads(gate_path.read_bytes())
    require(manifest["status"] == gate["status"] == "COMPILE_RESOURCE_PASS" and
            manifest["expected_cuda_version"] == "13.1.115" and gate["same_native_device_code"] and
            not gate["failures"], "reviewed native compile/resource gate required")
    require(manifest["probe_sha256"] == generate_host.FROZEN, "compile source set differs")
    linked = {}
    dependencies = {}
    for arm in ("control", "candidate"):
        path = objects / (arm + ".o")
        expected = manifest["arms"][arm]["object_sha256"]
        require(manifest["arms"][arm]["returncode"] == 0 and sha(path) == expected,
                "reviewed native object differs: " + arm)
        linked[arm] = {"path": str(path), "sha256": expected}
        for name, expected_dependency in manifest["arms"][arm]["dependency_sha256"].items():
            require(sha(pathlib.Path(name)) == expected_dependency, "native dependency differs: " + name)
            dependencies[name] = expected_dependency
    nvcc, dump, nm = (shutil.which(name) for name in ("nvcc", "cuobjdump", "nm"))
    require(nvcc and dump and nm, "reviewed CUDA toolchain and GNU nm are required")
    version = invoke([nvcc, "--version"], root)
    require(re.search(r"\bV13\.1\.115\b", version), "CUDA compiler version differs")
    for item in linked.values():
        undefined = invoke([nm, "-u", item["path"]], root)
        require(re.search(r"\bU __cudaLaunchKernel\b", undefined), "reviewed native launch ABI differs")
    out = (args.output_root or objects / "host").resolve()
    require(out.is_relative_to(objects / "host") and not out.is_relative_to(root) and
            not out.is_relative_to(HERE), "host outputs must stay in the selected external build/host")
    out.mkdir(parents=True, exist_ok=True)
    generated = out / "generated"
    generated.mkdir(exist_ok=True)
    for name, data in files.items():
        (generated / name).write_bytes(data)
    flags = ["-O3", "-std=c++17", "-arch=sm_86", "--extended-lambda", "-lineinfo",
             "-Xptxas=-v", "--frandom-seed=mmq-two-input-host",
             "-DNINFER_GGUF_Q4_COOPERATIVE_DECODE=1", "-DNINFER_GGUF_Q6_COOPERATIVE_DECODE=1",
             "-DNINFER_GGUF_MMQ_STORE_FUSION=1"]
    for path in (HERE, generated, root / "src", root / "include",
                 root / "third_party/ggml-quants/include", root / "third_party/ggml-quants/src",
                 root / "third_party/ggml-quants/src/ggml-cuda"):
        flags.extend(["-I", str(path)])
    compile_command = [nvcc, *flags, "--generate-dependencies-with-compile", "--dependency-output",
                       str(out / "main.d"), "-c", str(HERE / "main.cu"), "-o", str(out / "main.o")]
    started = time.monotonic()
    invoke(compile_command, root, out / "compile-main.log")
    compile_seconds = time.monotonic() - started
    link_command = [nvcc, "-arch=sm_86", str(out / "main.o"),
                    linked["control"]["path"], linked["candidate"]["path"],
                    "-Xlinker=--wrap=__cudaLaunchKernel", "-lcuda", "-o", str(out / "probe")]
    started = time.monotonic()
    invoke(link_command, root, out / "link.log")
    link_seconds = time.monotonic() - started
    resources = invoke([dump, "--dump-resource-usage", str(out / "probe")], root,
                       out / "resources-linked.txt")
    sass = invoke([dump, "--dump-sass", str(out / "probe")], root, out / "sass-linked.txt")
    linked_gate = verify_linked_native(resources, sass, gate)
    require(sha(manifest_path) == args.compile_manifest_sha256 and sha(gate_path) == args.resource_gate_sha256,
            "parent compile receipts changed during host build")
    for item in linked.values():
        require(sha(pathlib.Path(item["path"])) == item["sha256"], "parent native object changed")
    for name, expected in dependencies.items():
        require(sha(pathlib.Path(name)) == expected, "parent native dependency changed")
    after_files, after_metadata = generate_host.prepare(root)
    require(after_files == files and after_metadata == metadata, "host sources changed during build")
    host_dependencies = kernel_build.dependency_paths((out / "main.d").read_text(), root)
    metadata.update({
        "status": "HOST_BUILD_PASS", "toolchain": version,
        "compile_manifest_path": str(manifest_path), "compile_manifest_sha256": args.compile_manifest_sha256,
        "resource_gate_path": str(gate_path), "resource_gate_sha256": args.resource_gate_sha256,
        "linked_existing_objects": linked, "linked_native_gate": linked_gate,
        "compile_command": compile_command, "link_command": link_command,
        "compile_seconds": compile_seconds, "link_seconds": link_seconds,
        "compile_link_seconds": compile_seconds + link_seconds,
        "generated_sha256": generated_sha,
        "host_dependency_sha256": {str(path): sha(path) for path in host_dependencies},
        "host_object_sha256": sha(out / "main.o"), "probe_sha256": sha(out / "probe"),
        "kernel_recompiled": False, "GPU_run": False,
    })
    (out / "host_build.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
