import argparse
import ctypes
import hashlib
import json
import pathlib
import subprocess
import time


ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source"
OUT = ROOT / "results/bf16-store-pr"
FLAGS = subprocess.CREATE_NO_WINDOW


def command(args):
    return subprocess.check_output(args, text=True, creationflags=FLAGS).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=("control", "candidate"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--parallel", type=int)
    args = parser.parse_args()
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p]
    kernel.CreateMutexW.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel.ReleaseMutex.argtypes = [ctypes.c_void_p]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    mutex = kernel.CreateMutexW(None, False, "Local\\NInferV14CpuVariantPreparation")
    if not mutex or kernel.WaitForSingleObject(mutex, 0) not in (0, 128):
        raise RuntimeError("another lab CPU worker owns the build window")
    receipt = OUT / (args.arm + "-build.json")
    if receipt.exists() and not args.resume:
        raise RuntimeError("preserve the existing build receipt")
    head = command(["git", "-C", str(SOURCE), "rev-parse", "HEAD"])
    if command(["git", "-C", str(SOURCE), "status", "--porcelain"]):
        raise RuntimeError("build source must be a clean pinned tree")
    expected = command(["git", "-C", str(SOURCE), "rev-parse",
                        "origin/master" if args.arm == "control" else "main"])
    if head != expected:
        raise RuntimeError("the selected source does not match the requested arm")
    config_sha = hashlib.sha256((ROOT / "broker/config.json").read_bytes()).hexdigest()
    record = {"status": "BUILDING", "arm": args.arm, "source": head,
              "production_config_sha256": config_sha, "gpu_used": False,
              "started_unix": time.time(), "commands": []}
    if args.resume:
        previous = json.loads(receipt.read_text())
        if previous["status"] != "FAILED" or previous["source"] != head:
            raise RuntimeError("resume requires the same failed source arm")
        record = previous
        record.setdefault("recovery", []).append({"previous_error": record.pop("error", None),
            "reason": "remove unrelated attention from Op qualification; preserve completed objects"})
        record["status"] = "BUILDING"
    record["scope"] = "public Linear/GGUF Op qualification; softmax_attention excluded from lab link"
    record["build_inputs_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (OUT / "build-inputs").iterdir() if p.is_file()}

    def invoke(argv, label):
        suffix = "-r" + str(len(record.get("recovery", [])) + 1) if args.resume else ""
        log = OUT / (args.arm + "-" + label + suffix + ".log")
        started = time.monotonic()
        with log.open("wb") as output:
            proc = subprocess.run(argv, stdout=output, stderr=subprocess.STDOUT,
                                  creationflags=FLAGS, check=False)
        record["commands"].append({"argv": argv, "exit": proc.returncode,
                                    "seconds": time.monotonic() - started, "log": log.name})
        receipt.write_text(json.dumps(record, indent=2) + "\n")
        if proc.returncode:
            raise RuntimeError(label + " failed; inspect " + str(log))

    prefix = ["wsl.exe", "-d", "Ubuntu-24.04", "--cd", "/", "--exec",
              "docker", "exec", "ninfer-builder"]
    try:
        receipt.write_text(json.dumps(record, indent=2) + "\n")
        invoke(prefix + ["cmake", "--fresh", "-S", "/lab/source", "-B", "/build", "-G", "Ninja",
                         "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CUDA_ARCHITECTURES=86",
                         "-DNINFER_BUILD_APPS=OFF", "-DBUILD_TESTING=ON",
                         "-DNINFER_BUILD_BENCHMARKS=ON", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
                         "-DCMAKE_PROJECT_INCLUDE=/lab/results/bf16-store-pr/build-inputs/target.cmake"],
               "configure")
        parallel = ["-j", str(args.parallel)] if args.parallel else ["-j"]
        invoke(prefix + ["cmake", "--build", "/build", "--target", "ninfer_pr_check", *parallel], "compile")
        invoke(prefix + ["sh", "-c", "mkdir -p /lab/results/bf16-store-pr/bin; cp /build/ninfer_pr_check "
                         "/lab/results/bf16-store-pr/bin/" + args.arm], "export")
        binary = OUT / "bin" / args.arm
        record.update(status="PASS_OP_BUILD_LINK", binary_bytes=binary.stat().st_size,
                      binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
    except BaseException as error:
        record.update(status="FAILED", error=str(error))
        raise
    finally:
        record["finished_unix"] = time.time()
        record["production_config_unchanged"] = config_sha == hashlib.sha256(
            (ROOT / "broker/config.json").read_bytes()).hexdigest()
        receipt.write_text(json.dumps(record, indent=2) + "\n")
        kernel.ReleaseMutex(mutex)
        kernel.CloseHandle(mutex)
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
