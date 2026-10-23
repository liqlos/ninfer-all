import ctypes
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/bf16-store-q6-t16"
PREFIX = ["wsl.exe", "-d", "Ubuntu-24.04", "--cd", "/", "--exec", "docker", "exec", "ninfer-builder"]


def main():
    arm = sys.argv[1]
    receipt = OUT / (arm + "-build.json")
    if receipt.exists():
        raise RuntimeError("preserve prior build")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p]
    kernel.CreateMutexW.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel.ReleaseMutex.argtypes = [ctypes.c_void_p]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    lock = kernel.CreateMutexW(None, False, "Local\\NInferV14CpuVariantPreparation")
    if not lock or kernel.WaitForSingleObject(lock, 0) not in (0, 128):
        raise RuntimeError("another build owns cache")
    record = {"status": "BUILDING", "commands": []}
    try:
        git = lambda *args: subprocess.check_output(["git", "-C", str(ROOT / "source"), *args], text=True, creationflags=subprocess.CREATE_NO_WINDOW).strip()
        record["source"] = git("rev-parse", "HEAD")
        record["diff"] = git("diff", "HEAD")
        record["production_config_sha256"] = hashlib.sha256((ROOT / "broker/config.json").read_bytes()).hexdigest()
        commands = [
            ["cmake", "-S", "/lab/source", "-B", "/build", "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
             "-DCMAKE_CUDA_ARCHITECTURES=86", "-DNINFER_BUILD_APPS=OFF", "-DBUILD_TESTING=ON",
             "-DNINFER_BUILD_BENCHMARKS=ON", "-DCMAKE_PROJECT_INCLUDE=/lab/results/bf16-store-pr/build-inputs/target.cmake"],
            ["cmake", "--build", "/build", "--target", "ninfer_pr_check", "-j", "8"],
            ["cp", "/build/ninfer_pr_check", "/lab/results/bf16-store-q6-t16/bin/" + arm],
        ]
        for index, command in enumerate(commands):
            started = time.monotonic()
            log = arm + f"-build-{index}.log"
            with (OUT / log).open("wb") as file:
                result = subprocess.run(PREFIX + command, stdout=file, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
            record["commands"].append({"argv": command, "seconds": time.monotonic() - started, "exit": result.returncode, "log": log})
            if result.returncode:
                raise RuntimeError("build failed: " + log)
        record["binary_sha256"] = hashlib.sha256((OUT / "bin" / arm).read_bytes()).hexdigest()
        record["status"] = "PASS_OP_BUILD_LINK"
    except BaseException as error:
        record.update(status="FAILED", error=str(error))
        raise
    finally:
        receipt.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        kernel.ReleaseMutex(lock)
        kernel.CloseHandle(lock)
    print(record["status"], flush=True)


if __name__ == "__main__":
    main()
