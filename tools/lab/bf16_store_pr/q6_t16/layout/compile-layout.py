import hashlib
import ctypes
import json
import pathlib
import re
import subprocess
import time


ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "results/bf16-store-q6-t16"
PREFIX = ["wsl.exe", "-d", "Ubuntu-24.04", "--cd", "/", "--exec", "docker", "exec", "ninfer-builder"]


def invoke(args, log):
    started = time.monotonic()
    with (OUT / log).open("wb") as file:
        result = subprocess.run(PREFIX + args, stdout=file, stderr=subprocess.STDOUT,
                                creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode:
        raise RuntimeError(log + " failed: " + str(result.returncode))
    return {"argv": args, "log": log, "exit": result.returncode,
            "seconds": time.monotonic() - started}


def output(args):
    return subprocess.check_output(PREFIX + args, text=True,
                                   creationflags=subprocess.CREATE_NO_WINDOW)


def main():
    receipt = OUT / "layout-architecture-compile.json"
    if receipt.exists():
        raise RuntimeError("preserve existing architecture qualification")
    source = subprocess.check_output(["git", "-C", str(ROOT / "source"), "rev-parse", "HEAD"],
                                     text=True, creationflags=subprocess.CREATE_NO_WINDOW).strip()
    tested = json.loads((OUT / "layout-build.json").read_text(encoding="utf-8"))
    diff = subprocess.check_output(["git", "-C", str(ROOT / "source"), "diff", "HEAD"], text=True,
                                   creationflags=subprocess.CREATE_NO_WINDOW).strip()
    if source != tested["source"] or diff != tested["diff"]:
        raise RuntimeError("architecture qualification requires the GPU-tested working tree")
    record = {"status": "RUNNING", "source": source, "compile_only": True,
              "non86_GPU_run": False, "working_diff_sha256": hashlib.sha256(diff.encode()).hexdigest(), "configurations": []}
    receipt.write_text(json.dumps(record, indent=2) + "\n")
    try:
        record["toolchain_preparation"] = invoke(["bash", "/lab/results/bf16-store-pr/build-inputs/cuda12.sh"], "layout-cuda12-toolchain.log")
        configs = [("layout-cuda12", "/build/bf16-store-toolchain/root/usr/local/cuda-12.9", "70;75;80;86;89"),
                   ("layout-cuda13", "/usr/local/cuda", "120a")]
        for label, toolkit, arches in configs:
            build = "/build/bf16-store-matrix"
            config = {"name": label, "architectures": arches, "compiler": output([toolkit + "/bin/nvcc", "--version"])}
            record["configurations"].append(config)
            config["configure"] = invoke(["cmake", "--fresh", "-S", "/lab/source", "-B", build,
                "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CUDA_ARCHITECTURES=" + arches,
                "-DCMAKE_CUDA_COMPILER=" + toolkit + "/bin/nvcc", "-DCUDAToolkit_ROOT=" + toolkit,
                "-DNINFER_BUILD_APPS=OFF", "-DBUILD_TESTING=ON", "-DNINFER_BUILD_BENCHMARKS=OFF",
                "-DNINFER_SM120_NATIVE=OFF", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"], label + "-configure.log")
            all_targets = output(["ninja", "-C", build, "-t", "targets", "all"])
            expression = re.compile(r"(?:ggml_mmq_[^/]+\.cu|ggml_bridge\.cu|test_gguf_mmq_store\.cu)\.o$")
            targets = [line.split(": ")[0] for line in all_targets.splitlines()
                       if expression.search(line.split(": ")[0])]
            if len(targets) != 19:
                raise RuntimeError("expected 17 MMQ units, bridge and changed exact test")
            config["targets"] = targets
            config["compile"] = invoke(["ninja", "-C", build, "-j", "4", *targets], label + "-compile.log")
            config["objects"] = json.loads(output(["python3", "-c",
                "import hashlib,json,pathlib,sys; print(json.dumps([dict(path=p,bytes=(q:=pathlib.Path(sys.argv[1])/p).stat().st_size,sha256=hashlib.sha256(q.read_bytes()).hexdigest()) for p in sys.argv[2:]]))",
                build, *targets]))
            receipt.write_text(json.dumps(record, indent=2) + "\n")
        record["status"] = "PASS_AFFECTED_UNITS_COMPILE_ALL_SIX_ARCHITECTURES"
    except BaseException as error:
        record.update(status="FAILED", error=str(error))
        raise
    finally:
        receipt.write_text(json.dumps(record, indent=2) + "\n")
    print(record["status"], flush=True)


if __name__ == "__main__":
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p]
    kernel.CreateMutexW.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel.ReleaseMutex.argtypes = [ctypes.c_void_p]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handles = []
    try:
        for name in ("Local\\NInferV17RuntimeScreen", "Local\\NInferV14RuntimeMeasurement",
                     "Local\\NInferV15ExportedOperatorSeries", "Local\\NInferV14CpuVariantPreparation"):
            handle = kernel.CreateMutexW(None, False, name)
            if not handle or kernel.WaitForSingleObject(handle, 0) not in (0, 128):
                raise RuntimeError("another lab worker owns " + name)
            handles.append(handle)
        main()
    finally:
        for handle in reversed(handles):
            kernel.ReleaseMutex(handle)
            kernel.CloseHandle(handle)
