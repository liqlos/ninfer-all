import csv
import ctypes
import hashlib
import json
import runpy
import sqlite3
import subprocess
import sys
import time
import uuid
from pathlib import Path


H = runpy.run_path(str(Path(__file__).with_name("qualify-bf16-store-gpu.py")))
ROOT, WSL, IMAGE, OWNER = (H[k] for k in ("ROOT", "WSL", "IMAGE", "OWNER"))
api, run, admission = (H[k] for k in ("api", "run", "admission"))
OUT = ROOT / "results/bf16-store-q6-t16"
MOUNT = "/mnt/c/Users/aleksei/Projects/NInfer/results/bf16-store-q6-t16"


def main():
    mode, tag, *arms = sys.argv[1:]
    receipt = OUT / (tag + ".json")
    if receipt.exists():
        raise RuntimeError("preserve prior series")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p]
    kernel.CreateMutexW.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel.ReleaseMutex.argtypes = [ctypes.c_void_p]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    locks = []
    reservation = None
    containers = []
    record = {"status": "RUNNING", "mode": mode, "runs": [], "profiles": []}
    config_sha = hashlib.sha256((ROOT / "broker/config.json").read_bytes()).hexdigest()

    def save():
        receipt.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    def execute(arm, label, args, profile=False):
        admission(reservation)
        name = "ninfer-q6-" + uuid.uuid4().hex[:12]
        mounts = ["--mount", "type=bind,source=" + MOUNT + "/bin,target=/bin-pr,readonly",
                  "--mount", "type=bind,source=" + MOUNT + ",target=/results"]
        if profile:
            mounts += ["--mount", "type=bind,source=/mnt/c/Users/aleksei/Projects/NInfer/tools/nsys,target=/nsys/target-linux-x64,readonly"]
            argv = ["--entrypoint", "/bin/bash", IMAGE, "-c",
                    'ln -s /nsys/target-linux-x64/nsys /tmp/ninfer-nsys; exec "$@"', "profile",
                    "/tmp/ninfer-nsys", "profile", "--trace=cuda", "--sample=none", "--cpuctxsw=none",
                    "--capture-range=cudaProfilerApi", "--capture-range-end=stop", "-o", "/results/" + label,
                    "/bin-pr/" + arm, *args]
        else:
            argv = ["--entrypoint", "/bin-pr/" + arm, IMAGE, *args]
        container = run(WSL + ["run", "-d", "--name", name, "--network", "none", "--gpus", "all",
            "--label", "io.codex.ninfer-gpu=true", "--label", "io.codex.ninfer-owner=" + OWNER,
            *mounts, *argv])
        containers.append(container)
        code = run(WSL + ["wait", container], timeout=900)
        log = run(WSL + ["logs", container])
        (OUT / (label + ".log")).write_text(log + "\n", encoding="utf-8")
        run(WSL + ["rm", container])
        containers.remove(container)
        if code != "0":
            raise RuntimeError(label + " failed: " + log[-1000:])
        return {"arm": arm, "label": label, "exit": code}

    try:
        for name in ("Local\\NInferV17RuntimeScreen", "Local\\NInferV14RuntimeMeasurement",
                     "Local\\NInferV15ExportedOperatorSeries", "Local\\NInferV14CpuVariantPreparation"):
            handle = kernel.CreateMutexW(None, False, name)
            if not handle or kernel.WaitForSingleObject(handle, 0) not in (0, 128):
                raise RuntimeError("another lab worker owns " + name)
            locks.append(handle)
        record["binary_sha256"] = {arm: hashlib.sha256((OUT / "bin" / arm).read_bytes()).hexdigest() for arm in arms}
        admission()
        reservation = uuid.uuid4().hex
        grant = api("/v1/photo-gpu-reservations", "POST", {"owner": OWNER, "reservation_id": reservation, "ttl_seconds": 3600})
        if grant["reservation_id"] != reservation or grant["state"] != "active":
            raise RuntimeError("reservation mismatch")
        if mode == "profile":
            for arm in arms:
                label = tag + "-" + arm
                row = execute(arm, label, ["bench", "--qtype", "q6_k", "--n", "5120", "--k", "17408",
                    "--t", "16", "--warmup", "30", "--profile"], True)
                run(WSL + ["exec", "ninfer-builder", "/opt/nvidia/nsight-compute/2025.4.1/host/linux-desktop-glibc_2_11_3-x64/QdstrmImporter",
                    "-i", "/lab/results/bf16-store-q6-t16/" + label + ".qdstrm", "-o", "/lab/results/bf16-store-q6-t16/" + label + ".nsys-rep"], timeout=180)
                run(WSL + ["exec", "ninfer-builder", "/lab/tools/nsys/nsys", "export", "--type=sqlite",
                    "--output=/lab/results/bf16-store-q6-t16/" + label + ".sqlite", "/lab/results/bf16-store-q6-t16/" + label + ".nsys-rep"], timeout=180)
                db = sqlite3.connect(OUT / (label + ".sqlite"))
                row["kernels"] = [{"name": name, "nanoseconds": ns, "registers": regs, "grid": grid, "block": block}
                    for name, ns, regs, grid, block in db.execute("SELECT s.value,k.end-k.start,k.registersPerThread,k.gridX,k.blockX*k.blockY*k.blockZ FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName ORDER BY k.start")]
                db.close()
                record["profiles"].append(row)
                save()
                for suffix in (".qdstrm", ".nsys-rep", ".sqlite"):
                    (OUT / (label + suffix)).unlink()
        elif mode == "tests":
            for arm in arms:
                for command in (["oracle"] if arm == "control" else ["store", "oracle"]):
                    record["runs"].append(execute(arm, tag + "-" + arm + "-" + command, [command]))
                    save()
        else:
            points = [("q6_k", 5120, 17408, 16)] if mode in ("pilot", "confirm") else [("q6_k", 5120, 17408, 128), ("q6_k", 5120, 17408, 512), ("q6_k", 5120, 17408, 1024), ("q4_k", 17408, 5120, 16), ("q4_k", 17408, 5120, 128), ("q4_k", 17408, 5120, 512), ("q4_k", 17408, 5120, 1024)]
            orders = ["ABBA", "BAAB"] * (4 if mode == "confirm" else 1)
            record["protocol"] = {"points": points, "orders": orders, "warmup": 50, "repeat": 500, "flush_mib": 256, "execution": "eager", "scope": "public Linear Op"}
            save()
            if len(arms) != 2:
                raise RuntimeError("screen requires two arms")
            for q, n, k, t in points:
                for b, order in enumerate(orders):
                    for position, choice in enumerate(order):
                        arm = arms[choice == "B"]
                        label = f"{tag}-{q}-t{t}-b{b}-p{position}-{arm}"
                        row = execute(arm, label, ["bench", "--qtype", q, "--n", str(n), "--k", str(k), "--t", str(t), "--warmup", "50", "--repeat", "500", "--flush-mib", "256", "--execution", "eager", "--csv-out", "/results/" + label + ".csv"])
                        with (OUT / (label + ".csv")).open(newline="") as file:
                            data = list(csv.DictReader(file))
                        if len(data) != 1:
                            raise RuntimeError("unexpected benchmark rows")
                        row.update(qtype=q, n=n, k=k, t=t, block=b, position=position, median_us=float(data[0]["median_us"]))
                        record["runs"].append(row)
                        save()
        record["status"] = "PASS"
    except BaseException as error:
        record.update(status="FAILED", error=str(error))
        raise
    finally:
        errors = []
        for container in containers:
            try:
                run(WSL + ["rm", "-f", container])
            except Exception as error:
                errors.append(str(error))
        if reservation:
            try:
                record["release"] = api("/v1/photo-gpu-reservations/" + reservation, "DELETE", {"owner": OWNER})
            except Exception as error:
                errors.append(str(error))
        record["cleanup_errors"] = errors
        record["production_config_unchanged"] = hashlib.sha256((ROOT / "broker/config.json").read_bytes()).hexdigest() == config_sha
        save()
        for handle in reversed(locks):
            kernel.ReleaseMutex(handle)
            kernel.CloseHandle(handle)
    print(json.dumps({"status": record["status"], "runs": len(record["runs"]), "profiles": record["profiles"]}), flush=True)


if __name__ == "__main__":
    main()
