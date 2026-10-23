import ctypes
import hashlib
import json
import runpy
import sqlite3
import time
import uuid

from pathlib import Path


HELPERS = runpy.run_path(str(Path(__file__).with_name("qualify-bf16-store-gpu.py")))
ROOT, OUT, WSL, IMAGE, OWNER, MOUNT = (HELPERS[k] for k in ("ROOT", "OUT", "WSL", "IMAGE", "OWNER", "MOUNT"))
api, run, admission = (HELPERS[k] for k in ("api", "run", "admission"))


def main():
    receipt = OUT / "route.json"
    previous = json.loads(receipt.read_text()) if receipt.exists() else None
    if previous and previous["status"] != "FAILED":
        raise RuntimeError("preserve existing route receipt")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p]
    kernel.CreateMutexW.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel.ReleaseMutex.argtypes = [ctypes.c_void_p]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    locks = []
    for name in ("Local\\NInferV17RuntimeScreen", "Local\\NInferV14RuntimeMeasurement",
                 "Local\\NInferV15ExportedOperatorSeries", "Local\\NInferV14CpuVariantPreparation"):
        handle = kernel.CreateMutexW(None, False, name)
        if not handle or kernel.WaitForSingleObject(handle, 0) not in (0, 128):
            raise RuntimeError("another lab worker owns " + name)
        locks.append(handle)
    config_hash = hashlib.sha256((ROOT / "broker/config.json").read_bytes()).hexdigest()
    build = json.loads((OUT / "candidate-build.json").read_text())
    if hashlib.sha256((OUT / "bin/candidate").read_bytes()).hexdigest() != build["binary_sha256"]:
        raise RuntimeError("candidate binary changed")
    record = {"status": "RUNNING", "binary_sha256": build["binary_sha256"],
              "source": build["source"], "profile_not_timing": True, "points": []}
    if previous:
        record["previous_attempt"] = previous
        record["points"] = previous["points"]
    reservation = None
    container = None
    try:
        record["admission_before"] = admission()
        reservation = uuid.uuid4().hex
        grant = api("/v1/photo-gpu-reservations", "POST",
                    {"owner": OWNER, "reservation_id": reservation, "ttl_seconds": 900})
        if grant["reservation_id"] != reservation or grant["owner"] != OWNER or grant["state"] != "active":
            raise RuntimeError("reservation mismatch")
        record["reservation"] = grant
        for qtype, n, k in (("q4_k", 17408, 5120), ("q6_k", 5120, 17408)):
            for t in (128, 512):
                admission(reservation)
                label = f"route-{qtype}-t{t}"
                if any(x["qtype"] == qtype and x["t"] == t for x in record["points"]):
                    for suffix in (".sqlite", ".nsys-rep", ".qdstrm"):
                        (OUT / "gpu" / (label + suffix)).unlink(missing_ok=True)
                    continue
                profile = ["/tmp/ninfer-nsys", "profile", "--trace=cuda", "--sample=none", "--cpuctxsw=none",
                           "--capture-range=cudaProfilerApi", "--capture-range-end=stop", "--force-overwrite=false",
                           "-o", "/results/" + label, "/bin-pr/candidate", "bench", "--qtype", qtype,
                           "--n", str(n), "--k", str(k), "--t", str(t), "--warmup", "20", "--profile"]
                name = "ninfer-pr-route-" + uuid.uuid4().hex[:10]
                docker = WSL + ["run", "-d", "--name", name, "--network", "none", "--gpus", "all",
                    "--label", "io.codex.ninfer-gpu=true", "--label", "io.codex.ninfer-owner=" + OWNER,
                    "--mount", "type=bind,source=" + MOUNT + "/bin,target=/bin-pr,readonly",
                    "--mount", "type=bind,source=" + MOUNT + "/gpu,target=/results",
                    "--mount", "type=bind,source=/mnt/c/Users/aleksei/Projects/NInfer/tools/nsys,target=/nsys/target-linux-x64,readonly",
                    "--entrypoint", "/bin/bash", IMAGE, "-c",
                    'ln -s /nsys/target-linux-x64/nsys /tmp/ninfer-nsys; exec "$@"', "route", *profile]
                if not (OUT / "gpu" / (label + ".qdstrm")).exists():
                    container = run(docker)
                    code = run(WSL + ["wait", container], timeout=300)
                    log = run(WSL + ["logs", container])
                    (OUT / "gpu" / (label + ".log")).write_text(log + "\n")
                    run(WSL + ["rm", container])
                    container = None
                    if code != "0":
                        raise RuntimeError(label + " failed: " + log[-2000:])
                run(WSL + ["exec", "ninfer-builder",
                    "/opt/nvidia/nsight-compute/2025.4.1/host/linux-desktop-glibc_2_11_3-x64/QdstrmImporter",
                    "-i", "/lab/results/bf16-store-pr/gpu/" + label + ".qdstrm",
                    "-o", "/lab/results/bf16-store-pr/gpu/" + label + ".nsys-rep"], timeout=180)
                exported = run(WSL + ["exec", "ninfer-builder", "/lab/tools/nsys/nsys", "export", "--type=sqlite",
                    "--output=/lab/results/bf16-store-pr/gpu/" + label + ".sqlite",
                    "/lab/results/bf16-store-pr/gpu/" + label + ".nsys-rep"], timeout=180)
                db = OUT / "gpu" / (label + ".sqlite")
                with sqlite3.connect(db) as connection:
                    rows = connection.execute("SELECT s.value,COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName GROUP BY s.value").fetchall()
                connection.close()
                kernels = [{"name": name, "count": count} for name, count in rows]
                mmq = [x for x in kernels if "mul_mat_q" in x["name"] and "__nv_bfloat16" in x["name"]]
                fixup = [x for x in kernels if "fixup" in x["name"] and "__nv_bfloat16" in x["name"]]
                separate = [x for x in kernels if "store_plane" in x["name"]]
                if not mmq or separate or (t == 128 and not fixup) or (t == 512 and fixup):
                    raise RuntimeError(label + " unexpected kernel route: " + str(kernels))
                record["points"].append({"qtype": qtype, "n": n, "k": k, "t": t, "kernels": kernels,
                    "bf16_mmq": True, "bf16_fixup": bool(fixup), "separate_store_plane": False})
                receipt.write_text(json.dumps(record, indent=2) + "\n")
                for suffix in (".sqlite", ".nsys-rep", ".qdstrm"):
                    (OUT / "gpu" / (label + suffix)).unlink()
        record["status"] = "PASS_PUBLIC_BF16_MMQ_AND_FIXUP_ROUTE"
    except BaseException as error:
        record.update(status="FAILED", error=str(error))
        raise
    finally:
        errors = []
        if container:
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
        record["production_config_unchanged"] = config_hash == hashlib.sha256((ROOT / "broker/config.json").read_bytes()).hexdigest()
        record["finished_unix"] = time.time()
        receipt.write_text(json.dumps(record, indent=2) + "\n")
        for handle in reversed(locks):
            kernel.ReleaseMutex(handle)
            kernel.CloseHandle(handle)
    print(record["status"], flush=True)


if __name__ == "__main__":
    main()
