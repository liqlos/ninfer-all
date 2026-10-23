import csv
import ctypes
import hashlib
import json
import pathlib
import subprocess
import time
import urllib.request
import uuid


ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results/bf16-store-pr"
GPU = OUT / "gpu"
BASE = "http://127.0.0.1:8096"
OWNER = "ninfer-bf16-store-pr"
IMAGE = "sha256:501c92a90c3d85adc616455a6783bb5ae8b82008f469cf8dc3d6da538f75475b"
WSL = ["wsl.exe", "-d", "Ubuntu-24.04", "--cd", "/", "--exec", "docker"]
MOUNT = "/mnt/c/Users/aleksei/Projects/NInfer/results/bf16-store-pr"


def api(path, method="GET", body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=25) as response:
        return json.load(response)


def run(argv, timeout=60):
    result = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, timeout=timeout, creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode:
        raise RuntimeError(str(argv) + "\n" + result.stdout)
    return result.stdout.strip()


def admission(reservation=None):
    status = api("/v1/status")
    a = status["admission"]
    cap = status["capacity_estimate"][0]
    if status["leases"] or status["agents"] or status["queue"] or a["active_batches"]:
        raise RuntimeError("foreign broker work is active")
    if status["inference"]["active_requests"] or status["inference"]["active_sequences"]:
        raise RuntimeError("foreign inference is active")
    if a["runtime_maintenance"]["active"] or status["external_growth_pressure"]["active"]:
        raise RuntimeError("maintenance or GPU pressure is active")
    if cap["measurement"] != "measured" or not 0 <= cap["sample_age_seconds"] <= 15:
        raise RuntimeError("GPU budget is stale")
    if cap["available_llm_budget_bytes"] < 2 * 1024**3 or not cap["external_reserve_satisfied"]:
        raise RuntimeError("insufficient GPU budget")
    if reservation:
        if not a["draining"] or a["photo_gpu_reservation"]["owner"] != OWNER:
            raise RuntimeError("owned reservation is not the active drain")
    elif a["draining"] or a["photo_gpu_reservation"]["active"]:
        raise RuntimeError("broker is already reserved")
    if run(WSL + ["ps", "-q", "--filter", "label=io.codex.ninfer-gpu=true"]):
        raise RuntimeError("another lab GPU container is active")
    return status


def main():
    if (OUT / "qualification.json").exists():
        raise RuntimeError("preserve completed or interrupted qualification")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p]
    kernel.CreateMutexW.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel.ReleaseMutex.argtypes = [ctypes.c_void_p]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    locks = []
    for name in ("Local\\NInferV17RuntimeScreen", "Local\\NInferV14RuntimeMeasurement",
                 "Local\\NInferV15ExportedOperatorSeries", "Local\\NInferV14CpuVariantPreparation"):
        mutex = kernel.CreateMutexW(None, False, name)
        if not mutex or kernel.WaitForSingleObject(mutex, 0) not in (0, 128):
            raise RuntimeError("another lab worker owns " + name)
        locks.append(mutex)
    GPU.mkdir(exist_ok=True)
    config_hash = hashlib.sha256((ROOT / "broker/config.json").read_bytes()).hexdigest()
    builds = {}
    for arm in ("control", "candidate"):
        build = json.loads((OUT / (arm + "-build.json")).read_text())
        binary = OUT / "bin" / arm
        if build["status"] != "PASS_OP_BUILD_LINK" or build["binary_sha256"] != hashlib.sha256(binary.read_bytes()).hexdigest():
            raise RuntimeError("unqualified or changed binary " + arm)
        if build["production_config_sha256"] != config_hash:
            raise RuntimeError("production configuration changed")
        builds[arm] = {k: build[k] for k in ("source", "binary_sha256", "binary_bytes")}
    protocol = {"scope": "public Linear Op, synthetic GGUF weights, cold-cache eager execution",
        "builds": builds, "points": [("q4_k", 17408, 5120), ("q6_k", 5120, 17408)],
        "tokens": [16, 128, 512, 1024], "orders": ["ABBA", "BAAB", "ABBA"],
        "warmup": 20, "repeat": 50, "flush_mib": 256, "profile_in_timing": False}
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    record = {"status": "RUNNING", "protocol": protocol, "tests": [], "benchmarks": [],
              "production_config_sha256": config_hash, "started_unix": time.time()}
    reservation = None
    containers = []

    def save():
        (OUT / "qualification.json").write_text(json.dumps(record, indent=2) + "\n")

    def execute(arm, label, argv):
        admission(reservation)
        name = "ninfer-pr-" + uuid.uuid4().hex[:12]
        docker = WSL + ["run", "-d", "--name", name, "--network", "none", "--gpus", "all",
            "--label", "io.codex.ninfer-gpu=true", "--label", "io.codex.ninfer-owner=" + OWNER,
            "--mount", "type=bind,source=" + MOUNT + "/bin,target=/bin-pr,readonly",
            "--mount", "type=bind,source=" + MOUNT + "/gpu,target=/results",
            "--entrypoint", "/bin-pr/" + arm, IMAGE, *argv]
        container = run(docker)
        containers.append(container)
        started = time.monotonic()
        code = run(WSL + ["wait", container], timeout=900)
        output = run(WSL + ["logs", container])
        (GPU / (label + ".log")).write_text(output + "\n")
        run(WSL + ["rm", container])
        containers.remove(container)
        if code != "0":
            raise RuntimeError(label + " exited " + code + ": " + output[-1600:])
        return {"label": label, "arm": arm, "exit": code, "seconds": time.monotonic() - started}

    try:
        record["admission_before"] = admission()
        reservation = uuid.uuid4().hex
        grant = api("/v1/photo-gpu-reservations", "POST",
                    {"owner": OWNER, "reservation_id": reservation, "ttl_seconds": 3600})
        if grant["reservation_id"] != reservation or grant["owner"] != OWNER or grant["state"] != "active":
            raise RuntimeError("reservation identity mismatch")
        record["reservation"] = grant
        save()
        for arm, command in (("control", "oracle"), ("candidate", "store"), ("candidate", "oracle")):
            record["tests"].append(execute(arm, arm + "-" + command, [command]))
            save()
        for qtype, n, k in protocol["points"]:
            for t in protocol["tokens"]:
                for block, order in enumerate(protocol["orders"]):
                    for position, choice in enumerate(order):
                        arm = "control" if choice == "A" else "candidate"
                        label = f"{qtype}-t{t}-b{block}-s{position}-{arm}"
                        row = execute(arm, label, ["bench", "--qtype", qtype, "--n", str(n),
                            "--k", str(k), "--t", str(t), "--warmup", "20", "--repeat", "50",
                            "--flush-mib", "256", "--execution", "eager", "--csv-out", "/results/" + label + ".csv"])
                        with (GPU / (label + ".csv")).open(newline="") as file:
                            rows = list(csv.DictReader(file))
                        if len(rows) != 1 or int(rows[0]["T"]) != t:
                            raise RuntimeError("benchmark point mismatch")
                        row.update(qtype=qtype, n=n, k=k, t=t, block=block, position=position,
                                   median_us=float(rows[0]["median_us"]))
                        record["benchmarks"].append(row)
                        save()
        record["status"] = "PASS_GPU_TESTS_AND_FIXED_OP_SCREEN"
    except BaseException as error:
        record.update(status="FAILED", error=str(error))
        raise
    finally:
        cleanup_errors = []
        for container in containers:
            try:
                run(WSL + ["rm", "-f", container])
            except Exception as error:
                cleanup_errors.append(str(error))
        if reservation:
            try:
                record["release"] = api("/v1/photo-gpu-reservations/" + reservation, "DELETE", {"owner": OWNER})
            except Exception as error:
                cleanup_errors.append(str(error))
        record["cleanup_errors"] = cleanup_errors
        if cleanup_errors:
            record["status"] = "CLEANUP_REQUIRES_ATTENTION"
        record["production_config_unchanged"] = config_hash == hashlib.sha256((ROOT / "broker/config.json").read_bytes()).hexdigest()
        record["finished_unix"] = time.time()
        save()
        for mutex in reversed(locks):
            kernel.ReleaseMutex(mutex)
            kernel.CloseHandle(mutex)
    print(json.dumps({"status": record["status"], "tests": record["tests"], "runs": len(record["benchmarks"])}), flush=True)


if __name__ == "__main__":
    main()
