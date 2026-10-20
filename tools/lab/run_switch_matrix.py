import copy
import ctypes
import importlib.util
import json
import pathlib
import re
import socket
import sys
import time
import traceback
import urllib.request
import uuid

import build_switch_matrix as matrix
import profile_long_context as lab


ROOT, OUT = lab.ROOT, matrix.OUT
PROMPTS = ROOT / "results/dp4a-k5120-confirmation-20261019/dp4a-k5120-confirmation-20261019-prompts.json"
MODEL = ROOT / "models/stable/qwen38-jonathan-q4km-proposal-vision-q.ninfer"
MODEL_SHA = "781fcd7da81eb72a123091054ec9b2bd163798a374a127b0f90f68dae3edbf27"
BENCH = ROOT / "scripts/bench-jonathan-spec-short-20261018.py"
BENCH_SHA = "9395ab89a9c460fb3b66e7bc0d35111ae78b7c4a8cd9e910abaf8cf22200a6ff"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def serving_admission(reservation):
    status = lab.api("/v1/status")
    grant = status["admission"]["photo_gpu_reservation"]
    allowed = {"dwm", "explorer", "ChatGPT", "Codex", "chrome", "msedge", "firefox",
               "csrss", "vmmemWSL", "vmmem"}
    checks = {
        "broker_idle": not status["agents"] and not status["queue"] and not status["leases"],
        "inference_idle": not status["inference"]["active_requests"] and not status["inference"]["active_sequences"],
        "owned_reservation": grant["active"] and grant["owner"] == lab.OWNER,
        "no_batches": not status["admission"]["active_batches"],
        "allowed_processes": all(p["process"] in allowed for p in status["windows_gpu"]["process_memory"]),
        "no_external_growth": not status.get("external_growth_pressure", {}).get("active"),
    }
    if not all(checks.values()):
        raise RuntimeError(json.dumps({"admission_checks": checks,
                                      "process_memory": status["windows_gpu"]["process_memory"],
                                      "reservation": grant,
                                      "external_growth_pressure": status.get("external_growth_pressure")}))


def main():
    arm, mode, tag, operator_tag = sys.argv[1:5]
    control_tag = sys.argv[5] if len(sys.argv) == 6 else "w-screen-r1"
    assert len(sys.argv) in (5, 6)
    assert arm in matrix.VARIANTS and mode in {"profile", "screen"}
    assert all(re.fullmatch(r"[a-z0-9-]+", s) for s in (tag, operator_tag, control_tag))
    receipt = OUT / (tag + "-runtime.json")
    assert not receipt.exists()
    assert not list(OUT.glob(tag + "-*"))
    record = {"arm": arm, "mode": mode, "started_at": lab.now(),
              "exploratory": mode == "screen", "performance_claim": False}
    reservation, ident, handles = None, None, []
    config = lab.sha(ROOT / "broker/config.json")
    lab.OWNER = "ninfer-switch-matrix"
    k = ctypes.WinDLL("kernel32")
    k.CreateMutexW.restype = ctypes.c_void_p
    k.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    k.ReleaseMutex.argtypes = [ctypes.c_void_p]
    k.CloseHandle.argtypes = [ctypes.c_void_p]
    try:
        for name in matrix.MUTEXES:
            h = k.CreateMutexW(None, False, "Local\\" + name)
            if k.WaitForSingleObject(h, 0) not in (0, 128):
                k.CloseHandle(h)
                raise RuntimeError("Laboratory mutex busy: " + name)
            handles.append(h)
        record["admission"] = lab.admission()
        with socket.socket() as sock:
            assert sock.connect_ex(("127.0.0.1", 18042)) != 0
        export = load(ROOT / "artifacts/experimental/export-manifest.json")
        assert export["arm"] == arm
        exe = ROOT / "artifacts/experimental/ninfer-serve"
        exe_sha = lab.sha(exe)
        assert exe_sha == export["files"]["ninfer-serve"]
        record["export"] = export
        build = load(ROOT / "artifacts/experimental/build-receipt.json")
        assert build["export_sha256"] == export["files"] and build["arm"] == arm
        record["build_options"] = build["options"]
        record["build_receipt_sha256"] = lab.sha(ROOT / "artifacts/experimental/build-receipt.json")
        if mode == "screen":
            route = load(OUT / (arm + "-route.json"))
            assert route["status"] == "PASS" and route["executable_sha256"] == exe_sha
        gate = load(OUT / (operator_tag + "-qualification.json"))
        assert gate["status"] == "PASS" and gate["arm"] == arm
        assert gate["export"]["files"] == export["files"]
        assert lab.sha(MODEL) == MODEL_SHA and lab.sha(BENCH) == BENCH_SHA
        protocol = load(OUT / "protocol.json")
        assert protocol["model_sha256"] == MODEL_SHA
        assert protocol["bench_sha256"] == BENCH_SHA
        assert protocol["prompts_sha256"] == lab.sha(PROMPTS)
        record["protocol_sha256"] = lab.sha(OUT / "protocol.json")
        golden_path = OUT / (control_tag + "-runtime.json")
        golden_record = load(golden_path) if golden_path.exists() else None
        if golden_record is not None:
            assert golden_record["status"] == "SCREENED" and golden_record["arm"] == "w"
            assert golden_record["mode"] == "screen"
            assert golden_record["protocol_sha256"] == record["protocol_sha256"]
            assert golden_record["model_sha256"] == MODEL_SHA
            assert golden_record["prompts_sha256"] == lab.sha(PROMPTS)
            record["control_receipt_sha256"] = lab.sha(golden_path)
        else:
            assert arm == "w" and (mode == "profile" or tag == control_tag)
        golden = golden_record["samples"] if golden_record else None
        record["control_tag"] = control_tag
        prompts = load(PROMPTS)
        record.update(model_sha256=MODEL_SHA, bench_sha256=BENCH_SHA,
                      prompts_sha256=lab.sha(PROMPTS))
        spec = importlib.util.spec_from_file_location("frozen_bench", BENCH)
        bench = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bench)
        reservation = uuid.uuid4().hex
        granted = lab.api("/v1/photo-gpu-reservations", {
            "owner": lab.OWNER, "reservation_id": reservation, "ttl_seconds": 1800}, "POST", 90)
        assert granted["reservation_id"] == reservation and granted["state"] == "active"
        record["reservation_id"] = reservation
        record["owned_admission"] = lab.admission(True)
        args = ["/usr/local/bin/ninfer-serve", "/model/jonathan-full-q4km.ninfer",
                "--host", "0.0.0.0", "--port", "8080", "--model-id", "qwen3.8-27b",
                "--max-context", "65536", "--kv-capacity", "65536", "--max-concurrency", "1",
                "--max-pending-requests", "32", "--pending-timeout-ms", "600000",
                "--kv-dtype", "int8", "--spec", "mtp", "--draft-tokens", "3",
                "--prefill-chunk", "1024", "--no-thinking", "--structured-output",
                "--vision", "--vision-residency", "overlay", "--host-kv-mib", "0",
                "--host-state-slots", "0", "--request-log-jsonl", "/results/" + tag + "-events.jsonl"]
        record["command"] = args
        command = ["create", "--name", "ninfer-lab-matrix-" + reservation[:8],
                   "--label", "io.codex.ninfer-owner=" + reservation,
                   "--gpus", "all", "--ipc=host", "-p", "127.0.0.1:18042:8080",
                   "--mount", "type=bind,source=" + lab.WSL_ROOT + "/artifacts/experimental/ninfer-serve,target=/usr/local/bin/ninfer-serve,readonly",
                   "--mount", "type=bind,source=" + lab.WSL_ROOT + "/models/stable/" + MODEL.name + ",target=/model/jonathan-full-q4km.ninfer,readonly",
                   "--mount", "type=bind,source=" + lab.WSL_ROOT + "/results/switch-matrix,target=/results",
                   "-e", "CUDA_CACHE_PATH=/tmp/ninfer-matrix"]
        if mode == "profile":
            assert lab.sha(ROOT / "tools/nsys/nsys") == "ed489f1cdacf354aa6d91201e3f367863a4195153d5aa75a773719235b49315f"
            command += ["--mount", "type=bind,source=" + lab.WSL_ROOT + "/tools/nsys,target=/nsys/target-linux-x64,readonly",
                        "--entrypoint", "/bin/sh", lab.IMAGE, "-c",
                        'ln -s /nsys/target-linux-x64/nsys /tmp/ninfer-nsys; exec /tmp/ninfer-nsys "$@"',
                        "nsys", "profile", "--trace=cuda,nvtx", "--sample=none", "--cpuctxsw=none",
                        "--cuda-graph-trace=node", "--force-overwrite=false",
                        "-o", "/results/" + tag + "-nsys", *args]
        else:
            command += ["--entrypoint", args[0], lab.IMAGE, *args[1:]]
        ident = lab.docker(*command).stdout.strip()
        assert len(ident) == 64
        record["container_id"] = ident
        lab.docker("start", ident)
        for _ in range(240):
            try:
                with urllib.request.urlopen("http://127.0.0.1:18042/health", timeout=2):
                    break
            except Exception:
                assert json.loads(lab.docker("inspect", ident).stdout)[0]["State"]["Running"]
                time.sleep(1)
        else:
            raise RuntimeError("Lab readiness timed out")
        running_proof = 'for p in /proc/[0-9]*; do [ "$(readlink "$p/exe" 2>/dev/null)" = /usr/local/bin/ninfer-serve ] && sha256sum "$p/exe"; done; true'
        actual = lab.docker("exec", ident, "sh", "-c", running_proof).stdout.split()
        assert len(actual) == 2 and actual[0] == exe_sha
        record["running_executable_sha256"] = actual[0]
        warm = copy.deepcopy(prompts["categories"][0]["request_body"])
        warm["max_tokens"] = 32
        warm["messages"][-1]["content"] = "Switch matrix warmup\n" + warm["messages"][-1]["content"]
        serving_admission(reservation)
        record["warmup"] = lab.request(warm)
        assert record["warmup"]["usage"]["completion_tokens"] == 32
        print("Lab ready; collecting " + mode + " requests for " + arm, flush=True)
        events = OUT / (tag + "-events.jsonl")
        rows, used, prior = [], set(), 0
        selected = prompts["categories"] if mode == "screen" else prompts["categories"][:1]
        for item in selected:
            serving_admission(reservation)
            offset = bench.request_log_offset(events)
            observed, request_sha = bench.stream_request(18042, item["request_body"])
            assert request_sha == item["request_body_sha256"]
            done = bench.correlate_request_done(events, offset, observed, "mtp3", prior,
                                               observed["usage"], observed["timings"], used)
            prior = done["request_id"]
            assert observed["http_status"] == 200
            assert observed["usage"]["prompt_tokens"] == item["prompt_tokens"]
            assert observed["usage"]["completion_tokens"] == 512
            row = {name: observed[name] for name in ("elapsed_seconds", "ttft_content_seconds",
                   "usage", "timings", "finish_reason", "output_sha256")}
            row.update(category=item["category"], request_body_sha256=request_sha, request_done=done)
            if golden is not None:
                reference = next(r for r in golden if r["category"] == item["category"])
                assert all(row[field] == reference[field] for field in
                           ("output_sha256", "usage", "finish_reason", "request_body_sha256"))
                assert done["speculative"] == reference["request_done"]["speculative"]
            elif arm != "w":
                raise RuntimeError("Current W exact-output control is missing")
            rows.append(row)
            record["samples"] = rows
            print(item["category"] + " " + str(round(observed["elapsed_seconds"], 3)) + " s", flush=True)
        record["status"] = "CAPTURED" if mode == "profile" else "SCREENED"
    except BaseException as exc:
        record["status"] = "INCOMPLETE"
        record["failure"] = type(exc).__name__ + ": " + str(exc)
        record["failure_traceback"] = traceback.format_exc()
    finally:
        if ident:
            try:
                item = json.loads(lab.docker("inspect", ident).stdout)[0]
                assert item["Config"]["Labels"]["io.codex.ninfer-owner"] == reservation
                assert item["Image"] == lab.IMAGE
                if item["State"]["Running"]:
                    shell = 'for p in /proc/[0-9]*; do [ "$(readlink "$p/exe" 2>/dev/null)" = /usr/local/bin/ninfer-serve ] || continue; [ "$(sha256sum "$p/exe" | cut -d" " -f1)" = ' + exe_sha + ' ] || exit 3; kill -TERM "${p#/proc/}"; done; true'
                    lab.docker("exec", ident, "sh", "-c", shell)
                    try:
                        lab.docker("wait", ident, timeout=55)
                    except Exception:
                        lab.docker("stop", "--time", "5", ident)
                log = lab.docker("logs", ident, check=False)
                (OUT / (tag + "-server.log")).write_text(log.stdout + log.stderr, encoding="utf-8")
                lab.docker("rm", ident)
                record["owned_container_removed"] = True
                if mode == "profile" and not record.get("failure"):
                    capture = OUT / (tag + "-nsys.nsys-rep")
                    if not capture.exists():
                        capture = OUT / (tag + "-nsys.qdstrm")
                    record["capture"] = {"file": capture.name, "sha256": lab.sha(capture)}
            except BaseException as exc:
                record["cleanup_failure"] = str(exc)
        if reservation and (ident is None or record.get("owned_container_removed")):
            try:
                released = lab.api("/v1/photo-gpu-reservations/" + reservation, {"owner": lab.OWNER}, "DELETE")
                assert released["reservation_id"] == reservation and released["state"] == "released"
                record["reservation_released"] = True
            except BaseException as exc:
                record["cleanup_failure"] = str(exc)
        for h in reversed(handles):
            k.ReleaseMutex(h)
            k.CloseHandle(h)
        record["production_config_unchanged"] = lab.sha(ROOT / "broker/config.json") == config
        if record.get("cleanup_failure") or not record["production_config_unchanged"]:
            record["status"] = "INCOMPLETE"
        record["finished_at"] = lab.now()
        with receipt.open("x", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        print(json.dumps({key: record.get(key) for key in
              ("status", "failure", "cleanup_failure", "owned_container_removed", "reservation_released")}), flush=True)
    if record.get("failure") or record.get("cleanup_failure"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
