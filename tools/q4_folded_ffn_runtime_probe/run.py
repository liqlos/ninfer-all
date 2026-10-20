import argparse
import copy
import ctypes
import importlib.util
import json
import pathlib
import re
import socket
import subprocess
import sys
import time
import traceback
import urllib.request
import uuid

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lab"))
import build_switch_matrix as matrix
import profile_long_context as lab
from run_switch_matrix import serving_admission

ROOT = lab.ROOT
OUT = ROOT / "results/q4-folded-ffn-runtime"
PROTOCOL = OUT / "protocol-r2.json"
PROTOCOL_SHA = "1ff33e339731bfe6a7ac22f8364ce8e21fbb05e93c243ea0e29f721c5e2a5574"
BUILD = OUT / "build-r2.json"
BUILD_SHA = "6f7426bc1e6dcad9a4cdf347f1d97e23bd0fae6ed88ed5f6859cbb485d8d6e89"
CONFIRM_PROTOCOL = OUT / "confirmation-protocol-r2.json"
CONFIRM_PROTOCOL_SHA = "529472b6d838b9abd1a78cac1f857417722f21121d69de8b85d319620b1eb710"
EXPORT = ROOT / "artifacts/experimental/folded16-runtime-serve"
EXE = "/usr/local/bin/ninfer-serve"
LABEL = "io.codex.ninfer-owner"
ENV = "NINFER_LAB_FOLDED16_HOST_ADMISSIONS"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def file_identity(path):
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    return {"path": str(resolved), **{key: getattr(stat, key) for key in
            ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")}}


def unreserved_admission(confirmation):
    if not confirmation:
        return lab.admission()
    assert lab.sha(CONFIRM_PROTOCOL) == CONFIRM_PROTOCOL_SHA
    policy = load(CONFIRM_PROTOCOL)["pre_admission_wait"]
    started, attempts, announced = time.monotonic(), 0, False
    while True:
        attempts += 1
        try:
            result = lab.admission()
            assert result["host"]["free_commit"] >= policy["start_free_commit_GiB"] * 2**30
            result["pre_admission_wait_seconds"] = time.monotonic() - started
            result["pre_admission_attempts"] = attempts
            return result
        except AssertionError:
            if time.monotonic() - started >= policy["maximum_seconds_per_gate"]:
                raise
            if not announced:
                print("Waiting for full-model admission; no GPU reservation held.", flush=True)
                announced = True
            time.sleep(policy["poll_seconds"])


def owned(ident, token):
    item = json.loads(lab.docker("inspect", ident).stdout)[0]
    assert (item["Id"] == ident or item["Name"] == "/" + ident) and item["Image"] == lab.IMAGE
    assert item["Config"]["Labels"][LABEL] == token
    return item


def remove_owned(ident, token, exe_sha=None):
    item = owned(ident, token)
    if item["State"]["Running"]:
        if exe_sha:
            shell = 'for p in /proc/[0-9]*; do [ "$(readlink "$p/exe" 2>/dev/null)" = ' + EXE + ' ] || continue; [ "$(sha256sum "$p/exe" | cut -d" " -f1)" = ' + exe_sha + ' ] || exit 3; kill -TERM "${p#/proc/}"; done; true'
            lab.docker("exec", ident, "sh", "-c", shell)
            try:
                lab.docker("wait", ident, timeout=55)
            except Exception:
                lab.docker("stop", "--time", "5", ident)
        else:
            lab.docker("stop", "--time", "5", ident)
    log = lab.docker("logs", ident, check=False)
    lab.docker("rm", ident)
    return log.stdout + log.stderr


def main():
    global PROTOCOL, PROTOCOL_SHA
    parser = argparse.ArgumentParser(description="One serialized lab-only folded16 request/profile launch.")
    parser.add_argument("arm", choices=("off", "on"))
    parser.add_argument("mode", choices=("screen", "profile", "confirm"))
    parser.add_argument("tag")
    parser.add_argument("control_tag")
    parser.add_argument("--route", type=pathlib.Path)
    args = parser.parse_args()
    if args.mode == "confirm":
        PROTOCOL, PROTOCOL_SHA = CONFIRM_PROTOCOL, CONFIRM_PROTOCOL_SHA
    assert not sys.flags.optimize, "Run without Python -O"
    assert all(re.fullmatch(r"[a-z0-9-]+", value) for value in (args.tag, args.control_tag))
    receipt = OUT / (args.tag + "-runtime.json")
    record = {"schema": "folded16-runtime-launch-r2", "arm": args.arm, "mode": args.mode,
              "tag": args.tag, "control_tag": args.control_tag, "started_at": lab.now(),
              "status": "INCOMPLETE", "lab_only": True, "product_integration": False,
              "diagnostic_only": args.mode == "profile", "performance_claim": False,
              "host_admissions_enabled": args.mode == "profile", "host_admissions_count_gpu_replays": False,
              "phases_seconds": {}, "runner_sha256": lab.sha(pathlib.Path(__file__))}
    lab.OWNER = "ninfer-folded16-runtime"
    token, reservation, ident, probe = uuid.uuid4().hex, None, None, None
    handles, claimed, config, exe_sha = [], False, None, None
    k = ctypes.WinDLL("kernel32")
    k.CreateMutexW.restype = ctypes.c_void_p
    k.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    k.ReleaseMutex.argtypes = [ctypes.c_void_p]
    k.CloseHandle.argtypes = [ctypes.c_void_p]
    try:
        for name in matrix.MUTEXES:
            h = k.CreateMutexW(None, False, "Local\\" + name)
            if not h or k.WaitForSingleObject(h, 0) not in (0, 128):
                if h:
                    k.CloseHandle(h)
                raise RuntimeError("Laboratory mutex busy: " + name)
            handles.append(h)
        assert not receipt.exists() and not list(OUT.glob(args.tag + "-*"))
        claimed = True
        config = lab.sha(ROOT / "broker/config.json")
        record["admission"] = unreserved_admission(args.mode == "confirm")
        initial = lab.api("/v1/status")
        assert initial["model_state"] == "idle" and not initial["instances"]
        assert all(not m["replicas"] and not m["ready_replicas"] for m in initial["models"])
        record["initial_model_unloaded"] = True
        with socket.socket() as sock:
            assert sock.connect_ex(("127.0.0.1", 18042)) != 0
        assert lab.sha(PROTOCOL) == PROTOCOL_SHA and lab.sha(BUILD) == BUILD_SHA
        protocol, build = load(PROTOCOL), load(BUILD)
        assert protocol["created_before_inference"] and protocol["image"] == lab.IMAGE
        assert (protocol.get("stage") == "independent_confirmation") == (args.mode == "confirm")
        assert build["status"] == "LAB_ONLY_OFF_ON_WRAPPERS_LINKED_NOT_GPU_QUALIFIED"
        assert build["build"]["status"] == "LAB_WRAPPERS_LINKED_NOT_RUNTIME_QUALIFIED"
        assert all(lab.sha(HERE / name) == digest for name, digest in build["source_sha256"].items())
        output = build["build"]["outputs"][args.arm]
        assert output["path"] == "/build/q4-folded-ffn/runtime/ninfer-serve-" + args.arm
        exe_sha = output["sha256"]
        static_path = OUT / "static-link-r2.json"
        static = load(static_path)
        assert static["status"] == "PASS_STATIC_WRAPPED_FFN_CALL_AND_IDENTICAL_LINKED_DEVICE_KERNELS"
        assert all(static["arms"][arm]["binary_sha256"] == entry["sha256"]
                   for arm, entry in build["build"]["outputs"].items())
        record["static_link_receipt"] = {"file": static_path.name, "sha256": lab.sha(static_path)}
        record.update(protocol_sha256=PROTOCOL_SHA, build_receipt_sha256=BUILD_SHA,
                      executable_sha256=exe_sha, model_sha256=protocol["model"]["sha256"],
                      bench_sha256=protocol["bench"]["sha256"], image=protocol["image"], selection=protocol["selection"])
        control_path = OUT / (args.control_tag + "-runtime.json")
        control = load(control_path) if control_path.exists() else None
        if control:
            assert control["status"] == "SCREENED" and control["arm"] == "off"
            assert control["mode"] == ("confirm" if args.mode == "confirm" else "screen") and control["sequence_index"] == 0
            assert control["tag"] == control["control_tag"] == args.control_tag
            assert all(control[key] == record[key] for key in
                       ("protocol_sha256", "build_receipt_sha256", "model_sha256", "bench_sha256", "runner_sha256"))
            assert control["executable_sha256"] == build["build"]["outputs"]["off"]["sha256"]
            record["reference_receipt_sha256"] = lab.sha(control_path)
        else:
            assert args.arm == "off" and args.mode != "profile" and args.tag == args.control_tag
        if args.mode != "profile":
            previous = [load(p) for p in OUT.glob("*-runtime.json")]
            previous = sorted((r for r in previous if r.get("mode") == args.mode
                               and r.get("control_tag") == args.control_tag
                               and r.get("protocol_sha256") == PROTOCOL_SHA), key=lambda r: r["started_at"])
            assert all(r["status"] == "SCREENED" for r in previous), "Prior screen failed; do not extend"
            assert [r["arm"] for r in previous] == protocol["order"][:len(previous)]
            assert len(previous) < len(protocol["order"]) and args.arm == protocol["order"][len(previous)]
            record["sequence_index"] = len(previous)
        if args.arm == "on" and args.mode != "profile":
            assert args.route is not None, "ON timing requires the independent current CUDA route PASS"
            route_path = args.route if args.route.is_absolute() else ROOT / args.route
            route = load(route_path)
            assert route["status"] == "PASS"
            assert all(route[key] == record[key] for key in
                       ("executable_sha256", "protocol_sha256", "reference_receipt_sha256"))
            record["route_receipt"] = {"path": str(route_path), "sha256": lab.sha(route_path)}
        started = time.monotonic()
        model, bench_path = ROOT / protocol["model"]["path"], ROOT / protocol["bench"]["path"]
        record["model_file_identity"] = file_identity(model)
        if not control or record["model_file_identity"] != control["model_file_identity"]:
            assert lab.sha(model) == record["model_sha256"]
            assert file_identity(model) == record["model_file_identity"]
            record["model_verification"] = "full_sha256"
        else:
            assert control["model_verification"] == "full_sha256"
            record["model_verification"] = "first_off_sha256_and_unchanged_file_identity"
        assert lab.sha(bench_path) == record["bench_sha256"]
        record["phases_seconds"]["model_identity"] = time.monotonic() - started
        spec = importlib.util.spec_from_file_location("frozen_bench", bench_path)
        bench = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bench)
        requests = []
        for item in protocol["requests"]:
            manifest = ROOT / item["manifest"]
            assert lab.sha(manifest) == item["manifest_sha256"]
            body = next(r for r in load(manifest)["categories"] if r["category"] == item["category"])[item["body_key"]]
            assert bench.digest(bench.json_bytes(body)) == item["body_sha256"]
            assert all(body[key] == value for key, value in protocol["sampling"].items())
            requests.append((item, body))
        record["inputs"] = protocol["requests"]
        builder = json.loads(lab.docker("inspect", matrix.BUILDER).stdout)[0]
        assert builder["Id"] == build["builder_before"]["id"]
        assert builder["Image"] == build["builder_before"]["image"] and not builder["HostConfig"].get("DeviceRequests")
        started = time.monotonic()
        assert lab.docker("exec", matrix.BUILDER, "sha256sum", output["path"]).stdout.split()[0] == exe_sha
        temporary = EXPORT.with_name(EXPORT.name + "." + args.tag + ".new")
        with temporary.open("xb") as destination:
            try:
                copied = subprocess.run(["wsl.exe", "-d", "Ubuntu-24.04", "--exec", "/usr/bin/docker",
                                         "exec", matrix.BUILDER, "cat", output["path"]], stdout=destination,
                                        stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW, timeout=120)
                assert copied.returncode == 0, copied.stderr.decode(errors="replace")[-2000:]
            except BaseException:
                destination.close()
                temporary.unlink()
                raise
        try:
            with temporary.open("rb") as source:
                assert source.read(4) == b"\x7fELF"
            assert lab.sha(temporary) == exe_sha
            temporary.replace(EXPORT)
        finally:
            if temporary.exists():
                temporary.unlink()
        bind = "type=bind,source=" + lab.WSL_ROOT + "/artifacts/experimental/folded16-runtime-serve,target=" + EXE + ",readonly"
        probe = "ninfer-folded16-bind-" + token[:8]
        probe = lab.docker("create", "--name", probe, "--label", LABEL + "=" + token,
                           "--network", "none", "-e", "NVIDIA_VISIBLE_DEVICES=void", "--mount", bind,
                           "--entrypoint", "sha256sum", lab.IMAGE, EXE).stdout.strip()
        binding = owned(probe, token)
        assert not binding["HostConfig"].get("DeviceRequests")
        assert any(m["Destination"] == EXE and m["Type"] == "bind" and not m["RW"] for m in binding["Mounts"])
        assert lab.docker("start", "-a", probe).stdout.split()[0] == exe_sha
        remove_owned(probe, token)
        probe = None
        record["export"] = {"path": str(EXPORT), "native_path": output["path"], "sha256": exe_sha,
                            "elf_magic": True, "fresh_readonly_bind_sha256": exe_sha}
        record["phases_seconds"]["export_and_bind"] = time.monotonic() - started
        record["admission_after_export"] = unreserved_admission(args.mode == "confirm")
        reservation = token
        grant = lab.api("/v1/photo-gpu-reservations", {"owner": lab.OWNER,
                       "reservation_id": reservation, "ttl_seconds": 1800}, "POST", 90)
        assert grant["reservation_id"] == reservation and grant["owner"] == lab.OWNER and grant["state"] == "active"
        record["reservation_id"] = reservation
        record["owned_admission"] = lab.admission(True)
        runtime = protocol["runtime"]
        server = [EXE, "/model/jonathan-full-q4km.ninfer", "--host", "0.0.0.0", "--port", "8080",
                  "--model-id", "qwen3.8-27b", "--max-context", str(runtime["context"]), "--kv-capacity", str(runtime["context"]),
                  "--max-concurrency", str(runtime["concurrency"]), "--max-pending-requests", "32", "--pending-timeout-ms", "600000",
                  "--kv-dtype", runtime["kv"].split()[0], "--spec", "mtp", "--draft-tokens", str(runtime["MTP_tokens"]),
                  "--prefill-chunk", str(runtime["requested_prefill"]), "--no-thinking", "--structured-output", "--vision",
                  "--vision-residency", runtime["vision_residency"], "--host-kv-mib", str(runtime["host_kv_mib"]),
                  "--host-state-slots", str(runtime["host_state_slots"]),
                  "--request-log-jsonl", "/results/" + args.tag + "-events.jsonl"]
        record["command"] = server
        container_name = "ninfer-folded16-runtime-" + token[:8]
        command = ["create", "--name", container_name, "--label", LABEL + "=" + token,
                   "--gpus", "all", "--ipc=host", "-p", "127.0.0.1:18042:8080", "--mount", bind,
                   "--mount", "type=bind,source=" + lab.WSL_ROOT + "/" + protocol["model"]["path"] + ",target=" + server[1] + ",readonly",
                   "--mount", "type=bind,source=" + lab.WSL_ROOT + "/results/q4-folded-ffn-runtime,target=/results",
                   "-e", "CUDA_CACHE_PATH=/tmp/ninfer-folded16", "-e", ENV + "=" + str(int(args.mode == "profile"))]
        if args.mode == "profile":
            assert lab.sha(ROOT / "tools/nsys/nsys") == "ed489f1cdacf354aa6d91201e3f367863a4195153d5aa75a773719235b49315f"
            command += ["--mount", "type=bind,source=" + lab.WSL_ROOT + "/tools/nsys,target=/nsys/target-linux-x64,readonly",
                        "--entrypoint", "/bin/sh", lab.IMAGE, "-c",
                        'ln -s /nsys/target-linux-x64/nsys /tmp/ninfer-nsys; exec /tmp/ninfer-nsys "$@"',
                        "nsys", "profile", "--trace=cuda,nvtx", "--sample=none", "--cpuctxsw=none",
                        "--cuda-graph-trace=node", "--force-overwrite=false", "-o", "/results/" + args.tag + "-nsys", *server]
        else:
            command += ["--entrypoint", EXE, lab.IMAGE, *server[1:]]
        started = time.monotonic()
        ident = container_name
        ident = lab.docker(*command).stdout.strip()
        item = owned(ident, token)
        assert any(m["Destination"] == EXE and not m["RW"] for m in item["Mounts"])
        record["container_id"] = ident
        lab.docker("start", ident)
        for _ in range(240):
            try:
                with urllib.request.urlopen("http://127.0.0.1:18042/health", timeout=2):
                    break
            except Exception:
                assert owned(ident, token)["State"]["Running"]
                time.sleep(1)
        else:
            raise RuntimeError("Lab readiness timed out")
        proof = 'for p in /proc/[0-9]*; do [ "$(readlink "$p/exe" 2>/dev/null)" = ' + EXE + ' ] && sha256sum "$p/exe"; done; true'
        actual = lab.docker("exec", ident, "sh", "-c", proof).stdout.split()
        assert len(actual) == 2 and actual[0] == exe_sha
        record["running_executable_sha256"] = actual[0]
        record["phases_seconds"]["load_and_startup"] = time.monotonic() - started
        warm = copy.deepcopy(next(body for item, body in requests if item["name"] == protocol["warmup"]["source"]))
        warm["max_tokens"] = protocol["warmup"]["max_tokens"]
        warm["messages"][-1]["content"] = protocol["warmup"]["prefix"] + warm["messages"][-1]["content"]
        serving_admission(reservation, attempts=2)
        record["warmup"] = lab.request(warm)
        assert record["warmup"]["usage"]["completion_tokens"] == protocol["warmup"]["max_tokens"]
        assert record["warmup"]["request_sha256"] == bench.digest(bench.json_bytes(warm))
        record["phases_seconds"]["warmup"] = record["warmup"]["wall_seconds_instrumented"]
        events, used, prior = OUT / (args.tag + "-events.jsonl"), set(), 0
        record["samples"] = []
        for item, body in requests:
            if args.mode == "profile" and item["name"] != protocol["untimed_profile_request"]:
                continue
            serving_admission(reservation)
            offset = bench.request_log_offset(events)
            observed, body_sha = bench.stream_request(18042, body)
            serving_admission(reservation)
            assert body_sha == item["body_sha256"]
            done = bench.correlate_request_done(events, offset, observed, "mtp3", prior,
                                               observed["usage"], observed["timings"], used)
            prior = done["request_id"]
            assert observed["http_status"] == 200 and observed["usage"]["prompt_tokens"] == item["prompt_tokens"]
            assert observed["usage"]["completion_tokens"] == protocol["sampling"]["max_tokens"]
            row = {key: observed[key] for key in ("elapsed_seconds", "ttft_content_seconds", "usage",
                   "timings", "finish_reason", "output_sha256")}
            row.update(category=item["category"], name=item["name"], request_body_sha256=body_sha, request_done=done)
            record["samples"].append(row)
            if control:
                reference = next(r for r in control["samples"] if r["category"] == item["category"])
                assert all(row[key] == reference[key] for key in
                           ("output_sha256", "usage", "finish_reason", "request_body_sha256"))
                assert done["speculative"] == reference["request_done"]["speculative"]
            print(item["name"] + (" exact PASS" if control else " reference collected") + (" (instrumented)" if args.mode == "profile" else
                  " " + str(round(observed["elapsed_seconds"], 3)) + " s"), flush=True)
        record["status"] = "CAPTURED" if args.mode == "profile" else "SCREENED"
    except BaseException as exc:
        record.update(status="INCOMPLETE", failure=type(exc).__name__ + ": " + str(exc),
                      failure_traceback=traceback.format_exc())
    finally:
        for container, runtime in ((probe, False), (ident, True)):
            if container:
                try:
                    log = remove_owned(container, token, exe_sha if runtime else None)
                    if runtime:
                        record["owned_container_removed"] = True
                        with (OUT / (args.tag + "-server.log")).open("x", encoding="utf-8") as stream:
                            stream.write(log)
                        if args.mode == "profile" and not record.get("failure"):
                            capture = next(p for p in (OUT / (args.tag + "-nsys.nsys-rep"),
                                           OUT / (args.tag + "-nsys.qdstrm")) if p.is_file())
                            record["capture"] = {"file": capture.name, "sha256": lab.sha(capture)}
                except BaseException as exc:
                    record.setdefault("cleanup_failures", []).append(str(exc))
        if reservation and (ident is None or record.get("owned_container_removed")):
            try:
                released = lab.api("/v1/photo-gpu-reservations/" + reservation, {"owner": lab.OWNER}, "DELETE")
                assert released["reservation_id"] == reservation and released["state"] == "released"
                record["reservation_released"] = True
            except BaseException as exc:
                record.setdefault("cleanup_failures", []).append(str(exc))
        try:
            if claimed:
                record["production_config_unchanged"] = lab.sha(ROOT / "broker/config.json") == config
                if record.get("cleanup_failures") or not record["production_config_unchanged"]:
                    record["status"] = "INCOMPLETE"
                record["finished_at"] = lab.now()
                with receipt.open("x", encoding="utf-8") as stream:
                    json.dump(record, stream, indent=2)
        finally:
            for h in reversed(handles):
                k.ReleaseMutex(h)
                k.CloseHandle(h)
        print(json.dumps({key: record.get(key) for key in
              ("status", "failure", "cleanup_failures", "owned_container_removed", "reservation_released")}), flush=True)
    if record["status"] == "INCOMPLETE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
