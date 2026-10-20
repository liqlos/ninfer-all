import argparse
import copy
import ctypes
import importlib.util
import json
import math
import pathlib
import re
import shutil
import socket
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
import uuid


sys.dont_write_bytecode = True
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "lab"))
import profile_long_context as lab
import replay_sse_boundary as guard
import run_switch_matrix as screen


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


common = module("model_storage_common", HERE.parent / "q4_t4_chain_probe/run.py")
require = common.require
OUT = ROOT / "results/model-storage"
IMAGE = "sha256:0400a382e1b5ce533cd169379afe3c3b99f73c908b8168f9e554bb03616f0c5b"
EXECUTABLE = ROOT / "artifacts/experimental/ninfer-serve"
EXECUTABLE_SHA = "73697224f35d50d113122db42de5877e89a3246f8019530ab92146639f580b02"
EXE = "/usr/local/bin/ninfer-serve"
MODEL_SHA = "781fcd7da81eb72a123091054ec9b2bd163798a374a127b0f90f68dae3edbf27"
MODEL_BYTES = 17695430144
MODEL_NAME = "qwen38-jonathan-q4km-proposal-vision-q.ninfer"
MODEL_WINDOWS = ROOT / "models/stable" / MODEL_NAME
MODEL_TARGET = "/model/jonathan-full-q4km.ninfer"
SOURCES = {"9p": lab.WSL_ROOT + "/models/stable/" + MODEL_NAME,
           "ext4": "/root/ninfer-stage/model/" + MODEL_NAME}
ORDER = ["9p", "ext4", "ext4", "9p"]
INPUT_PROTOCOL = ROOT / "results/q4-folded-ffn-runtime/protocol-r2.json"
INPUT_SHA = "1ff33e339731bfe6a7ac22f8364ce8e21fbb05e93c243ea0e29f721c5e2a5574"
READ_9P = ROOT / "results/weight-read/screen-r1.json"
READ_EXT4 = ROOT / "results/weight-read/ext4-screen-r1.json"
LABEL = "io.codex.ninfer-owner"
EXPERIMENT = "model-storage"
PORT = 18042
IDENTITY_KEYS = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
HELPERS = [pathlib.Path(lab.__file__), pathlib.Path(guard.__file__), pathlib.Path(screen.__file__),
           HERE.parent / "lab/build_switch_matrix.py", pathlib.Path(common.__file__)]


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    with path.open("x", encoding="utf-8") as destination:
        json.dump(value, destination, ensure_ascii=False, indent=2)
        destination.write("\n")


def file_identity(path):
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    return {"path": str(resolved), **{key: getattr(stat, key) for key in IDENTITY_KEYS}}


def model_identities():
    code = (
        "import json,os,sys; paths=json.loads(sys.argv[1]); "
        "keys=('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns'); "
        "print(json.dumps({arm:{'path':os.path.realpath(path),"
        "'identity':{key:getattr(os.stat(path),key) for key in keys}} "
        "for arm,path in paths.items()}))"
    )
    native = json.loads(lab.run(["wsl.exe", "-d", "Ubuntu-24.04", "-u", "root", "--exec",
                                "/usr/bin/python3", "-B", "-c", code,
                                json.dumps(SOURCES)], timeout=20).stdout)
    for arm, row in native.items():
        require(row["path"] == SOURCES[arm] and row["identity"]["st_size"] == MODEL_BYTES,
                "Model source path/size differs: " + arm)
    return {"windows": file_identity(MODEL_WINDOWS), "native": native}


def inputs():
    require(lab.IMAGE == IMAGE and lab.sha(INPUT_PROTOCOL) == INPUT_SHA, "Input protocol/image changed")
    source = load(INPUT_PROTOCOL)
    require(source["model"]["sha256"] == MODEL_SHA and source["image"] == IMAGE, "Model/image differs")
    require(source["runtime"] == {"context": 65536, "kv": "int8 K8/V8", "concurrency": 1,
                                  "MTP_tokens": 3, "requested_prefill": 1024,
                                  "vision_residency": "overlay", "host_kv_mib": 0,
                                  "host_state_slots": 0}, "Frozen runtime differs")
    require(lab.sha(EXECUTABLE) == EXECUTABLE_SHA, "W executable differs")
    with EXECUTABLE.open("rb") as executable:
        require(executable.read(4) == b"\x7fELF", "Executable is not ELF")
    read9, read4 = load(READ_9P), load(READ_EXT4)
    require(read9["status"] == "CPU_DIRECT_READ_SCREEN_PASS" and
            read9["model_sha256_from_unchanged_native_census"] == MODEL_SHA and
            read9["summary"]["model_identity_unchanged"] is True, "9p qualification differs")
    require(read4["status"] == "PASS" and read4["production_unchanged"] is True and
            read4["ext4_identity"]["sha256"] == MODEL_SHA and
            read4["ext4_identity"]["identity_unchanged"] is True, "Ext4 qualification differs")
    identities = model_identities()
    require(all(identities["windows"][key] == value
                for key, value in read9["model_windows_identity"].items()), "Qualified Windows model changed")
    require(identities["native"]["ext4"]["identity"] == read4["ext4_identity"]["identity"],
            "Qualified ext4 model changed")
    require(all(identities["native"]["9p"]["identity"][key] == value
                for key, value in read9["summary"]["model_identity"].items() if key != "st_dev"),
            "Qualified 9p source changed")
    bench_path = ROOT / source["bench"]["path"]
    require(lab.sha(bench_path) == source["bench"]["sha256"], "Frozen benchmark changed")
    bench = module("model_storage_frozenbench", bench_path)
    requests = []
    for item in source["requests"]:
        manifest = ROOT / item["manifest"]
        require(lab.sha(manifest) == item["manifest_sha256"], "Request manifest changed")
        body = next(row for row in load(manifest)["categories"]
                    if row["category"] == item["category"])[item["body_key"]]
        require(bench.digest(bench.json_bytes(body)) == item["body_sha256"], "Request bytes changed")
        require(all(body[key] == value for key, value in source["sampling"].items()), "Sampling changed")
        requests.append((item, body))
    require([item["name"] for item, body in requests] == ["RU4K", "JSON32K"], "Workload order differs")
    pins = {str(path.relative_to(ROOT)): lab.sha(path)
            for path in [pathlib.Path(__file__), *HELPERS, INPUT_PROTOCOL, READ_9P, READ_EXT4, bench_path,
                         *[ROOT / item["manifest"] for item, body in requests]]}
    return source, bench, requests, identities, pins


def unchanged(protocol):
    require(all(lab.sha(ROOT / path) == expected for path, expected in protocol["source_pins"].items()),
            "Frozen source/input receipt changed")
    require(file_identity(EXECUTABLE) == protocol["executable_identity"] and
            lab.sha(EXECUTABLE) == EXECUTABLE_SHA, "Experimental executable changed")
    actual = model_identities()
    require(actual == protocol["model_identities"], "Qualified model identity changed")
    return actual


def admission(token=None, ident=None):
    if ident is None:
        result = common.admission(lab, guard, token)
    else:
        result = screen.serving_admission(token)
        state = lab.api("/v1/status")
        require(state["windows_gpu"]["available"] and
                state["windows_gpu"]["external_reserve_fraction"] == 0.3 and
                not state["admission"]["runtime_maintenance"]["active"], "Serving reserve/maintenance guard")
        capacity = state["capacity_estimate"]
        require(len(capacity) == 1 and capacity[0]["measurement"] == "measured" and
                capacity[0]["sample_age_seconds"] <= 15 and capacity[0]["external_reserve_satisfied"],
                "Stale or unsatisfied serving GPU reserve")
        for current in lab.docker("ps", "-q").stdout.split():
            item = guard.inspect(lab, current)
            if item["HostConfig"].get("DeviceRequests") or item["HostConfig"].get("Devices"):
                require(item["Id"] == ident and item["Image"] == IMAGE and
                        (item["Config"].get("Labels") or {}).get(LABEL) == token,
                        "Foreign GPU container")
        script = ("$m=Get-CimInstance Win32_PerfRawData_PerfOS_Memory;"
                  "@{free_commit=([uint64]$m.CommitLimit-[uint64]$m.CommittedBytes);"
                  "commit_limit=[uint64]$m.CommitLimit;committed_bytes=[uint64]$m.CommittedBytes;"
                  "available_bytes=[uint64]$m.AvailableBytes}|ConvertTo-Json -Compress")
        host = json.loads(lab.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-WindowStyle",
                                   "Hidden", "-Command", script]).stdout)
        host["c_free_bytes"] = shutil.disk_usage("C:/").free
        result.update(host=host, builder=guard.builder_idle(lab), capacity=capacity[0])
    require(ident is not None or result["host"]["free_commit"] >= 40 * 2**30,
            "Prelaunch full-model host budget failed")
    require(result["host"]["c_free_bytes"] >= 20 * 2**30, "C: budget failed")
    return result


def owned(target, name, token, ident=None):
    value = guard.inspect(lab, target)
    labels = value["Config"].get("Labels") or {}
    require(value["Name"] == "/" + name and value["Image"] == IMAGE and
            labels.get(LABEL) == token and labels.get("io.codex.ninfer-experiment") == EXPERIMENT and
            (ident is None or value["Id"] == ident), "Container ownership mismatch")
    return value


def remove_owned(name, token, ident=None):
    found = lab.docker("inspect", ident or name, check=False)
    if found.returncode:
        require("no such object" in found.stderr.lower() or "no such container" in found.stderr.lower(),
                "Cannot establish cleanup ownership")
        return ""
    item = owned(ident or name, name, token, ident)
    if item["State"]["Running"]:
        lab.docker("stop", "--time", "45", item["Id"], timeout=60)
    log = lab.docker("logs", item["Id"], check=False)
    lab.docker("rm", item["Id"])
    return log.stdout + log.stderr


def executable_bind():
    return "type=bind,source=" + lab.WSL_ROOT + "/artifacts/experimental/ninfer-serve,target=" + EXE + ",readonly"


def fresh_bind(token):
    name, ident = "ninfer-model-storage-bind-" + token[:12], None
    try:
        ident = lab.docker("create", "--name", name, "--label", LABEL + "=" + token,
                           "--label", "io.codex.ninfer-experiment=" + EXPERIMENT,
                           "--runtime", "runc", "--network", "none", "--ulimit", "core=0",
                           "-e", "NVIDIA_VISIBLE_DEVICES=void",
                           "--mount", executable_bind(), "--entrypoint", "sha256sum", IMAGE, EXE).stdout.strip()
        item = owned(ident, name, token, ident)
        require(not item["HostConfig"].get("DeviceRequests") and not item["HostConfig"].get("Devices"),
                "Executable bind proof is not CPU-only")
        mounts = [row for row in item["Mounts"] if row["Destination"] == EXE]
        require(len(mounts) == 1 and mounts[0]["Type"] == "bind" and not mounts[0]["RW"] and
                mounts[0]["Source"] == lab.WSL_ROOT + "/artifacts/experimental/ninfer-serve", "Executable bind differs")
        digest = lab.docker("start", "-a", ident).stdout.split()
        require(len(digest) == 2 and digest[0] == EXECUTABLE_SHA, "Fresh executable bind SHA differs")
        require(owned(ident, name, token, ident)["State"]["ExitCode"] == 0, "Bind proof failed")
        return {"container_id": ident, "sha256": digest[0], "mount": mounts[0], "GPU_run": False}
    finally:
        remove_owned(name, token, ident)


def native_start(data):
    rows = [json.loads(line) for line in data.splitlines() if line.strip()]
    starts = [row for row in rows if row.get("event") == "server_start"]
    require(len(starts) == 1, "Need one complete server_start")
    row = starts[0]
    artifact, engine = row["artifact"], row["engine"]
    expected = {"max_context": 65536, "max_concurrency": 1, "kv_capacity": 65536,
                "kv_cache": "int8-group64", "speculative_backend": "mtp", "speculative_draft_window": 3,
                "mtp_policy": "fixed", "proposal_head": "full", "prefill_chunk": 1024,
                "vision": True, "cuda_graph": True}
    require(all(engine.get(key) == value for key, value in expected.items()), "Native Engine settings differ")
    require(engine["context_cache"]["host_state_slots"] == 0 and
            engine["context_cache"]["host_kv_capacity_bytes"] == 0, "Host state/KV differs")
    require(artifact["path"] == MODEL_TARGET and artifact["size_bytes"] == MODEL_BYTES and
            artifact["pinned_weight_bytes"] > 0 and artifact["overlay_window_bytes"] > 0,
            "Native artifact/overlay differs")
    require(all(type(artifact[key]) in (int, float) and math.isfinite(artifact[key]) and artifact[key] > 0
                for key in ("load_seconds", "upload_seconds")), "Invalid native load measurements")
    return row


def warm_done(data, observed, warm):
    rows = [json.loads(line) for line in data.splitlines() if line.strip()]
    events = [row for row in rows if row.get("event", "").startswith("request_")]
    require([row["event"] for row in events] == ["request_start", "request_done"], "Warmup event window differs")
    start, done = events
    request, result, spec = done["request"], done["result"], done["speculative"]
    require(start["server_instance_id"] == done["server_instance_id"] and
            start["request"]["request_id"] == request["request_id"], "Warmup request identity differs")
    require(request["requested_output_tokens"] == 32 and request["stream"] is True and
            all(request["sampling"][key] == warm[key] for key in ("temperature", "top_k", "seed")),
            "Warmup request settings differ")
    require(result["completion_tokens"] == observed["usage"]["completion_tokens"] == 32 and
            result["prompt_tokens"] == observed["usage"]["prompt_tokens"] and
            result["prefix_cache_hit_tokens"] == 0 and
            observed["usage"]["prompt_tokens_details"]["cached_tokens"] == 0 and
            observed["timings"]["cache_n"] == 0, "Warmup usage/cache differs")
    require(observed["finish_reason"] == "length" and result["finish_reason"] == "output_limit" and
            spec["backend"] == "mtp" and spec["draft_window"] == 3, "Warmup finish/MTP differs")
    return done


def equal_results(actual, reference):
    for key in ("output_sha256", "usage", "finish_reason", "request_body_sha256"):
        require(actual[key] == reference[key], "Result differs: " + key)
    require(actual["speculative"] == reference["speculative"], "Full MTP workload differs")


def sample(observed, body_sha, speculative):
    require(observed["http_status"] == 200 and observed["saw_done"], "Request did not complete successfully")
    return {**{key: observed[key] for key in ("elapsed_seconds", "ttft_content_seconds", "usage", "timings",
                                             "finish_reason", "output_sha256", "output", "response_id")},
            "request_body_sha256": body_sha, "speculative": speculative}


def bounded_request(bench, body):
    result = {}

    def receive():
        try:
            result["response"] = bench.stream_request(PORT, body)
        except BaseException as error:
            result["error"] = error

    worker = threading.Thread(target=receive, daemon=True)
    worker.start()
    worker.join(180)
    require(not worker.is_alive(), "Request exceeded fixed 180-second deadline; stop owned server")
    if "error" in result:
        raise result["error"]
    return result["response"]


def launch(index, arm, directory, protocol, protocol_sha, source, bench, requests, reference, production_before):
    token = uuid.uuid4().hex
    name, ident = "ninfer-model-storage-" + token[:12], None
    reservation, created = False, False
    stem = f"{index + 1:02d}-{arm}"
    events = directory / (stem + "-events.jsonl")
    record = {"schema": "model-storage-launch-v1", "started_at": lab.now(), "status": "INCOMPLETE",
              "index": index, "arm": arm, "model_source": SOURCES[arm], "owner": lab.OWNER,
              "reservation_id": token, "protocol_sha256": protocol_sha}
    try:
        require(lab.sha(directory / "protocol.json") == protocol_sha, "Immutable protocol changed")
        record["model_identity_before"] = unchanged(protocol)
        record["admission_before_bind"] = admission()
        record["fresh_readonly_executable_bind"] = fresh_bind(token)
        record["admission_before_reservation"] = admission()
        reservation = True
        grant = lab.api("/v1/photo-gpu-reservations", {"owner": lab.OWNER, "reservation_id": token,
                        "ttl_seconds": 1800}, "POST", 90)
        require(grant["reservation_id"] == token and grant["owner"] == lab.OWNER and grant["state"] == "active",
                "Reservation grant differs")
        record["grant"] = grant
        record["owned_admission"] = admission(token)
        require(model_identities() == record["model_identity_before"], "Model changed before launch")
        with socket.socket() as sock:
            require(sock.connect_ex(("127.0.0.1", PORT)) != 0, "Lab port is already listening")
        server = [EXE, MODEL_TARGET, "--host", "0.0.0.0", "--port", "8080", "--model-id", "qwen3.8-27b",
                  "--max-context", "65536", "--kv-capacity", "65536", "--max-concurrency", "1",
                  "--max-pending-requests", "32", "--pending-timeout-ms", "600000", "--kv-dtype", "int8",
                  "--spec", "mtp", "--draft-tokens", "3", "--prefill-chunk", "1024", "--no-thinking",
                  "--structured-output", "--vision", "--vision-residency", "overlay", "--host-kv-mib", "0",
                  "--host-state-slots", "0", "--request-log-jsonl", "/results/" + events.name]
        command = ["create", "--name", name, "--label", LABEL + "=" + token,
                   "--label", "io.codex.ninfer-experiment=" + EXPERIMENT, "--gpus", "all", "--ipc=host",
                   "--ulimit", "core=0",
                   "-p", f"127.0.0.1:{PORT}:8080", "--mount", executable_bind(), "--mount",
                   "type=bind,source=" + SOURCES[arm] + ",target=" + MODEL_TARGET + ",readonly", "--mount",
                   "type=bind,source=" + lab.WSL_ROOT + "/" + directory.relative_to(ROOT).as_posix() + ",target=/results",
                   "-e", "CUDA_CACHE_PATH=/tmp/ninfer-model-storage", "--entrypoint", EXE, IMAGE, *server[1:]]
        record["command"] = server
        created = True
        ident = lab.docker(*command).stdout.strip()
        item = owned(ident, name, token, ident)
        model_mount = [row for row in item["Mounts"] if row["Destination"] == MODEL_TARGET]
        require(len(model_mount) == 1 and model_mount[0]["Type"] == "bind" and
                model_mount[0]["Source"] == SOURCES[arm] and not model_mount[0]["RW"], "Model bind differs")
        require(any(row["Destination"] == EXE and not row["RW"] for row in item["Mounts"]), "Executable bind is writable")
        record.update(container_id=ident, model_mount=model_mount[0])
        clocks = guard.clocks()
        start = time.monotonic_ns()
        lab.docker("start", ident)
        polls, next_check = 0, time.monotonic()
        while time.monotonic_ns() - start < 300 * 10**9:
            polls += 1
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1) as response:
                    health = json.load(response)
                    require(response.status == 200 and health == {"status": "ok"}, "Health did not report ready")
                break
            except (urllib.error.URLError, TimeoutError):
                if time.monotonic() >= next_check:
                    require(owned(ident, name, token, ident)["State"]["Running"], "Server exited before readiness")
                    next_check = time.monotonic() + 5
                time.sleep(0.1)
        else:
            raise RuntimeError("Startup exceeded fixed 300-second deadline")
        record["startup_seconds_windows"] = (time.monotonic_ns() - start) / 1e9
        record["startup_clock_durations_seconds"] = guard.durations(clocks, guard.clocks())
        record["health_poll_count"] = polls
        require(lab.docker("exec", ident, "readlink", "/proc/1/exe").stdout.strip() == EXE, "Runtime entrypoint differs")
        running = lab.docker("exec", ident, "sha256sum", "/proc/1/exe").stdout.split()
        require(len(running) == 2 and running[0] == EXECUTABLE_SHA, "Actual running executable differs")
        record["running_executable_sha256"] = running[0]
        record["server_start"] = native_start(events.read_bytes())
        if reference:
            first = reference["server_start"]
            actual_artifact = {key: value for key, value in record["server_start"]["artifact"].items()
                               if key not in ("load_seconds", "upload_seconds")}
            old_artifact = {key: value for key, value in first["artifact"].items()
                            if key not in ("load_seconds", "upload_seconds")}
            require(actual_artifact == old_artifact and record["server_start"]["engine"] == first["engine"],
                    "Native artifact bytes/Engine settings changed between sources")
        record["serving_admission"] = admission(token, ident)
        print(json.dumps({"phase": "ready", "launch": index + 1, "arm": arm,
                          "startup_seconds_windows": record["startup_seconds_windows"],
                          "resident_free_commit_bytes": record["serving_admission"]["host"]["free_commit"]}),
              flush=True)
        warm = copy.deepcopy(next(body for item, body in requests if item["name"] == source["warmup"]["source"]))
        warm["max_tokens"] = source["warmup"]["max_tokens"]
        warm["messages"][-1]["content"] = source["warmup"]["prefix"] + warm["messages"][-1]["content"]
        offset = bench.request_log_offset(events)
        observed, body_sha = bounded_request(bench, warm)
        require(body_sha == bench.digest(bench.json_bytes(warm)), "Warmup request bytes differ")
        warm_native = warm_done(events.read_bytes()[offset:], observed, warm)
        require(warm_native["server_instance_id"] == record["server_start"]["server_instance_id"],
                "Warmup server identity differs")
        record["warmup"] = sample(observed, body_sha, warm_native["speculative"])
        record["warmup"]["request_done"] = warm_native
        if reference:
            equal_results(record["warmup"], reference["warmup"])
        used, prior = {warm_native["request"]["request_id"]}, warm_native["request"]["request_id"]
        record["samples"] = []
        for item, body in requests:
            admission(token, ident)
            offset = bench.request_log_offset(events)
            observed, body_sha = bounded_request(bench, body)
            admission(token, ident)
            require(body_sha == item["body_sha256"] and observed["usage"]["prompt_tokens"] == item["prompt_tokens"] and
                    observed["usage"]["completion_tokens"] == 512, "Frozen workload bytes/usage differ")
            done = bench.correlate_request_done(events, offset, observed, "mtp3", prior,
                                                observed["usage"], observed["timings"], used)
            require(done["server_instance_id"] == record["server_start"]["server_instance_id"], "Request server identity differs")
            prior = done["request_id"]
            row = sample(observed, body_sha, done["speculative"])
            row.update(name=item["name"], request_done=done)
            record["samples"].append(row)
            if reference:
                equal_results(row, next(value for value in reference["samples"] if value["name"] == item["name"]))
        record["status"] = "PASS"
    except BaseException as error:
        record.update(failure=type(error).__name__ + ": " + str(error), traceback=traceback.format_exc())
    finally:
        if created:
            try:
                log = remove_owned(name, token, ident)
                record["owned_container_removed"] = True
                with (directory / (stem + "-server.log")).open("x", encoding="utf-8") as destination:
                    destination.write(log)
            except BaseException as error:
                record.setdefault("cleanup_failures", []).append(str(error))
        if reservation and (not created or record.get("owned_container_removed")):
            try:
                released = lab.api("/v1/photo-gpu-reservations/" + token, {"owner": lab.OWNER}, "DELETE")
                require(released["reservation_id"] == token and released["state"] == "released", "Reservation not released")
                record["reservation_released"] = True
            except BaseException as error:
                record.setdefault("cleanup_failures", []).append(str(error))
        try:
            require(lab.sha(directory / "protocol.json") == protocol_sha, "Immutable protocol changed")
            record["model_identity_after"] = unchanged(protocol)
            record["production_unchanged"] = (lab.sha(ROOT / "broker/config.json") == protocol["production_config_sha256"] and
                                              common.production(lab) == production_before)
            require(record["production_unchanged"], "Production changed")
            record["builder_after"] = guard.builder_idle(lab)
            if events.exists():
                record["events"] = {"file": events.name, "bytes": events.stat().st_size, "sha256": lab.sha(events)}
        except BaseException as error:
            record.setdefault("cleanup_failures", []).append(str(error))
        if record.get("failure") or record.get("cleanup_failures"):
            record["status"] = "INCOMPLETE"
        record["finished_at"] = lab.now()
        write(directory / (stem + "-receipt.json"), record)
    return record


def summary(rows):
    require(len(rows) == 4 and [row["arm"] for row in rows] == ORDER and
            all(row["status"] == "PASS" and row.get("reservation_released") and
                row.get("owned_container_removed") for row in rows), "Incomplete fixed ABBA screen")
    values = {arm: [row["startup_seconds_windows"] for row in rows if row["arm"] == arm] for arm in SOURCES}
    require(all(math.isfinite(value) and value > 0 for times in values.values() for value in times), "Invalid startup time")
    gains = [100 * (1 - rows[b]["startup_seconds_windows"] / rows[a]["startup_seconds_windows"])
             for a, b in ((0, 1), (3, 2))]
    gain = 100 * (1 - sum(values["ext4"]) / sum(values["9p"]))
    return {"status": "PASS_FULL_LOAD_PATH_AND_REQUEST_EQUIVALENCE", "order": ORDER,
            "startup_seconds_windows": values, "startup_ratio_of_sums_time_reduction_percent": gain,
            "adjacent_pair_time_reduction_percent": gains,
            "usefulness_gate_pass": gain >= 10 and all(value > 0 for value in gains),
            "native_load_seconds_separate_clock": {arm: [row["server_start"]["artifact"]["load_seconds"]
                                                          for row in rows if row["arm"] == arm] for arm in SOURCES},
            "native_upload_seconds_separate_clock": {arm: [row["server_start"]["artifact"]["upload_seconds"]
                                                            for row in rows if row["arm"] == arm] for arm in SOURCES},
            "statistical_confirmation": False, "warm_request_speedup_claim": False,
            "host_storage_cache_controlled": False, "new_model_copy": False,
            "scope": "Fresh server startup to health-ready; OS/host cache state is not controlled."}


def main():
    parser = argparse.ArgumentParser(description="Fixed lab-only same-W 9p/ext4 full-load ABBA screen.")
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    require(sys.platform == "win32" and not sys.flags.optimize, "Use Windows Python without -O")
    require(re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", args.tag), "Invalid result tag")
    lab.OWNER = "ninfer-model-storage-" + args.tag
    directory = OUT / args.tag
    handles, claimed, rows = [], False, []
    record = {"schema": "model-storage-series-v1", "started_at": lab.now(), "status": "INCOMPLETE"}
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p]
    kernel.CreateMutexW.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel.ReleaseMutex.argtypes = [ctypes.c_void_p]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    try:
        require(len(guard.MUTEXES) == 5 and guard.MUTEXES == screen.matrix.MUTEXES, "Mutex coverage changed")
        for name in guard.MUTEXES:
            handle = kernel.CreateMutexW(None, False, "Local\\" + name)
            if not handle or kernel.WaitForSingleObject(handle, 0) not in (0, 128):
                if handle:
                    kernel.CloseHandle(handle)
                raise RuntimeError("Laboratory mutex busy: " + name)
            handles.append(handle)
        OUT.mkdir(parents=True, exist_ok=True)
        directory.mkdir()
        claimed = True
        before = common.production(lab)
        state = lab.api("/v1/status")
        require(before is None and state["model_state"] == "idle" and not state["instances"] and
                all(not row["replicas"] and not row["ready_replicas"] for row in state["models"]),
                "Production must already be unloaded; this runner does not unload it")
        record["initial_admission"] = admission()
        source, bench, requests, identities, pins = inputs()
        protocol = {"schema": "model-storage-protocol-v1", "created_before_inference": True,
                    "created_at": lab.now(), "order": ORDER, "executable_sha256": EXECUTABLE_SHA, "image": IMAGE,
                    "executable_identity": file_identity(EXECUTABLE), "model_sha256": MODEL_SHA,
                    "model_identities": identities, "source_pins": pins,
                    "production_config_sha256": lab.sha(ROOT / "broker/config.json"),
                    "requests": source["requests"], "sampling": source["sampling"], "warmup": source["warmup"],
                    "runtime": source["runtime"], "mutexes": guard.MUTEXES,
                    "startup_clock": "Windows time.monotonic_ns (QPC), separate from native Linux steady_clock",
                    "startup_boundary": "Immediately before docker start until first successful health response; create/bind/proof outside.",
                    "health_poll_seconds": 0.1, "startup_deadline_seconds": 300,
                    "request_deadline_seconds": 180,
                    "request_timeout": "Unmodified frozenbench in daemon thread; 180-second join deadline, then owned-server cleanup",
                    "memory_guards_gib": {"before_launch_free_commit": 40, "before_cpu_preparation_free_commit": 24,
                                          "resident_model_free_commit": "informational snapshot; original serving admission",
                                          "c_drive_free_always": 20},
                    "external_reserve_fraction_always": 0.3, "owned_reservation_ttl_seconds": 1800,
                    "lab_container_core_ulimit": 0,
                    "usefulness_gate": {"ratio_of_sums_startup_time_reduction_percent": 10,
                                        "both_adjacent_pairs_positive": True},
                    "statistical_confirmation": False, "host_storage_cache_controlled": False,
                    "timing_profiler_allowed": False, "warm_request_speedup_claim": False,
                    "no_optional_extension": True, "no_outlier_deletion": True, "new_model_copy": False}
        write(directory / "protocol.json", protocol)
        record["protocol_sha256"] = lab.sha(directory / "protocol.json")
        for index, arm in enumerate(ORDER):
            row = launch(index, arm, directory, protocol, record["protocol_sha256"], source, bench,
                         requests, rows[0] if rows else None, before)
            rows.append(row)
            record["completed_launches"] = len(rows)
            require(row["status"] == "PASS", "Launch failed; preserve receipts without extending the series")
            print(json.dumps({"launch": index + 1, "arm": arm, "status": row["status"],
                              "startup_seconds_windows": row["startup_seconds_windows"]}), flush=True)
        record.update(summary(rows))
        record["receipts"] = {f"{index + 1:02d}-{arm}-receipt.json": lab.sha(directory / f"{index + 1:02d}-{arm}-receipt.json")
                              for index, arm in enumerate(ORDER)}
    except BaseException as error:
        record.update(status="INCOMPLETE", failure=type(error).__name__ + ": " + str(error), traceback=traceback.format_exc())
    finally:
        try:
            if claimed:
                record["finished_at"] = lab.now()
                write(directory / "summary.json", record)
        finally:
            for handle in reversed(handles):
                kernel.ReleaseMutex(handle)
                kernel.CloseHandle(handle)
    print(json.dumps({key: record.get(key) for key in
                      ("status", "failure", "completed_launches", "usefulness_gate_pass")}), flush=True)
    if record["status"] == "INCOMPLETE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
