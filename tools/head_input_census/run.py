import argparse
import contextlib
import copy
import ctypes
import importlib.util
import json
import os
import pathlib
import re
import shutil
import socket
import struct
import sys
import threading
import time
import traceback
import urllib.request
import uuid


sys.dont_write_bytecode = True
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lab"))
import profile_long_context as lab


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


common = module("head_common", HERE.parent / "q4_t4_chain_probe/run.py")
guard = module("head_guard", HERE.parent / "lab/replay_sse_boundary.py")
prior = module("head_prior", HERE.parent / "mtp_proposal_census/run.py")
checker = module("head_compact_checker", HERE.parent / "mtp_proposal_census/check.py")
proof = module("head_array_proof", HERE / "proof.py")
ROOT = lab.ROOT
OUT = ROOT / "results/head-input-census"
EXPORT = ROOT / "artifacts/experimental/head-input-census-serve"
BASELINE = ROOT / "artifacts/experimental/ninfer-serve"
BASELINE_SHA = "73697224f35d50d113122db42de5877e89a3246f8019530ab92146639f580b02"
MODEL = "/root/ninfer-stage/model/qwen38-jonathan-q4km-proposal-vision-q.ninfer"
MODEL_SHA = "781fcd7da81eb72a123091054ec9b2bd163798a374a127b0f90f68dae3edbf27"
INPUT = ROOT / "results/mtp-proposal-census/protocol-r1.json"
EXE = prior.EXE
KIND = "ninfer_lab_head_input_census"
STAT_KEYS = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
HEADER = struct.Struct("<Q8IQ")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def checked(path, expected):
    require(common.valid_sha(expected) and lab.sha(path) == expected, "Receipt pin changed: " + str(path))
    return load(path)


def save(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def claim(tag):
    require(re.fullmatch(r"[a-z0-9-]+", tag) is not None, "Invalid exclusive tag")
    OUT.mkdir(parents=True, exist_ok=True)
    require(not list(OUT.glob(tag + "-*")), "Tag already has evidence: " + tag)
    path = OUT / (tag + "-claim.json")
    save(path, {"tag": tag, "created_at": lab.now(), "controller_pid": os.getpid()})
    return path


@contextlib.contextmanager
def mutexes():
    kernel = ctypes.WinDLL("kernel32")
    kernel.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p]
    kernel.CreateMutexW.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel.ReleaseMutex.argtypes = [ctypes.c_void_p]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handles = []
    try:
        require(len(common.MUTEXES) == 5, "Laboratory mutex contract changed")
        for name in common.MUTEXES:
            handle = kernel.CreateMutexW(None, False, "Local\\" + name)
            if not handle or kernel.WaitForSingleObject(handle, 0) not in (0, 128):
                if handle:
                    kernel.CloseHandle(handle)
                raise RuntimeError("Laboratory mutex busy: " + name)
            handles.append(handle)
        yield
    finally:
        for handle in reversed(handles):
            kernel.ReleaseMutex(handle)
            kernel.CloseHandle(handle)


def helpers():
    paths = [pathlib.Path(m.__file__) for m in
             (common, guard, lab, prior, prior.matrix, prior.screen, checker, proof)]
    paths += [HERE / "run.py", HERE / "link.py", HERE / "compile.py"]
    return {path.relative_to(ROOT).as_posix(): lab.sha(path) for path in paths}


def model_identity():
    script = ("import json,pathlib,sys; p=pathlib.Path(sys.argv[1]); "
              "assert str(p.resolve(strict=True))==sys.argv[1]; s=p.stat(); "
              "print(json.dumps({k:getattr(s,k) for k in " + repr(STAT_KEYS) + "}))")
    result = lab.run(["wsl.exe", "-d", "Ubuntu-24.04", "-u", "root", "--exec", "python3", "-c", script, MODEL])
    return json.loads(result.stdout)


def serving_gate(token):
    serving = prior.serving_gate(token)
    script = ("$m=Get-CimInstance Win32_PerfRawData_PerfOS_Memory;"
              "@{free_commit=([uint64]$m.CommitLimit-[uint64]$m.CommittedBytes)}|ConvertTo-Json -Compress")
    host = json.loads(lab.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-WindowStyle", "Hidden",
                              "-Command", script]).stdout)
    host["c_free_bytes"] = shutil.disk_usage("C:/").free
    require(host["c_free_bytes"] >= 20 * 2**30, "C: free space below 20 GiB")
    host["free_commit_is_informational_while_model_resident"] = True
    return {"serving": serving, "host": host}


def find_owned(name, token):
    result = lab.docker("inspect", name, check=False)
    if result.returncode:
        require("no such object" in result.stderr.lower() or "no such container" in result.stderr.lower(),
                "Cannot resolve owned container after create")
        return None
    return prior.owned(name, token)["Id"]


def source_pins(native):
    recipe = module("head_source_recipe", HERE / "compile.py")
    require(recipe.source_check(ROOT / "source")[1] == native["source_sha256"],
            "Compiled observer source changed")


def link_receipt(path, digest):
    result = checked(path, digest)
    require(result["status"] == "LINKED_EXPORTED_NOT_RUNTIME_QUALIFIED" and
            result["gpu_run"] is False and result["production_unchanged"] is True,
            "ON link/export is not qualified")
    native = result["native"]
    require(native["status"] == "ON_LINKED_NOT_RUNTIME_QUALIFIED" and
            native["baseline_sha256"] == BASELINE_SHA and native["native_input_count"] == 29 and
            native["output_path"] == "/build/head-input-census/ninfer-serve-on",
            "Unexpected native link contract")
    require(result["fresh_bind"]["image"] == lab.IMAGE and result["fresh_bind"]["gpu_requested"] is False and
            result["fresh_bind"]["readonly"] is True and
            result["fresh_bind"]["sha256"] == native["executable_sha256"], "Fresh export bind is unqualified")
    source_pins(native)
    return result


def freeze(args):
    require(args.model_receipt is not None and args.model_receipt_sha256 is not None,
            "Freeze requires the existing ext4 identity receipt and its SHA")
    linked = link_receipt(args.link, args.link_sha256)
    model = checked(args.model_receipt, args.model_receipt_sha256)
    identity = model["ext4_identity"]
    complete = [event for event in model["events"] if event.get("event") == "complete"]
    require(model["status"] == "PASS" and model["GPU_run"] is False and
            model["production_unchanged"] is True and identity["sha256"] == MODEL_SHA and
            identity["identity_unchanged"] is True and identity["identity"]["st_size"] == 17695430144 and
            len(complete) == 1 and complete[0]["status"] == "PASS" and
            complete[0]["model_identity"] == identity["identity"] and
            complete[0]["model_identity_unchanged"] is True, "Ext4 content identity is unqualified")
    require(lab.sha(prior.INPUT_PROTOCOL) == prior.INPUT_SHA, "Frozen workload source changed")
    frozen = load(prior.INPUT_PROTOCOL)
    require(frozen["model"]["sha256"] == MODEL_SHA and frozen["image"] == lab.IMAGE,
            "Frozen workload model or image differs")
    previous = load(INPUT)
    require(all(previous[key] == frozen[key] for key in
                ("model", "bench", "image", "requests", "runtime", "sampling")), "Workloads diverged")
    require(previous["warmup"] == {"source": "RU4K", "prefix": "MTP census warmup\n", "max_tokens": 32},
            "Qualified warmup profile changed")
    value = {key: copy.deepcopy(frozen[key]) for key in
             ("model", "bench", "image", "requests", "runtime", "sampling")}
    value.update(schema="head-input-census-r1", created_at=lab.now(), created_before_inference=True,
        diagnostic_only=True, performance_claim=False, order=["w", "on"],
        request_deadline_seconds=180,
        linked_checkpoints_per_request=16, warmup=previous["warmup"],
        baseline={"path": BASELINE.relative_to(ROOT).as_posix(), "sha256": BASELINE_SHA},
        native_model={"path": MODEL, "sha256": MODEL_SHA, "identity": identity["identity"],
            "identity_receipt": str(args.model_receipt.resolve()),
            "identity_receipt_sha256": args.model_receipt_sha256,
            "verification": "existing_full_sha256_plus_unchanged_before_after_stat"},
        link_receipt=str(args.link.resolve()), link_receipt_sha256=args.link_sha256,
        source_sha256=linked["native"]["source_sha256"], helper_sha256=helpers(),
        workload_source={"path": str(prior.INPUT_PROTOCOL), "sha256": prior.INPUT_SHA},
        warmup_source={"path": str(INPUT), "sha256": lab.sha(INPUT)},
        acceptance={"epochs": [1, 2, 3, 4], "qualified_request_epochs": [3, 4],
            "exact_output_usage_finish_cache_and_full_MTP": True,
            "native_request_windows": True, "full_array_replay_before_geometry": True,
            "geometry_is_not_a_certificate": True})
    claim(args.tag)
    path = OUT / (args.tag + "-protocol.json")
    save(path, value)
    print(json.dumps({"path": str(path), "sha256": lab.sha(path), "gpu_run": False}))


def adapt_observer_log(log):
    translated, observed = [], []
    for number, line in enumerate(log.splitlines(), 1):
        require("ninfer_lab_mtp_census" not in line, "Unexpected old observer in new executable")
        if KIND not in line:
            translated.append(line)
            continue
        value = json.loads(line)
        require(value.get("kind") in (KIND, KIND + "_declined"), "Malformed head observer kind")
        if value["kind"] == KIND:
            require("payload" in value, "Head observer omitted payload state")
            require((value["completed_linked_records"] > 0) == (value["payload"] is not None),
                    "Payload completeness disagrees with compact record")
            observed.append((number, value))
        else:
            require(value == {"kind": KIND + "_declined", "reason": "unsupported_request"},
                    "Unexpected head decline")
        adapted = {key: item for key, item in value.items() if key != "payload"}
        adapted["kind"] = adapted["kind"].replace(KIND, "ninfer_lab_mtp_census", 1)
        translated.append(json.dumps(adapted, ensure_ascii=False))
    return "\n".join(translated), observed


def qualify_capture(log, arm, warmup, samples, capture, tag):
    adapted, reports = adapt_observer_log(log)
    summaries = prior.qualify(adapted, "off" if arm == "w" else "on", warmup, samples, checker)
    if arm == "w":
        require(not reports and not capture.exists(), "W baseline produced observer artifacts")
        return {"summary": summaries, "payloads": []}, []
    require([value["epoch"] for _, value in reports] == [1, 2, 3, 4], "Head capture epoch sequence")
    payloads, full = [], []
    expected_files = set()
    for line, value in reports:
        payload = value["payload"]
        if value["epoch"] == 1:
            require(payload is None, "Inactive startup must not create a binary")
            continue
        count = value["completed_linked_records"]
        name = "head-input-epoch-" + str(value["epoch"]) + ".bin"
        expected_path = "/results/" + capture.name + "/" + name
        require(payload["path"] == expected_path and payload["schema"] == 1 and
                payload["initial_counts"] == "logical_vocab_before_first_saved_P_acceptance" and
                payload["sample_roles"] == ["Q0", "Q1", "Q2", "P0", "P1", "P2", "P3"] and
                payload["weight_content_identity_verified"] is False, "Unexpected payload provenance")
        path = capture / name
        require(path.is_file() and not path.is_symlink(), "Missing or symlinked payload")
        with path.open("rb") as stream:
            header = HEADER.unpack(stream.read(HEADER.size))
            stream.seek(HEADER.size + 248077 * 4 + 24)
            address = struct.unpack("<Q", stream.read(8))[0]
        size = 248077 * 4 + count * 7 * 502432
        require(header == (0x315355534E454348, 1, value["epoch"], count, 5120,
                           248320, 248077, 5760, 502432, size) and
                path.stat().st_size == payload["bytes"] == size + HEADER.size,
                "Binary header/length/count domain mismatch")
        require(address == payload["head_weight_address"] != 0, "Metadata and binary weight address differ")
        metadata = OUT / (tag + "-epoch-" + str(value["epoch"]) + "-metadata.json")
        save(metadata, value)
        info = {"epoch": value["epoch"], "linked_records": count, "server_log_line": line,
                "path": str(path), "sha256": lab.sha(path), "bytes": path.stat().st_size,
                "metadata": str(metadata), "metadata_sha256": lab.sha(metadata),
                "warmup_only": value["epoch"] == 2}
        payloads.append(info)
        expected_files.add(name)
        if value["epoch"] in (3, 4):
            full.append((value, path, info))
    require({path.name for path in capture.iterdir()} == expected_files, "Extra capture files")
    return {"summary": summaries, "payloads": payloads,
            "kind_adapter": "name_translation_and_payload_removal_for_existing_phase_checker"}, full


def warmup_done(bench, path, offset, observed):
    deadline = time.monotonic() + 2
    while True:
        data = path.read_bytes()
        require(len(data) >= offset, "Warmup native log shrank")
        segment = data[offset:]
        if segment.endswith(b"\n"):
            events = [json.loads(line) for line in segment.splitlines() if line.strip()]
            events = [event for event in events if event.get("event", "").startswith("request_")]
            if any(event["event"] == "request_done" for event in events):
                break
        require(time.monotonic() < deadline, "Warmup native window incomplete")
        time.sleep(0.05)
    require([e["event"] for e in events] == ["request_start", "request_done"], "Warmup native window")
    start, done = events
    request, result, speculative = done["request"], done["result"], done["speculative"]
    require(start["server_instance_id"] == done["server_instance_id"] and
            start["request"]["request_id"] == request["request_id"] == 1 and
            request["requested_output_tokens"] == 32 and request["stream"] is True,
            "Warmup request identity")
    for name, value in (("temperature", 0), ("top_k", 1), ("seed", 42),
                        ("presence_penalty", 1.5), ("frequency_penalty", 0)):
        require(request["sampling"][name] == value, "Warmup sampling mismatch")
    require(result["prompt_tokens"] == observed["usage"]["prompt_tokens"] and
            result["completion_tokens"] == observed["usage"]["completion_tokens"] == 32 and
            result["prefix_cache_hit_tokens"] == observed["usage"]["prompt_tokens_details"]["cached_tokens"] ==
            observed["timings"]["cache_n"] == 0 and result["finish_reason"] == "output_limit" and
            observed["finish_reason"] == "length" and speculative["backend"] == "mtp" and
            speculative["draft_window"] == 3, "Warmup result or MTP contract")
    return {"request_id": 1, "request": request, "result": result, "speculative": speculative,
            "server_instance_id": done["server_instance_id"], "byte_offset_before": offset,
            "byte_offset_after": len(data), "segment_sha256": bench.digest(segment)}


def compare(row, reference):
    for key in ("output", "output_sha256", "usage", "finish_reason", "request_body_sha256"):
        require(row[key] == reference[key], "Fresh W/ON request mismatch: " + key)
    for key in ("request", "result", "speculative"):
        require(row["request_done"][key] == reference["request_done"][key], "Fresh native mismatch: " + key)


def bounded_request(bench, body):
    result = {}

    def receive():
        try:
            result["response"] = bench.stream_request(18042, body)
        except BaseException as error:
            result["error"] = error

    worker = threading.Thread(target=receive, daemon=True)
    worker.start()
    worker.join(180)
    require(not worker.is_alive(), "Request exceeded 180-second deadline; stop owned server")
    if "error" in result:
        raise result["error"]
    return result["response"]


def runtime(args):
    require(args.protocol is not None and args.protocol_sha256 is not None, "Runtime requires frozen protocol")
    token = uuid.uuid4().hex
    lab.OWNER = "ninfer-head-input-" + token
    record = {"schema": "head-input-runtime-r1", "started_at": lab.now(), "status": "INCOMPLETE",
              "arm": args.arm, "tag": args.tag, "control_tag": args.control_tag,
              "diagnostic_only": True, "performance_claim": False,
              "controller_pid": os.getpid(), "owner": lab.OWNER, "samples": []}
    ident, created_name, reservation, claimed, full = None, None, None, False, []
    config, production, model_before = None, None, None
    with mutexes():
        try:
            claim(args.tag)
            claimed = True
            protocol = checked(args.protocol, args.protocol_sha256)
            linked = link_receipt(args.link, args.link_sha256)
            require(protocol["schema"] == "head-input-census-r1" and protocol["created_before_inference"] and
                    protocol["diagnostic_only"] and not protocol["performance_claim"] and
                    protocol["order"] == ["w", "on"] and protocol["linked_checkpoints_per_request"] == 16 and
                    protocol["link_receipt_sha256"] == args.link_sha256 and protocol["helper_sha256"] == helpers() and
                    protocol["source_sha256"] == linked["native"]["source_sha256"], "Frozen protocol differs")
            require(lab.datetime.datetime.fromisoformat(protocol["created_at"]) <=
                    lab.datetime.datetime.fromisoformat(record["started_at"]), "Protocol created after run")
            require(protocol["native_model"]["path"] == MODEL and protocol["native_model"]["sha256"] == MODEL_SHA and
                    protocol["baseline"] == {"path": BASELINE.relative_to(ROOT).as_posix(), "sha256": BASELINE_SHA},
                    "Native model or W identity changed")
            checked(pathlib.Path(protocol["native_model"]["identity_receipt"]),
                    protocol["native_model"]["identity_receipt_sha256"])
            frozen = checked(prior.INPUT_PROTOCOL, prior.INPUT_SHA)
            require(all(protocol[key] == frozen[key] for key in
                        ("model", "bench", "image", "requests", "runtime", "sampling")), "Workload changed")
            require(protocol["image"] == lab.IMAGE and protocol["warmup"] ==
                    {"source": "RU4K", "prefix": "MTP census warmup\n", "max_tokens": 32}, "Runtime or warmup changed")
            record.update(protocol_sha256=args.protocol_sha256, link_receipt_sha256=args.link_sha256,
                helper_sha256=protocol["helper_sha256"], source_sha256=protocol["source_sha256"],
                model_sha256=MODEL_SHA, model_verification=protocol["native_model"]["verification"])
            previous = [load(path) for path in OUT.glob("*-runtime.json")]
            previous = [r for r in previous if r.get("protocol_sha256") == args.protocol_sha256]
            require([r["arm"] for r in previous] == ([] if args.arm == "w" else ["w"]), "Fresh W then ON order")
            reference = None
            if args.arm == "on":
                path = OUT / (args.control_tag + "-runtime.json")
                reference = load(path)
                require(previous == [reference] and reference["status"] == "REFERENCE_COLLECTED" and
                        reference["tag"] == reference["control_tag"] == args.control_tag,
                        "Missing fresh same-protocol W reference")
                record["reference_receipt_sha256"] = lab.sha(path)
                require(importlib.util.find_spec("numpy") is not None, "NumPy missing for post-capture CPU gate")
            else:
                require(args.control_tag == args.tag, "W control tag must name itself")
            config, production = lab.sha(ROOT / "broker/config.json"), common.production(lab)
            record["admission"] = lab.admission()
            initial = lab.api("/v1/status")
            require(initial["model_state"] == "idle" and not initial["instances"] and
                    all(not model["replicas"] and not model["ready_replicas"] for model in initial["models"]),
                    "Existing model must already be unloaded")
            with socket.socket() as sock:
                require(sock.connect_ex(("127.0.0.1", 18042)) != 0, "Lab port18042 busy")
            model_before = model_identity()
            require(model_before == protocol["native_model"]["identity"], "Ext4 model changed since full hash")
            record["model_identity_before"] = model_before
            binary = BASELINE if args.arm == "w" else EXPORT
            binary_sha = BASELINE_SHA if args.arm == "w" else linked["native"]["executable_sha256"]
            require(lab.sha(binary) == binary_sha, "Exported executable changed")
            record.update(executable_sha256=binary_sha, exported_identity=prior.file_identity(binary))
            bench_path = ROOT / protocol["bench"]["path"]
            require(lab.sha(bench_path) == protocol["bench"]["sha256"], "Frozen benchmark changed")
            bench = module("head_frozen_bench", bench_path)
            requests = []
            for item in protocol["requests"]:
                manifest = checked(ROOT / item["manifest"], item["manifest_sha256"])
                body = next(row for row in manifest["categories"] if row["category"] == item["category"])[item["body_key"]]
                require(bench.digest(bench.json_bytes(body)) == item["body_sha256"], "Frozen request bytes changed")
                requests.append((item, body))
            capture = OUT / (args.tag + "-capture")
            if args.arm == "on":
                capture.mkdir(exist_ok=False)
            reservation = token
            grant = lab.api("/v1/photo-gpu-reservations", {"owner": lab.OWNER,
                "reservation_id": token, "ttl_seconds": 1800}, "POST", 90)
            require(grant["reservation_id"] == token and grant["owner"] == lab.OWNER and
                    grant["state"] == "active", "Own reservation was not granted")
            record["reservation"] = grant
            record["owned_admission"] = lab.admission(True)
            server = [EXE, "/model/jonathan-full-q4km.ninfer", "--host", "0.0.0.0", "--port", "8080",
                "--model-id", "qwen3.8-27b", "--max-context", "65536", "--kv-capacity", "65536",
                "--max-concurrency", "1", "--max-pending-requests", "32", "--pending-timeout-ms", "600000",
                "--kv-dtype", "int8", "--spec", "mtp", "--draft-tokens", "3", "--prefill-chunk", "1024",
                "--no-thinking", "--structured-output", "--vision", "--vision-residency", "overlay",
                "--host-kv-mib", "0", "--host-state-slots", "0", "--request-log-jsonl",
                "/results/" + args.tag + "-events.jsonl"]
            created_name = "ninfer-head-input-" + token[:8]
            command = ["create", "--name", created_name, "--label", prior.LABEL + "=" + token,
                "--gpus", "all", "--ipc=host", "--ulimit", "core=0", "-p", "127.0.0.1:18042:8080", "--mount",
                "type=bind,source=" + lab.WSL_ROOT + "/" + binary.relative_to(ROOT).as_posix() + ",target=" + EXE + ",readonly",
                "--mount", "type=bind,source=" + MODEL + ",target=" + server[1] + ",readonly", "--mount",
                "type=bind,source=" + lab.WSL_ROOT + "/results/head-input-census,target=/results",
                "-e", "CUDA_CACHE_PATH=/tmp/ninfer-head-input"]
            if args.arm == "on":
                command += ["-e", "NINFER_HEAD_INPUT_CENSUS_DIR=/results/" + capture.name]
            ident = lab.docker(*command, "--entrypoint", EXE, lab.IMAGE, *server[1:]).stdout.strip()
            record["container_id"] = prior.owned(ident, token)["Id"]
            lab.docker("start", ident)
            for attempt in range(240):
                try:
                    with urllib.request.urlopen("http://127.0.0.1:18042/health", timeout=2):
                        break
                except Exception:
                    require(prior.owned(ident, token)["State"]["Running"], "Owned server exited during startup")
                    if attempt % 10 == 0:
                        serving_gate(token)
                    time.sleep(1)
            else:
                raise RuntimeError("Owned server readiness timed out")
            item = prior.owned(ident, token)
            require(item["State"]["Pid"] > 0 and item["Path"] == EXE and item["Args"] == server[1:], "Actual server command")
            require(any(limit == {"Name": "core", "Hard": 0, "Soft": 0}
                        for limit in item["HostConfig"].get("Ulimits", [])), "Owned core dump limit differs")
            require(any(m["Destination"] == server[1] and m["Source"] == MODEL and not m["RW"] and m["Type"] == "bind"
                        for m in item["Mounts"]), "Actual model bind is not requested read-only ext4 file")
            require(any(m["Destination"] == EXE and not m["RW"] and m["Type"] == "bind" for m in item["Mounts"]),
                    "Executable bind is not read-only")
            require(lab.docker("exec", ident, "sha256sum", "/proc/1/exe").stdout.split() == [binary_sha, "/proc/1/exe"] and
                    lab.docker("exec", ident, "cat", "/proc/1/cmdline").stdout.rstrip("\0").split("\0") == server,
                    "Actual process executable or argv differs")
            bound = list(map(int, lab.docker("exec", ident, "stat", "-Lc", "%d %i %s %Y %Z", server[1]).stdout.split()))
            require(bound == [model_before[k] for k in STAT_KEYS[:3]] +
                    [model_before[k] // 10**9 for k in STAT_KEYS[3:]], "Actual model inode/size/timestamps differ")
            record["process"] = {"host_pid": item["State"]["Pid"], "container_pid": 1,
                "sha256": binary_sha, "command": server, "bound_model_stat_seconds": bound}
            events = OUT / (args.tag + "-events.jsonl")
            warm = copy.deepcopy(requests[0][1])
            warm["max_tokens"] = 32
            warm["messages"][-1]["content"] = protocol["warmup"]["prefix"] + warm["messages"][-1]["content"]
            all_requests = [({"name": "warmup", "body_sha256": bench.digest(bench.json_bytes(warm))}, warm), *requests]
            used, previous_id = set(), 0
            for index, (expected, body) in enumerate(all_requests):
                admission_before = serving_gate(token)
                offset = bench.request_log_offset(events)
                observed, body_sha = bounded_request(bench, body)
                admission_after = serving_gate(token)
                require(observed["http_status"] == 200 and body_sha == expected["body_sha256"], "Request transport/bytes")
                done = (warmup_done(bench, events, offset, observed) if index == 0 else
                    bench.correlate_request_done(events, offset, observed, "mtp3", previous_id,
                        observed["usage"], observed["timings"], used))
                require(done["request_id"] == index + 1, "Native request sequence changed")
                if index:
                    require(done["server_instance_id"] == record["warmup"]["request_done"]["server_instance_id"],
                            "Native server instance changed between requests")
                previous_id = done["request_id"]
                if index:
                    require(observed["usage"]["prompt_tokens"] == expected["prompt_tokens"] and
                            observed["usage"]["completion_tokens"] == 512, "Request token counts changed")
                require(done["request"]["sampling"]["presence_penalty"] == 1.5 and
                        done["request"]["sampling"]["frequency_penalty"] == 0, "Native penalties changed")
                row = {key: observed[key] for key in ("output", "output_sha256", "usage", "finish_reason")}
                row.update(name=expected["name"], request_body_sha256=body_sha, request_done=done,
                           admission_before=admission_before, admission_after=admission_after)
                save(OUT / (args.tag + "-" + expected["name"].lower() + "-response.json"), observed)
                if reference:
                    compare(row, reference["warmup"] if index == 0 else reference["samples"][index - 1])
                if index == 0:
                    record["warmup"] = row
                else:
                    record["samples"].append(row)
                print(expected["name"] + (" exact W/ON PASS" if reference else " fresh W reference collected"), flush=True)
            record["request_log_sha256"] = lab.sha(events)
            record["status"] = "REQUESTS_COMPLETE_CAPTURE_PENDING"
        except BaseException as exc:
            record.update(failure=type(exc).__name__ + ": " + str(exc), failure_traceback=traceback.format_exc())
        finally:
            if created_name and not ident:
                try:
                    ident = find_owned(created_name, token)
                except BaseException as exc:
                    record.setdefault("cleanup_or_capture_failures", []).append(str(exc))
            if ident:
                try:
                    log = prior.remove_owned(ident, token, record.get("executable_sha256"))
                    record["owned_container_removed"] = True
                    path = OUT / (args.tag + "-server.log")
                    with path.open("x", encoding="utf-8") as stream:
                        stream.write(log)
                    record["server_log_sha256"] = lab.sha(path)
                    if record["status"] == "REQUESTS_COMPLETE_CAPTURE_PENDING":
                        record["capture"], full = qualify_capture(log, args.arm, record["warmup"],
                            record["samples"], capture, args.tag)
                        record["status"] = "CAPTURE_BOUND_CPU_PENDING" if args.arm == "on" else "REFERENCE_COLLECTED"
                except BaseException as exc:
                    record.setdefault("cleanup_or_capture_failures", []).append(type(exc).__name__ + ": " + str(exc))
            if reservation and (record.get("owned_container_removed") or
                                (ident is None and not record.get("cleanup_or_capture_failures"))):
                try:
                    released = lab.api("/v1/photo-gpu-reservations/" + reservation, {"owner": lab.OWNER}, "DELETE")
                    require(released["reservation_id"] == reservation and released["state"] == "released", "Own lease release")
                    record["reservation_released"] = True
                except BaseException as exc:
                    record.setdefault("cleanup_or_capture_failures", []).append(str(exc))
            if claimed:
                try:
                    record["production_unchanged"] = (config is not None and
                        lab.sha(ROOT / "broker/config.json") == config and common.production(lab) == production)
                    record["model_identity_after"] = model_identity() if model_before is not None else None
                    record["model_identity_unchanged"] = model_before is not None and record["model_identity_after"] == model_before
                    require(record["production_unchanged"] and record["model_identity_unchanged"], "Production/model identity changed")
                    require(helpers() == record["helper_sha256"] and prior.file_identity(binary) == record["exported_identity"] and
                            lab.sha(binary) == record["executable_sha256"], "Controller or exported binary changed")
                    if record["status"] == "CAPTURE_BOUND_CPU_PENDING" and not record.get("cleanup_or_capture_failures"):
                        record["cpu_admission"] = common.admission(lab, guard)
                        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
                        os.environ.setdefault("OMP_NUM_THREADS", "1")
                        record["cpu_replay"] = []
                        for metadata, binary, info in full:
                            result = proof.analyze(metadata, binary)
                            require(result["status"] == "CAPTURE_NATIVE_REPLAY_AND_GEOMETRY_PASS" and
                                    result["records"] == 16 and result["epoch"] == metadata["epoch"] and
                                    result["binary_sha256"] == info["sha256"] and not result["per_row_certificate"] and
                                    result["geometry_is_nonrigorous_diagnostic"] and not result["native_lowering_proved"] and
                                    not result["performance_claim"],
                                    "CPU replay payload identity or certificate flag")
                            path = OUT / (args.tag + "-epoch-" + str(metadata["epoch"]) + "-cpu.json")
                            save(path, result)
                            record["cpu_replay"].append({"epoch": metadata["epoch"], "path": str(path),
                                "sha256": lab.sha(path), "status": result["status"]})
                        require(len(record["cpu_replay"]) == 2, "Both full capture windows need CPU replay")
                        record["status"] = "DIAGNOSTIC_CAPTURE_AND_CPU_REPLAY_PASS"
                except BaseException as exc:
                    record.setdefault("cleanup_or_capture_failures", []).append(type(exc).__name__ + ": " + str(exc))
                if record.get("cleanup_or_capture_failures") or "failure" in record:
                    record["status"] = "INCOMPLETE"
                record["finished_at"] = lab.now()
                save(OUT / (args.tag + "-runtime.json"), record)
    print(json.dumps({key: record.get(key) for key in ("status", "failure", "cleanup_or_capture_failures", "reservation_released")}))
    require(record["status"] in ("REFERENCE_COLLECTED", "DIAGNOSTIC_CAPTURE_AND_CPU_REPLAY_PASS"), "Diagnostic incomplete")


def main():
    parser = argparse.ArgumentParser(description="Fresh native W then ON diagnostic; no timing or certificate claim.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze", action="store_true")
    mode.add_argument("--arm", choices=("w", "on"))
    parser.add_argument("--tag", required=True)
    parser.add_argument("--control-tag")
    for name in ("link", "protocol", "model-receipt"):
        parser.add_argument("--" + name, type=pathlib.Path, required=name == "link")
        parser.add_argument("--" + name + "-sha256", required=name == "link")
    args = parser.parse_args()
    require(not sys.flags.optimize and os.name == "nt", "Use native Windows Python without -O")
    if args.freeze:
        freeze(args)
    else:
        args.control_tag = args.control_tag or args.tag
        require(re.fullmatch(r"[a-z0-9-]+", args.control_tag) is not None, "Invalid control tag")
        runtime(args)


if __name__ == "__main__":
    main()
