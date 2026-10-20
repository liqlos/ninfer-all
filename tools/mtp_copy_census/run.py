import argparse
import copy
import ctypes
import importlib.util
import json
import os
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
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE.parent / "lab"))
import build_switch_matrix as matrix
import profile_long_context as lab
import run_switch_matrix as screen

ROOT = lab.ROOT
OUT = ROOT / "results/mtp-copy-census"
EXPORT = ROOT / "artifacts/experimental/mtp-copy-census-serve"
EXE = "/usr/local/bin/ninfer-serve"
LABEL = "io.codex.ninfer-owner"
INPUT_PROTOCOL = ROOT / "results/q4-folded-ffn-runtime/protocol-r2.json"
INPUT_SHA = "1ff33e339731bfe6a7ac22f8364ce8e21fbb05e93c243ea0e29f721c5e2a5574"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def file_identity(path):
    path = path.resolve(strict=True)
    stat = path.stat()
    return {"path": str(path), **{key: getattr(stat, key) for key in
            ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")}}


def owned(ident, token):
    item = json.loads(lab.docker("inspect", ident).stdout)[0]
    assert (item["Id"] == ident or item["Name"] == "/" + ident) and item["Image"] == lab.IMAGE
    assert item["Config"]["Labels"][LABEL] == token
    return item


def remove_owned(ident, token, executable_sha=None):
    item = owned(ident, token)
    if item["State"]["Running"]:
        if executable_sha:
            assert lab.docker("exec", ident, "sha256sum", "/proc/1/exe").stdout.split()[0] == executable_sha
        lab.docker("stop", "--time", "45", ident)
    logs = lab.docker("logs", ident, check=False)
    lab.docker("rm", ident)
    return logs.stdout + logs.stderr


def serving_gate(reservation):
    result = screen.serving_admission(reservation)
    state = lab.api("/v1/status")
    grant = state["admission"]["photo_gpu_reservation"]
    assert grant["active"] and grant["owner"] == lab.OWNER
    assert state["windows_gpu"]["external_reserve_fraction"] == 0.3
    assert not state["admission"]["runtime_maintenance"]["active"]
    for ident in lab.docker("ps", "-q").stdout.split():
        item = json.loads(lab.docker("inspect", ident).stdout)[0]
        if item["HostConfig"].get("DeviceRequests"):
            assert item["Config"]["Labels"].get(LABEL) == reservation and item["Image"] == lab.IMAGE
    return result


def qualify(log, arm, warmup, samples, checker, tag):
    lines = log.splitlines()
    prefix, marker = "MTP_COPY_CENSUS ", "MTP_COPY_CENSUS_BIND "
    observations = [(i, json.loads(line[len(prefix):])) for i, line in enumerate(lines) if line.startswith(prefix)]
    bindings = [(i, json.loads(line[len(marker):])) for i, line in enumerate(lines) if line.startswith(marker)]
    assert sum("MTP_COPY_CENSUS" in line for line in lines) == len(observations) + len(bindings), "Malformed observer output"
    if arm == "off":
        assert not observations and not bindings, "OFF unexpectedly emitted observer records"
        return []
    assert arm == "on" and len(observations) == len(bindings) == 4, "Need exactly four native wait epochs"
    positions, reports = zip(*observations)
    marker_positions, marker_records = zip(*bindings)
    assert [r["epoch"] for r in reports] == [1, 2, 3, 4], "Missing, extra, duplicate or reordered epochs"
    for expected, report, bind in zip(range(1, 5), reports, marker_records):
        assert report["schema"] == 1 and report["kind"] == "request" and report["diagnostic_only"] is True
        assert report["error"] == [] and report["binding"]["wait_id"] == expected
        assert report["binding"]["captured_on_program_thread"] is True and report["binding"]["sequence_pointer_cleared"] is True
        assert bind == {"epoch": expected, "wait_id": expected, "active": report["active"]}
    startup = reports[0]
    assert startup["active"] is False and startup["first_ledger"] == [] and startup["rows"] == []
    assert len(startup["result"]["generated_token_ids"]) == 4 and startup["result"]["finish_reason"] == "output_limit"
    assert startup["result"]["prompt_tokens"] > 0
    assert [s["name"] for s in samples] == ["RU4K", "JSON32K"]
    assert [s["request_done"]["request_id"] for s in samples] == [2, 3]
    request_lines = []
    for i, line in enumerate(lines):
        if "req#" in line:
            match = re.search(r"\bINFO\s+req#([0-9]+) (started|done) \|", line)
            assert match, "Unexpected request lifecycle line"
            request_lines.append((int(match[1]), match[2], i))
    assert [(i, phase) for i, phase, _ in request_lines] == [
        (1, "started"), (1, "done"), (2, "started"), (2, "done"), (3, "started"), (3, "done")]
    s1, d1, s2, d2, s3, d3 = [i for _, _, i in request_lines]

    def unique(pattern):
        matches = [i for i, line in enumerate(lines) if re.search(pattern, line)]
        assert len(matches) == 1, "Missing or duplicate startup/stop boundary"
        return matches[0]

    engine = unique(r"\bINFO\s+engine ready \|")
    startup_done = unique(r"\bINFO\s+warmup complete \|")
    listening = unique(r"\bINFO\s+listening on http://0\.0\.0\.0:8080 \| model qwen3\.8-27b \|")
    stopped = unique(r"\bINFO\s+server stopped$")
    b0, b1, b2, b3 = marker_positions
    assert engine < b0 < startup_done < listening < s1 < b1 < d1 < s2 < b2 < d2 < s3 < b3 < d3 < stopped, "Native wait marker outside its request phase"
    validated = []
    for report, sample, count, start, done in zip(reports[1:], [warmup, *samples], (32, 512, 512), (s1, s2, s3), (d1, d2, d3)):
        assert report["active"] is True and sample["usage"]["completion_tokens"] == count
        native = report["result"]
        assert native["prompt_tokens"] == sample["usage"]["prompt_tokens"] and len(native["generated_token_ids"]) == count
        assert native["finish_reason"] == "output_limit" and native["has_media"] is False
        assert f"| max output {count} | thinking off" in lines[start]
        assert f"| prompt {sample['usage']['prompt_tokens']:,} | output {count} | cache 0 (" in lines[done]
        validated.append(checker.validate_capture(report))
    result = []
    for report, data, sample, position, bound, start, done in zip(
            reports[2:], validated[1:], samples, positions[2:], (b2, b3), (s2, s3), (d2, d3)):
        event_spec = sample["request_done"]["speculative"]
        assert all(event_spec[key] == report["result"]["speculative"][key] for key in
                   ("rounds", "drafted_tokens", "accepted_tokens", "fallback_steps", "accepted_per_position"))
        path = OUT / (tag + "-" + sample["name"].lower() + "-native.json")
        with path.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, separators=(",", ":"))
            stream.write("\n")
        summary = {key: value for key, value in data.items() if key not in ("rows", "generated")}
        summary.update(name=sample["name"], request_id=sample["request_done"]["request_id"],
                       request_body_sha256=sample["request_body_sha256"], capture_path=str(path), capture_sha256=lab.sha(path),
                       stdout_record_line=position + 1, stderr_binding_line=bound + 1,
                       request_start_line=start + 1, request_done_line=done + 1,
                       startup_binding_line=b0 + 1, binding_uses_stderr_order=True)
        result.append(summary)
    return result


def main():
    parser = argparse.ArgumentParser(description="One serialized diagnostic census launch; fresh OFF then ON.")
    parser.add_argument("arm", choices=("off", "on"))
    parser.add_argument("tag")
    parser.add_argument("control_tag")
    for name in ("protocol", "build"):
        parser.add_argument("--" + name, type=pathlib.Path, required=True)
        parser.add_argument("--" + name + "-sha256", required=True)
    args = parser.parse_args()
    assert not sys.flags.optimize, "Run without Python -O"
    assert all(re.fullmatch(r"[a-z0-9-]+", x) for x in (args.tag, args.control_tag))
    assert all(re.fullmatch(r"[0-9a-f]{64}", x) for x in (args.protocol_sha256, args.build_sha256))
    receipt = OUT / (args.tag + "-runtime.json")
    token = uuid.uuid4().hex
    lab.OWNER = "ninfer-mtp-copy-census-" + token
    record = {"schema": "mtp-copy-census-runtime-r1", "started_at": lab.now(), "status": "INCOMPLETE",
              "arm": args.arm, "tag": args.tag, "control_tag": args.control_tag, "diagnostic_only": True,
              "performance_claim": False, "host_only_observer": True,
              "native_id_source": "GenerationHandle::wait/GenerationResult.generated_token_ids",
              "controller_pid": os.getpid(), "owner": lab.OWNER,
              "runner_sha256": lab.sha(pathlib.Path(__file__)), "samples": [],
              "expected_epochs": {"native_startup": 1, "warmup": 2, "RU4K": 3, "JSON32K": 4},
              "warmup_census_qualified": False, "native_startup_census_qualified": False}
    handles, reservation, ident, probe, claimed, executable_sha, config_sha = [], None, None, None, False, None, None
    k = ctypes.WinDLL("kernel32")
    k.CreateMutexW.restype = ctypes.c_void_p
    k.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    k.ReleaseMutex.argtypes = [ctypes.c_void_p]
    k.CloseHandle.argtypes = [ctypes.c_void_p]
    try:
        for name in matrix.MUTEXES:
            handle = k.CreateMutexW(None, False, "Local\\" + name)
            if not handle or k.WaitForSingleObject(handle, 0) not in (0, 128):
                if handle:
                    k.CloseHandle(handle)
                raise RuntimeError("Laboratory mutex busy: " + name)
            handles.append(handle)
        OUT.mkdir(parents=True, exist_ok=True)
        assert not receipt.exists() and not list(OUT.glob(args.tag + "-*"))
        claimed = True
        config_sha = lab.sha(ROOT / "broker/config.json")
        assert lab.sha(args.protocol) == args.protocol_sha256 and lab.sha(args.build) == args.build_sha256
        protocol, build = load(args.protocol), load(args.build)
        assert protocol["created_before_inference"] and protocol["diagnostic_only"]
        assert lab.datetime.datetime.fromisoformat(protocol["created_at"]) <= lab.datetime.datetime.fromisoformat(record["started_at"])
        assert protocol["schema"] == "mtp-copy-census-protocol-r1" and protocol["order"] == ["off", "on"]
        assert protocol["capture_all_actual_mtp_runs"] is True and protocol["native_generated_ids_required"] is True
        assert protocol["acceptance"]["epochs"] == [1, 2, 3, 4]
        assert protocol["acceptance"]["qualified_request_epochs"] == [3, 4]
        assert protocol["build_receipt_sha256"] == args.build_sha256 and lab.sha(INPUT_PROTOCOL) == INPUT_SHA
        frozen = load(INPUT_PROTOCOL)
        assert all(protocol[key] == frozen[key] for key in ("model", "bench", "image", "requests", "runtime", "sampling"))
        assert protocol["warmup"]["source"] == "RU4K" and protocol["warmup"]["max_tokens"] == 32
        assert isinstance(protocol["warmup"]["prefix"], str) and protocol["warmup"]["prefix"]
        built = build["build"]
        assert build["returncode"] == 0 and build["gpu_used"] is False and build["core_source_modified"] is False
        assert built["status"] == "BUILT_NOT_RUNTIME_QUALIFIED" and built["diagnostic_only"] and not built["gpu_run"]
        assert built["host_only"] is True
        recipe = module("census_recipe", HERE / "build.py")
        assert recipe.source_check(ROOT / "source") == built["source_sha256"]
        assert built["abi_symbols"] == recipe.SYMBOLS and built["original_input_count"] == len(built["native_input_sha256"]) == 29
        output = built["outputs"][args.arm]
        assert all(built["outputs"][arm]["path"] == "/build/mtp-copy-census/ninfer-serve-" + arm for arm in ("off", "on"))
        executable_sha = output["sha256"]
        assert re.fullmatch(r"[0-9a-f]{64}", executable_sha)
        record.update(protocol_path=str(args.protocol.resolve()), protocol_sha256=args.protocol_sha256,
                      build_receipt_path=str(args.build.resolve()), build_receipt_sha256=args.build_sha256,
                      source_sha256=built["source_sha256"], executable_sha256=executable_sha,
                      model_sha256=protocol["model"]["sha256"], bench_sha256=protocol["bench"]["sha256"],
                      helper_sha256={pathlib.Path(m.__file__).name: lab.sha(pathlib.Path(m.__file__)) for m in (matrix, lab, screen)})
        previous = [load(path) for path in OUT.glob("*-runtime.json")]
        previous = [r for r in previous if r.get("protocol_sha256") == args.protocol_sha256]
        assert [r["arm"] for r in previous] == ([] if args.arm == "off" else ["off"])
        control = previous[0] if previous else None
        if control:
            control_path = OUT / (args.control_tag + "-runtime.json")
            assert control == load(control_path) and control["status"] == "REFERENCE_COLLECTED"
            assert control["tag"] == control["control_tag"] == args.control_tag
            assert control["executable_sha256"] == built["outputs"]["off"]["sha256"]
            assert all(control[key] == record[key] for key in ("protocol_sha256", "build_receipt_sha256",
                       "source_sha256", "model_sha256", "bench_sha256", "helper_sha256", "runner_sha256"))
            record["reference_receipt_sha256"] = lab.sha(control_path)
        else:
            assert args.arm == "off" and args.tag == args.control_tag
        record["admission"] = lab.admission()
        initial = lab.api("/v1/status")
        assert initial["model_state"] == "idle" and not initial["instances"]
        assert all(not m["replicas"] and not m["ready_replicas"] for m in initial["models"])
        record["initial_model_unloaded"] = True
        with socket.socket() as sock:
            assert sock.connect_ex(("127.0.0.1", 18042)) != 0
        model = ROOT / protocol["model"]["path"]
        record["model_file_identity"] = file_identity(model)
        if control and control["model_file_identity"] == record["model_file_identity"]:
            assert control["model_verification"] == "full_sha256"
            record["model_verification"] = "first_off_sha256_and_unchanged_file_identity"
        else:
            assert lab.sha(model) == record["model_sha256"] and file_identity(model) == record["model_file_identity"]
            record["model_verification"] = "full_sha256"
        bench_path = ROOT / protocol["bench"]["path"]
        assert lab.sha(bench_path) == record["bench_sha256"]
        bench, checker = module("frozen_bench", bench_path), module("copy_census_analysis", HERE / "analysis.py")
        requests = []
        for item in protocol["requests"]:
            manifest = ROOT / item["manifest"]
            assert lab.sha(manifest) == item["manifest_sha256"]
            body = next(r for r in load(manifest)["categories"] if r["category"] == item["category"])[item["body_key"]]
            assert bench.digest(bench.json_bytes(body)) == item["body_sha256"]
            requests.append((item, body))
        record["inputs"] = protocol["requests"]
        builder = json.loads(lab.docker("inspect", matrix.BUILDER).stdout)[0]
        assert builder["Id"] == build["builder_before"]["id"] and builder["Image"] == build["builder_before"]["image"]
        assert builder["State"]["Running"] and builder["State"]["Pid"] > 0 and not builder["HostConfig"].get("DeviceRequests")
        processes = lab.docker("top", builder["Id"], "-eo", "pid,args").stdout.splitlines()[1:]
        assert [p.split(None, 1)[1] for p in processes if p.strip()] == ["sleep infinity"]
        record["builder"] = {"id": builder["Id"], "image": builder["Image"], "pid": builder["State"]["Pid"], "processes": processes}
        assert lab.docker("exec", builder["Id"], "sha256sum", output["path"]).stdout.split()[0] == executable_sha
        temporary = EXPORT.with_name(EXPORT.name + "." + token + ".new")
        EXPORT.parent.mkdir(parents=True, exist_ok=True)
        try:
            with temporary.open("xb") as stream:
                subprocess.run(["wsl.exe", "-d", "Ubuntu-24.04", "--exec", "/usr/bin/docker", "exec",
                                builder["Id"], "cat", output["path"]], stdout=stream, stderr=subprocess.PIPE,
                               creationflags=subprocess.CREATE_NO_WINDOW, timeout=120, check=True)
            with temporary.open("rb") as stream:
                assert stream.read(4) == b"\x7fELF"
            assert lab.sha(temporary) == executable_sha
            temporary.replace(EXPORT)
        finally:
            temporary.unlink(missing_ok=True)
        bind = "type=bind,source=" + lab.WSL_ROOT + "/artifacts/experimental/mtp-copy-census-serve,target=" + EXE + ",readonly"
        probe = "ninfer-copy-census-bind-" + token[:8]
        probe = lab.docker("create", "--name", probe, "--label", LABEL + "=" + token, "--network", "none",
                           "-e", "NVIDIA_VISIBLE_DEVICES=void", "--mount", bind, "--entrypoint", "sha256sum", lab.IMAGE, EXE).stdout.strip()
        item = owned(probe, token)
        assert not item["HostConfig"].get("DeviceRequests")
        assert any(m["Destination"] == EXE and m["Type"] == "bind" and not m["RW"] for m in item["Mounts"])
        assert lab.docker("start", "-a", probe).stdout.split()[0] == executable_sha
        remove_owned(probe, token)
        probe = None
        record["export"] = {"identity": file_identity(EXPORT), "native_path": output["path"],
                            "elf_magic": True, "fresh_readonly_bind_sha256": executable_sha}
        record["admission_after_export"] = lab.admission()
        reservation = token
        grant = lab.api("/v1/photo-gpu-reservations", {"owner": lab.OWNER, "reservation_id": token, "ttl_seconds": 1800}, "POST", 90)
        assert grant["reservation_id"] == token and grant["owner"] == lab.OWNER and grant["state"] == "active"
        record["reservation"] = grant
        record["owned_admission"] = lab.admission(True)
        server = [EXE, "/model/jonathan-full-q4km.ninfer", "--host", "0.0.0.0", "--port", "8080",
                  "--model-id", "qwen3.8-27b", "--max-context", "65536", "--kv-capacity", "65536",
                  "--max-concurrency", "1", "--max-pending-requests", "32", "--pending-timeout-ms", "600000",
                  "--kv-dtype", "int8", "--spec", "mtp", "--draft-tokens", "3", "--prefill-chunk", "1024",
                  "--no-thinking", "--structured-output", "--vision", "--vision-residency", "overlay",
                  "--host-kv-mib", "0", "--host-state-slots", "0", "--request-log-jsonl", "/results/" + args.tag + "-events.jsonl"]
        record["command"] = server
        ident = "ninfer-mtp-copy-census-" + token[:8]
        ident = lab.docker("create", "--name", ident, "--label", LABEL + "=" + token, "--gpus", "all", "--ipc=host",
                           "-p", "127.0.0.1:18042:8080", "--mount", bind, "--mount",
                           "type=bind,source=" + lab.WSL_ROOT + "/" + protocol["model"]["path"] + ",target=" + server[1] + ",readonly",
                           "--mount", "type=bind,source=" + lab.WSL_ROOT + "/results/mtp-copy-census,target=/results",
                           "-e", "CUDA_CACHE_PATH=/tmp/ninfer-mtp-copy-census", "--entrypoint", EXE, lab.IMAGE, *server[1:]).stdout.strip()
        record["container_id"] = owned(ident, token)["Id"]
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
        item = owned(ident, token)
        assert item["State"]["Pid"] > 0 and item["Path"] == EXE and item["Args"] == server[1:]
        assert all(any(m["Destination"] == dest and m["Type"] == "bind" and not m["RW"] for m in item["Mounts"])
                   for dest in (EXE, server[1]))
        actual = lab.docker("exec", ident, "sha256sum", "/proc/1/exe").stdout.split()
        assert actual == [executable_sha, "/proc/1/exe"]
        assert lab.docker("exec", ident, "cat", "/proc/1/cmdline").stdout.rstrip("\0").split("\0") == server
        record["process"] = {"host_pid": item["State"]["Pid"], "container_pid": 1, "sha256": actual[0], "command": server}
        warm = copy.deepcopy(requests[0][1])
        warm["max_tokens"] = 32
        warm["messages"][-1]["content"] = protocol["warmup"]["prefix"] + warm["messages"][-1]["content"]
        serving_gate(reservation)
        record["warmup"] = lab.request(warm)
        assert record["warmup"]["usage"]["completion_tokens"] == 32
        assert record["warmup"]["request_sha256"] == bench.digest(bench.json_bytes(warm))
        events, used, prior = OUT / (args.tag + "-events.jsonl"), set(), 0
        for expected, body in requests:
            serving_gate(reservation)
            offset = bench.request_log_offset(events)
            observed, body_sha = bench.stream_request(18042, body)
            serving_gate(reservation)
            assert observed["http_status"] == 200 and body_sha == expected["body_sha256"]
            assert observed["usage"]["prompt_tokens"] == expected["prompt_tokens"] and observed["usage"]["completion_tokens"] == 512
            done = bench.correlate_request_done(events, offset, observed, "mtp3", prior, observed["usage"], observed["timings"], used)
            assert done["request"]["sampling"]["presence_penalty"] == 1.5 and done["request"]["sampling"]["frequency_penalty"] == 0
            prior = done["request_id"]
            row = {key: observed[key] for key in ("output", "output_sha256", "usage", "finish_reason")}
            row.update(name=expected["name"], request_body_sha256=body_sha, request_done=done)
            if control:
                reference = next(r for r in control["samples"] if r["name"] == row["name"])
                assert all(row[key] == reference[key] for key in ("output", "output_sha256", "usage", "finish_reason", "request_body_sha256"))
                assert done["speculative"] == reference["request_done"]["speculative"]
                assert done["result"] == reference["request_done"]["result"]
            record["samples"].append(row)
            print(expected["name"] + (" exact PASS; census pending" if control else " fresh reference collected"), flush=True)
        record["status"] = "REQUESTS_COMPLETE_CENSUS_PENDING"
    except BaseException as exc:
        record.update(status="INCOMPLETE", failure=type(exc).__name__ + ": " + str(exc), failure_traceback=traceback.format_exc())
    finally:
        for container, runtime in ((probe, False), (ident, True)):
            if container:
                try:
                    log = remove_owned(container, token, executable_sha if runtime else None)
                    if runtime:
                        record["owned_container_removed"] = True
                        log_path = OUT / (args.tag + "-server.log")
                        with log_path.open("x", encoding="utf-8") as stream:
                            stream.write(log)
                        record["server_log_sha256"] = lab.sha(log_path)
                        if record["status"] == "REQUESTS_COMPLETE_CENSUS_PENDING":
                            record["census"] = qualify(log, args.arm, record["warmup"], record["samples"], checker, args.tag)
                            assert "generated_token_ids" not in events.read_text(encoding="utf-8"), "Native IDs leaked into request events"
                            record["request_log_sha256"] = lab.sha(events)
                            record["status"] = "DIAGNOSTIC_QUALIFIED" if args.arm == "on" else "REFERENCE_COLLECTED"
                except BaseException as exc:
                    record.setdefault("cleanup_or_census_failures", []).append(type(exc).__name__ + ": " + str(exc))
        if reservation and (ident is None or record.get("owned_container_removed")):
            try:
                released = lab.api("/v1/photo-gpu-reservations/" + reservation, {"owner": lab.OWNER}, "DELETE")
                assert released["reservation_id"] == reservation and released["state"] == "released"
                record["reservation_released"] = True
            except BaseException as exc:
                record.setdefault("cleanup_or_census_failures", []).append(str(exc))
        try:
            if claimed:
                record["production_config_unchanged"] = lab.sha(ROOT / "broker/config.json") == config_sha
                record["model_identity_unchanged"] = "model_file_identity" not in record or file_identity(model) == record["model_file_identity"]
                if record.get("cleanup_or_census_failures") or not record["production_config_unchanged"] or not record["model_identity_unchanged"]:
                    record["status"] = "INCOMPLETE"
                record["finished_at"] = lab.now()
                with receipt.open("x", encoding="utf-8") as stream:
                    json.dump(record, stream, indent=2, ensure_ascii=False)
        finally:
            for handle in reversed(handles):
                k.ReleaseMutex(handle)
                k.CloseHandle(handle)
        print(json.dumps({key: record.get(key) for key in ("status", "failure", "cleanup_or_census_failures", "reservation_released")}), flush=True)
    if record["status"] not in ("REFERENCE_COLLECTED", "DIAGNOSTIC_QUALIFIED"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
