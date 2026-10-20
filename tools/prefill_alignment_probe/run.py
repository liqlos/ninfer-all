import ctypes
import json
import pathlib
import sys
import time
import uuid


ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "source/tools/lab"))
import build_switch_matrix as matrix
import profile_long_context as lab
import run_switch_matrix as runtime


OUT = ROOT / "results/prefill-alignment"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    assert len(sys.argv) in (2, 3)
    tag = sys.argv[1]
    protocol_path = OUT / "short-r1-execution-protocol.json"
    protocol = load(protocol_path)
    arms = [arm for arm in protocol["order"] if arm["tag"] == tag]
    assert len(arms) == 1
    arm = arms[0]
    expected_anchor = int(sys.argv[2]) if len(sys.argv) == 3 else None
    assert (expected_anchor is None) == (tag == "short-r1-a1")
    assert expected_anchor is None or 0 <= expected_anchor < 248320
    receipt_path = OUT / (tag + "-receipt.json")
    assert not receipt_path.exists() and not (OUT / (tag + "-output.jsonl")).exists()
    record = {"tag": tag, "started_at": lab.now(), "status": "INCOMPLETE",
              "scope": "correctness_only_no_performance_claim", "arm": arm,
              "expected_anchor": expected_anchor, "protocol_sha256": lab.sha(protocol_path)}
    config = lab.sha(ROOT / "broker/config.json")
    reservation, ident, handles = None, None, []
    k = ctypes.WinDLL("kernel32")
    k.CreateMutexW.restype = ctypes.c_void_p
    k.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    k.ReleaseMutex.argtypes = [ctypes.c_void_p]
    k.CloseHandle.argtypes = [ctypes.c_void_p]
    try:
        for name in matrix.MUTEXES:
            handle = k.CreateMutexW(None, False, "Local\\" + name)
            if k.WaitForSingleObject(handle, 0) not in (0, 128):
                k.CloseHandle(handle)
                raise RuntimeError("Laboratory mutex busy: " + name)
            handles.append(handle)
        record["admission_before"] = lab.admission()
        build = load(OUT / "build-r1.json")
        export = load(OUT / "export-r1.json")
        assert build["status"] == "CPU_BUILT_NOT_GPU_QUALIFIED"
        assert export["status"] == "EXPORTED_NOT_GPU_QUALIFIED"
        assert export["build_receipt_sha256"] == lab.sha(OUT / "build-r1.json")
        binary = ROOT / "artifacts/experimental/prefill-alignment-probe"
        assert lab.sha(binary) == export["binary_sha256"] == build["binary_sha256"]
        for name, digest in build["probe_sources"].items():
            assert pathlib.Path(name).name == name
            assert lab.sha(pathlib.Path(__file__).parent / name) == digest
        input_path = OUT / "raw-short-r1-effective-input.json"
        assert lab.sha(input_path) == protocol["effective_input_sha256"]
        base = load(OUT / "protocol-r1.json")
        assert lab.sha(ROOT / "models/stable/qwen38-jonathan-q4km-proposal-vision-q.ninfer") == base["model_sha256"]
        record.update(binary_sha256=export["binary_sha256"],
                      build_receipt_sha256=lab.sha(OUT / "build-r1.json"),
                      export_receipt_sha256=lab.sha(OUT / "export-r1.json"),
                      input_sha256=lab.sha(input_path))
        lab.OWNER = "ninfer-prefill-alignment"
        reservation = uuid.uuid4().hex
        grant = lab.api("/v1/photo-gpu-reservations", {
            "owner": lab.OWNER, "reservation_id": reservation, "ttl_seconds": 900}, "POST", 90)
        assert grant["state"] == "active" and grant["reservation_id"] == reservation
        record["reservation_id"] = reservation
        record["owned_admission"] = lab.admission(True)
        args = ["/lab/results/prefill-alignment/raw-short-r1-effective-input.json"]
        if expected_anchor is not None:
            args.append(str(expected_anchor))
        ident = lab.docker("create", "--name", "ninfer-lab-prefill-" + reservation[:8],
            "--label", "io.codex.ninfer-owner=" + reservation, "--gpus", "all", "--ipc=host",
            "--mount", "type=bind,source=" + lab.WSL_ROOT + ",target=/lab,readonly",
            "--mount", "type=bind,source=" + lab.WSL_ROOT +
            "/models/stable/qwen38-jonathan-q4km-proposal-vision-q.ninfer,target=/model/jonathan-full-q4km.ninfer,readonly",
            "-e", "NINFER_PREFILL_ALIGN=" + str(arm["align"]),
            "-e", "CUDA_CACHE_PATH=/tmp/ninfer-prefill-alignment",
            "--entrypoint", "/lab/artifacts/experimental/prefill-alignment-probe",
            lab.IMAGE, *args).stdout.strip()
        assert len(ident) == 64
        record["container_id"] = ident
        item = json.loads(lab.docker("inspect", ident).stdout)[0]
        assert item["Image"] == lab.IMAGE
        assert item["Config"]["Entrypoint"] == ["/lab/artifacts/experimental/prefill-alignment-probe"]
        assert all(not mount["RW"] for mount in item["Mounts"])
        lab.docker("start", ident)
        proof = lab.docker("exec", ident, "sha256sum", "/proc/1/exe").stdout.split()
        assert len(proof) == 2 and proof[0] == export["binary_sha256"]
        record["running_executable_sha256"] = proof[0]
        print(json.dumps({"tag": tag, "stage": "running", "binary_sha256": proof[0]}), flush=True)
        started = time.monotonic()
        while True:
            record["last_serving_admission"] = runtime.serving_admission(reservation)
            state = json.loads(lab.docker("inspect", ident).stdout)[0]["State"]
            if not state["Running"]:
                break
            if time.monotonic() - started > 300:
                raise TimeoutError("Standalone state gate exceeded 300 seconds")
            time.sleep(3)
        logs = lab.docker("logs", ident)
        (OUT / (tag + "-output.jsonl")).write_text(logs.stdout, encoding="utf-8")
        record["stderr"] = logs.stderr
        record["exit_code"] = state["ExitCode"]
        assert state["ExitCode"] == 0, logs.stderr[-3000:] + logs.stdout[-3000:]
        rows = [json.loads(line) for line in logs.stdout.splitlines() if line.startswith("{")]
        assert len(rows) == 1
        record["output_sha256"] = lab.sha(OUT / (tag + "-output.jsonl"))
        assert lab.sha(binary) == export["binary_sha256"]
        record["status"] = "RAN_PENDING_STATE_COMPARISON"
    except BaseException as exc:
        record["failure"] = repr(exc)
        raise
    finally:
        try:
            if ident:
                item = json.loads(lab.docker("inspect", ident).stdout)[0]
                assert item["Id"] == ident and item["Image"] == lab.IMAGE
                assert item["Config"]["Labels"]["io.codex.ninfer-owner"] == reservation
                if item["State"]["Running"]:
                    lab.docker("stop", "--time", "5", ident)
                lab.docker("rm", ident)
                record["owned_container_removed"] = True
            if reservation:
                released = lab.api("/v1/photo-gpu-reservations/" + reservation,
                                   {"owner": lab.OWNER}, "DELETE")
                assert released["state"] == "released" and released["reservation_id"] == reservation
                record["reservation_released"] = True
        except BaseException as exc:
            record["cleanup_failure"] = repr(exc)
        for handle in reversed(handles):
            k.ReleaseMutex(handle)
            k.CloseHandle(handle)
        record["production_config_unchanged"] = lab.sha(ROOT / "broker/config.json") == config
        record["finished_at"] = lab.now()
        with receipt_path.open("x", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2)
        print(json.dumps({key: record.get(key) for key in
                          ("tag", "status", "failure", "cleanup_failure", "owned_container_removed",
                           "reservation_released", "production_config_unchanged")}), flush=True)
    if record["status"] != "RAN_PENDING_STATE_COMPARISON" or record.get("cleanup_failure"):
        raise SystemExit(1)
    assert record["production_config_unchanged"]


if __name__ == "__main__":
    main()
