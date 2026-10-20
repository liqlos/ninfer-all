import ctypes
import json
import pathlib
import subprocess
import sys
import time
import uuid


sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "source/tools/lab"))
import profile_long_context as lab
import replay_sse_boundary as guard
import run_switch_matrix as matrix


OUT = ROOT / "results/q6-word-group"
IMAGE = "sha256:501c92a90c3d85adc616455a6783bb5ae8b82008f469cf8dc3d6da538f75475b"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate(rows, launch_id):
    assert rows[0]["event"] == "start" and rows[0]["cc"] == 860
    assert rows[0]["head_vector_primary_writer"] == "StoreBF16"
    assert rows[0]["down_vector_primary_writer"] == "AccumulateBF16"
    assert not rows[0]["kernel_objects_recompiled"]
    kinds = [r["event"] for r in rows]
    assert "failure" not in kinds
    roundtrips = [r for r in rows if r["event"] == "full_fixture_literal_roundtrip"]
    assert {r["fixture"] for r in roundtrips} == {"head", "down"}
    for row in roundtrips:
        assert row["scales_and_d_bits_exact"] and not row["encoded_file_written"]
        assert row["original_bytes"] == row["packed_bytes"]
    cases = [r for r in rows if r["event"] == "case_correctness"]
    assert len(cases) == 10 and len({r["case"] for r in cases}) == 10
    assert sum(r["odd_start_crossing_slab"] for r in cases) == 5
    for row in cases:
        assert row["launch_id"] == launch_id
        assert row["actual_full_parent"] and row["original_parent_base_retained"]
        assert row["native_FP64_samples"] == row["candidate_FP64_samples"] == 64
        assert row["FP64_max_bound_fraction"] <= 1
        assert row["oracle_uses_original_native_bytes"] and not row["oracle_uses_candidate_codec"]
        assert row["common_native_activation_unchanged"] and row["all_buffer_canaries_intact"]
        modes = row["output_modes"]
        assert {r["mode"] for r in modes} == {"PlainFP32", "StoreBF16", "AccumulateBF16"}
        for mode in modes:
            assert mode["full_output_bit_exact"]
            assert mode["public_guard_checks"]["rejected_without_launch"]
            assert mode["public_guard_checks"]["output_unchanged"]
            for arm in ("native_route", "candidate_route"):
                assert mode[arm]["outside_timing_events"]
    qualified = [r for r in rows if r["event"] == "qualification_complete"]
    assert len(qualified) == 1 and qualified[0]["correctness_pass"]
    assert qualified[0]["cases"] == 10 and not qualified[0]["timing_has_started"]
    at = kinds.index("qualification_complete")
    assert max(i for i, kind in enumerate(kinds) if kind == "case_correctness") < at
    screens = [r for r in rows if r["event"] == "exploratory_short_screen"]
    assert len(screens) == 7 and min(i for i, k in enumerate(kinds) if k == "exploratory_short_screen") > at
    expected = {("FULL_head_vector_T1", "StoreBF16"), ("FULL_head_vector_T4", "StoreBF16"),
                ("FULL_down_vector_T1", "AccumulateBF16"), ("FULL_down_vector_T4", "AccumulateBF16")}
    expected |= {("FULL_down_MMQ_T1280", mode) for mode in ("PlainFP32", "StoreBF16", "AccumulateBF16")}
    assert {(r["case"], r["mode"]) for r in screens} == expected
    for row in screens:
        assert row["launch_id"] == launch_id and row["calls_per_sample"] == 32
        assert row["warmup_samples"] == 4 and row["warmup_order"] == "ABAB"
        assert row["evolved_32_call_outputs_bit_exact"] and not row["diagnostic_name_lookup_during_events"]
        assert row["common_activation_packed_before_events"] and row["carry_reset_before_each_event"]
        assert not row["statistical_confirmation"] and not row["end_to_end_claim"]
        assert [r["order"] for r in row["blocks"]] == ["ABBA", "BAAB", "ABBA"]
        a, b = [], []
        for block in row["blocks"]:
            assert len(block["ms_per_call"]) == 4 and all(x > 0 for x in block["ms_per_call"])
            for arm, value in zip(block["order"], block["ms_per_call"]):
                (a if arm == "A" else b).append(value)
        assert abs(sum(a) / len(a) - row["native_mean_ms"]) < 1e-8
        assert abs(sum(b) / len(b) - row["candidate_mean_ms"]) < 1e-8
    assert rows[-1]["event"] == "complete" and rows[-1]["correctness_pass"]
    assert rows[-1]["short_screen_endpoints"] == 7
    assert rows[-1]["all_correctness_passed_before_timing"]
    return {"correctness_cases": len(cases), "timing_endpoints": len(screens)}


def main():
    assert sys.platform == "win32"
    tag = sys.argv[1] if len(sys.argv) == 2 else "screen-r1"
    assert tag.replace("-", "").isalnum()
    receipt = OUT / (tag + "-receipt.json")
    output = OUT / (tag + ".jsonl")
    assert not receipt.exists() and not output.exists()
    protocol = load(OUT / "screen-protocol-r1.json")
    qualification = load(OUT / "host-export-r1.json")
    assert qualification["status"] == "EXPORTED_LINKED_Q6_HOST_NOT_GPU_QUALIFIED"
    binary = ROOT / "artifacts/experimental/q6-word-group-probe"
    assert lab.sha(binary) == qualification["binary_sha256"]
    for name, digest in qualification["host_source_sha256"].items():
        assert lab.sha(pathlib.Path(__file__).parent / name) == digest
    for name, digest in qualification["native_source_sha256"].items():
        assert lab.sha(ROOT / "source" / name) == digest
    record = {"status": "INCOMPLETE", "started_at": lab.now(), "exploratory_only": True,
              "binary_sha256": qualification["binary_sha256"],
              "protocol_sha256": lab.sha(OUT / "screen-protocol-r1.json"),
              "export_receipt_sha256": lab.sha(OUT / "host-export-r1.json"),
              "controller_sha256": lab.sha(pathlib.Path(__file__)), "performance_claim": False}
    lab.OWNER = "ninfer-q6-word-group"
    config = lab.sha(ROOT / "broker/config.json")
    handles, reservation, ident = [], None, None
    kernel = ctypes.WinDLL("kernel32")
    kernel.CreateMutexW.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel.ReleaseMutex.argtypes = [ctypes.c_void_p]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    try:
        for name in guard.MUTEXES:
            handle = kernel.CreateMutexW(None, False, "Local\\" + name)
            assert handle
            if kernel.WaitForSingleObject(handle, 0) not in (0, 128):
                kernel.CloseHandle(handle)
                raise RuntimeError("Laboratory mutex busy: " + name)
            handles.append(handle)
        record["admission"] = lab.admission()
        record["fixtures"] = {}
        for name, digest in protocol["fixtures"].items():
            assert lab.sha(ROOT / name) == digest
            record["fixtures"][name] = digest
        reservation = uuid.uuid4().hex
        grant = lab.api("/v1/photo-gpu-reservations", {"owner": lab.OWNER,
                       "reservation_id": reservation, "ttl_seconds": 900}, "POST", 90)
        assert grant["state"] == "active" and grant["reservation_id"] == reservation
        record["reservation_id"] = reservation
        record["owned_admission"] = lab.admission(True)
        launch_id = tag + "-" + reservation
        ident = lab.docker("create", "--name", "ninfer-lab-q6-word-group-" + reservation[:8],
            "--label", "io.codex.ninfer-owner=" + reservation, "--gpus", "all",
            "--mount", "type=bind,source=" + lab.WSL_ROOT + ",target=/lab,readonly",
            "--entrypoint", "/lab/artifacts/experimental/q6-word-group-probe", IMAGE,
            "--fixtures-root", "/lab/fixtures", "--launch-id", launch_id).stdout.strip()
        assert len(ident) == 64
        record["container_id"] = ident
        item = json.loads(lab.docker("inspect", ident).stdout)[0]
        assert item["Image"] == IMAGE
        assert item["Config"]["Entrypoint"] == ["/lab/artifacts/experimental/q6-word-group-probe"]
        assert any(m["Destination"] == "/lab" and not m["RW"] and m["Source"] == lab.WSL_ROOT
                   for m in item["Mounts"])
        assert lab.sha(binary) == qualification["binary_sha256"]
        command = ["wsl.exe", "-d", "Ubuntu-24.04", "--exec", "/usr/bin/docker", "start", "-a", ident]
        start = time.monotonic()
        with output.open("x", encoding="utf-8") as stdout:
            process = subprocess.Popen(command, stdout=stdout, stderr=subprocess.PIPE,
                                       text=True, creationflags=subprocess.CREATE_NO_WINDOW)
            try:
                deadline = time.monotonic() + 15
                while process.poll() is None and time.monotonic() < deadline:
                    proof = lab.docker("exec", ident, "sha256sum", "/proc/1/exe", check=False)
                    if proof.returncode == 0:
                        record["running_executable_sha256"] = proof.stdout.split()[0]
                        assert record["running_executable_sha256"] == qualification["binary_sha256"]
                        break
                    time.sleep(0.05)
                assert record.get("running_executable_sha256") == qualification["binary_sha256"]
                _, stderr = process.communicate(timeout=450)
                record.update(exit_code=process.returncode, stderr=stderr,
                              wall_seconds=time.monotonic() - start)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate(timeout=10)
        record["after_run_admission"] = matrix.serving_admission(reservation)
        assert lab.sha(binary) == qualification["binary_sha256"]
        rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.startswith("{")]
        assert record["exit_code"] == 0, rows[-1] if rows else record["stderr"]
        record["validation"] = validate(rows, launch_id)
        record["output_sha256"] = lab.sha(output)
        record["status"] = "PASS_EXPORTED_OPERATOR_CORRECTNESS_AND_EXPLORATORY_SCREEN"
    except BaseException as exc:
        record["failure"] = type(exc).__name__ + ": " + str(exc)
    finally:
        try:
            if ident:
                item = json.loads(lab.docker("inspect", ident).stdout)[0]
                assert item["Image"] == IMAGE and item["Id"] == ident
                assert item["Config"]["Labels"]["io.codex.ninfer-owner"] == reservation
                if item["State"]["Running"]:
                    lab.docker("stop", "--time", "5", ident)
                lab.docker("rm", ident)
                record["owned_container_removed"] = True
            if reservation and (ident is None or record.get("owned_container_removed")):
                released = lab.api("/v1/photo-gpu-reservations/" + reservation,
                                   {"owner": lab.OWNER}, "DELETE")
                assert released["reservation_id"] == reservation and released["state"] == "released"
                record["reservation_released"] = True
        except BaseException as exc:
            record["cleanup_failure"] = str(exc)
        for handle in reversed(handles):
            kernel.ReleaseMutex(handle)
            kernel.CloseHandle(handle)
        record["production_config_unchanged"] = lab.sha(ROOT / "broker/config.json") == config
        record["finished_at"] = lab.now()
        with receipt.open("x", encoding="utf-8") as target:
            json.dump(record, target, indent=2)
        print(json.dumps(record, indent=2), flush=True)
    if record["status"] != "PASS_EXPORTED_OPERATOR_CORRECTNESS_AND_EXPLORATORY_SCREEN" or record.get("cleanup_failure"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
