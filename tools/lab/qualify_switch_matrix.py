import ctypes
import hashlib
import json
import pathlib
import re
import sys
import uuid

import build_switch_matrix as matrix
import profile_long_context as lab


ROOT = lab.ROOT
OUT = ROOT / "results/switch-matrix"
ARTIFACTS = ROOT / "artifacts/experimental"
OPERATOR_MANIFEST = ROOT / "fixtures/operators/manifest.json"
WEIGHT_MANIFEST = ROOT / "fixtures/weights/manifest.json"
HEAD_RECEIPT = ROOT / "provenance/q6-full-head-fixture-20261017.json"
OPERATOR_FIXTURE = "q6-full-k17408.bin"
HEAD_FIXTURE = "q6-head-full-k5120.bin"
OPERATOR_SHA = "a381c6be36f572170ccc9c7e4a3f8b25e5061e4eb633649399f3ec9759cb43e2"
HEAD_SHA = "2155d6e1f7c5549077438c4760e6e150945dd5e406484b29162dff7b03d05da0"
BUILDER_IMAGE = "sha256:501c92a90c3d85adc616455a6783bb5ae8b82008f469cf8dc3d6da538f75475b"
OWNER = "ninfer-switch-matrix"
LAB_CONTROLLERS = {
    "tools/lab/build_switch_matrix.py", "tools/lab/profile_long_context.py",
    "tools/lab/qualify_switch_matrix.py", "tools/lab/run_switch_matrix.py",
    "tools/lab/switch_matrix_route.py", "tools/lab/analyze_switch_matrix.py",
}


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def build_inputs_sha(build):
    files = {name: sha for name, sha in build["source_files"].items()
             if name not in LAB_CONTROLLERS}
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def reference_inputs_sha(reference):
    matches = []
    for path in OUT.glob("*-build.json"):
        build = load(path)
        if (build.get("status") == "BUILT_EXPORTED_NOT_GPU_QUALIFIED" and
                build.get("arm") == "w" and
                build.get("source_manifest_sha256") == reference["source_manifest_sha256"]):
            matches.append(build_inputs_sha(build))
    assert matches and len(set(matches)) == 1, "Pinned W build inputs missing or ambiguous"
    return matches[0]


def verify_export(arm):
    manifest_path = ARTIFACTS / "export-manifest.json"
    build_path = ARTIFACTS / "build-receipt.json"
    manifest, build = load(manifest_path), load(build_path)
    assert manifest["schema"] == "switch-matrix-export-v1" and manifest["arm"] == arm
    assert build["arm"] == arm and build["status"] == "BUILT_EXPORTED_NOT_GPU_QUALIFIED"
    assert build["source_manifest_sha256"] == manifest["source_manifest_sha256"]
    assert manifest["files"] == build["export_sha256"]
    opts = build["options"]
    assert opts["NINFER_GGUF_MMQ_STORE_FUSION"] is True
    assert opts["NINFER_GGUF_DP4A_SHORT_CHAINS"] is (arm in {"q4", "both"})
    assert opts["NINFER_GGUF_Q6_MMQ_A_X4"] is (arm in {"q6", "both"})
    expected_sources = build["source_files"]
    for name in ("tests/ops/linear/test_gguf_mmq_store.cu", "tests/ops/linear/tests.cmake"):
        assert lab.sha(ROOT / "source" / name) == expected_sources[name]
    for name, expected in manifest["files"].items():
        path = ARTIFACTS / name
        assert path.is_file() and lab.sha(path) == expected, name
    return manifest, build, lab.sha(manifest_path), lab.sha(build_path)


def verify_fixtures():
    operator_manifest = load(OPERATOR_MANIFEST)
    operator_entry = next(x for x in operator_manifest["fixtures"] if x["file"] == OPERATOR_FIXTURE)
    head_receipt = load(HEAD_RECEIPT)
    assert operator_entry["bytes"] == 73113600 and operator_entry["sha256"] == OPERATOR_SHA
    assert head_receipt["status"] == "PASS_FULL_REAL_HEAD_FIXTURE"
    assert head_receipt["file"] == HEAD_FIXTURE and head_receipt["bytes"] == 1042944000
    assert head_receipt["sha256"] == HEAD_SHA and not head_receipt["synthetic_fallback"]
    weight_manifest = load(WEIGHT_MANIFEST)
    runtime_entries = {x["file"]: x for x in weight_manifest["samples"]}
    needed = {"q4-k5120.bin", "q4-k6144.bin", "q4-k17408.bin", "q4-swiglu-k5120.bin",
              "q6-k5120.bin", "q6-k17408.bin"}
    assert needed <= runtime_entries.keys()
    runtime_fixtures = {}
    for name in sorted(needed):
        entry = runtime_entries[name]
        path = ROOT / "fixtures/weights" / name
        assert path.stat().st_size == entry["bytes"] and lab.sha(path) == entry["sha256"]
        runtime_fixtures[name] = {"bytes": entry["bytes"], "sha256": entry["sha256"]}
    fixtures = {
        OPERATOR_FIXTURE: {"path": ROOT / "fixtures/operators" / OPERATOR_FIXTURE,
                           "sha256": OPERATOR_SHA, "bytes": 73113600},
        HEAD_FIXTURE: {"path": ROOT / "fixtures/weights" / HEAD_FIXTURE,
                       "sha256": HEAD_SHA, "bytes": 1042944000},
    }
    for item in fixtures.values():
        assert item["path"].stat().st_size == item["bytes"]
        assert lab.sha(item["path"]) == item["sha256"]
    hashes = {name: {"bytes": item["bytes"], "sha256": item["sha256"]}
              for name, item in fixtures.items()}
    hashes.update(runtime_fixtures)
    return hashes, {"operators_manifest_sha256": lab.sha(OPERATOR_MANIFEST),
                    "weights_manifest_sha256": lab.sha(WEIGHT_MANIFEST),
                    "full_head_receipt_sha256": lab.sha(HEAD_RECEIPT)}


def serving_admission(reservation):
    status = lab.api("/v1/status")
    assert not status["agents"] and not status["queue"] and not status["leases"]
    assert not status["inference"]["active_requests"]
    grant = status["admission"]["photo_gpu_reservation"]
    assert grant["active"] and grant["owner"] == OWNER
    assert not status["admission"]["active_batches"]
    assert not status["inference"]["active_sequences"]
    assert not status.get("external_growth_pressure", {}).get("active")
    assert status["windows_gpu"]["available"]
    assert status["windows_gpu"]["external_reserve_fraction"] == 0.3
    allowed = {"dwm", "explorer", "ChatGPT", "Codex", "chrome", "msedge", "firefox",
               "csrss", "vmmemWSL", "vmmem"}
    assert all(p.get("process") in allowed for p in status["windows_gpu"]["process_memory"])
    capacities = status["capacity_estimate"]
    assert len(capacities) == 1
    capacity = capacities[0]
    assert capacity["sample_age_seconds"] <= 15 and capacity["measurement"] == "measured"
    assert capacity["external_reserve_satisfied"]
    assert capacity["available_llm_budget_bytes"] >= 24_000_000_000
    return {"at": lab.now(), "capacity": capacity,
            "process_memory": status["windows_gpu"]["process_memory"]}


def main():
    arm, tag = sys.argv[1:]
    assert arm in matrix.VARIANTS and re.fullmatch(r"[a-z0-9-]+", tag)
    OUT.mkdir(parents=True, exist_ok=True)
    receipt_path = OUT / (tag + "-qualification.json")
    dump_path = OUT / (tag + "-q6-runtime.bin")
    q4_dump_path = OUT / (tag + "-q4-mmq.bin")
    assert not receipt_path.exists() and not dump_path.exists() and not q4_dump_path.exists()
    record = {"schema": "ninfer-switch-matrix-q6-qualification-v2", "arm": arm,
              "tag": tag, "started_at": lab.now(), "performance_claim": False}
    reservation, ident, handles = None, None, []
    config_sha = lab.sha(ROOT / "broker/config.json")
    lab.OWNER = OWNER
    kernel = None
    reference_path = OUT / "q6-runtime-gate-reference-w-v2.json"
    record["reference_file"] = reference_path.name
    try:
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateMutexW.restype = ctypes.c_void_p
        kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        kernel.ReleaseMutex.argtypes = [ctypes.c_void_p]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        for name in matrix.MUTEXES:
            handle = kernel.CreateMutexW(None, False, "Local\\" + name)
            if kernel.WaitForSingleObject(handle, 0) not in (0, 128):
                kernel.CloseHandle(handle)
                raise RuntimeError("Laboratory mutex busy: " + name)
            handles.append(handle)
        record["admission_before"] = lab.admission()
        manifest, build, manifest_sha, build_sha = verify_export(arm)
        fixture_hashes, fixture_receipts = verify_fixtures()
        record.update(export_manifest_sha256=manifest_sha, build_receipt_sha256=build_sha,
                      export=manifest, source_manifest_sha256=manifest["source_manifest_sha256"],
                      compiled_inputs_sha256=build_inputs_sha(build),
                      source_comparison_excludes=sorted(LAB_CONTROLLERS),
                      fixture_sha256=fixture_hashes,
                      fixture_receipts=fixture_receipts)
        if arm != "w" or reference_path.exists():
            reference = load(reference_path)
            assert reference["status"] == "PASS" and reference["arm"] == "w"
            assert reference_inputs_sha(reference) == build_inputs_sha(build), \
                "Compiled source/test inputs differ from W"
            assert reference["fixture_sha256"] == fixture_hashes
        reservation = uuid.uuid4().hex
        granted = lab.api("/v1/photo-gpu-reservations", {
            "owner": OWNER, "reservation_id": reservation, "ttl_seconds": 1800}, "POST", 90)
        assert granted["reservation_id"] == reservation and granted["owner"] == OWNER
        assert granted["state"] == "active"
        record["reservation_id"] = reservation
        record["admission_owned"] = lab.admission(True)
        record["admission_before_tests"] = serving_admission(reservation)
        container_name = "ninfer-q6-gate-" + reservation[:10]
        record["container_name"] = container_name
        container_args = [
            "create", "--name", container_name, "--label", "io.codex.ninfer-experiment=q6-switch-matrix",
            "--label", "io.codex.ninfer-owner=" + reservation, "--gpus", "all", "--ipc=host",
            "--mount", "type=bind,source=" + lab.WSL_ROOT + "/artifacts/experimental/ninfer_tests,target=/lab/ninfer_tests,readonly",
            "--mount", "type=bind,source=" + lab.WSL_ROOT + "/fixtures/operators,target=/lab/fixtures/operators,readonly",
            "--mount", "type=bind,source=" + lab.WSL_ROOT + "/fixtures/weights,target=/lab/fixtures/weights,readonly",
            "--mount", "type=bind,source=" + lab.WSL_ROOT + "/results/switch-matrix,target=/results",
            "-e", "CUDA_CACHE_PATH=/tmp/ninfer-q6-gate-" + reservation[:10],
            "--entrypoint", "/bin/sh", BUILDER_IMAGE, "-c", "exec sleep infinity"]
        ident = lab.docker(*container_args).stdout.strip()
        assert re.fullmatch(r"[0-9a-f]{64}", ident)
        record["container_id"] = ident
        item = json.loads(lab.docker("inspect", ident).stdout)[0]
        assert item["Id"] == ident and item["Name"] == "/" + container_name
        assert item["Image"] == BUILDER_IMAGE
        assert item["Config"]["Labels"]["io.codex.ninfer-experiment"] == "q6-switch-matrix"
        assert item["Config"]["Labels"]["io.codex.ninfer-owner"] == reservation
        lab.docker("start", ident)
        binary_proof = lab.docker("exec", ident, "sha256sum", "/lab/ninfer_tests").stdout.split()[0]
        assert binary_proof == manifest["files"]["ninfer_tests"]
        record["mounted_ninfer_tests_sha256"] = binary_proof
        fixture_paths = ["/lab/fixtures/operators/" + OPERATOR_FIXTURE,
                         "/lab/fixtures/weights/" + HEAD_FIXTURE]
        fixture_paths.extend("/lab/fixtures/weights/" + name
                             for name in sorted(fixture_hashes)
                             if name.startswith("q4-") or name.startswith("q6-k"))
        fixture_proof = lab.docker("exec", ident, "sha256sum", *fixture_paths).stdout
        mounted = {line.split()[1]: line.split()[0] for line in fixture_proof.splitlines()}
        for name, item in fixture_hashes.items():
            path = ("/lab/fixtures/operators/" + name if name == OPERATOR_FIXTURE else
                    "/lab/fixtures/weights/" + name)
            assert mounted[path] == item["sha256"], path
        record["mounted_fixture_sha256"] = {path: mounted[path] for path in fixture_paths}
        tests = []
        record["tests"] = tests
        store = lab.docker("exec", ident, "/lab/ninfer_tests", "ninfer_linear_gguf_mmq_store_test",
                           timeout=300, check=False)
        tests.append({"name": "ninfer_linear_gguf_mmq_store_test", "exit_code": store.returncode,
                      "stdout_sha256": hashlib.sha256(store.stdout.encode()).hexdigest()})
        assert store.returncode == 0, store.stdout[-2000:] + store.stderr[-2000:]
        serving_admission(reservation)
        gate = lab.docker("exec", ident, "/lab/ninfer_tests", "ninfer_linear_gguf_mmq_store_test",
                          "--q6-runtime-gate", "/lab/fixtures/operators", "/lab/fixtures/weights",
                          "/results/" + dump_path.name, timeout=900, check=False)
        tests.append({"name": "ninfer_linear_gguf_mmq_store_test --q6-runtime-gate",
                      "exit_code": gate.returncode,
                      "stdout_sha256": hashlib.sha256(gate.stdout.encode()).hexdigest()})
        assert gate.returncode == 0 and "fixtures=real_only prefix_rows=10240 status=PASS" in gate.stdout, \
            gate.stdout[-3000:] + gate.stderr[-3000:]
        expected_macro = "NINFER_GGUF_Q6_MMQ_A_X4=" + (
            "ON" if arm in {"q6", "both"} else "OFF")
        assert "q6_runtime_gate " + expected_macro in gate.stdout
        q6_cases = [line for line in gate.stdout.splitlines()
                    if line.startswith("q6_runtime_gate case=")]
        assert len(q6_cases) == 11 and all("route=PASS" in line or "status=PASS" in line
                                          for line in q6_cases)
        assert sum("T1280 epilogue=" in line and "store_route=FUSED" in line for line in q6_cases) == 4
        assert sum("T1281 epilogue=" in line and "store_route=FALLBACK" in line for line in q6_cases) == 2
        for case, epilogue, route in (
            ("K17408_N5120_T768", "Accumulate", "FUSED"),
            ("K5120_N10240_T1280", "Store", "FUSED"),
            ("K5120_N10240_T768", "Store", "FUSED"),
            ("K5120_N1024_T1280", "Store", "FUSED"),
            ("K5120_N1024_T768", "Store", "FALLBACK"),
        ):
            assert any(f"case={case} epilogue={epilogue} store_route={route}" in line
                       for line in q6_cases), (case, epilogue, route)
        record["q6_runtime_cases"] = q6_cases
        assert dump_path.is_file()
        runtime_sha = lab.sha(dump_path)
        record["q6_runtime_dump_sha256"] = runtime_sha
        record["q6_runtime_dump_bytes"] = dump_path.stat().st_size
        serving_admission(reservation)
        q4 = lab.docker("exec", ident, "/lab/ninfer_tests", "ninfer_linear_gguf_test",
                        "--real-shapes", "--weight-dir", "/lab/fixtures/weights",
                        "--output", "/results/" + q4_dump_path.name, timeout=900, check=False)
        tests.append({"name": "ninfer_linear_gguf_test --real-shapes",
                      "exit_code": q4.returncode,
                      "stdout_sha256": hashlib.sha256(q4.stdout.encode()).hexdigest()})
        assert q4.returncode == 0 and "OK GGUF real shapes" in q4.stdout, \
            q4.stdout[-3000:] + q4.stderr[-3000:]
        real_shape_lines = [line[line.index("gguf real-shape "):] for line in q4.stdout.splitlines()
                            if "gguf real-shape " in line]
        record["real_shape_cases"] = real_shape_lines
        assert len(real_shape_lines) == 7 and all("source=synthetic" not in line for line in
                                                   real_shape_lines if "q4_k" in line)
        assert any("q4_k K=5120" in line and "source=q4-k5120.bin" in line
                   for line in real_shape_lines)
        assert any("q4_k K=6144" in line and "source=q4-k6144.bin" in line
                   for line in real_shape_lines)
        assert any("q4_k K=17408" in line and "source=q4-k17408.bin" in line
                   for line in real_shape_lines)
        assert any("q4_k K=5120 SwiGLU source=q4-swiglu-k5120.bin" in line
                   for line in real_shape_lines)
        assert any("q6_k K=5120" in line and "source=q6-k5120.bin" in line
                   for line in real_shape_lines)
        assert any("q6_k K=17408" in line and "source=q6-k17408.bin" in line
                   for line in real_shape_lines)
        assert any("q6_k K=6144" in line and "source=synthetic" in line
                   for line in real_shape_lines)
        assert q4_dump_path.is_file()
        record["q4_real_shapes_dump_sha256"] = lab.sha(q4_dump_path)
        record["q4_real_shapes_dump_bytes"] = q4_dump_path.stat().st_size
        record["tests"] = tests
        if arm == "w":
            record["w_control_sha256"] = reference["q6_runtime_dump_sha256"] if reference_path.exists() else runtime_sha
            record["q6_outputs_exact_match"] = runtime_sha == record["w_control_sha256"]
            record["q4_outputs_exact_match"] = (
                record["q4_real_shapes_dump_sha256"] == reference["q4_real_shapes_dump_sha256"]
                if reference_path.exists() else True)
            assert record["q6_outputs_exact_match"] and record["q4_outputs_exact_match"]
            if not reference_path.exists():
                record["pending_w_reference"] = {
                    "schema": "ninfer-switch-matrix-q6-output-reference-v2", "status": "PASS",
                    "arm": "w", "source_manifest_sha256": manifest["source_manifest_sha256"],
                    "fixture_sha256": fixture_hashes, "q6_runtime_dump_sha256": runtime_sha,
                    "q6_runtime_dump_bytes": dump_path.stat().st_size,
                    "q4_real_shapes_dump_sha256": record["q4_real_shapes_dump_sha256"],
                    "q4_real_shapes_dump_bytes": q4_dump_path.stat().st_size}
        else:
            record["w_control_sha256"] = reference["q6_runtime_dump_sha256"]
            record["q6_outputs_exact_match"] = runtime_sha == reference["q6_runtime_dump_sha256"]
            record["q4_outputs_exact_match"] = (
                record["q4_real_shapes_dump_sha256"] == reference["q4_real_shapes_dump_sha256"])
            assert record["q6_outputs_exact_match"], "Q6 runtime dump differs from exact W control"
            assert record["q4_outputs_exact_match"], "Q4 real-shape dump differs from exact W control"
        record["outputs_exact_match"] = record["q6_outputs_exact_match"] and record["q4_outputs_exact_match"]
        record["status"] = "PASS"
    except BaseException as exc:
        record["status"] = "INCOMPLETE"
        record["failure"] = type(exc).__name__ + ": " + str(exc)
    finally:
        container_gone = ident is None
        if ident:
            try:
                item = json.loads(lab.docker("inspect", ident).stdout)[0]
                assert item["Id"] == ident and item["Name"] == "/" + container_name
                assert item["Image"] == BUILDER_IMAGE
                labels = item["Config"]["Labels"]
                assert labels["io.codex.ninfer-experiment"] == "q6-switch-matrix"
                assert labels["io.codex.ninfer-owner"] == reservation
                if item["State"]["Running"]:
                    lab.docker("stop", "--time", "10", ident, timeout=30)
                item = json.loads(lab.docker("inspect", ident).stdout)[0]
                assert item["Id"] == ident and not item["State"]["Running"]
                lab.docker("rm", ident)
                container_gone = True
                record["owned_container_removed"] = True
            except BaseException as exc:
                record["cleanup_failure"] = type(exc).__name__ + ": " + str(exc)
        for path in (dump_path, q4_dump_path):
            try:
                path.unlink(missing_ok=True)
            except BaseException as exc:
                record["dump_cleanup_failure"] = str(exc)
        if reservation and container_gone:
            try:
                released = lab.api("/v1/photo-gpu-reservations/" + reservation,
                                   {"owner": OWNER}, "DELETE")
                assert released["reservation_id"] == reservation and released["state"] == "released"
                record["reservation_released"] = True
            except BaseException as exc:
                record["cleanup_failure"] = type(exc).__name__ + ": " + str(exc)
        for handle in reversed(handles):
            kernel.ReleaseMutex(handle)
            kernel.CloseHandle(handle)
        record["production_config_unchanged"] = lab.sha(ROOT / "broker/config.json") == config_sha
        record["finished_at"] = lab.now()
        if (record.get("dump_cleanup_failure") or record.get("cleanup_failure") or
                not record.get("reservation_released") or not record["production_config_unchanged"]):
            record["status"] = "INCOMPLETE"
        if arm == "w" and record["status"] == "PASS" and not reference_path.exists():
            try:
                with reference_path.open("x", encoding="utf-8") as f:
                    json.dump(record["pending_w_reference"], f, indent=2)
            except BaseException as exc:
                record["status"] = "INCOMPLETE"
                record["cleanup_failure"] = "W reference write failed: " + str(exc)
        record.pop("pending_w_reference", None)
        with receipt_path.open("x", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        print(json.dumps({key: record.get(key) for key in
                          ("status", "q6_outputs_exact_match", "q4_outputs_exact_match", "failure", "cleanup_failure",
                           "dump_cleanup_failure", "owned_container_removed", "reservation_released")}),
              flush=True)
    if record["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
