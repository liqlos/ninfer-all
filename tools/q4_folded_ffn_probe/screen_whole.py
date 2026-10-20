import json
import pathlib
import subprocess
import sys
import time


ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "source/tools/ffn_pair_probe"))
sys.path.insert(0, str(ROOT / "source/tools/lab"))
import run
import run_switch_matrix as matrix


OUT = ROOT / "results/q4-folded-ffn"
load = lambda p: json.loads(p.read_text(encoding="utf-8"))
tag = sys.argv[1]
protocol = load(OUT / "whole-protocol-r1.json")
assert tag == protocol["tag"]
qualification = load(OUT / "whole-link-export-r1.json")
assert qualification["status"] == "EXPORTED_WHOLE_FFN_LINKED_IDENTITY_PASS_NOT_GPU_QUALIFIED"
assert run.lab.sha(OUT / "pack-body-r2.json") == qualification["pack_body_receipt_sha256"]
assert load(OUT / "pack-body-r2.json")["status"] == "PASS_STATIC_PACK_RESOURCE_AND_TIMED_STORE_GATE"
binary = ROOT / "artifacts/experimental/q4-folded-ffn-probe"
expected = qualification["binary_sha256"]
assert run.lab.sha(binary) == expected
for name, digest in qualification["probe_sources"].items():
    assert run.lab.sha(pathlib.Path(__file__).parent / name) == digest
manifest = load(ROOT / "results/switch-matrix/w-stable-r2-build.json")["source_files"]
for name, digest in manifest.items():
    if name.endswith((".cu", ".cuh", ".h", ".hpp")):
        assert run.lab.sha(ROOT / "source" / name) == digest
binding = OUT / (tag + "-binding.json")
assert not binding.exists()
record = {"status": "INCOMPLETE", "binary_sha256": expected,
          "protocol_sha256": run.lab.sha(OUT / "whole-protocol-r1.json"),
          "qualification_sha256": run.lab.sha(OUT / "whole-link-export-r1.json"),
          "runner_sha256": run.lab.sha(pathlib.Path(__file__)),
          "started_at": run.lab.now(), "performance_claim": False}
docker = run.lab.docker


def checked_docker(*args, **kwargs):
    if args[0] == "create":
        args = (*args, str(protocol["input_seed"]))
    if args[:2] != ("start", "-a"):
        return docker(*args, **kwargs)
    ident = args[2]
    item = json.loads(docker("inspect", ident).stdout)[0]
    assert item["Config"]["Entrypoint"] == ["/lab/artifacts/experimental/q4-folded-ffn-probe"]
    assert any(m["Destination"] == "/lab" and not m["RW"] and
               m["Source"] == run.lab.WSL_ROOT for m in item["Mounts"])
    record["before_run"] = matrix.serving_admission(None)
    assert run.lab.sha(binary) == expected
    command = ["wsl.exe", "-d", "Ubuntu-24.04", "--exec", "/usr/bin/docker", *args]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and process.poll() is None:
            proof = docker("exec", ident, "sha256sum", "/proc/1/exe", check=False)
            if proof.returncode == 0:
                record["running_executable_sha256"] = proof.stdout.split()[0]
                assert record["running_executable_sha256"] == expected
                break
            time.sleep(0.05)
        assert record.get("running_executable_sha256") == expected
        stdout, stderr = process.communicate(timeout=180)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=10)
    record["after_run"] = matrix.serving_admission(None)
    assert run.lab.sha(binary) == expected
    if process.returncode == 0:
        rows = [json.loads(line) for line in stdout.splitlines() if line.startswith("{")]
        correctness = [r for r in rows if r["kind"] == "correctness"]
        assert [r["case"] for r in correctness] == protocol["cases"]
        for row in correctness:
            assert row["status"] == "PASS" and row["exact_packed_bytes"] == 25067520
            for key in ("exact_gate_up_fp32", "exact_intermediate_bf16", "exact_down_fp32",
                        "exact_down_bf16", "diagnostic_matches_timed_pack", "nonzero_carry_expected",
                        "canaries", "activation_slack_untouched", "invalid_pack_api_guards"):
                assert row[key], key
            assert row["gate_fp64_samples"] == 80 and row["down_fp64_samples"] == 64
            assert row["producer_oracle_groups"] > 0 and row["carry_sensitive_outputs"] > 0
            assert row["input_seed"] == protocol["input_seed"]
            assert row["down_blocks"] == 400 and not row["down_fixup"]
        timings = [r for r in rows if r["kind"] == "timing"]
        assert len(timings) == 6
        for case in protocol["cases"]:
            part = [r for r in timings if r["case"] == case]
            assert [r["order"] for r in part] == protocol["orders"]
            for row in part:
                assert row["repetitions_per_sample"] == 32 and row["warmup_samples"] == 4
                assert row["common_input_pack_outside_events"] and row["carry_reset_outside_events"]
                assert row["evolving_carry_exact"] and row["evolving_carry_calls"] == 32
                assert row["scope"] == protocol["timing_scope"]
                assert len(row["samples_ms"]) == 4 and all(x > 0 for x in row["samples_ms"])
                for arm, key in (("A", "baseline_ms"), ("B", "candidate_ms")):
                    mean = sum(x for a, x in zip(row["order"], row["samples_ms"]) if a == arm) / 2
                    assert abs(mean - row[key]) < 1e-8
        record["correctness"] = correctness
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


run.OUT_SUBDIR = "q4-folded-ffn"
run.PROBE_NAME = "q4-folded-ffn-probe"
run.PROBE_INPUTS = ["/lab/fixtures/operators"]
run.FIXTURES = [(ROOT / name, digest) for name, digest in protocol["fixtures"].items()]
run.EXPECTED_TIMING_BLOCKS = 6
run.lab.OWNER = "ninfer-q4-folded-whole"
run.lab.docker = checked_docker
try:
    run.main()
    receipt = load(OUT / (tag + "-receipt.json"))
    assert receipt["status"] == "PASS" and receipt["production_config_unchanged"]
    assert receipt["owned_container_removed"] and receipt["reservation_released"]
    assert not receipt.get("cleanup_failure")
    record["status"] = "PASS_EXPORTED_WHOLE_FFN_OPERATOR_BINDING"
    record["screen_receipt_sha256"] = run.lab.sha(OUT / (tag + "-receipt.json"))
finally:
    run.lab.docker = docker
    record["finished_at"] = run.lab.now()
    with binding.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
