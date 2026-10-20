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
protocol = load(OUT / "protocol-r1.json")
assert tag == protocol["tag"]
qualification = load(OUT / "link-export-r1.json")
assert qualification["status"] == "EXPORTED_LINKED_KERNEL_IDENTITY_PASS_NOT_GPU_QUALIFIED"
assert load(OUT / "body-r1.json")["status"] == "PASS_STATIC_RESOURCE_AND_TIMED_STORE_GATE"
assert qualification["body_receipt_sha256"] == run.lab.sha(OUT / "body-r1.json")
assert run.lab.sha(OUT / "build-r1.json") == qualification["build_receipt_sha256"]
binary = ROOT / "artifacts/experimental/q4-folded-ffn-probe"
expected = qualification["binary_sha256"]
assert run.lab.sha(binary) == expected
build = qualification
for name, digest in build["probe_sources"].items():
    assert run.lab.sha(pathlib.Path(__file__).parent / name) == digest
manifest = load(ROOT / "results/switch-matrix/w-stable-r2-build.json")["source_files"]
for name, digest in manifest.items():
    if name.endswith((".cu", ".cuh", ".h", ".hpp")):
        assert run.lab.sha(ROOT / "source" / name) == digest
binding = OUT / (tag + "-binding.json")
assert not binding.exists()
record = {"status": "INCOMPLETE", "binary_sha256": expected,
          "protocol_sha256": run.lab.sha(OUT / "protocol-r1.json"),
          "qualification_sha256": run.lab.sha(OUT / "link-export-r1.json"),
          "runner_sha256": run.lab.sha(pathlib.Path(__file__)),
          "started_at": run.lab.now(), "performance_claim": False}
docker = run.lab.docker


def checked_docker(*args, **kwargs):
    if args[0] == "create":
        args = (*args, str(protocol["input_seed"]))
    if args[:2] != ("start", "-a"):
        return docker(*args, **kwargs)
    ident = args[2]
    item = load_container(ident)
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
    rows = [json.loads(s) for s in stdout.splitlines() if s.startswith("{")]
    correctness = next(row for row in rows if row["kind"] == "correctness")
    assert correctness["exact_bf16_all_outputs"] and correctness["exact_fp32_planes"]
    assert correctness["output_canaries"] and correctness["independent_fp64_samples"] == 80
    assert correctness["invalid_api_guards"] == "PASS"
    assert correctness["input_seed"] == protocol["input_seed"]
    warmups = [r for r in rows if r["kind"] == "warmup"]
    samples = [r for r in rows if r["kind"] == "sample"]
    timings = [r for r in rows if r["kind"] == "timing"]
    assert len(warmups) == 4 and len(samples) == 12 and len(timings) == 3
    assert [r["order"] for r in timings] == protocol["orders"]
    assert [("B" if r["candidate"] else "A") for r in warmups] == protocol["warmup_order"]
    assert all(r["repetitions"] == 32 and r["ms"] > 0 for r in warmups + samples)
    for block, timing in enumerate(timings):
        part = [r for r in samples if r["block"] == block]
        assert [r["sample"] for r in part] == list(range(4))
        assert ''.join("AB"[r["arm"]] for r in part) == timing["order"]
        for arm, field in ((0, "baseline_ms"), (1, "candidate_ms")):
            actual = sum(r["ms"] for r in part if r["arm"] == arm) / 2
            assert abs(actual - timing[field]) < 1e-8
    record["correctness"] = correctness
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def load_container(ident):
    return json.loads(docker("inspect", ident).stdout)[0]


run.OUT_SUBDIR = "q4-folded-ffn"
run.PROBE_NAME = "q4-folded-ffn-probe"
run.lab.OWNER = "ninfer-q4-folded-ffn"
run.lab.docker = checked_docker
try:
    run.main()
    receipt = load(OUT / (tag + "-receipt.json"))
    assert receipt["status"] == "PASS" and receipt["production_config_unchanged"]
    assert receipt["owned_container_removed"] and receipt["reservation_released"]
    assert not receipt.get("cleanup_failure")
    record["status"] = "PASS_EXPORTED_OPERATOR_DISCRIMINATION_BINDING"
    record["screen_receipt_sha256"] = run.lab.sha(OUT / (tag + "-receipt.json"))
finally:
    run.lab.docker = docker
    record["finished_at"] = run.lab.now()
    with binding.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
