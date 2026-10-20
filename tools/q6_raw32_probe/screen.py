import json
import math
import pathlib
import re
import sys


ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "source/tools/ffn_pair_probe"))
sys.path.insert(0, str(ROOT / "source/tools/lab"))
import run
import run_switch_matrix as matrix


OUT = ROOT / "results/q6-raw32"
TAG = sys.argv[1] if len(sys.argv) == 2 else "screen"
assert re.fullmatch(r"[a-z0-9-]+", TAG)
binding_path = OUT / (TAG + "-binding.json")
assert not binding_path.exists()
load = lambda path: json.loads(path.read_text(encoding="utf-8"))
build = load(OUT / "build-r2.json")
body = load(OUT / "body-r2.json")
export = load(OUT / "export-r2.json")
assert build["status"] == "CPU_BUILT_NOT_GPU_QUALIFIED"
assert body["status"] == "PASS_STATIC_CUBIN_BODY_QUALIFICATION"
assert export["status"] == "EXPORTED_NOT_GPU_QUALIFIED"
assert export["build_receipt_sha256"] == run.lab.sha(OUT / "build-r2.json")
assert export["body_receipt_sha256"] == run.lab.sha(OUT / "body-r2.json")
for name, digest in build["probe_sources"].items():
    assert pathlib.Path(name).name == name
    assert run.lab.sha(pathlib.Path(__file__).parent / name) == digest
loader = ROOT / "source/third_party/ggml-quants/src/ggml-cuda/mmq-load-tiles.cuh"
assert body["source_sha256"] == run.lab.sha(loader)
assert body["generated_header_sha256"] == build["generated_header_sha256"]
binary = ROOT / "artifacts/experimental/q6-raw32-probe"
assert run.lab.sha(binary) == export["binary_sha256"] == body["binary_sha256"]
source_receipt = ROOT / "results/switch-matrix/w-stable-r2-build.json"
for name, digest in load(source_receipt)["source_files"].items():
    if name.endswith((".cu", ".cuh", ".h", ".hpp")):
        assert run.lab.sha(ROOT / "source" / name) == digest
record = {
    "started_at": run.lab.now(), "status": "INCOMPLETE", "performance_claim": False,
    "binary_sha256": export["binary_sha256"],
    "runner_sha256": run.lab.sha(pathlib.Path(__file__)),
    "shared_runner_sha256": run.lab.sha(ROOT / "source/tools/ffn_pair_probe/run.py"),
    "compiled_source_receipt_sha256": run.lab.sha(source_receipt),
    "export_receipt_sha256": run.lab.sha(OUT / "export-r2.json"),
}
expected = {(17408, 5120, 1280, "accumulate"), (5120, 10240, 1280, "store"),
            (5120, 1024, 1280, "store")}
docker = run.lab.docker


def checked_docker(*args, **kwargs):
    if args[:2] == ("start", "-a"):
        assert "before_run" not in record
        record["before_run"] = matrix.serving_admission(None)
        assert run.lab.sha(binary) == record["binary_sha256"]
        item = json.loads(docker("inspect", args[2]).stdout)[0]
        assert item["Config"]["Entrypoint"] == ["/lab/artifacts/experimental/q6-raw32-probe"]
        assert any(m["Destination"] == "/lab" and m["Source"] == run.lab.WSL_ROOT
                   and not m["RW"] for m in item["Mounts"])
        record["mounted_container_id"] = item["Id"]
    result = docker(*args, **kwargs)
    if args[:2] == ("start", "-a"):
        record["after_run"] = matrix.serving_admission(None)
        assert run.lab.sha(binary) == record["binary_sha256"]
        rows = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
        correct = [row for row in rows if row["kind"] == "correctness"]
        assert len(correct) == len(expected)
        assert {(r["k"], r["rows"], r["columns"], r["output_mode"]) for r in correct} == expected
        for row in correct:
            assert row["status"] == "PASS" and row["exact_fp32_all_outputs"]
            assert row["exact_bf16_all_outputs"] and row["output_canaries"]
            assert row["independent_fp64_samples"] >= 40
            assert row["invalid_api_guards"] == "PASS"
            if row["output_mode"] == "accumulate":
                assert row["accumulate_matches_fp32_carry"] is True
            timing = [r for r in rows if r["kind"] == "timing" and r["case"] == row["case"]]
            assert len(timing) == 3 and {r["block"] for r in timing} == {0, 1, 2}
            for sample in timing:
                assert sample["order"] in {"ABBA", "BAAB"}
                assert sample["repetitions_per_sample"] > 0
                assert all(math.isfinite(sample[k]) and sample[k] > 0
                           for k in ("baseline_ms", "candidate_ms"))
        record["correctness"] = correct
    return result


run.OUT_SUBDIR = "q6-raw32"
run.PROBE_NAME = "q6-raw32-probe"
run.FIXTURES = [
    (ROOT / "fixtures/operators/q6-full-k17408.bin",
     "a381c6be36f572170ccc9c7e4a3f8b25e5061e4eb633649399f3ec9759cb43e2"),
    (ROOT / "fixtures/weights/q6-head-full-k5120.bin",
     "2155d6e1f7c5549077438c4760e6e150945dd5e406484b29162dff7b03d05da0"),
]
run.PROBE_INPUTS = ["/lab/fixtures/operators", "/lab/fixtures/weights"]
run.EXPECTED_TIMING_BLOCKS = 9
run.lab.OWNER = "ninfer-q6-raw32"
run.lab.docker = checked_docker
try:
    run.main()
    receipt = OUT / (TAG + "-receipt.json")
    completed = load(receipt)
    assert completed["status"] == "PASS" and completed["production_config_unchanged"]
    assert completed["owned_container_removed"] and completed["reservation_released"]
    assert not completed.get("cleanup_failure")
    assert "before_run" in record and "after_run" in record
    record["screen_receipt_sha256"] = run.lab.sha(receipt)
    record["status"] = "PASS_OPERATOR_SCREEN_BINDING"
finally:
    run.lab.docker = docker
    record["finished_at"] = run.lab.now()
    with binding_path.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
