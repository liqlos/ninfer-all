import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "source/tools/ffn_pair_probe"))
sys.path.insert(0, str(ROOT / "source/tools/lab"))
import run
import run_switch_matrix as matrix

out = ROOT / "results/q4-biased-dot"
tag = sys.argv[1] if len(sys.argv) == 2 else "screen"
assert re.fullmatch(r"[a-z0-9-]+", tag)
binding = out / (tag + "-binding.json")
assert not binding.exists()
load = lambda path: json.loads(path.read_text(encoding="utf-8"))
export = load(out / "export-r1.json")
build = load(out / "build-r1.json")
body = load(out / "body-r2.json")
assert build["status"] == "CPU_BUILT_NOT_GPU_QUALIFIED"
assert export["status"] == "EXPORTED_NOT_GPU_QUALIFIED"
assert body["status"] == "PASS_STATIC_CUBIN_BODY_QUALIFICATION"
assert export["build_receipt_sha256"] == run.lab.sha(out / "build-r1.json")
assert export["body_receipt_sha256"] == run.lab.sha(out / "body-r2.json")
for name in ("main.cu", "build.sh", "generate.py"):
    assert run.lab.sha(pathlib.Path(__file__).parent / name) == build["probe_sources"][name]
assert body["source_sha256"] == run.lab.sha(ROOT / "source/third_party/ggml-quants/src/ggml-cuda/mmq-vec-dot.cuh")
assert body["generated_header_sha256"] == build["generated_header_sha256"]
binary = ROOT / "artifacts/experimental/q4-biased-dot-probe"
assert run.lab.sha(binary) == export["binary_sha256"] == body["binary_sha256"]
source_receipt = ROOT / "results/switch-matrix/both-stable-r1-build.json"
for name, digest in load(source_receipt)["source_files"].items():
    if name.endswith((".cu", ".cuh", ".h", ".hpp")):
        assert run.lab.sha(ROOT / "source" / name) == digest
record = {"started_at": run.lab.now(), "binary_sha256": export["binary_sha256"],
          "runner_sha256": run.lab.sha(pathlib.Path(__file__)),
          "compiled_source_receipt_sha256": run.lab.sha(source_receipt),
          "export_receipt_sha256": run.lab.sha(out / "export-r1.json"),
          "generated_header_sha256": build["generated_header_sha256"],
          "bias_bits": "0x4b400000", "status": "INCOMPLETE", "performance_claim": False}
docker = run.lab.docker


def checked_docker(*args, **kwargs):
    if args[:2] == ("start", "-a"):
        assert "before_run" not in record
        record["before_run"] = matrix.serving_admission(None)
        assert run.lab.sha(binary) == record["binary_sha256"]
        item = json.loads(docker("inspect", args[2]).stdout)[0]
        assert item["Config"]["Entrypoint"] == ["/lab/artifacts/experimental/q4-biased-dot-probe"]
        assert any(m["Destination"] == "/lab" and m["Source"] == run.lab.WSL_ROOT
                   and not m["RW"] for m in item["Mounts"])
        record["mounted_container_id"] = item["Id"]
    result = docker(*args, **kwargs)
    if args[:2] == ("start", "-a"):
        record["after_run"] = matrix.serving_admission(None)
        assert run.lab.sha(binary) == record["binary_sha256"]
        rows = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
        correctness = next(row for row in rows if row["kind"] == "correctness")
        assert correctness["exact_fp32_planes"] is True
        assert correctness["exact_bf16_all_outputs"] is True
        assert correctness["output_canaries"] is True
        assert correctness["independent_fp64_samples"] == 40
        assert correctness["invalid_api_guards"] == "PASS"
        record["correctness"] = correctness
    return result


run.OUT_SUBDIR = "q4-biased-dot"
run.PROBE_NAME = "q4-biased-dot-probe"
run.lab.OWNER = "ninfer-q4-biased-dot"
run.lab.docker = checked_docker
try:
    run.main()
    receipt = out / (tag + "-receipt.json")
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
    with binding.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
