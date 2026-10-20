import hashlib
import json
import math
import pathlib
import statistics
import sys

import scipy
from scipy.stats import t


ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = ROOT / "results/q6-dense-entry"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    protocol_path = OUT / "confirmation-r2-protocol.json"
    protocol = load(protocol_path)
    assert protocol["status"] == "FIXED_BEFORE_CONFIRMATION"
    assert len(protocol["tags"]) == 6 and len(protocol["cases"]) == 3
    export = load(OUT / "export-r2.json")
    result = {
        "scope": "operator_only_not_request_or_runtime_integration",
        "protocol": protocol,
        "protocol_sha256": sha(protocol_path),
        "analyzer_sha256": sha(pathlib.Path(__file__)),
        "python": sys.version,
        "scipy": scipy.__version__,
        "binary_sha256": export["binary_sha256"],
        "launches": [],
        "endpoints": {},
        "no_outlier_removal": True,
        "selection_data_excluded": True,
    }
    values = {case: [] for case in protocol["cases"]}
    containers = set()
    for tag in protocol["tags"]:
        receipt_path = OUT / (tag + "-receipt.json")
        binding_path = OUT / (tag + "-binding.json")
        output_path = OUT / (tag + ".jsonl")
        receipt, binding = load(receipt_path), load(binding_path)
        assert receipt["status"] == "PASS" and binding["status"] == "PASS_OPERATOR_SCREEN_BINDING"
        assert receipt["binary_sha256"] == binding["binary_sha256"] == result["binary_sha256"]
        assert binding["screen_receipt_sha256"] == sha(receipt_path)
        assert binding["export_receipt_sha256"] == sha(OUT / "export-r2.json")
        assert receipt["owned_container_removed"] and receipt["reservation_released"]
        assert receipt["production_config_unchanged"] and not receipt.get("cleanup_failure")
        assert receipt["container_id"] not in containers
        containers.add(receipt["container_id"])
        assert protocol["created_at"] < receipt["started_at"]
        rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
        timings = [r for r in rows if r["kind"] == "timing"]
        assert len(timings) == 9 and len(binding["correctness"]) == 3
        for case in protocol["cases"]:
            blocks = [r for r in timings if r["case"] == case]
            assert len(blocks) == 3 and {r["block"] for r in blocks} == {0, 1, 2}
            for block in blocks:
                assert block["repetitions_per_sample"] == 32 and block["warmup_samples"] == 4
                outer = (block["block"] + int("candidate-first" in tag)) % 2 == 0
                assert block["order"] == ("ABBA" if outer else "BAAB")
                assert all(math.isfinite(block[k]) and block[k] > 0
                           for k in ("baseline_ms", "candidate_ms"))
            values[case].append(statistics.mean(math.log(r["baseline_ms"] / r["candidate_ms"])
                                                for r in blocks))
        result["launches"].append({"tag": tag, "receipt_sha256": sha(receipt_path),
                                   "binding_sha256": sha(binding_path),
                                   "output_sha256": sha(output_path), "timings": timings})
    critical = float(t.ppf(1.0 - 0.05 / (2 * len(values)), 5))
    for case, logs in values.items():
        mean = statistics.mean(logs)
        se = statistics.stdev(logs) / math.sqrt(len(logs))
        lower, upper = mean - critical * se, mean + critical * se
        result["endpoints"][case] = {
            "launch_log_ratios": logs, "n": len(logs), "df": 5, "t_critical": critical,
            "familywise_coverage": 0.95, "multiplicity": "Bonferroni_3_predeclared_endpoints",
            "speed_increase_percent": 100 * math.expm1(mean),
            "speed_increase_ci_percent": [100 * math.expm1(v) for v in (lower, upper)],
            "time_decrease_percent": -100 * math.expm1(-mean),
            "time_decrease_ci_percent": [-100 * math.expm1(-v) for v in (lower, upper)],
            "positive_operator_effect_confirmed": lower > 0,
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
