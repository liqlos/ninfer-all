import argparse
import datetime
import json
import math
import pathlib
import statistics

import profile_long_context as lab


OUT = lab.ROOT / "results/switch-matrix"


def analyze(records, protocol, protocol_sha):
    assert len(records) == len(protocol["screen_schedule"]) == 8
    assert [r["arm"] for r in records] == protocol["screen_schedule"]
    categories = ["ru", "en", "code", "json"]
    baseline = records[0]
    previous_finish = None
    binaries = {}
    warmup = None
    for record in records:
        assert record["status"] == "SCREENED" and record["mode"] == "screen"
        assert record["exploratory"] and not record["performance_claim"]
        assert record["owned_container_removed"] and record["reservation_released"]
        assert record["production_config_unchanged"] and not record.get("cleanup_failure")
        assert record["protocol_sha256"] == protocol_sha
        for key in ("model_sha256", "bench_sha256", "prompts_sha256"):
            assert record[key] == protocol[key]
        started = datetime.datetime.fromisoformat(record["started_at"])
        finished = datetime.datetime.fromisoformat(record["finished_at"])
        assert started < finished
        assert previous_finish is None or previous_finish < started
        previous_finish = finished
        arm = record["arm"]
        binary = record["running_executable_sha256"]
        assert binary == record["export"]["files"]["ninfer-serve"]
        assert binaries.setdefault(arm, binary) == binary
        expected_on = set(protocol["common_on"] + protocol["arms"][arm])
        actual_on = {name for name, value in record["build_options"].items()
                     if value and name not in {"NINFER_BUILD_APPS", "NINFER_BUILD_BENCHMARKS"}}
        assert actual_on == expected_on
        observed_warmup = {key: record["warmup"][key]
                           for key in ("request_sha256", "usage", "output_sha256", "finish_reason")}
        assert observed_warmup["usage"]["completion_tokens"] == 32
        if warmup is None:
            warmup = observed_warmup
        assert observed_warmup == warmup
        assert [row["category"] for row in record["samples"]] == categories
        for row, control in zip(record["samples"], baseline["samples"]):
            assert math.isfinite(row["elapsed_seconds"]) and row["elapsed_seconds"] > 0
            for key in ("output_sha256", "usage", "finish_reason", "request_body_sha256"):
                assert row[key] == control[key]
            assert row["request_done"]["speculative"] == control["request_done"]["speculative"]
    scores = {}
    for arm in ("q4", "q6", "both"):
        positions = [i for i, record in enumerate(records) if record["arm"] == arm]
        halves = []
        for position, control_position in zip(positions, (0, 7)):
            candidate = records[position]
            control = records[control_position]
            values = [math.log(a["elapsed_seconds"] / b["elapsed_seconds"])
                      for a, b in zip(control["samples"], candidate["samples"])]
            halves.append({"mean_log_speed_ratio": statistics.mean(values),
                           "gain_percent": math.expm1(statistics.mean(values)) * 100,
                           "category_gain_percent": dict(zip(
                               categories, [math.expm1(value) * 100 for value in values]))})
        mean = statistics.mean(h["mean_log_speed_ratio"] for h in halves)
        gain = math.expm1(mean) * 100
        scores[arm] = {"halves": halves, "gain_percent": gain,
                       "eligible": gain > 0.3 and all(h["mean_log_speed_ratio"] > 0 for h in halves)}
    eligible = [arm for arm, score in scores.items() if score["eligible"]]
    selected = max(eligible, key=lambda arm: scores[arm]["gain_percent"]) if eligible else None
    return {"schema": "ninfer-switch-matrix-selection-v1", "exploratory": True,
            "performance_claim": False, "protocol_sha256": protocol_sha, "scores": scores,
            "selected_arm": selected, "confirmation_pairs": 6 if selected else 0,
            "confirmation_order": ["AB", "BA", "AB", "BA", "AB", "BA"] if selected else [],
            "interpretation": "Descriptive screening only; candidate selection is not confirmation"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tags", nargs=8)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    assert all(pathlib.Path(tag).name == tag for tag in args.tags)
    paths = [OUT / (tag + "-runtime.json") for tag in args.tags]
    protocol_path = OUT / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    result = analyze([json.loads(path.read_text(encoding="utf-8")) for path in paths],
                     protocol, lab.sha(protocol_path))
    result["inputs"] = {path.name: lab.sha(path) for path in paths}
    output = OUT / args.output
    assert output.parent == OUT and output.suffix == ".json"
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
