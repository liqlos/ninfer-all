import datetime
import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import run as controller


def validated(paths, protocol_path, protocol_sha, mode):
    protocol = controller.load(protocol_path)
    assert controller.lab.sha(protocol_path) == protocol_sha
    records = [controller.load(path) for path in paths]
    assert [r["arm"] for r in records] == protocol["order"]
    reference = records[0]
    assert reference["protocol_sha256"] == protocol_sha
    assert reference["build_receipt_sha256"] == controller.BUILD_SHA
    assert reference["model_sha256"] == protocol["model"]["sha256"]
    assert reference["bench_sha256"] == protocol["bench"]["sha256"]
    build = controller.load(controller.BUILD)
    prior_finish = None
    command = None
    for index, record in enumerate(records):
        assert record["status"] == "SCREENED" and record["mode"] == mode
        assert not record["diagnostic_only"] and not record["host_admissions_enabled"]
        assert record["owned_container_removed"] and record["reservation_released"]
        assert record["production_config_unchanged"] and not record.get("cleanup_failures")
        assert record["sequence_index"] == index and record["control_tag"] == reference["tag"]
        for key in ("protocol_sha256", "build_receipt_sha256", "model_sha256", "bench_sha256", "runner_sha256"):
            assert record[key] == reference[key]
        assert record["running_executable_sha256"] == record["executable_sha256"] == record["export"]["sha256"]
        assert record["executable_sha256"] == build["build"]["outputs"][record["arm"]]["sha256"]
        if index:
            assert record["reference_receipt_sha256"] == controller.lab.sha(paths[0])
        started = datetime.datetime.fromisoformat(record["started_at"])
        finished = datetime.datetime.fromisoformat(record["finished_at"])
        assert started < finished and (prior_finish is None or prior_finish < started)
        prior_finish = finished
        argv = record["command"][:]
        argv[argv.index("--request-log-jsonl") + 1] = "<request-log>"
        command = argv if command is None else command
        assert argv == command
        assert [r["name"] for r in record["samples"]] == [r["name"] for r in protocol["requests"]]
        for row, baseline in zip(record["samples"], reference["samples"]):
            assert math.isfinite(row["elapsed_seconds"]) and row["elapsed_seconds"] > 0
            for key in ("output_sha256", "usage", "finish_reason", "request_body_sha256"):
                assert row[key] == baseline[key]
            assert row["request_done"]["speculative"] == baseline["request_done"]["speculative"]
        if record["arm"] == "on":
            route_path = pathlib.Path(record["route_receipt"]["path"])
            assert controller.lab.sha(route_path) == record["route_receipt"]["sha256"]
            route = controller.load(route_path)
            assert route["status"] == "PASS"
            assert all(route[key] == record[key] for key in
                       ("executable_sha256", "protocol_sha256", "reference_receipt_sha256"))
    return protocol, records


def main():
    out = controller.OUT
    decision = out / "screen-r2-decision.json"
    assert not decision.exists()
    tags = ["r2-w1", "r2-c1", "r2-c2", "r2-w2"]
    paths = [out / (tag + "-runtime.json") for tag in tags]
    protocol, records = validated(paths, controller.PROTOCOL, controller.PROTOCOL_SHA, "screen")
    scores = {}
    for category, request in enumerate(protocol["requests"]):
        a = [records[i]["samples"][category]["elapsed_seconds"] for i in (0, 3)]
        b = [records[i]["samples"][category]["elapsed_seconds"] for i in (1, 2)]
        scores[request["name"]] = {
            "off_seconds": a, "on_seconds": b,
            "pair_time_reduction_percent": [100 * (x - y) / x for x, y in zip(a, b)],
            "ratio_of_sums_time_reduction_percent": 100 * (sum(a) - sum(b)) / sum(a),
        }
    rule = protocol["selection"]
    passed = (scores["JSON32K"]["ratio_of_sums_time_reduction_percent"] >=
              rule["JSON32K_min_ratio_of_sums_time_reduction_percent"]
              and all(x > 0 for x in scores["JSON32K"]["pair_time_reduction_percent"])
              and scores["RU4K"]["ratio_of_sums_time_reduction_percent"] >= -rule["RU4K_max_regression_percent"])
    result = {
        "status": "SELECTED_FOR_INDEPENDENT_CONFIRMATION" if passed else "REJECTED_AT_REQUEST_SCREEN",
        "created_at": controller.lab.now(), "selection_pass": passed,
        "protocol_sha256": controller.PROTOCOL_SHA,
        "receipts": {path.name: controller.lab.sha(path) for path in paths},
        "scores": scores, "selection": rule, "independent_pair_units": 2,
        "statistical_confirmation": False, "confidence_interval_claim": False,
        "historical_W_percent_not_added": True,
        "action": rule["action_if_pass"] if passed else rule["action_if_fail"],
    }
    with decision.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
