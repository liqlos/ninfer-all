import json
import math
import pathlib
import statistics
import subprocess
import sys

import analyze
import run as controller


OUT = controller.OUT
TAGS = [f"confirm-r1-{i:02d}" for i in range(1, 17)]
PATHS = [OUT / (tag + "-runtime.json") for tag in TAGS]
ROUTE = OUT / "confirm-r1-route.json"
DECISION = OUT / "confirmation-r1-decision.json"


def reduction(log_ratio):
    return -100 * math.expm1(-log_ratio)


def scores(records, protocol):
    result = {}
    for category, request in enumerate(protocol["requests"]):
        pairs = []
        for index, order in enumerate(protocol["pairs"]):
            pair = records[2 * index:2 * index + 2]
            a = next(r for r in pair if r["arm"] == "off")["samples"][category]["elapsed_seconds"]
            b = next(r for r in pair if r["arm"] == "on")["samples"][category]["elapsed_seconds"]
            pairs.append({"pair": index + 1, "order": "AB" if order[0] == "off" else "BA",
                          "off_seconds": a, "on_seconds": b, "log_ratio": math.log(a / b),
                          "time_reduction_percent": 100 * (a - b) / a})
        values = [p["log_ratio"] for p in pairs]
        mean, sd = statistics.mean(values), statistics.stdev(values)
        half = protocol["selection"]["student_t_critical"] * sd / math.sqrt(len(values))
        subgroups = {order: reduction(statistics.mean(p["log_ratio"] for p in pairs if p["order"] == order))
                     for order in ("AB", "BA")}
        subgroups.update(first_four=reduction(statistics.mean(values[:4])),
                         last_four=reduction(statistics.mean(values[4:])))
        result[request["name"]] = {
            "pairs": pairs, "mean_log_ratio": mean, "sd_log_ratio": sd,
            "point_time_reduction_percent": reduction(mean),
            "CI95_time_reduction_percent": [reduction(mean - half), reduction(mean + half)],
            "secondary_ratio_of_sums_time_reduction_percent":
                100 * (sum(p["off_seconds"] for p in pairs) - sum(p["on_seconds"] for p in pairs)) /
                sum(p["off_seconds"] for p in pairs),
            "descriptive_subgroup_point_reduction_percent": subgroups,
        }
    return result


def decision():
    assert not DECISION.exists()
    protocol, records = analyze.validated(PATHS, controller.CONFIRM_PROTOCOL,
                                          controller.CONFIRM_PROTOCOL_SHA, "confirm")
    assert len(records) == 16 and len(protocol["pairs"]) == 8
    values = scores(records, protocol)
    rule = protocol["selection"]
    checks = {
        "JSON_CI_lower_positive": values["JSON32K"]["CI95_time_reduction_percent"][0] > 0,
        "JSON_point_at_least_0_5_percent": values["JSON32K"]["point_time_reduction_percent"] >=
            rule["JSON_point_time_reduction_min_percent"],
        "RU_mean_regression_within_0_3_percent": values["RU4K"]["point_time_reduction_percent"] >=
            -rule["RU_point_regression_max_percent"],
    }
    result = {
        "status": "CONFIRMED_JSON_ENDPOINT" if all(checks.values()) else "NOT_SELECTED_AFTER_FIXED_CONFIRMATION",
        "created_at": controller.lab.now(), "checks": checks, "scores": values,
        "protocol_sha256": controller.CONFIRM_PROTOCOL_SHA,
        "receipts": {path.name: controller.lab.sha(path) for path in PATHS},
        "independent_pair_units": 8, "selection_data_included": False,
        "CI_assumptions": rule["CI_assumptions"],
        "RU_noninferiority_proven_by_mean_guard": False,
        "true_gain_at_least_0_5_percent_proven_by_point_gate": False,
        "historical_W_percent_not_added": True, "production_change": False,
        "action": rule["action_if_pass"] if all(checks.values()) else rule["action_if_fail"],
    }
    with DECISION.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result), flush=True)


def route_binding(protocol):
    old_path = OUT / "r2-on-profile-route.json"
    old = controller.load(old_path)
    old_protocol = controller.load(controller.PROTOCOL)
    fresh = controller.load(PATHS[0])
    previous = controller.load(OUT / "r2-w1-runtime.json")
    assert old["status"] == "PASS" and fresh["status"] == "SCREENED"
    assert old["executable_sha256"] == controller.load(controller.BUILD)["build"]["outputs"]["on"]["sha256"]
    for key in ("model", "bench", "image", "requests", "sampling", "runtime", "warmup"):
        assert old_protocol[key] == protocol[key]
    for a, b in zip(fresh["samples"], previous["samples"]):
        assert all(a[key] == b[key] for key in ("name", "output_sha256", "usage", "finish_reason", "request_body_sha256"))
        assert a["request_done"]["speculative"] == b["request_done"]["speculative"]
    derived = dict(old, protocol_sha256=controller.CONFIRM_PROTOCOL_SHA,
                   reference_receipt_sha256=controller.lab.sha(PATHS[0]),
                   reused_route_receipt_sha256=controller.lab.sha(old_path),
                   proof_reuse="Identical ON executable/model/requests/runtime/bench, new OFF output/workload exact. No new profiling or timing-data reuse.")
    with ROUTE.open("x", encoding="utf-8") as stream:
        json.dump(derived, stream, indent=2)


def main():
    assert sys.platform == "win32" and len(sys.argv) == 1 and not sys.flags.optimize
    driver_receipt = OUT / "confirmation-r1-driver.json"
    assert not driver_receipt.exists() and not ROUTE.exists() and not DECISION.exists()
    assert not any(PATHS[i].exists() or list(OUT.glob(TAGS[i] + "-*")) for i in range(16))
    assert controller.lab.sha(controller.CONFIRM_PROTOCOL) == controller.CONFIRM_PROTOCOL_SHA
    protocol = controller.load(controller.CONFIRM_PROTOCOL)
    assert protocol["pairs"] == [["off", "on"], ["on", "off"], ["on", "off"], ["off", "on"]] * 2
    assert protocol["order"] == [arm for pair in protocol["pairs"] for arm in pair]
    assert controller.load(OUT / "screen-r2-decision.json")["selection_pass"]
    record = {"status": "INCOMPLETE", "started_at": controller.lab.now(),
              "protocol_sha256": controller.CONFIRM_PROTOCOL_SHA, "completed": [],
              "driver_sha256": controller.lab.sha(pathlib.Path(__file__)),
              "runner_sha256": controller.lab.sha(pathlib.Path(controller.__file__)),
              "analyzer_sha256": controller.lab.sha(pathlib.Path(analyze.__file__))}
    try:
        for index, (tag, arm) in enumerate(zip(TAGS, protocol["order"])):
            if index == 1:
                route_binding(protocol)
            command = [sys.executable, "-B", str(pathlib.Path(controller.__file__)), arm, "confirm", tag, TAGS[0]]
            if arm == "on":
                command += ["--route", str(ROUTE)]
            with (OUT / (f"confirmation-controller-{index + 1:02d}.log")).open("x", encoding="utf-8") as log:
                child = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NO_WINDOW)
            launch = controller.load(PATHS[index])
            assert child.returncode == 0 and launch["status"] == "SCREENED", launch.get("failure")
            assert launch["mode"] == "confirm" and launch["sequence_index"] == index
            record["completed"].append({"tag": tag, "receipt_sha256": controller.lab.sha(PATHS[index])})
            print(json.dumps({"completed_launches": index + 1, "total": 16, "arm": arm,
                              "at": controller.lab.now()}), flush=True)
        decision()
        record["status"] = "COMPLETE_FIXED_SERIES"
    except BaseException as exc:
        record["failure"] = type(exc).__name__ + ": " + str(exc)
        raise
    finally:
        record["finished_at"] = controller.lab.now()
        with driver_receipt.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2)


if __name__ == "__main__":
    main()
