import argparse
import datetime
import hashlib
import json
import math
import pathlib
import re
import statistics

import profile_long_context as lab
import qualify_switch_matrix as qualification


OUT = lab.ROOT / "results/switch-matrix"


def analyze(records, protocol, protocol_sha, control_tag, control_sha, allow_recipe_rebuilds=False):
    assert len(records) == len(protocol["screen_schedule"]) == 8
    assert [r["arm"] for r in records] == protocol["screen_schedule"]
    categories = ["ru", "en", "code", "json"]
    baseline = records[0]
    previous_finish = None
    binaries = {}
    warmup = None
    command = None
    for record in records:
        assert record["status"] == "SCREENED" and record["mode"] == "screen"
        assert record["exploratory"] and not record["performance_claim"]
        assert record["owned_container_removed"] and record["reservation_released"]
        assert record["production_config_unchanged"] and not record.get("cleanup_failure")
        assert record["protocol_sha256"] == protocol_sha
        assert record["control_tag"] == control_tag
        if record is not baseline:
            assert record["control_receipt_sha256"] == control_sha
        argv = record["command"]
        log_index = argv.index("--request-log-jsonl") + 1
        normalized = argv[:log_index] + ["<request-log>"] + argv[log_index + 1:]
        if command is None:
            command = normalized
        assert normalized == command
        for key, value in {"--max-context": "65536", "--kv-capacity": "65536",
                           "--max-concurrency": "1", "--kv-dtype": "int8", "--spec": "mtp",
                           "--draft-tokens": "3", "--prefill-chunk": "1024"}.items():
            assert argv.count(key) == 1 and argv[argv.index(key) + 1] == value
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
        previous_binary = binaries.setdefault(arm, binary)
        assert allow_recipe_rebuilds or previous_binary == binary
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
            assert row["usage"]["completion_tokens"] == 512
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
            "artifact_scope": "fixed_source_recipe" if allow_recipe_rebuilds else "same_executable_per_arm",
            "interpretation": "Descriptive screening only; candidate selection is not confirmation"}


def validate_evidence(records):
    builds = [json.loads(path.read_text(encoding="utf-8")) for path in OUT.glob("*-build.json")]
    cache = (lab.ROOT / "artifacts/experimental/CMakeCache.txt").read_bytes()
    inputs = set()
    for record in records:
        def option(match):
            name = match[1].decode()
            assert name in record["build_options"]
            return match[1] + b":BOOL=" + (b"ON" if record["build_options"][name] else b"OFF")
        expected_cache = re.sub(rb"^(NINFER_\w+):BOOL=(?:ON|OFF)$", option, cache, flags=re.M)
        assert hashlib.sha256(expected_cache).hexdigest() == record["export"]["files"]["CMakeCache.txt"]
        related = {}
        for key in ("qualification_receipt", "route_receipt"):
            binding = record[key]
            assert pathlib.Path(binding["file"]).name == binding["file"]
            path = OUT / binding["file"]
            assert lab.sha(path) == binding["sha256"]
            related[key] = json.loads(path.read_text(encoding="utf-8"))
            assert related[key]["status"] == "PASS"
        gate, route = related["qualification_receipt"], related["route_receipt"]
        assert gate["schema"] == "ninfer-switch-matrix-q6-qualification-v2"
        assert gate["arm"] == record["arm"] and gate["outputs_exact_match"]
        assert gate["export"]["files"] == record["export"]["files"]
        assert route["executable_sha256"] == record["running_executable_sha256"]
        matches = [build for build in builds
                   if build.get("status") == "BUILT_EXPORTED_NOT_GPU_QUALIFIED"
                   and build.get("arm") == record["arm"]
                   and build.get("export_sha256") == record["export"]["files"]
                   and build.get("source_manifest_sha256") == record["export"]["source_manifest_sha256"]]
        assert matches
        for build in matches:
            assert build["cuda_symbol_seeds"] == ["ninfer-ggml_vec_q4_k", "ninfer-ggml_mmq_q6_k"]
            assert build["options"] == record["build_options"]
            compiled = qualification.build_inputs_sha(build)
            assert compiled == gate["compiled_inputs_sha256"]
            inputs.add(compiled)
    assert len(inputs) == 1
    return inputs.pop()


def validate_rebuild_policy(name, protocol_sha, compiled):
    path = OUT / name
    assert path.parent == OUT and path.suffix == ".json"
    policy = json.loads(path.read_text(encoding="utf-8"))
    assert policy["schema"] == "ninfer-exploratory-rebuild-policy-v1"
    assert policy["protocol_sha256"] == protocol_sha
    assert policy["compiled_inputs_sha256"] == compiled
    assert policy["scope"] == "exploratory_fixed_source_recipe"
    assert policy["per_executable_qualification_and_route_required"]
    assert policy["selection_rules_unchanged"]
    assert not policy["historical_binary_equivalence_claimed"]
    assert policy["confirmation_requires_independent_qualified_fixed_recipe"]
    binding = policy["diagnostic"]
    diagnostic = OUT / binding["file"]
    assert diagnostic.parent == OUT and lab.sha(diagnostic) == binding["sha256"]
    evidence = json.loads(diagnostic.read_text(encoding="utf-8"))
    assert evidence["status"] == "PASS_NONALLOC_STRTAB_VARIATION_DIAGNOSIS"
    assert evidence["compiled_inputs_sha256"] == compiled
    inventories = []
    for filename, digest in evidence["inventories"].items():
        inventory = OUT / filename
        assert inventory.parent == OUT and lab.sha(inventory) == digest
        inventories.append(json.loads(inventory.read_text(encoding="utf-8")))
    assert len(inventories) == 2
    before, after = inventories
    for kind in ("object", "serve"):
        left, right = before[kind]["sections"], after[kind]["sections"]
        assert left.keys() == right.keys()
        assert {key for key in left if left[key] != right[key]} == {".strtab"}
        assert left[".strtab"]["flags"] == right[".strtab"]["flags"] == 0
    assert before["cubins"] and before["cubins"] == after["cubins"]
    return {"file": path.name, "sha256": lab.sha(path), "limitation": policy["limitation"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tags", nargs=8)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rebuild-policy")
    args = parser.parse_args()
    assert all(pathlib.Path(tag).name == tag for tag in args.tags)
    paths = [OUT / (tag + "-runtime.json") for tag in args.tags]
    protocol_path = OUT / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    compiled = validate_evidence(records)
    policy = validate_rebuild_policy(args.rebuild_policy, lab.sha(protocol_path), compiled) if args.rebuild_policy else None
    result = analyze(records, protocol, lab.sha(protocol_path), args.tags[0], lab.sha(paths[0]), policy is not None)
    if policy:
        result["rebuild_policy"] = policy
    result["compiled_inputs_sha256"] = compiled
    result["inputs"] = {path.name: lab.sha(path) for path in paths}
    output = OUT / args.output
    assert output.parent == OUT and output.suffix == ".json"
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
