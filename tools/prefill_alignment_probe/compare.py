import hashlib
import json
import pathlib
import struct


ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = ROOT / "results/prefill-alignment"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def token_hash(tokens):
    return hashlib.sha256(struct.pack("<" + "i" * len(tokens), *tokens)).hexdigest()


def snapshot_components(snapshot):
    result = dict(snapshot["components"])
    for group in ("main_kv", "mtp_kv"):
        for plane in snapshot[group]["planes"]:
            key = group + "." + str(plane["plane_index"])
            assert key not in result
            result[key] = plane
    return result


def validate(report, arm, data, expected_anchor):
    errors = []
    if report["status"] != "complete":
        errors.append("probe_status=" + report["status"])
    assert report["schema"] == "ninfer.prefill-alignment-state.v1"
    assert report["input_contract"] == "raw_token_ids" and not report["state_dump_written"]
    assert report["native_logits_reprojection_equal"] is True
    assert report["actual_prefill_chunk"] == arm["expected_chunk"]
    assert report["actual_context_capacity"] == report["actual_kv_capacity"] == 65536
    if report["status"] == "complete":
        assert report["cleanup"] == "consumed" and report["forced_execution_width"] == 1
    tokens, forced = data["token_ids"], data["forced_tokens"]
    assert report["token_ids_sha256"] == token_hash(tokens)
    assert report["forced_tokens_sha256"] == token_hash(forced)
    assert report["prompt_tokens"] == len(tokens) and report["forced_tokens"] == forced
    chunk = arm["expected_chunk"]
    expected_chunks = [chunk] * (len(tokens) // chunk)
    if len(tokens) % chunk:
        expected_chunks.append(len(tokens) % chunk)
    assert report["actual_chunks"] == expected_chunks
    profile = report["profile"]
    assert profile["max_context"] == 65536 and profile["max_concurrency"] == 1
    assert profile["requested_prefill_chunk"] == 1024 and profile["draft_tokens"] == 3
    assert profile["backend"] == profile["kv_storage"] == 1 and profile["proposal_head"] == 0
    assert profile["context_cache"] is False and profile["device_profile"] == "auto"
    assert profile["enable_vision"] and profile["vision_residency"] == "overlay"
    assert profile["host_kv_mib"] == profile["host_state_slots"] == 0
    assert profile["prefill_cublas"] is False and profile["gdn_state_fp16"] is False
    assert report["routes"]["mode"] == "auto" and report["routes"]["multiprocessors"] == 82
    if report["prefill_anchor"] != expected_anchor:
        errors.append("sampled_anchor_differs")
    for phase, expected in (("prefill", tokens + [report["prefill_anchor"]]),
                            ("continuation", tokens + [report["prefill_anchor"]] + forced)):
        if phase not in report["snapshots"]:
            errors.append("missing_snapshot=" + phase)
            continue
        snapshot = report["snapshots"][phase]
        frontier = len(expected) - 1
        assert snapshot["execution_frontier"] == frontier
        assert snapshot["ledger_frontier"] == len(expected)
        assert snapshot["text_kv_valid"] == snapshot["mtp_kv_valid"] == frontier
        assert snapshot["ledger_sha256"] == token_hash(expected)
        assert snapshot["physical_prefix_sha256"] == token_hash(expected[:-1])
        assert snapshot["unexecuted_anchor"] == expected[-1]
        assert snapshot["tail_hidden_valid"] and not snapshot["state_fork_pending"]
        expected_drafts = 3 if phase == "prefill" else 0
        assert snapshot["mtp_draft_count"] == expected_drafts
        assert len(snapshot["mtp_drafts"]) == expected_drafts
        assert snapshot["gdn_layers"] == 48
        assert snapshot["sampler"]["token_counts_active"] is True
        assert all(snapshot["sampler"][key] == value for key, value in data["sampling"].items())
        names = {"conv." + str(i) for i in range(48)} | {"gdn." + str(i) for i in range(48)}
        names |= {"continuation_hidden", "tail_hidden", "target_logits", "token_counts"}
        assert set(snapshot["components"]) == names
        assert len(snapshot["main_kv"]["planes"]) == 64
        assert len(snapshot["mtp_kv"]["planes"]) == 4
        for group in ("main_kv", "mtp_kv"):
            assert snapshot[group]["frontier"] == frontier
            assert snapshot[group]["logical_pages"] == (frontier + 63) // 64
        components = snapshot_components(snapshot)
        assert len(components) == 168
        for component in components.values():
            assert component["bytes"] > 0 and len(component["sha256"]) == 64
            if component["dtype"] in {"fp32", "fp16", "bf16"}:
                assert component["numeric"]["nonfinite"] == 0
    return errors


def differences(reference, candidate):
    result = []
    for phase in reference["snapshots"]:
        if phase not in candidate["snapshots"]:
            result.append({"phase": phase, "missing_snapshot": True})
            continue
        a = snapshot_components(reference["snapshots"][phase])
        b = snapshot_components(candidate["snapshots"][phase])
        assert set(a) == set(b)
        changed = []
        for name in sorted(a):
            assert all(a[name][key] == b[name][key] for key in ("bytes", "dtype", "shape"))
            if a[name]["sha256"] != b[name]["sha256"]:
                changed.append(name)
        result.append({"phase": phase, "components_checked": len(a), "different_components": changed})
    return result


def main():
    protocol_path = OUT / "short-r1-execution-protocol.json"
    protocol = load(protocol_path)
    data_path = OUT / "raw-short-r1-effective-input.json"
    assert sha(data_path) == protocol["effective_input_sha256"]
    data = load(data_path)
    reports, receipts, bindings = [], [], []
    binary = load(OUT / "export-r1.json")["binary_sha256"]
    for arm in protocol["order"]:
        tag = arm["tag"]
        receipt_path = OUT / (tag + "-receipt.json")
        output_path = OUT / (tag + "-output.jsonl")
        receipt = load(receipt_path)
        assert receipt["owned_container_removed"] and receipt["reservation_released"]
        assert receipt["production_config_unchanged"] and not receipt.get("cleanup_failure")
        assert receipt["binary_sha256"] == receipt["running_executable_sha256"] == binary
        assert receipt["protocol_sha256"] == sha(protocol_path)
        assert receipt["input_sha256"] == protocol["effective_input_sha256"]
        assert receipt["output_sha256"] == sha(output_path)
        rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()
                if line.startswith("{")]
        assert len(rows) == 1
        if rows[0]["status"] == "complete":
            assert receipt["status"] == "RAN_PENDING_STATE_COMPARISON" and receipt["exit_code"] == 0
        reports.append(rows[0])
        receipts.append(receipt)
        bindings.append({"tag": tag, "receipt_sha256": sha(receipt_path),
                         "output_sha256": sha(output_path)})
    assert len({receipt["container_id"] for receipt in receipts}) == 3
    baseline, candidate, repeated = reports
    errors = [validate(report, arm, data, baseline["prefill_anchor"])
              for report, arm in zip(reports, protocol["order"])]
    assert all(report["profile"] == baseline["profile"] and report["routes"] == baseline["routes"]
               and report["artifact_id"] == baseline["artifact_id"]
               and report["positions_sha256"] == baseline["positions_sha256"] for report in reports)
    control = differences(baseline, repeated)
    treatment = differences(baseline, candidate)
    for phase in baseline["snapshots"]:
        assert all(report["snapshots"][phase]["mtp_drafts"] ==
                   baseline["snapshots"][phase]["mtp_drafts"] for report in reports)
    control_equal = not errors[0] and not errors[2] and all(
        not row.get("different_components") and not row.get("missing_snapshot") for row in control)
    candidate_equal = control_equal and not errors[1] and all(
        not row.get("different_components") and not row.get("missing_snapshot") for row in treatment)
    print(json.dumps({"status": "PASS_EXACT_STATE_GATE" if candidate_equal else "STATE_GATE_NOT_PASSED",
                      "scope": "raw4096_prefill_and_forced16_committed_state_only",
                      "performance_claim": False, "timing_allowed": False,
                      "long_state_gate_allowed": candidate_equal,
                      "control_deterministic": control_equal, "candidate_exact": candidate_equal,
                      "validation_errors": errors, "control_differences": control,
                      "candidate_differences": treatment, "bindings": bindings,
                      "protocol_sha256": sha(protocol_path), "binary_sha256": binary,
                      "analyzer_sha256": sha(pathlib.Path(__file__)),
                      "anchor": baseline["prefill_anchor"]}, indent=2))


if __name__ == "__main__":
    main()
