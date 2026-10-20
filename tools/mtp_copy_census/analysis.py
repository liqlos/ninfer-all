import argparse
import hashlib
import json
import pathlib


DOMAIN = 248077
MAX_SUFFIX = 256
MAX_HISTORY = 65536
POLICIES = (8, 16, 32)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def token_ids(values):
    require(isinstance(values, (list, tuple)) and
            all(type(value) is int and 0 <= value < DOMAIN for value in values),
            "Invalid native token IDs")
    return values


def copy_candidate(ledger, minimum=8, maximum=MAX_SUFFIX, history=MAX_HISTORY, full=True):
    token_ids(ledger)
    require(type(minimum) is int and type(maximum) is int and
            1 <= minimum <= maximum <= MAX_SUFFIX, "Invalid suffix bounds")
    require(type(history) is int and 1 <= history <= MAX_HISTORY, "Invalid history bound")
    require(type(full) is bool, "Invalid full-draft policy")
    size = len(ledger)
    length = min(size, maximum)
    if size < 2 or minimum > length:
        return None
    pattern = list(reversed(ledger[-length:]))
    prefix = [0] * length
    for index in range(1, length):
        matched = prefix[index - 1]
        while matched and pattern[matched] != pattern[index]:
            matched = prefix[matched - 1]
        if pattern[matched] == pattern[index]:
            matched += 1
        prefix[index] = matched
    oldest_end = max(size - 1 - history, 0)
    oldest_scan = max(oldest_end - length + 1, 0)
    matched = best_length = 0
    best_end = None
    scan_start = size - (4 if full else 2)
    for index in range(scan_start, oldest_scan - 1, -1):
        value = ledger[index]
        while matched and (matched == length or pattern[matched] != value):
            matched = prefix[matched - 1]
        if pattern[matched] == value:
            matched += 1
        candidate_end = index + matched - 1
        available = size - candidate_end - 1
        if (candidate_end >= oldest_end and matched >= minimum and matched > best_length and
                available >= (3 if full else 1)):
            best_length, best_end = matched, candidate_end
            if best_length == length:
                break
    if best_end is None:
        return None
    follow = best_end + 1
    count = min(3, size - follow)
    require(not full or count == 3, "Full candidate has insufficient known continuation")
    source_start = best_end + 1 - best_length
    require(source_start >= 0 and list(ledger[source_start:follow]) == list(ledger[-best_length:]),
            "Candidate suffix identity differs")
    return {"source_start": source_start, "source_end_exclusive": follow,
            "suffix_length": best_length, "candidate_ids": list(ledger[follow:follow + count])}


def common_prefix(left, right):
    count = 0
    for a, b in zip(left, right):
        if a != b:
            break
        count += 1
    return count


def local_comparison(ledger, generated_future, native_drafts, minimum=8):
    selected = copy_candidate(ledger, minimum=minimum)
    token_ids(generated_future)
    token_ids(native_drafts)
    require(len(generated_future) >= 4 and len(native_drafts) == 3,
            "Need a full K3 verification and four known future output tokens")
    result = {"copy": selected,
              "native_prefix_matches": common_prefix(native_drafts, generated_future[:3])}
    if selected is not None:
        candidate = selected["candidate_ids"]
        result["finalized_output_prefix_matches"] = common_prefix(candidate, generated_future[:3])
        result["finalized_output_match_difference"] = (
            result["finalized_output_prefix_matches"] - result["native_prefix_matches"])
        lower = upper = 3
        changed = None
        for index, (proposed, original, target) in enumerate(zip(candidate, native_drafts, generated_future)):
            if proposed != target:
                lower = upper = index
                break
            if proposed != original:
                lower, upper, changed = index + 1, 3, index
                break
        result.update(checkpoint_accepted_lower=lower, checkpoint_accepted_upper=upper,
                      checkpoint_licensed_difference_lower=lower - result["native_prefix_matches"],
                      checkpoint_licensed_difference_upper=upper - result["native_prefix_matches"],
                      first_new_prefix_position=changed,
                      acceptance_after_new_prefix_observed=False)
    return result


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def integer(value, minimum=0):
    require(type(value) is int and value >= minimum, "Invalid integer field")
    return value


def validate_capture(record):
    require(record["schema"] == 1 and record["kind"] == "request" and record["active"] is True,
            "Need an active native request capture")
    require(record["error"] == [], "Native collector reported an error")
    sampling = record["sampling"]
    require(all(sampling[k] == v for k, v in {"temperature": 0, "top_k": 1,
            "presence_penalty": 1.5, "frequency_penalty": 0}.items()), "Sampling contract differs")
    result = record["result"]
    prompt_count = integer(result["prompt_tokens"], 1)
    generated = token_ids(result["generated_token_ids"])
    require(len(generated) >= 4 and result["finish_reason"] == "output_limit",
            "This census requires an output-limit request with at least four tokens")
    initial = list(token_ids(record["first_ledger"]))
    initial_generated = len(initial) - prompt_count
    require(1 <= initial_generated < len(generated), "Initial native ledger does not cover a decode frontier")
    require(initial[prompt_count:] == list(generated[:initial_generated]), "Initial native output IDs differ")
    rows = record["rows"]
    require(isinstance(rows, list) and rows, "Missing native MTP round boundaries")
    total = initial[:prompt_count] + list(generated)
    ledger = list(initial)
    previous_size = len(initial)
    counts = {"rounds": 0, "drafted_tokens": 0, "accepted_tokens": 0, "fallback_steps": 0}
    accepted_positions = [0, 0, 0]
    validated = []
    for index, row in enumerate(rows):
        require(row["round"] == index + 1, "Duplicate, missing or reordered native round")
        delta = list(token_ids(row["committed_delta"]))
        if index == 0:
            require(not delta, "First round unexpectedly duplicates the initial ledger")
        else:
            require(1 <= len(delta) <= 4, "Invalid committed span between native rounds")
            ledger.extend(delta)
        e, s = integer(row["E"]), integer(row["S"])
        require(s == e + 1 == len(ledger) and row["mtp_kv_valid"] == e,
                "Native frontier/KV/ledger relation differs")
        require(ledger == total[:s] and row["anchor"] == ledger[-1], "Native ledger prefix or anchor differs")
        require(row["verify_window"] == row["draft_window"] == 3, "Fixed physical T4/K3 route differs")
        budget = integer(row["remaining_budget"], 1)
        require(budget == len(total) - s and e < 65536, "Native remaining budget or context differs")
        extent = integer(row["extent"])
        count = integer(row["mtp_draft_count"])
        drafts = token_ids(row["drafts"])
        require(count <= 3 and len(drafts) == count and
                extent == min(count, 3, budget - 1, 65536 - e - 1), "Native logical draft extent differs")
        future = total[s:]
        matches = common_prefix(drafts[:extent], future[:extent])
        next_size = integer(rows[index + 1]["S"]) if index + 1 < len(rows) else len(total)
        require(next_size - s == matches + 1, "Native acceptance does not replay to its committed ledger")
        if extent:
            counts["rounds"] += 1
            counts["drafted_tokens"] += extent
            counts["accepted_tokens"] += matches
            for position in range(matches):
                accepted_positions[position] += 1
        else:
            counts["fallback_steps"] += 1
        validated.append({"round": index + 1, "ledger": list(ledger), "generated_offset": s - prompt_count,
                          "extent": extent, "budget": budget, "drafts": list(drafts[:extent]),
                          "native_prefix_matches": matches,
                          "eligible_copy_first": index > 0 and extent == 3 and budget >= 4})
        require(index == 0 or s > previous_size, "Native frontier did not advance")
        previous_size = s
    spec = result["speculative"]
    require(spec["enabled"] is True and spec["backend"] == "mtp" and spec["draft_window"] == 3,
            "Native MTP result route differs")
    require(spec["adaptive"] is False and spec["window_transitions"] == 0 and
            all(value == 0 for value in spec["rounds_per_window"]), "Unexpected adaptive policy")
    require(all(spec[k] == value for k, value in counts.items()), "Native MTP counters do not replay")
    require(spec["accepted_per_position"] == accepted_positions, "Native acceptance histogram differs")
    require(all(result[k] == value for k, value in counts.items() if k != "fallback_steps"),
            "Native flat/result counters differ")
    require(record["validation"]["monotonic_prefix"] is True and
            record["validation"]["native_result_prefixes"] is True, "Collector validation incomplete")
    return {"prompt_tokens": prompt_count, "generated_tokens": len(generated),
            "initial_generated_tokens": initial_generated, "physical_MTP_calls": len(rows),
            "native_counters": counts, "native_accepted_per_position": accepted_positions,
            "rows": validated, "generated": list(generated)}


def analyze_capture(record, minimum=8):
    data = validate_capture(record)
    eligible = [row for row in data["rows"] if row["eligible_copy_first"]]
    samples = []
    for row in eligible:
        future = data["generated"][row["generated_offset"]:]
        compared = local_comparison(row["ledger"], future, row["drafts"], minimum)
        if compared["copy"] is not None:
            samples.append({"round": row["round"], "generated_offset": row["generated_offset"], **compared})
    return {"minimum_suffix": minimum, "maximum_suffix": MAX_SUFFIX, "history_limit": MAX_HISTORY,
            "full_three_known_tokens_only": True, "longest_then_most_recent": True,
            "eligible_baseline_checkpoints": len(eligible), "causal_copy_admissions": len(samples),
            "admission_fraction": len(samples) / len(eligible) if eligible else 0,
            "all_three_equal_finalized_output_continuation": sum(
                row["finalized_output_prefix_matches"] == 3 for row in samples),
            "checkpoint_licensed_increase_proven": sum(
                row["checkpoint_licensed_difference_lower"] > 0 for row in samples),
            "checkpoint_licensed_decrease_proven": sum(
                row["checkpoint_licensed_difference_upper"] < 0 for row in samples),
            "checkpoint_licensed_equal_proven": sum(
                row["checkpoint_licensed_difference_lower"] == row["checkpoint_licensed_difference_upper"] == 0
                for row in samples),
            "checkpoint_licensed_difference_lower_sum": sum(
                row["checkpoint_licensed_difference_lower"] for row in samples),
            "checkpoint_licensed_difference_upper_sum": sum(
                row["checkpoint_licensed_difference_upper"] for row in samples),
            "optimistic_avoided_head_calls_on_original_checkpoints": 3 * len(samples),
            "optimistic_avoided_AR_calls_on_original_checkpoints": 2 * len(samples),
            "new_trajectory_or_throughput_claim": False, "terminal_drain_saving_included": False,
            "first_ready_draft_round_excluded": True, "samples": samples}


def main():
    parser = argparse.ArgumentParser(description="Causal history-only copy census; no model execution or speed claim.")
    parser.add_argument("capture", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Immutable analysis output already exists")
    record = json.loads(args.capture.read_text(encoding="utf-8"))
    validated = validate_capture(record)
    result = {"schema": "mtp-copy-local-checkpoint-analysis-r1", "capture_sha256": sha(args.capture),
              "analyzer_sha256": sha(pathlib.Path(__file__)), "diagnostic_only": True,
              "capture_summary": {k: v for k, v in validated.items() if k not in ("rows", "generated")},
              "policies": [analyze_capture(record, minimum) for minimum in POLICIES],
              "interpretation": "Each candidate is selected only from its original committed ledger. Finalized-output agreement is not counterfactual acceptance. Checkpoint acceptance bounds stop at the first new draft prefix; later acceptance is unobserved. Original checkpoints are not stitched into a new rollout. Counts are not a speed estimate."}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": "ANALYZED", "output_sha256": sha(args.output)}))


if __name__ == "__main__":
    main()
