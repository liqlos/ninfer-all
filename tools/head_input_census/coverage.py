import argparse
import collections
import hashlib
import importlib.util
import json
import os
import pathlib
import sys


sys.dont_write_bytecode = True
HERE = pathlib.Path(__file__).resolve().parent
CAPACITIES = (1024, 3072, 8192)
ANCHORS = (8, 16)
ROW_BYTES = 4200


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked(path, digest):
    require(sha(path) == digest, "Pinned evidence changed: " + str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def recent_ids(ledger, limit):
    result, seen = [], set()
    for token in reversed(ledger):
        if token not in seen:
            result.append(token)
            seen.add(token)
            if len(result) == limit:
                break
    return result


def shortlist(bank, ledger, capacity, event):
    require(all(head["event"] < event for head in bank), "A future or current unpaid head entered the bank")
    sources = [head["top_ids"] for head in reversed(bank)]
    if ledger is not None:
        sources.insert(0, recent_ids(ledger, capacity))
    selected, visits, depth = set(), 0, 0
    while len(selected) < capacity:
        advanced = False
        for source in sources:
            if depth < len(source):
                advanced = True
                visits += 1
                selected.add(int(source[depth]))
                if len(selected) == capacity:
                    break
        if not advanced:
            break
        depth += 1
    return sorted(selected), visits


def acceptance_support(record, candidates, choices):
    limit = min(len(candidates), record["extent"], record["accepted"] + 1)
    supported = selected = 0
    for column in range(limit):
        winner = record["p"][column]["adjusted"][0]["id"]
        if winner not in candidates[column]:
            break
        supported += 1
    for column in range(limit):
        winner = record["p"][column]["adjusted"][0]["id"]
        if choices[column] != winner:
            break
        selected += 1
    return {"native_accepted": record["accepted"], "valid_baseline_target_columns_checked": limit,
            "inventory_supported_known_prefix_ceiling": supported,
            "selected_known_accepted_prefix": selected,
            "support_is_oracle_inventory_assessment_not_a_selection_rule": True,
            "blocked_by_inventory_at_column": supported if supported < limit else None,
            "unknown_after_repairing_first_native_rejection":
                record["accepted"] < record["extent"] and selected > record["accepted"],
            "future_target_columns_used_for_bank_or_selection": False}


def add_prior_targets(data, bank, index):
    if index:
        for column in range(data.records[index - 1]["extent"] + 1):
            bank.append(data.head(index - 1, 3 + column))


def replay(data, capacity, anchors, use_ledger, policy, geometry=False):
    bank = collections.deque(maxlen=anchors)
    rows, dense, sparse, equal, candidate_rows, visits, target_admissions = [], 0, 0, 0, 0, 0, 0
    total = sum(record["extent"] for record in data.records)
    for index, record in enumerate(data.records):
        add_prior_targets(data, bank, index)
        if index:
            target_admissions += data.records[index - 1]["extent"] + 1
        ledger = data.ledger(index) if use_ledger else None
        candidates, choices = [], []
        for column in range(record["extent"]):
            event = data.event(index, column)
            paid = index == 0 or (policy == "first_q_each_round_dense" and column == 0)
            if paid:
                token = data.records[index]["q"][column]["actual"]
                dense += 1
                candidates.append({token})
                choices.append(token)
                bank.append(data.head(index, column))
                equal += 1
                continue
            ids, scanned = shortlist(bank, ledger, capacity, event)
            require(ids, "No causal candidate; define a dense fallback before evaluating this query")
            token = data.choose(index, column, ids)
            sparse += 1
            candidate_rows += len(ids)
            visits += scanned
            candidates.append(set(ids))
            choices.append(token)
            same = token == record["q"][column]["actual"]
            detail = {"producer_round": record["producer_round"], "column": column,
                      "bank_heads": [head["name"] for head in bank], "candidate_count": len(ids),
                      "proposal_matches_native": same, "selected": token,
                      "native_raw_q_top1": record["q"][column]["actual"]}
            if geometry:
                detail["geometry"] = data.geometry(index, column, bank)
            rows.append(detail)
            if not same:
                return {"policy": policy, "status": "STOPPED_AT_FIRST_PROPOSAL_DIFFERENCE",
                        "same_native_proposal_prefix_heads": equal, "dense_q_heads_paid": dense,
                        "sparse_q_heads_evaluated": sparse, "candidate_rows_scored": candidate_rows,
                        "shortlist_rank_id_visits": visits, "paid_target_heads_admitted": target_admissions,
                        "unknown_subsequent_q_heads": total - dense - sparse, "first_difference": detail,
                        "last_round_causal_acceptance_assessment": acceptance_support(record, candidates, choices),
                        "new_trajectory_known": False, "later_baseline_heads_admitted": False,
                        "sparse_q_heads_added_as_full_anchors": 0, "sparse_events": rows}
            equal += 1
    return {"policy": policy, "status": "NATIVE_PROPOSAL_PREFIX_MATCHES_THROUGH_CAPTURE",
            "same_native_proposal_prefix_heads": equal, "dense_q_heads_paid": dense,
            "sparse_q_heads_evaluated": sparse, "candidate_rows_scored": candidate_rows,
            "shortlist_rank_id_visits": visits, "paid_target_heads_admitted": target_admissions,
            "unknown_subsequent_q_heads": 0, "first_difference": None,
            "new_trajectory_known": False, "sparse_q_heads_added_as_full_anchors": 0,
            "sparse_events": rows, "exact_native_row_scores_assumed": True}


def conditioned(data, capacity, anchors, use_ledger, geometry=False):
    bank = collections.deque(maxlen=anchors)
    rows, hits, queries, candidate_rows, oracle_saving_rows, visits, admissions = [], 0, 0, 0, 0, 0, 0
    for index, record in enumerate(data.records):
        if not index:
            continue
        add_prior_targets(data, bank, index)
        admissions += data.records[index - 1]["extent"] + 1
        ledger = data.ledger(index) if use_ledger else None
        ids, scanned = shortlist(bank, ledger, capacity, data.event(index, 0))
        require(ids, "Paid prior target inventory unexpectedly empty")
        present = set(ids)
        selected = [data.choose(index, column, ids) for column in range(record["extent"])]
        covered = [q["actual"] in present for q in record["q"]]
        hits += sum(covered)
        queries += len(covered)
        candidate_rows += len(ids) * len(covered)
        oracle_saving_rows += sum(covered) * (data.physical - len(ids))
        visits += scanned
        row = {"producer_round": record["producer_round"], "bank_heads": [head["name"] for head in bank],
               "bank_not_yet_full": len(bank) < anchors, "candidate_count": len(ids),
               "raw_q_top1_covered": covered, "selected": selected,
               "causal_acceptance_support": acceptance_support(record, [present] * len(covered), selected)}
        if use_ledger:
            row["committed_ledger_unique_ids"] = len(set(ledger))
        if geometry:
            row["geometry"] = [data.geometry(index, column, bank) for column in range(record["extent"])]
        rows.append(row)
    return {"scope": "ONE_STEP_CONDITIONAL_ON_EACH_NATIVE_BASELINE_PREFIX",
            "not_a_counterfactual_continuous_trajectory": True,
            "bootstrap_q_heads_excluded_missing_P1": data.records[0]["extent"],
            "steady_q_heads": queries, "raw_q_top1_hits": hits, "raw_q_top1_misses": queries - hits,
            "raw_q_top1_coverage": hits / queries if queries else None,
            "candidate_rows_if_all_steady_q_are_sparse": candidate_rows,
            "shortlist_rank_id_visits_with_one_build_per_round": visits,
            "paid_target_heads_admitted": admissions, "dense_q_anchors_admitted": 0,
            "strict_exact_no_fallback_claim_rejected": hits < queries,
            "oracle_gross_q_weight_bytes_saved_upper_bound": oracle_saving_rows * ROW_BYTES,
            "oracle_bound_scope": "fixed_target_bank; oracle chooses skips before work; misses dense; all overhead ignored",
            "oracle_fallback_trigger_is_not_implementable_from_this_census": True,
            "rounds": rows}


class Capture:
    def __init__(self, metadata, binary, prompt_tokens, proof):
        import numpy as np
        self.np, self.proof = np, proof
        self.metadata, self.records = metadata, metadata["records"]
        self.physical, self.logical = proof.PHYSICAL, proof.LOGICAL
        with binary.open("rb") as stream:
            header = proof.HEADER.unpack(stream.read(proof.HEADER.size))
        count = len(self.records)
        require(header == (proof.MAGIC, 1, metadata["epoch"], count, proof.K, proof.PHYSICAL,
                           proof.LOGICAL, proof.PACKED, proof.SAMPLE,
                           proof.LOGICAL * 4 + count * 7 * proof.SAMPLE), "Capture header changed")
        require(binary.stat().st_size == proof.HEADER.size + header[-1], "Capture length changed")
        dtype = np.dtype([("epoch", "<u4"), ("round", "<u4"), ("role", "<i4"),
            ("frontier", "<i4"), ("anchor", "<i4"), ("extent", "<i4"), ("weight", "<u8"),
            ("packed", "u1", (proof.PACKED,)), ("logits", "<u2", (proof.PHYSICAL,))])
        require(dtype.itemsize == proof.SAMPLE, "Sample layout changed")
        self.samples = np.memmap(binary, dtype=dtype, mode="r", offset=proof.HEADER.size + proof.LOGICAL * 4,
                                 shape=(count, 7))
        first = metadata["payload"]["first_ledger"]
        ledger_rows = metadata["payload"]["ledger_rows"]
        require(len(first) == prompt_tokens + 1 and metadata["first_frontier"] == prompt_tokens and
                len(ledger_rows) >= count + 1, "Native prompt ledger boundary changed")
        self.initial = list(first) + list(ledger_rows[1]["committed_delta"])
        self.prompt_tokens, self.head_cache, self.vector_cache, self.geometry_cache = prompt_tokens, {}, {}, {}
        for index, record in enumerate(self.records):
            require(record["producer_round"] == index + 1 and record["round"] == index + 2,
                    "Unsupported nonconsecutive capture")
            ledger = self.ledger(index)
            require(len(ledger) == record["frontier"] + 1 and ledger[-1] == record["anchor"] and
                    all(0 <= token < self.logical for token in ledger), "Causal ledger reconstruction failed")
            require(ledger_rows[index + 1]["frontier"] == record["frontier"] and
                    (index == 0 or ledger_rows[index + 1]["committed_delta"] == self.records[index - 1]["licensed"]),
                    "Future or unrelated ledger delta")

    def ledger(self, index):
        values = list(self.initial)
        for previous in self.records[:index]:
            values.extend(previous["licensed"])
        return values

    def event(self, index, role):
        record = self.records[index]
        return (record["producer_round"], 1, role) if role < 3 else (record["round"], 0, role - 3)

    def raw(self, index, role):
        return (self.samples[index, role]["logits"][:self.logical].astype(self.np.uint32) << 16).view(self.np.float32)

    def head(self, index, role):
        key = (index, role)
        if key not in self.head_cache:
            np = self.np
            values = self.raw(index, role)
            require(bool(np.all(np.isfinite(values))), "Nonfinite anchor logits")
            count = min(max(CAPACITIES), len(values))
            threshold = np.partition(values, len(values) - count)[len(values) - count]
            better = np.flatnonzero(values > threshold)
            equal = np.flatnonzero(values == threshold)[:count - len(better)]
            ids = np.concatenate((better, equal))
            ids = ids[np.lexsort((ids, -values[ids]))]
            event = self.event(index, role)
            self.head_cache[key] = {"key": key, "event": event, "name":
                ("Q" if role < 3 else "P") + str(event[0]) + "." + str(event[2]),
                "top_ids": ids.tolist()}
        return self.head_cache[key]

    def choose(self, index, column, ids):
        values = self.raw(index, column)
        require(ids == sorted(set(ids)) and ids and ids[0] >= 0 and ids[-1] < self.logical,
                "Candidate ID domain/order")
        return ids[int(self.np.argmax(values[ids]))]

    def vector(self, index, role):
        key = (index, role)
        if key not in self.vector_cache:
            np, proof = self.np, self.proof
            packed = self.samples[index, role]["packed"]
            qs = packed[:proof.K].view(np.int8).astype(np.float64).reshape(proof.K // 32, 32)
            scale = packed[proof.K:].view("<f2").reshape(proof.K // 32, 2)[:, 0].astype(np.float64)
            require(bool(np.all(np.isfinite(scale))), "Nonfinite Q8 scale")
            self.vector_cache[key] = (qs * scale[:, None]).reshape(proof.K)
        return self.vector_cache[key]

    def geometry(self, index, column, bank):
        key = (index, column, tuple(head["key"] for head in bank))
        if key not in self.geometry_cache:
            np = self.np
            x = self.vector(index, column)
            matrix = np.stack([self.vector(*head["key"]) for head in bank], axis=1)
            coefficients, _, rank, _ = np.linalg.lstsq(matrix, x, rcond=None)
            norm = float(np.linalg.norm(x))
            residual = float(np.linalg.norm(x - matrix @ coefficients))
            nearest = float(np.min(np.linalg.norm(matrix - x[:, None], axis=0)))
            self.geometry_cache[key] = {"rank": int(rank), "anchors": len(bank),
                "target_norm": norm, "residual_norm": residual, "nearest_anchor_drift_norm": nearest,
                "relative_residual": residual / norm if norm else None,
                "relative_nearest_drift": nearest / norm if norm else None,
                "coefficient_l1": float(np.sum(np.abs(coefficients))),
                "anchor_norm_inflation": float(np.abs(coefficients) @ np.linalg.norm(matrix, axis=0)),
                "diagnostic_not_outward_or_per_row_certificate": True}
        return self.geometry_cache[key]


def cost_model(capacity, anchors, proof):
    return {"q6_bytes_per_row": ROW_BYTES, "dense_q_physical_weight_bytes": proof.PHYSICAL * ROW_BYTES,
            "sparse_q_weight_bytes_floor_at_capacity": capacity * ROW_BYTES,
            "ranking_ids_bank_bytes_at_capacity": anchors * capacity * 4,
            "retained_Q8_inputs_bytes": anchors * proof.PACKED,
            "optional_full_BF16_logit_bank_bytes_logical": anchors * proof.LOGICAL * 2,
            "minimum_raw_logit_bytes_scanned_per_new_head_ranking": proof.LOGICAL * 2,
            "all_vocabulary_certificate_logit_reads_per_query_if_used": anchors * proof.LOGICAL * 2,
            "fallback_cost_formula": "shortlist construction + sparse head + certificate + fallback_probability * dense head",
            "unpriced_required_work": ["top-M selection and ranking cache writes", "ledger inventory maintenance",
                "candidate deduplication", "indexed Q6 row transaction amplification", "sparse argmax",
                "fallback decisions and graph launch/control", "bank replacement and optional full-logit retention",
                "rigorous numerical bounds if claiming exact head elimination"],
            "target_T4_and_transformer_work_unchanged": True,
            "row_byte_floor_is_not_DRAM_traffic_or_latency": True, "whole_request_speedup": None}


def analyze(runtime_path, runtime_sha, epochs, geometry):
    runtime = checked(runtime_path, runtime_sha)
    require(runtime["status"] == "DIAGNOSTIC_CAPTURE_AND_CPU_REPLAY_PASS" and runtime["arm"] == "on" and
            runtime["diagnostic_only"] and runtime["performance_claim"] is False and
            runtime["production_unchanged"] and runtime["model_identity_unchanged"] and
            runtime["reservation_released"], "Need a qualified completed native W/ON capture")
    require(sha(HERE / "proof.py") == runtime["source_sha256"]["proof.py"], "Qualified proof source changed")
    proof = module("head_coverage_proof", HERE / "proof.py")
    checker_path = HERE.parent / "mtp_proposal_census/check.py"
    require(sha(checker_path) == runtime["helper_sha256"]["source/tools/mtp_proposal_census/check.py"],
            "Qualified compact checker source changed")
    checker = module("head_coverage_compact", checker_path)
    output = []
    for epoch in epochs:
        payload = next(value for value in runtime["capture"]["payloads"] if value["epoch"] == epoch)
        replay_receipt = next(value for value in runtime["cpu_replay"] if value["epoch"] == epoch)
        cpu = checked(pathlib.Path(replay_receipt["path"]), replay_receipt["sha256"])
        require(cpu["status"] == "CAPTURE_NATIVE_REPLAY_AND_GEOMETRY_PASS" and cpu["epoch"] == epoch and
                cpu["records"] == 16 and cpu["binary_sha256"] == payload["sha256"] and
                cpu["per_row_certificate"] is False and cpu["performance_claim"] is False and
                payload["warmup_only"] is False, "Full-array replay is not qualified for this epoch")
        metadata = checked(pathlib.Path(payload["metadata"]), payload["metadata_sha256"])
        binary = pathlib.Path(payload["path"])
        require(sha(binary) == payload["sha256"], "Qualified binary changed")
        checker.check_request(metadata)
        sample = runtime["samples"][epoch - 3]
        require(sample["name"] == ("RU4K" if epoch == 3 else "JSON32K"), "Request/epoch binding changed")
        data = Capture(metadata, binary, sample["usage"]["prompt_tokens"], proof)
        settings = []
        for anchors in ANCHORS:
            for capacity in CAPACITIES:
                for use_ledger in (False, True):
                    settings.append({"shortlist_capacity": capacity, "anchor_capacity": anchors,
                        "inventory": "native_ledger_and_paid_heads" if use_ledger else "paid_heads_only",
                        "baseline_conditioned": conditioned(data, capacity, anchors, use_ledger, geometry),
                        "causal_replays": [replay(data, capacity, anchors, use_ledger, policy, geometry)
                            for policy in ("bootstrap_dense_only", "first_q_each_round_dense")],
                        "cost_model": cost_model(capacity, anchors, proof)})
        output.append({"epoch": epoch, "request": sample["name"], "binary_sha256": payload["sha256"],
            "metadata_sha256": payload["metadata_sha256"], "full_array_replay_sha256": replay_receipt["sha256"],
            "native_prompt_ids": data.prompt_tokens, "prompt_inventory_source": "first_ledger native IDs; no retokenization",
            "initial_missing_paid_target_head": "P1; its committed delta is known before Q1, its logits and Q8 input are absent",
            "bootstrap_unobserved_prefill_heads": True, "uncaptured_prior_heads_are_not_synthesized": True,
            "penalty_counts_are_emission_only_and_not_used_as_prompt_inventory": True,
            "unverified_q_tail_columns_excluded": sum(r["q_extent"] - r["extent"] for r in data.records),
            "settings": settings})
    return {"schema": "head-causal-coverage-r1", "status": "DIAGNOSTIC_CAUSAL_COVERAGE_COMPLETE",
            "runtime_receipt_sha256": runtime_sha, "source_sha256": sha(pathlib.Path(__file__)),
            "gpu_run": False, "performance_claim": False, "per_row_certificate": False,
            "geometry_requested": geometry, "shortlist_policy":
                "FIFO paid heads; rank-round-robin newest head first; optional recent unique native ledger as first stream; global cap",
            "candidate_ranking": "raw BF16 descending, lower logical token ID on ties",
            "new_bank_Q_heads": "only bootstrap or predeclared Q0 dense calls; never skipped Q heads",
            "causal_order": "P(r) and acceptance precede Q(r); joined P(r+1) is evaluation only",
            "early_kill": "A top1 miss rejects exact no-fallback replacement for that fixed inventory; geometry alone cannot reject all certificates",
            "limitations": ["native sparse row scores are assumed; no new head kernel is numerically qualified",
                "conditional one-step coverage does not measure a modified trajectory",
                "replay stops at first changed proposal, even when an earlier rejection might preserve committed tokens",
                "further acceptance after fixing the first native rejection needs new forwards",
                "an oracle fallback saving bound is not a realizable policy or a whole-request forecast",
                "no row norms or native arithmetic certificate are provided"], "captures": output}


def self_test():
    class Tiny:
        physical = 12

        def __init__(self):
            self.records = []
            for index in range(3):
                self.records.append({"producer_round": index + 1, "round": index + 2, "extent": 3,
                    "accepted": 1, "q": [{"actual": token} for token in (1, 2, 3)],
                    "p": [{"adjusted": [{"id": token}]} for token in (1, 4, 9, 10)]})
            self.admitted = []

        def event(self, index, role):
            return (index + 1, 1, role) if role < 3 else (index + 2, 0, role - 3)

        def ledger(self, index):
            return [11, 8, 1]

        def head(self, index, role):
            self.admitted.append((index, role))
            first = self.records[index]["q"][role]["actual"] if role < 3 else 1
            return {"key": (index, role), "event": self.event(index, role),
                    "name": str((index, role)), "top_ids": [first, 7, 8, 9]}

        def choose(self, index, column, ids):
            native = self.records[index]["q"][column]["actual"]
            return native if native in ids else min(ids)

    data = Tiny()
    result = replay(data, 1, 8, False, "bootstrap_dense_only")
    require(result["dense_q_heads_paid"] == 3 and result["sparse_q_heads_evaluated"] == 2 and
            result["same_native_proposal_prefix_heads"] == 4 and result["unknown_subsequent_q_heads"] == 4,
            "Replay did not stop at first changed Q prefix")
    require(all(index == 0 for index, _ in data.admitted) and len(data.admitted) == 7,
            "Skipped or future baseline Q/P entered the bank")
    paid = Tiny()
    scheduled = replay(paid, 1, 8, False, "first_q_each_round_dense")
    require(scheduled["dense_q_heads_paid"] == 4 and (1, 0) in paid.admitted and
            (1, 1) not in paid.admitted, "Dense fallback schedule or skipped-head ownership")
    future = data.head(1, 3)
    try:
        shortlist([future], None, 1, data.event(1, 0))
    except ValueError:
        pass
    else:
        raise ValueError("Joined future P(r+1) was accepted")
    record = data.records[0]
    support = acceptance_support(record, [{1}, {4}, {9}], [1, 4, 9])
    require(support["inventory_supported_known_prefix_ceiling"] == 2 and
            support["selected_known_accepted_prefix"] == 2 and
            support["unknown_after_repairing_first_native_rejection"], "Future acceptance suffix leaked")
    a = {"event": (1, 0, 0), "top_ids": [4, 5, 6]}
    b = {"event": (1, 0, 1), "top_ids": [7, 4, 8]}
    require(shortlist([a, b], None, 3, (1, 1, 0))[0] == [4, 5, 7], "Global shortlist cap or rank order")
    require(shortlist([a, b], [10, 10, 9], 3, (1, 1, 0))[0] == [4, 7, 9], "Native ledger stream order")
    conditional = conditioned(Tiny(), 1, 8, False)
    require(conditional["steady_q_heads"] == 6 and conditional["raw_q_top1_hits"] == 2 and
            conditional["dense_q_anchors_admitted"] == 0, "Conditional census conflated with replay")
    return {"status": "SMALL_SYNTHETIC_CAUSAL_CONTROLS_PASS", "gpu_run": False,
            "controls": ["future P exclusion", "no skipped Q reuse", "stop after proposal divergence",
                "predeclared dense fallback", "acceptance-prefix information ceiling",
                "global shortlist cap", "causal ledger stream", "conditional versus trajectory separation"]}


def main():
    parser = argparse.ArgumentParser(description="CPU-only causal coverage; default runs small synthetic controls.")
    parser.add_argument("--runtime", type=pathlib.Path)
    parser.add_argument("--runtime-sha256")
    parser.add_argument("--epoch", choices=(3, 4), type=int, action="append")
    parser.add_argument("--geometry", action="store_true")
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    require(not sys.flags.optimize and bool(args.runtime) == bool(args.runtime_sha256),
            "Use Python without -O; runtime path and SHA must be paired")
    require(args.runtime is not None or not (args.epoch or args.geometry), "Analysis options require qualified capture")
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    result = analyze(args.runtime, args.runtime_sha256, sorted(set(args.epoch or (3, 4))), args.geometry) if args.runtime else self_test()
    encoded = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
        print(json.dumps({"path": str(args.output.resolve()), "sha256": sha(args.output), "status": result["status"]}))
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
