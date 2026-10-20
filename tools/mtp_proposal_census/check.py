import argparse
import copy
import json
import math
import pathlib
import struct


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def number(value):
    return struct.unpack("<f", struct.pack("<I", value))[0]


def check_score(score, prefix):
    require(0 <= score["id"] < 248077, "Token outside public domain")
    raw = number(score["raw_bits"])
    adjusted = number(score["adjusted_bits"])
    require(math.isfinite(raw) and math.isfinite(adjusted), "Non-finite score")
    require(score["raw_bits"] & 65535 == 0, "Raw score is not expanded BF16")
    require(score["base_count"] >= 0 and score["prefix_count"] == prefix.count(score["id"]),
            "Invalid committed or prefix count")
    expected = raw - (1.5 if score["base_count"] + score["prefix_count"] > 0 else 0.0)
    require(bits(expected) == score["adjusted_bits"], "Penalty arithmetic differs")


def check_column(column, prefix):
    known = {}
    for kind, key in (("raw", "raw_bits"), ("adjusted", "adjusted_bits")):
        entries = column[kind]
        require(len(entries) == 2 and entries[0]["id"] != entries[1]["id"], "Invalid top2")
        require(entries == sorted(entries, key=lambda x: (-number(x[key]), x["id"])),
                "Top2 ordering differs")
        for score in entries:
            check_score(score, prefix)
            require(score["id"] not in known or known[score["id"]] == score,
                    "Same token has inconsistent scores")
            known[score["id"]] = score
    for score in column["candidates"]:
        check_score(score, prefix)
        require(score["id"] not in known or known[score["id"]] == score,
                "Candidate score differs from top2")
        known[score["id"]] = score
    for kind, key in (("raw", "raw_bits"), ("adjusted", "adjusted_bits")):
        observed_top2 = sorted(known.values(), key=lambda x: (-number(x[key]), x["id"]))[:2]
        require(column[kind] == observed_top2, "Top2 contradicts another recorded score")
    return known


def check_record(r):
    n = r["extent"]
    require(1 <= n <= r["q_extent"] <= 3, "Invalid linked extent")
    require(r["epoch"] == r["q_epoch"] and r["producer_round"] + 1 == r["round"],
            "Wrong request or producer round")
    require(r["frontier"] == r["q_frontier"] and r["anchor"] == r["q_anchor"],
            "Wrong proposal frontier")
    require(len(r["drafts"]) == len(r["q"]) == n and len(r["p"]) == n + 1,
            "Invalid record dimensions")
    known = {}
    for col, q in enumerate(r["q"]):
        check_column(q, r["drafts"][:col])
        require(q["actual"] == q["raw"][0]["id"] == r["drafts"][col], "Wrong actual draft")
        fixed = list(dict.fromkeys(s["id"] for s in q["raw"] + q["adjusted"]))
        require([s["id"] for s in q["candidates"]] == fixed, "Q candidate set was not frozen")
    for col, p in enumerate(r["p"]):
        values = check_column(p, r["drafts"][:col])
        for ident, score in values.items():
            require(ident not in known or known[ident] == score["base_count"],
                    "Global counts changed between target columns")
            known[ident] = score["base_count"]
        require(p["actual_raw"] == p["raw"][0]["id"], "Wrong native raw target winner")
        require(p["causal"] == (col <= r["accepted"]), "Wrong causal column label")
        qset = r["q"][col]["candidates"] if col < n else []
        require([s["id"] for s in p["candidates"]] == [s["id"] for s in qset],
                "P introduced a future candidate")
        for qscore, pscore in zip(qset, p["candidates"]):
            require(qscore["base_count"] == pscore["base_count"] and
                    qscore["prefix_count"] == pscore["prefix_count"], "Q/P count join differs")
        winner = p["adjusted"][0]["id"]
        expected_q = next((s for s in qset if s["id"] == winner), None)
        require(p["q_score_at_target_winner"] == expected_q, "Missing Q value was fabricated")
    accepted = 0
    while accepted < n and r["drafts"][accepted] == r["p"][accepted]["adjusted"][0]["id"]:
        accepted += 1
    licensed = r["drafts"][:accepted] + [r["p"][accepted]["adjusted"][0]["id"]]
    require(r["accepted"] == accepted and r["licensed_count"] == len(licensed) and
            r["licensed"] == licensed, "Native acceptance differs from processed target")
    require(r["post_frontier"] == r["frontier"] + len(licensed) and
            r["post_anchor"] == licensed[-1], "Wrong native post-accept state")
    require(r["causal_columns"] == (1 << len(licensed)) - 1, "Wrong causal mask")
    return known


def check_request(report):
    require(report["error"] == 0 and report["active"] == 1, "Device observer declined or failed")
    require(report["valid_rows"] == 248077 and report["physical_rows"] == 248320,
            "Vocabulary profile differs")
    require(report["native_q_policy"] == "raw_argmax" and report["q_adjusted_is_counterfactual"]
            and report["target_policy"] == "processed_greedy", "Proposal policy differs")
    require(report["sampling"] == {"temperature": 0.0, "top_k": 1,
                                   "presence_penalty": 1.5, "frequency_penalty": 0.0},
            "Sampling profile differs")
    records = report["records"]
    require(report["completed_linked_records"] == len(records) == 16, "Need 16 linked checkpoints")
    require([r["round"] for r in records] == list(range(2, 18)), "Expected first 16 linked rounds")
    summary = {"epoch": report["epoch"], "linked_rounds": 16, "first_rejections": 0,
               "raw_q_equals_raw_p_but_penalty_changes_p": 0,
               "adjusted_q_matches_first_rejection": 0, "raw_q_second_matches_first_rejection": 0,
               "adjusted_q_changes_already_accepted_prefix": 0}
    previous = None
    previous_counts = {}
    for r in records:
        require(r["epoch"] == report["epoch"], "Mixed request epochs")
        counts = check_record(r)
        if previous is not None:
            require(r["frontier"] == previous["post_frontier"] and
                    r["anchor"] == previous["post_anchor"], "Non-contiguous committed frontier")
            for ident in counts.keys() & previous_counts.keys():
                require(counts[ident] == previous_counts[ident] + previous["licensed"].count(ident),
                        "Cross-round generated counts differ")
        a = r["accepted"]
        for col in range(a):
            summary["adjusted_q_changes_already_accepted_prefix"] += (
                r["q"][col]["adjusted"][0]["id"] != r["drafts"][col])
        if a < r["extent"]:
            q, p = r["q"][a], r["p"][a]
            winner = p["adjusted"][0]["id"]
            summary["first_rejections"] += 1
            summary["raw_q_equals_raw_p_but_penalty_changes_p"] += (
                q["actual"] == p["raw"][0]["id"] and p["raw"][0]["id"] != winner)
            summary["adjusted_q_matches_first_rejection"] += q["adjusted"][0]["id"] == winner
            summary["raw_q_second_matches_first_rejection"] += q["raw"][1]["id"] == winner
        previous, previous_counts = r, counts
    return summary


def self_check():
    def scored(ident, raw, count, prefix=0):
        return {"id": ident, "raw_bits": bits(raw),
                "adjusted_bits": bits(raw - (1.5 if count + prefix else 0)),
                "base_count": count, "prefix_count": prefix}
    a, b = scored(20, 5, 1), scored(21, 4, 0)
    q = {"raw": [a, b], "adjusted": [b, a], "candidates": [a, b], "actual": 20}
    p = {**copy.deepcopy(q), "actual_raw": 20, "causal": True, "q_score_at_target_winner": b}
    del p["actual"]
    bonus_a = scored(20, 5, 1, 1)
    bonus = {"raw": [bonus_a, b], "adjusted": [b, bonus_a], "candidates": [],
             "actual_raw": 20, "causal": False, "q_score_at_target_winner": None}
    r = {"epoch": 1, "q_epoch": 1, "round": 2, "producer_round": 1,
         "frontier": 64, "q_frontier": 64, "anchor": 7, "q_anchor": 7,
         "extent": 1, "q_extent": 1, "drafts": [20], "q": [q], "p": [p, bonus],
         "accepted": 0, "licensed_count": 1, "licensed": [21],
         "post_frontier": 65, "post_anchor": 21, "causal_columns": 1}
    check_record(r)
    mutations = [
        lambda x: x.update(q_epoch=2),
        lambda x: x.update(q_frontier=63),
        lambda x: x.update(accepted=1),
        lambda x: x["p"][0].update(q_score_at_target_winner=None),
        lambda x: x["p"][0]["candidates"][0].update(base_count=2),
        lambda x: x["q"][0]["candidates"].append(scored(22, 3, 0)),
    ]
    for mutate in mutations:
        broken = copy.deepcopy(r)
        mutate(broken)
        try:
            check_record(broken)
        except ValueError:
            continue
        raise ValueError("Malformed checkpoint was accepted")
    for extra in (scored(22, 6, 0), scored(22, 4.5, 0), scored(19, 5, 1)):
        broken_column = copy.deepcopy(q)
        broken_column["candidates"].append(extra)
        try:
            check_column(broken_column, [])
        except ValueError:
            continue
        raise ValueError("Contradictory observed top2 was accepted")
    return {"positive": 1, "negative": len(mutations) + 3, "cuda_run": False}


def main():
    parser = argparse.ArgumentParser(description="Validate compact census JSON lines without CUDA.")
    parser.add_argument("path", type=pathlib.Path, nargs="?")
    parser.add_argument("--epoch", action="append", type=int)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        print(json.dumps(self_check()))
        return
    require(args.path is not None and args.epoch, "A census log and explicit target epochs are required")
    require(all(epoch > 0 for epoch in args.epoch) and len(args.epoch) == len(set(args.epoch)),
            "Target epochs must be positive and unique")
    selected = []
    for line in args.path.read_text(encoding="utf-8").splitlines():
        try:
            report = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(report, dict) or report.get("kind") != "ninfer_lab_mtp_census":
            continue
        if report["epoch"] not in args.epoch:
            continue
        selected.append(report)
    epochs = [r["epoch"] for r in selected]
    require(epochs and len(epochs) == len(set(epochs)), "Missing or duplicate request census")
    require(set(epochs) == set(args.epoch), "Requested epoch is absent")
    print(json.dumps({"status": "COMPACT_CONTRACT_PASS", "diagnostic_only": True,
                      "full_output_equivalence_checked": False,
                      "requests": [check_request(r) for r in selected]}, indent=2))


if __name__ == "__main__":
    main()
