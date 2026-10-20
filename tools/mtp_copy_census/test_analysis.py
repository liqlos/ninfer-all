import copy
import itertools
import random
import unittest

import analysis


def reference(ledger, minimum, maximum=256, history=65536, full=True):
    best = None
    size = len(ledger)
    for end in range(size - 2, max(size - 2 - history, -1), -1):
        available = size - end - 1
        if available < (3 if full else 1):
            continue
        length = 0
        while (length < min(maximum, size, end + 1) and
               ledger[end - length] == ledger[size - 1 - length]):
            length += 1
        if length >= minimum and (best is None or length > best["suffix_length"]):
            best = {"source_start": end - length + 1, "source_end_exclusive": end + 1,
                    "suffix_length": length, "candidate_ids": list(ledger[end + 1:end + 4])}
    return best


class CopyTests(unittest.TestCase):
    def test_exhaustive_small_histories(self):
        for size in range(10):
            for ledger in itertools.product(range(2), repeat=size):
                for minimum in (1, 2, 3):
                    for full in (False, True):
                        self.assertEqual(analysis.copy_candidate(ledger, minimum=minimum, full=full),
                                         reference(ledger, minimum, full=full))

    def test_random_bounds_and_history(self):
        rng = random.Random(0x3090)
        for _ in range(200):
            ledger = [rng.randrange(4) for _ in range(rng.randrange(2, 180))]
            minimum = rng.randrange(1, 9)
            maximum = rng.randrange(minimum, 40)
            history = rng.randrange(1, 120)
            self.assertEqual(analysis.copy_candidate(ledger, minimum, maximum, history),
                             reference(ledger, minimum, maximum, history))

    def test_longest_before_recent_and_no_future_choice(self):
        ledger = [1, 2, 3, 4, 7, 8, 9, 0, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4]
        self.assertEqual(analysis.copy_candidate(ledger, 2)["candidate_ids"], [7, 8, 9])
        ledger = [1, 2, 9, 9, 9, 0, 1, 2, 7, 8, 9, 0, 1, 2]
        a = analysis.local_comparison(ledger, [9, 9, 9, 5], [9, 9, 9], 2)
        b = analysis.local_comparison(ledger, [7, 8, 9, 5], [9, 9, 9], 2)
        self.assertEqual(a["copy"], b["copy"])
        self.assertEqual(a["copy"]["candidate_ids"], [7, 8, 9])
        self.assertEqual(a["local_licensed_difference"], -3)
        self.assertEqual(b["local_licensed_difference"], 3)

    def test_partial_continuation_and_overlapping_known_history(self):
        self.assertIsNone(analysis.copy_candidate([1] * 4, 2))
        self.assertEqual(analysis.copy_candidate([1] * 5, 2)["candidate_ids"], [1] * 3)
        self.assertEqual(analysis.copy_candidate([1] * 5, 2, full=False)["candidate_ids"], [1])

    def test_invalid_domain_and_bounds(self):
        for ledger in ([True], [-1], [248077], [1.0], "12"):
            with self.assertRaises(ValueError):
                analysis.copy_candidate(ledger)
        for bounds in ((0, 256, 65536), (8, 7, 65536), (8, 257, 65536), (8, 256, 0)):
            with self.assertRaises(ValueError):
                analysis.copy_candidate([], *bounds)


def capture_fixture():
    result = {"prompt_tokens": 2, "generated_token_ids": list(range(1, 10)),
              "finish_reason": "output_limit", "rounds": 3, "drafted_tokens": 8,
              "accepted_tokens": 5,
              "speculative": {"enabled": True, "backend": "mtp", "draft_window": 3,
                              "rounds": 3, "drafted_tokens": 8, "accepted_tokens": 5,
                              "fallback_steps": 0, "accepted_per_position": [3, 2, 0],
                              "adaptive": False, "window_transitions": 0, "rounds_per_window": [0, 0, 0]}}
    rows = []
    for index, (frontier, budget, extent, drafts, delta, anchor) in enumerate([
            (2, 8, 3, [2, 3, 99], [], 1),
            (5, 5, 3, [5, 99, 99], [2, 3, 4], 4),
            (7, 3, 2, [7, 8, 42], [5, 6], 6)]):
        rows.append({"round": index + 1, "E": frontier, "S": frontier + 1,
                     "anchor": anchor, "mtp_kv_valid": frontier, "verify_window": 3,
                     "draft_window": 3, "remaining_budget": budget, "extent": extent,
                     "mtp_draft_count": 3, "drafts": drafts, "committed_delta": delta})
    return {"schema": 1, "kind": "request", "epoch": 1, "active": True, "error": [],
            "sampling": {"temperature": 0, "top_k": 1, "presence_penalty": 1.5, "frequency_penalty": 0},
            "first_ledger": [50, 51, 1], "rows": rows, "result": result,
            "validation": {"monotonic_prefix": True, "native_result_prefixes": True}}


class NativeCaptureTests(unittest.TestCase):
    def test_exact_native_replay_and_terminal_clipping(self):
        record = capture_fixture()
        replay = analysis.validate_capture(record)
        self.assertEqual(replay["native_accepted_per_position"], [3, 2, 0])
        self.assertEqual(replay["initial_generated_tokens"], 1)
        self.assertEqual([r["eligible_copy_first"] for r in replay["rows"]], [False, True, False])
        summary = analysis.analyze_capture(record)
        self.assertEqual(summary["eligible_baseline_checkpoints"], 1)
        self.assertFalse(summary["new_trajectory_or_throughput_claim"])
        self.assertFalse(summary["terminal_drain_saving_included"])

    def test_incorrect_frontiers_ids_counter_or_epoch_rejected(self):
        original = capture_fixture()
        variants = []
        for field, value in (("S", 7), ("E", 4), ("anchor", 99), ("mtp_kv_valid", 4),
                             ("remaining_budget", 6), ("extent", 2), ("round", 3),
                             ("committed_delta", [2, 3, 99]), ("drafts", [5, 6, 7])):
            record = copy.deepcopy(original)
            record["rows"][1][field] = value
            variants.append(record)
        for field, value in (("rounds", 2), ("accepted_tokens", 6), ("accepted_per_position", [3, 3, 0]),
                             ("adaptive", True), ("fallback_steps", 1)):
            record = copy.deepcopy(original)
            record["result"]["speculative"][field] = value
            variants.append(record)
        record = copy.deepcopy(original)
        record["first_ledger"][-1] = 99
        variants.append(record)
        record = copy.deepcopy(original)
        record["rows"] = record["rows"][:2]
        variants.append(record)
        for record in variants:
            with self.subTest(record=record), self.assertRaises(ValueError):
                analysis.validate_capture(record)

    def test_first_MTP_boundary_does_not_define_prompt_length(self):
        record = capture_fixture()
        record["first_ledger"].append(2)
        record["rows"][0].update(E=3, S=4, anchor=2, mtp_kv_valid=3,
                                  remaining_budget=7, drafts=[3, 99, 99])
        record["rows"][1]["committed_delta"] = [3, 4]
        record["result"]["accepted_tokens"] = 4
        record["result"]["speculative"].update(accepted_tokens=4, accepted_per_position=[3, 1, 0])
        summary = analysis.validate_capture(record)
        self.assertEqual(summary["prompt_tokens"], 2)
        self.assertEqual(summary["initial_generated_tokens"], 2)


if __name__ == "__main__":
    unittest.main()
