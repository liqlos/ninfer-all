import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import run as controller


class PhaseBindingTests(unittest.TestCase):
    def setUp(self):
        common = {"kind": "ninfer_lab_mtp_census", "schema": 1, "diagnostic_only": True,
                  "bootstrap_omitted": True, "physical_rows": 248320, "valid_rows": 248077,
                  "native_q_policy": "raw_argmax", "q_adjusted_is_counterfactual": True,
                  "target_policy": "processed_greedy", "error": 0}
        empty = {**common, "epoch": 1, "active": 0, "reason": "request_boundary_partial",
                 "device_observed_rounds": 0, "completed_linked_records": 0, "records": [],
                 "first_frontier": 0, "first_anchor": 0, "first_budget": 0,
                 "sampling": {"temperature": 0.0, "top_k": 0, "presence_penalty": 0.0, "frequency_penalty": 0.0}}
        reports = [empty]
        for epoch, frontier, budget, count in ((2, 4095, 31, 10), (3, 4089, 511, 16), (4, 32756, 511, 16)):
            reports.append({**common, "epoch": epoch, "active": 1, "first_frontier": frontier,
                            "first_budget": budget, "first_anchor": 42, "completed_linked_records": count,
                            "device_observed_rounds": count + 1, "records": [None] * count,
                            "reason": "request_boundary_partial" if epoch == 2 else "complete"})
        self.warmup = {"usage": {"prompt_tokens": 4095, "completion_tokens": 32}}
        self.samples = [{"name": name, "usage": {"prompt_tokens": prompt, "completion_tokens": 512},
                         "request_done": {"request_id": ident}, "request_body_sha256": name}
                        for ident, name, prompt in ((2, "RU4K", 4089), (3, "JSON32K", 32756))]
        self.lines = [
            "INFO  engine ready | model", json.dumps({"kind": "ninfer_lab_mtp_census_declined", "reason": "unsupported_request"}),
            "INFO  warmup complete | 679 ms", "INFO  listening on http://0.0.0.0:8080 | model qwen3.8-27b | auth disabled",
            "INFO  req#1 started | max output 32 | thinking off", json.dumps(reports[0]),
            "INFO  req#1 done | prompt 4,095 | output 32 | cache 0 (0.0%)",
            "INFO  req#2 started | max output 512 | thinking off", json.dumps(reports[1]), json.dumps(reports[2]),
            "INFO  req#2 done | prompt 4,089 | output 512 | cache 0 (0.0%)",
            "INFO  req#3 started | max output 512 | thinking off", json.dumps(reports[3]),
            "INFO  req#3 done | prompt 32,756 | output 512 | cache 0 (0.0%)", "INFO  server stopped"]
        self.checker = SimpleNamespace(check_request=Mock(side_effect=lambda r: {"epoch": r["epoch"]}))

    def qualify(self, lines, arm="on"):
        return controller.qualify("\n".join(lines), arm, self.warmup, self.samples, self.checker)

    def test_exact_startup_and_three_request_phases(self):
        result = self.qualify(self.lines)
        self.assertEqual([(r["epoch"], r["request_id"]) for r in result], [(3, 2), (4, 3)])
        self.assertEqual([call.args[0]["epoch"] for call in self.checker.check_request.call_args_list], [3, 4])

    def test_off_has_no_observer_records(self):
        self.assertEqual(self.qualify([line for line in self.lines if "ninfer_lab_mtp_census" not in line], "off"), [])
        self.checker.check_request.assert_not_called()
        with self.assertRaises(AssertionError):
            self.qualify(self.lines, "off")

    def test_reject_incomplete_duplicate_or_wrong_phase(self):
        variants = {}
        for name, index in (("decline", 1), ("inactive", 5), ("warmup", 8), ("RU", 9), ("JSON", 12), ("ready", 3)):
            variants["missing_" + name] = self.lines[:index] + self.lines[index + 1:]
        for name, index in (("decline", 1), ("inactive", 5), ("RU", 9), ("JSON", 12), ("ready", 3)):
            variants["duplicate_" + name] = self.lines[:index] + [self.lines[index]] + self.lines[index:]
        variants["decline_after_listening"] = [self.lines[0], self.lines[2], self.lines[3], self.lines[1], *self.lines[4:]]
        variants["inactive_after_warmup"] = self.lines[:5] + [self.lines[6], self.lines[5]] + self.lines[7:]
        variants["RU_after_request_done"] = self.lines[:9] + [self.lines[10], self.lines[9]] + self.lines[11:]
        variants["unsupported_real_request"] = self.lines[:12] + [self.lines[1]] + self.lines[12:]
        variants["extra_request"] = self.lines[:-1] + ["INFO  req#4 started | max output 512 | thinking off", self.lines[-1]]
        for name, index, changes in (
                ("nonempty_startup", 5, {"records": [None]}), ("startup_active", 5, {"active": 1}),
                ("startup_sampling", 5, {"sampling": {}}), ("wrong_epoch", 12, {"epoch": 3}),
                ("wrong_frontier", 12, {"first_frontier": 4089}), ("wrong_budget", 12, {"first_budget": 31}),
                ("unsupported_full_epoch", 12, {"active": 0}), ("partial_full_epoch", 12, {"reason": "request_boundary_partial"})):
            lines = list(self.lines)
            report = json.loads(lines[index])
            report.update(changes)
            lines[index] = json.dumps(report)
            variants[name] = lines
        for name, lines in variants.items():
            with self.subTest(name=name), self.assertRaises((AssertionError, ValueError)):
                self.qualify(lines)

    def test_reject_request_identity_change(self):
        self.samples = copy.deepcopy(self.samples)
        self.samples[1]["request_done"]["request_id"] = 4
        with self.assertRaises(AssertionError):
            self.qualify(self.lines)


if __name__ == "__main__":
    unittest.main()
