"""Unit tests for paired outcome analysis; no network or third-party packages."""
import json
import math
import tempfile
import unittest
from pathlib import Path

from agent_outcome_ledger import evaluate, load_jsonl, parse_outcome


def row(case_id="case-1", variant="baseline", **overrides):
    record = {
        "case_id": case_id,
        "variant": variant,
        "expected_action": "ESCALATE",
        "proposed_action": "ESCALATE",
        "accepted_by_human": True,
        "resolved": True,
        "auto_executed": False,
        "critical": False,
        "source_sha": "abcdef1",
        "verified_sha": "abcdef1",
        "evidence_id": "trace-1",
        "minutes_to_resolution": 20.0 if variant == "baseline" else 12.0,
        "model_cost_usd": 0.20 if variant == "baseline" else 0.21,
    }
    record.update(overrides)
    return parse_outcome(record)


def pairs(n=10):
    return [row(str(i), variant) for i in range(n) for variant in ("baseline", "candidate")]


class OutcomeLedgerTests(unittest.TestCase):
    def test_pilot_on_paired_improvement(self):
        result = evaluate(pairs())
        self.assertEqual(result["decision"], "PILOT")
        self.assertEqual(result["median_paired_resolution_delta_minutes"], -8)

    def test_small_sample_review(self):
        self.assertEqual(evaluate(pairs(2))["decision"], "REVIEW")

    def test_missing_pair_rejected(self):
        with self.assertRaisesRegex(ValueError, "same non-empty"):
            evaluate(pairs()[:-1])

    def test_duplicate_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            evaluate(pairs() + [row("0", "baseline")])

    def test_critical_accepted_wrong_action_blocks(self):
        records = pairs()
        records[-1] = row("9", "candidate", critical=True, proposed_action="MERGE", expected_action="BLOCK_MERGE")
        records[-2] = row("9", "baseline", critical=True, expected_action="BLOCK_MERGE", proposed_action="BLOCK_MERGE")
        self.assertEqual(evaluate(records)["decision"], "BLOCK")

    def test_critical_wrong_unaccepted_review(self):
        records = pairs()
        records[-1] = row("9", "candidate", critical=True, proposed_action="MERGE", expected_action="BLOCK_MERGE", accepted_by_human=False)
        records[-2] = row("9", "baseline", critical=True, expected_action="BLOCK_MERGE", proposed_action="BLOCK_MERGE")
        self.assertEqual(evaluate(records)["decision"], "REVIEW")

    def test_critical_auto_execution_blocks(self):
        records = pairs()
        records[-1] = row("9", "candidate", critical=True, proposed_action="MERGE", expected_action="BLOCK_MERGE", accepted_by_human=False, auto_executed=True)
        records[-2] = row("9", "baseline", critical=True, expected_action="BLOCK_MERGE", proposed_action="BLOCK_MERGE")
        self.assertEqual(evaluate(records)["decision"], "BLOCK")

    def test_source_sha_mismatch_blocks(self):
        records = pairs()
        records[-1] = row("9", "candidate", verified_sha="newsha")
        self.assertEqual(evaluate(records)["decision"], "BLOCK")

    def test_invalid_baseline_provenance_blocks(self):
        records = pairs()
        records[-2] = row("9", "baseline", verified_sha="wrong")
        self.assertEqual(evaluate(records)["decision"], "BLOCK")

    def test_source_drift_blocks(self):
        records = pairs()
        records[-1] = row("9", "candidate", source_sha="newsha", verified_sha="newsha")
        self.assertEqual(evaluate(records)["decision"], "BLOCK")

    def test_label_drift_blocks(self):
        records = pairs()
        records[-1] = row("9", "candidate", expected_action="OTHER")
        self.assertEqual(evaluate(records)["decision"], "BLOCK")

    def test_cost_regression_reviews(self):
        records = [row(str(i), v, model_cost_usd=0.8 if v == "candidate" else 0.2)
                   for i in range(10) for v in ("baseline", "candidate")]
        self.assertEqual(evaluate(records)["decision"], "REVIEW")

    def test_resolution_regression_reviews(self):
        records = [row(str(i), v, minutes_to_resolution=30 if v == "candidate" else 20)
                   for i in range(10) for v in ("baseline", "candidate")]
        self.assertEqual(evaluate(records)["decision"], "REVIEW")

    def test_correctness_regression_reviews(self):
        records = pairs()
        records[-1] = row("9", "candidate", resolved=False)
        self.assertEqual(evaluate(records)["decision"], "REVIEW")

    def test_invalid_nan(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            row(model_cost_usd=math.nan)

    def test_invalid_boolean(self):
        with self.assertRaisesRegex(ValueError, "boolean"):
            row(accepted_by_human=1)

    def test_invalid_extra_field(self):
        with self.assertRaisesRegex(ValueError, "field mismatch"):
            row(extra_field=1)

    def test_zero_correct_no_division(self):
        records = [row(str(i), v, resolved=False) for i in range(10) for v in ("baseline", "candidate")]
        report = evaluate(records)
        self.assertIsNone(report["candidate"]["cost_per_verified_correct_usd"])
        self.assertEqual(report["decision"], "REVIEW")

    def test_jsonl_load_and_validation(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "input.jsonl"
            p.write_text(json.dumps(row().__dict__) + "\n", encoding="utf-8")
            self.assertEqual(len(load_jsonl(p)), 1)
            p.write_text('{bad json}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "input.jsonl:1"):
                load_jsonl(p)

    def test_invalid_thresholds(self):
        with self.assertRaises(ValueError):
            evaluate(pairs(), min_cases=0)
        with self.assertRaises(ValueError):
            evaluate(pairs(), max_cost_ratio=math.inf)


if __name__ == "__main__":
    unittest.main()
