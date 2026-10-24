"""Unit tests for the evidence-backed evaluation regression gate."""

import dataclasses
import unittest

from eval_provenance_regression_gate import Case, evaluate


def fixture(**changes):
    case = Case(
        case_id="stale_success", expected_action="BLOCK_MERGE",
        observed_action="BLOCK_MERGE", source_sha="abc123", verified_sha="abc123",
        quality=0.99, latency_ms=100, cost_usd=0.10,
        evaluator_id="decision-v1", evaluator_version="1.0.0",
        execution_trace_id="trace-001", critical=True,
    )
    return dataclasses.replace(case, **changes)


class EvalProvenanceGateTests(unittest.TestCase):
    def test_pass(self):
        self.assertEqual(evaluate([fixture()], [fixture()]).decision, "PASS")

    def test_critical_wrong_action_blocks(self):
        result = evaluate([fixture()], [fixture(observed_action="MERGE")])
        self.assertEqual(result.decision, "BLOCK")

    def test_noncritical_wrong_action_reviews(self):
        result = evaluate([fixture(critical=False)],
                          [fixture(critical=False, observed_action="MERGE")])
        self.assertEqual(result.decision, "REVIEW")

    def test_sha_mismatch_blocks(self):
        self.assertEqual(evaluate([fixture()], [fixture(verified_sha="other")]).decision,
                         "BLOCK")

    def test_policy_violation_blocks(self):
        self.assertEqual(evaluate([fixture()], [fixture(policy_violations=1)]).decision,
                         "BLOCK")

    def test_missing_provenance_reviews(self):
        self.assertEqual(evaluate([fixture()], [fixture(execution_trace_id="")]).decision,
                         "REVIEW")

    def test_quality_regression_reviews(self):
        self.assertIn("quality_regression", evaluate([fixture()],
                      [fixture(quality=0.90)]).reasons)

    def test_cost_regression_reviews(self):
        self.assertIn("cost_regression", evaluate([fixture()],
                      [fixture(cost_usd=0.13)]).reasons)

    def test_latency_regression_reviews(self):
        self.assertIn("latency_regression", evaluate([fixture()],
                      [fixture(latency_ms=130)]).reasons)

    def test_dataset_drift_blocks(self):
        self.assertIn("dataset_case_ids_changed", evaluate([fixture()],
                      [fixture(case_id="other")]).reasons)

    def test_expected_action_mutation_blocks(self):
        self.assertEqual(evaluate([fixture()],
                         [fixture(expected_action="MERGE")]).decision, "BLOCK")

    def test_criticality_downgrade_blocks(self):
        self.assertIn("stale_success:criticality_downgraded",
                      evaluate([fixture()], [fixture(critical=False)]).reasons)

    def test_source_change_blocks(self):
        self.assertIn("stale_success:evaluation_source_changed",
                      evaluate([fixture()], [fixture(source_sha="new",
                                                     verified_sha="new")]).reasons)

    def test_duplicate_case_ids_rejected(self):
        with self.assertRaises(ValueError):
            evaluate([fixture(), fixture()], [fixture()])

    def test_nan_rejected(self):
        with self.assertRaises(ValueError):
            evaluate([fixture()], [fixture(quality=float("nan"))])

    def test_negative_cost_rejected(self):
        with self.assertRaises(ValueError):
            evaluate([fixture()], [fixture(cost_usd=-1)])

    def test_empty_dataset_rejected(self):
        with self.assertRaises(ValueError):
            evaluate([], [])

    def test_threshold_boundary_passes(self):
        self.assertEqual(evaluate([fixture()], [fixture(quality=0.97,
            cost_usd=0.12, latency_ms=120)]).decision, "PASS")

    def test_multiple_failures_block_precedence(self):
        result = evaluate([fixture()], [fixture(verified_sha="other",
            quality=0.8, cost_usd=0.2)])
        self.assertEqual(result.decision, "BLOCK")
        self.assertIn("quality_regression", result.reasons)


if __name__ == "__main__":
    unittest.main()
