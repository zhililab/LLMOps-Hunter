import unittest

from agent_release_gate import (
    EXIT_OK,
    EXIT_RELIABILITY,
    GatePolicy,
    Trace,
    deterministic_sample,
    evaluate_gate,
)


class AgentReleaseGateTest(unittest.TestCase):
    def setUp(self):
        self.baseline = [
            Trace("b1", 0.92, 1000, 0.020),
            Trace("b2", 0.91, 1050, 0.021),
            Trace("b3", 0.93, 1100, 0.020),
        ]

    def test_deterministic_sampling_is_repeatable(self):
        traces = [Trace(f"t{i}", 0.9, 1000, 0.02) for i in range(100)]
        first = deterministic_sample(traces, 0.25, seed="same")
        second = deterministic_sample(traces, 0.25, seed="same")
        self.assertEqual(
            [trace.trace_id for trace in first],
            [trace.trace_id for trace in second],
        )

    def test_healthy_candidate_passes(self):
        candidate = [
            Trace("c1", 0.93, 1000, 0.020),
            Trace("c2", 0.92, 1070, 0.021),
            Trace("c3", 0.94, 1110, 0.020),
        ]

        result = evaluate_gate(self.baseline, candidate)

        self.assertTrue(result.passed)
        self.assertEqual(EXIT_OK, result.exit_code)

    def test_sandbox_violation_fails_closed(self):
        candidate = [
            Trace("c1", 0.93, 1000, 0.020),
            Trace("c2", 0.92, 1070, 0.021, sandbox_violations=1),
            Trace("c3", 0.94, 1110, 0.020),
        ]

        result = evaluate_gate(
            self.baseline,
            candidate,
            policy=GatePolicy(max_sandbox_violations=0),
        )

        self.assertFalse(result.passed)
        self.assertEqual(EXIT_RELIABILITY, result.exit_code)
        self.assertTrue(any("sandbox violation" in reason for reason in result.reasons))


if __name__ == "__main__":
    unittest.main()
