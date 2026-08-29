import json
import tempfile
import unittest
from pathlib import Path

import agent_autonomy_policy_gate as gate


class AgentAutonomyPolicyGateTests(unittest.TestCase):
    def test_hard_violation_forces_read_only(self):
        evidence = gate.Evidence(
            sample_count=1000,
            task_success_rate=1.0,
            tool_success_rate=1.0,
            human_override_rate=0.0,
            rollback_rate=0.0,
            critical_policy_violations=0,
            sandbox_violations=1,
            unauthorized_tool_calls=0,
        )

        decision = gate.evaluate_request(
            gate.AutonomyLevel.EXECUTE_REVERSIBLE,
            evidence,
        )

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.maximum_allowed_level, "READ_ONLY")
        self.assertIn("Hard safety invariant violated", decision.reasons[0])

    def test_good_evidence_allows_reversible_execution(self):
        evidence = gate.Evidence(
            sample_count=250,
            task_success_rate=0.985,
            tool_success_rate=0.995,
            human_override_rate=0.01,
            rollback_rate=0.002,
            critical_policy_violations=0,
            sandbox_violations=0,
            unauthorized_tool_calls=0,
        )

        decision = gate.evaluate_request(
            gate.AutonomyLevel.EXECUTE_REVERSIBLE,
            evidence,
        )

        self.assertTrue(decision.allowed)
        self.assertEqual(decision.maximum_allowed_level, "EXECUTE_REVERSIBLE")
        self.assertEqual(decision.approved_level, "EXECUTE_REVERSIBLE")

    def test_privileged_execution_requires_stronger_evidence(self):
        evidence = gate.Evidence(
            sample_count=600,
            task_success_rate=0.996,
            tool_success_rate=0.999,
            human_override_rate=0.005,
            rollback_rate=0.001,
            critical_policy_violations=0,
            sandbox_violations=0,
            unauthorized_tool_calls=0,
        )

        decision = gate.evaluate_request(
            gate.AutonomyLevel.EXECUTE_PRIVILEGED,
            evidence,
        )

        self.assertTrue(decision.allowed)
        self.assertEqual(decision.maximum_allowed_level, "EXECUTE_PRIVILEGED")

    def test_cli_returns_one_when_request_exceeds_limit(self):
        payload = {
            "sample_count": 20,
            "task_success_rate": 0.99,
            "tool_success_rate": 0.99,
            "human_override_rate": 0.0,
            "rollback_rate": 0.0,
            "critical_policy_violations": 0,
            "sandbox_violations": 0,
            "unauthorized_tool_calls": 0,
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evidence.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            exit_code = gate.main(
                ["--evidence", str(path), "--request", "EXECUTE_REVERSIBLE"]
            )

        self.assertEqual(exit_code, 1)


if __name__ == "__main__":
    unittest.main()
