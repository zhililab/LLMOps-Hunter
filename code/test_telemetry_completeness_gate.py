import unittest
from telemetry_completeness_gate import Window, Decision, evaluate

class GateTests(unittest.TestCase):
    def test_missing_telemetry_alerts(self):
        self.assertEqual(evaluate(Window(10, 0, 400, 3, 0))[0], Decision.ALERT)
    def test_blocked_queue_alerts(self):
        self.assertIn("central_build_queue_blocked", evaluate(Window(2, 2, 10, 4, 16))[1])
    def test_partial_coverage_reviews(self):
        self.assertEqual(evaluate(Window(10, 8, 10, 0, 0))[0], Decision.REVIEW)
    def test_idle_is_healthy(self):
        self.assertEqual(evaluate(Window(0, 0, 999, 0, 0))[0], Decision.OK)
    def test_invalid_metrics_rejected(self):
        with self.assertRaises(ValueError):
            evaluate(Window(-1, 0, 0, 0, 0))

if __name__ == "__main__":
    unittest.main()
