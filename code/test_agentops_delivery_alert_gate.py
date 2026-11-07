import unittest
from agentops_delivery_alert_gate import Signal, Severity, assess

class GateTests(unittest.TestCase):
    def test_normal(self):
        self.assertEqual(assess(Signal(1, 20, 10, 10, 0, 1))[0], Severity.INFO)

    def test_blocked(self):
        self.assertEqual(assess(Signal(0, 20, 10, 10, 20, 2))[0], Severity.CRITICAL)

    def test_no_evidence(self):
        self.assertEqual(assess(Signal(0, 20, 10, 10, 20, 0))[0], Severity.WARNING)

    def test_spike(self):
        self.assertIn("failure_spike", assess(Signal(5, 10, 10, 10, 0, 1))[1])

    def test_stale(self):
        self.assertEqual(assess(Signal(0, 10, 10, 10, 0, 1, True))[0], Severity.CRITICAL)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            assess(Signal(2, 1, 10, 10, 0, 1))

if __name__ == "__main__":
    unittest.main()
