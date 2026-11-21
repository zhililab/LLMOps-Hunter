import unittest
from dataclasses import replace
from agent_budget_gate import Observation, Decision, evaluate

class GateTests(unittest.TestCase):
    def setUp(self):
        self.base = Observation('abc','abc','trace-1',True,0.05,0.2,'DIAGNOSE')
    def test_pass(self):
        self.assertEqual(evaluate(self.base)[0], Decision.PASS)
    def test_stale_blocks(self):
        self.assertEqual(evaluate(replace(self.base,current_sha='def'))[0], Decision.BLOCK)
    def test_privileged_blocks(self):
        self.assertEqual(evaluate(replace(self.base,proposed_action='MERGE'))[0], Decision.BLOCK)
    def test_over_budget_blocks(self):
        self.assertEqual(evaluate(replace(self.base,cost_usd=0.3))[0], Decision.BLOCK)
    def test_missing_usage_reviews(self):
        self.assertEqual(evaluate(replace(self.base,token_usage_known=False))[0], Decision.REVIEW)
    def test_missing_trace_reviews(self):
        self.assertEqual(evaluate(replace(self.base,trace_id=None))[0], Decision.REVIEW)
    def test_policy_blocks(self):
        self.assertEqual(evaluate(replace(self.base,policy_violations=1))[0], Decision.BLOCK)
    def test_missing_cost_reviews(self):
        self.assertEqual(evaluate(replace(self.base,cost_usd=None))[0], Decision.REVIEW)

if __name__ == '__main__':
    unittest.main()
