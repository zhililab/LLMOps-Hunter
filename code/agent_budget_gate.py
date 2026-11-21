"""Deterministic budget and evidence gate for EDA delivery agents."""
from dataclasses import dataclass
from enum import Enum

class Decision(str, Enum):
    PASS = 'PASS'
    REVIEW = 'REVIEW'
    BLOCK = 'BLOCK'

@dataclass(frozen=True)
class Observation:
    dispatch_sha: str
    current_sha: str
    trace_id: str | None
    token_usage_known: bool
    cost_usd: float | None
    budget_usd: float
    proposed_action: str
    policy_violations: int = 0

def evaluate(obs: Observation) -> tuple[Decision, tuple[str, ...]]:
    """Fail closed for unsafe actions; review incomplete accounting evidence."""
    block = []
    review = []
    if obs.policy_violations:
        block.append('policy_violation')
    if not obs.dispatch_sha or obs.dispatch_sha != obs.current_sha:
        block.append('stale_or_missing_source_sha')
    if obs.proposed_action in {'MERGE', 'DELETE', 'RESTART_PRODUCTION'}:
        block.append('privileged_action_requires_external_approval')
    if obs.budget_usd < 0 or (obs.cost_usd is not None and obs.cost_usd < 0):
        block.append('invalid_cost_data')
    if obs.cost_usd is not None and obs.cost_usd > obs.budget_usd:
        block.append('budget_exceeded')
    if not obs.trace_id:
        review.append('missing_trace_id')
    if not obs.token_usage_known or obs.cost_usd is None:
        review.append('unknown_token_usage_or_cost')
    if block:
        return Decision.BLOCK, tuple(block + review)
    if review:
        return Decision.REVIEW, tuple(review)
    return Decision.PASS, ()
