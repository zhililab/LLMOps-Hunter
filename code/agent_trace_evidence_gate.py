"""Deterministic evidence gate for agent-assisted CI decisions."""
from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class Decision(str, Enum):
    PASS = "PASS"
    REVIEW = "REVIEW"
    BLOCK = "BLOCK"


@dataclass(frozen=True)
class TraceEvidence:
    trace_id: str
    task_success: bool
    tool_error_count: int
    policy_violation_count: int
    evidence_coverage: float
    p95_latency_ms: int


def evaluate_trace(e: TraceEvidence) -> Decision:
    """Fail closed on hard invariants; degrade uncertain cases to REVIEW."""
    if e.policy_violation_count > 0:
        return Decision.BLOCK
    if not e.task_success or e.tool_error_count > 0:
        return Decision.REVIEW
    if e.evidence_coverage < 0.95 or e.p95_latency_ms > 30_000:
        return Decision.REVIEW
    return Decision.PASS


def aggregate(evidence: Iterable[TraceEvidence]) -> Decision:
    """Return the strictest decision across a release candidate."""
    decisions = [evaluate_trace(item) for item in evidence]
    if Decision.BLOCK in decisions:
        return Decision.BLOCK
    if Decision.REVIEW in decisions:
        return Decision.REVIEW
    return Decision.PASS


if __name__ == "__main__":
    sample = TraceEvidence(
        trace_id="central-build-set-001",
        task_success=True,
        tool_error_count=0,
        policy_violation_count=0,
        evidence_coverage=0.98,
        p95_latency_ms=12_000,
    )
    print(evaluate_trace(sample).value)
