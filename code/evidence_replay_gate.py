"""Evidence-first replay gate for AgentOps/LLMOps CI.

Turns historical production failures into deterministic regression cases.
LLMs may classify or summarize evidence, but hard delivery invariants stay in code.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class Decision(str, Enum):
    PASS = "PASS"
    REVIEW = "REVIEW"
    BLOCK = "BLOCK"


@dataclass(frozen=True)
class ReplayCase:
    case_id: str
    expected_action: str
    observed_action: str
    evidence_coverage: float
    hard_invariant_violations: tuple[str, ...] = ()


def evaluate(case: ReplayCase, min_evidence_coverage: float = 0.95) -> Decision:
    """Fail closed on hard invariants; review weak evidence or behavior drift."""
    if case.hard_invariant_violations:
        return Decision.BLOCK

    if case.evidence_coverage < min_evidence_coverage:
        return Decision.REVIEW

    if case.observed_action != case.expected_action:
        return Decision.REVIEW

    return Decision.PASS


def release_gate(cases: Iterable[ReplayCase]) -> Decision:
    """Aggregate replay cases into a release decision."""
    decisions = [evaluate(case) for case in cases]
    if Decision.BLOCK in decisions:
        return Decision.BLOCK
    if Decision.REVIEW in decisions:
        return Decision.REVIEW
    return Decision.PASS


if __name__ == "__main__":
    regression_cases = [
        ReplayCase(
            case_id="central-build-stale-success",
            expected_action="DO_NOT_MERGE",
            observed_action="DO_NOT_MERGE",
            evidence_coverage=1.0,
        ),
        ReplayCase(
            case_id="unauthorized-tool-call",
            expected_action="DENY",
            observed_action="DENY",
            evidence_coverage=1.0,
            hard_invariant_violations=("UNAUTHORIZED_TOOL_CALL",),
        ),
    ]

    print(release_gate(regression_cases).value)
