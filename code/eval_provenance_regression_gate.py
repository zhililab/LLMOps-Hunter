"""Fail-closed regression gate for evidence-backed agent evaluations.

Input: two JSON arrays with the same case IDs (baseline and candidate).
Exit: 0 PASS, 2 REVIEW, 3 BLOCK, 4 invalid input.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Case:
    case_id: str
    expected_action: str
    observed_action: str
    source_sha: str
    verified_sha: str
    quality: float
    latency_ms: float
    cost_usd: float
    evaluator_id: str
    evaluator_version: str
    execution_trace_id: str
    policy_violations: int = 0
    critical: bool = False


@dataclass(frozen=True)
class GateResult:
    decision: str
    reasons: tuple[str, ...]
    metrics: dict[str, float]


def _index(cases: list[Case]) -> dict[str, Case]:
    result: dict[str, Case] = {}
    for case in cases:
        if not case.case_id or case.case_id in result:
            raise ValueError("case IDs must be unique and nonempty")
        if not 0 <= case.quality <= 1 or not all(
            math.isfinite(v) and v >= 0
            for v in (case.quality, case.latency_ms, case.cost_usd)
        ):
            raise ValueError(f"invalid numeric metrics: {case.case_id}")
        if isinstance(case.policy_violations, bool) or case.policy_violations < 0:
            raise ValueError(f"invalid policy violations: {case.case_id}")
        result[case.case_id] = case
    if not result:
        raise ValueError("evaluation dataset cannot be empty")
    return result


def evaluate(
    baseline: list[Case],
    candidate: list[Case],
    max_quality_drop: float = 0.02,
    max_cost_ratio: float = 1.20,
    max_latency_ratio: float = 1.20,
) -> GateResult:
    """Use deterministic checks; model-generated scores are inputs, not authority."""
    if not 0 <= max_quality_drop <= 1 or min(max_cost_ratio, max_latency_ratio) < 1:
        raise ValueError("invalid thresholds")
    old, new = _index(baseline), _index(candidate)
    block: list[str] = []
    review: list[str] = []

    if old.keys() != new.keys():
        block.append("dataset_case_ids_changed")

    for case_id in sorted(old.keys() & new.keys()):
        before, after = old[case_id], new[case_id]
        if before.expected_action != after.expected_action:
            block.append(f"{case_id}:expected_action_changed")
        if before.critical and not after.critical:
            block.append(f"{case_id}:criticality_downgraded")
        if before.source_sha != after.source_sha:
            block.append(f"{case_id}:evaluation_source_changed")
        if not after.source_sha or after.verified_sha != after.source_sha:
            block.append(f"{case_id}:unverified_source_sha")
        if after.policy_violations:
            block.append(f"{case_id}:policy_violation")
        if after.observed_action != after.expected_action:
            target = block if after.critical else review
            target.append(f"{case_id}:incorrect_action")
        if not all((after.evaluator_id, after.evaluator_version, after.execution_trace_id)):
            review.append(f"{case_id}:missing_score_provenance")

    old_quality = sum(x.quality for x in old.values()) / len(old)
    new_quality = sum(x.quality for x in new.values()) / len(new)
    old_cost = sum(x.cost_usd for x in old.values())
    new_cost = sum(x.cost_usd for x in new.values())
    old_latency = sum(x.latency_ms for x in old.values()) / len(old)
    new_latency = sum(x.latency_ms for x in new.values()) / len(new)

    if old_quality - new_quality > max_quality_drop + 1e-12:
        review.append("quality_regression")
    if new_cost > old_cost * max_cost_ratio + 1e-12:
        review.append("cost_regression")
    if new_latency > old_latency * max_latency_ratio + 1e-12:
        review.append("latency_regression")

    decision = "BLOCK" if block else "REVIEW" if review else "PASS"
    return GateResult(
        decision=decision,
        reasons=tuple(block + review),
        metrics={
            "baseline_quality": round(old_quality, 4),
            "candidate_quality": round(new_quality, 4),
            "baseline_cost_usd": round(old_cost, 6),
            "candidate_cost_usd": round(new_cost, 6),
            "baseline_latency_ms": round(old_latency, 2),
            "candidate_latency_ms": round(new_latency, 2),
        },
    )


def load_cases(path: Path) -> list[Case]:
    data: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("input must be a JSON array")
    return [Case(**item) for item in data]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = evaluate(load_cases(args.baseline), load_cases(args.candidate))
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(json.dumps({"decision": "ERROR", "error": str(exc)}))
        return 4
    print(json.dumps({"decision": result.decision, "reasons": result.reasons,
                      "metrics": result.metrics}, indent=2))
    return {"PASS": 0, "REVIEW": 2, "BLOCK": 3}[result.decision]


if __name__ == "__main__":
    raise SystemExit(main())
