"""Compare paired, human-verified outcomes of an AI diagnosis workflow.

Only synthetic examples belong in a public repository. Production evidence,
source code, credentials, and incident details must remain in company systems.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Outcome:
    case_id: str
    variant: str
    expected_action: str
    proposed_action: str
    accepted_by_human: bool
    resolved: bool
    auto_executed: bool
    critical: bool
    source_sha: str
    verified_sha: str
    evidence_id: str
    minutes_to_resolution: float
    model_cost_usd: float

    @property
    def verified_correct(self) -> bool:
        return self.resolved and self.proposed_action == self.expected_action


def parse_outcome(raw: dict[str, Any]) -> Outcome:
    fields = Outcome.__dataclass_fields__
    if set(raw) != set(fields):
        raise ValueError(f"field mismatch: missing={sorted(set(fields) - set(raw))}, extra={sorted(set(raw) - set(fields))}")
    for key in ("case_id", "variant", "expected_action", "proposed_action", "source_sha", "verified_sha", "evidence_id"):
        if not isinstance(raw[key], str) or not raw[key].strip():
            raise ValueError(f"{key} must be a non-empty string")
    for key in ("accepted_by_human", "resolved", "auto_executed", "critical"):
        if type(raw[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    for key in ("minutes_to_resolution", "model_cost_usd"):
        if type(raw[key]) not in (int, float) or not math.isfinite(raw[key]) or raw[key] < 0:
            raise ValueError(f"{key} must be finite and non-negative")
    if raw["variant"] not in ("baseline", "candidate"):
        raise ValueError("variant must be baseline or candidate")
    return Outcome(**raw)


def load_jsonl(path: Path) -> list[Outcome]:
    records: list[Outcome] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError("record must be an object")
            records.append(parse_outcome(raw))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{path}:{lineno}: {exc}") from exc
    if not records:
        raise ValueError("no outcome records")
    return records


def _summary(rows: list[Outcome]) -> dict[str, Any]:
    correct = [r for r in rows if r.verified_correct]
    total_cost = sum(r.model_cost_usd for r in rows)
    return {
        "cases": len(rows),
        "verified_correct": len(correct),
        "verified_correct_rate": round(len(correct) / len(rows), 4),
        "human_acceptance_rate": round(sum(r.accepted_by_human for r in rows) / len(rows), 4),
        "total_model_cost_usd": round(total_cost, 4),
        "cost_per_verified_correct_usd": round(total_cost / len(correct), 4) if correct else None,
        "median_minutes_to_correct_resolution": round(statistics.median(r.minutes_to_resolution for r in correct), 3) if correct else None,
    }


def evaluate(rows: list[Outcome], *, min_cases: int = 10, max_cost_ratio: float = 1.25) -> dict[str, Any]:
    if min_cases < 1 or not math.isfinite(max_cost_ratio) or max_cost_ratio <= 0:
        raise ValueError("invalid evaluation thresholds")
    by_variant: dict[str, dict[str, Outcome]] = {"baseline": {}, "candidate": {}}
    for row in rows:
        group = by_variant[row.variant]
        if row.case_id in group:
            raise ValueError(f"duplicate case_id: {row.variant}/{row.case_id}")
        group[row.case_id] = row
    base, candidate = by_variant["baseline"], by_variant["candidate"]
    if not base or set(base) != set(candidate):
        raise ValueError("baseline and candidate must have the same non-empty case IDs")
    blocks: list[str] = []
    reviews: list[str] = []
    for case_id in sorted(base):
        before, after = base[case_id], candidate[case_id]
        if before.expected_action != after.expected_action or before.critical != after.critical:
            blocks.append(f"{case_id}:label_or_criticality_changed")
        if before.verified_sha != before.source_sha or before.source_sha != after.source_sha or after.verified_sha != after.source_sha:
            blocks.append(f"{case_id}:source_provenance_mismatch")
        if after.critical and after.proposed_action != after.expected_action:
            if after.accepted_by_human or after.auto_executed:
                blocks.append(f"{case_id}:unsafe_critical_action_accepted_or_executed")
            else:
                reviews.append(f"{case_id}:unsafe_critical_suggestion")
    before_summary = _summary(list(base.values()))
    after_summary = _summary(list(candidate.values()))
    common_correct = [case_id for case_id in sorted(base) if base[case_id].verified_correct and candidate[case_id].verified_correct]
    paired_delta = (statistics.median(candidate[c].minutes_to_resolution - base[c].minutes_to_resolution for c in common_correct)
                    if common_correct else None)
    gain = after_summary["verified_correct"] - before_summary["verified_correct"]
    if len(base) < min_cases:
        reviews.append(f"insufficient_paired_cases:{len(base)}<{min_cases}")
    if gain < 0:
        reviews.append("verified_correct_count_regressed")
    if paired_delta is not None and paired_delta > 0:
        reviews.append("paired_resolution_time_regressed")
    baseline_cost = before_summary["cost_per_verified_correct_usd"]
    candidate_cost = after_summary["cost_per_verified_correct_usd"]
    if candidate_cost is None or (baseline_cost is not None and candidate_cost > baseline_cost * max_cost_ratio):
        reviews.append("cost_per_correct_exceeded_budget")
    decision = "BLOCK" if blocks else "REVIEW" if reviews else "PILOT"
    return {
        "decision": decision,
        "baseline": before_summary,
        "candidate": after_summary,
        "verified_correct_gain": gain,
        "paired_both_correct_cases": len(common_correct),
        "median_paired_resolution_delta_minutes": round(paired_delta, 3) if paired_delta is not None else None,
        "blocks": blocks,
        "reviews": reviews,
        "caveat": "Descriptive paired replay, not a causal productivity estimate or permission to auto-merge.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--min-cases", type=int, default=10)
    parser.add_argument("--max-cost-ratio", type=float, default=1.25)
    args = parser.parse_args()
    try:
        report = evaluate(load_jsonl(args.jsonl), min_cases=args.min_cases, max_cost_ratio=args.max_cost_ratio)
    except (OSError, ValueError) as exc:
        print(json.dumps({"decision": "INVALID", "error": str(exc)}, ensure_ascii=False))
        return 4
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return {"PILOT": 0, "REVIEW": 2, "BLOCK": 3}[report["decision"]]


if __name__ == "__main__":
    raise SystemExit(main())
