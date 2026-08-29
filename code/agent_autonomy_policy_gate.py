#!/usr/bin/env python3
"""
Evidence-based autonomy gate for production AI agents.

The gate converts recent evaluation evidence into the maximum autonomy level
an agent is allowed to request. It is intentionally deterministic so the
decision can be reviewed, tested, and enforced in CI/CD or a runtime gateway.

Input JSON example:
{
  "sample_count": 250,
  "task_success_rate": 0.985,
  "tool_success_rate": 0.995,
  "human_override_rate": 0.01,
  "rollback_rate": 0.002,
  "critical_policy_violations": 0,
  "sandbox_violations": 0,
  "unauthorized_tool_calls": 0
}
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any


class AutonomyLevel(IntEnum):
    READ_ONLY = 0
    SUGGEST = 1
    EXECUTE_REVERSIBLE = 2
    EXECUTE_PRIVILEGED = 3

    @classmethod
    def parse(cls, value: str) -> "AutonomyLevel":
        normalized = value.strip().upper().replace("-", "_")
        aliases = {
            "READ": cls.READ_ONLY,
            "READ_ONLY": cls.READ_ONLY,
            "SUGGEST": cls.SUGGEST,
            "EXECUTE": cls.EXECUTE_REVERSIBLE,
            "EXECUTE_REVERSIBLE": cls.EXECUTE_REVERSIBLE,
            "PRIVILEGED": cls.EXECUTE_PRIVILEGED,
            "EXECUTE_PRIVILEGED": cls.EXECUTE_PRIVILEGED,
        }
        try:
            return aliases[normalized]
        except KeyError as exc:
            valid = ", ".join(level.name for level in cls)
            raise ValueError(f"Unknown autonomy level '{value}'. Valid values: {valid}") from exc


@dataclass(frozen=True)
class Evidence:
    sample_count: int
    task_success_rate: float
    tool_success_rate: float
    human_override_rate: float
    rollback_rate: float
    critical_policy_violations: int
    sandbox_violations: int
    unauthorized_tool_calls: int

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "Evidence":
        evidence = cls(
            sample_count=int(data["sample_count"]),
            task_success_rate=float(data["task_success_rate"]),
            tool_success_rate=float(data["tool_success_rate"]),
            human_override_rate=float(data["human_override_rate"]),
            rollback_rate=float(data["rollback_rate"]),
            critical_policy_violations=int(data["critical_policy_violations"]),
            sandbox_violations=int(data["sandbox_violations"]),
            unauthorized_tool_calls=int(data["unauthorized_tool_calls"]),
        )
        evidence.validate()
        return evidence

    def validate(self) -> None:
        if self.sample_count < 0:
            raise ValueError("sample_count must be >= 0")

        for name in (
            "task_success_rate",
            "tool_success_rate",
            "human_override_rate",
            "rollback_rate",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0.0 and 1.0")

        for name in (
            "critical_policy_violations",
            "sandbox_violations",
            "unauthorized_tool_calls",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")


@dataclass(frozen=True)
class Thresholds:
    reversible_min_samples: int = 100
    privileged_min_samples: int = 500
    reversible_task_success_rate: float = 0.97
    privileged_task_success_rate: float = 0.995
    reversible_tool_success_rate: float = 0.98
    privileged_tool_success_rate: float = 0.995
    reversible_max_human_override_rate: float = 0.05
    privileged_max_human_override_rate: float = 0.01
    reversible_max_rollback_rate: float = 0.02
    privileged_max_rollback_rate: float = 0.005


@dataclass(frozen=True)
class Decision:
    requested_level: str
    maximum_allowed_level: str
    approved_level: str
    allowed: bool
    reasons: tuple[str, ...]
    evidence: Evidence

    def to_json(self) -> str:
        payload = asdict(self)
        payload["reasons"] = list(self.reasons)
        return json.dumps(payload, indent=2, sort_keys=True)


def maximum_allowed_level(
    evidence: Evidence,
    thresholds: Thresholds = Thresholds(),
) -> tuple[AutonomyLevel, list[str]]:
    """
    Calculate the maximum autonomy level supported by current evidence.

    Hard safety invariants always win over aggregate quality metrics.
    """
    reasons: list[str] = []

    hard_violation_count = (
        evidence.critical_policy_violations
        + evidence.sandbox_violations
        + evidence.unauthorized_tool_calls
    )
    if hard_violation_count > 0:
        reasons.append(
            "Hard safety invariant violated: policy, sandbox, or authorization breach detected."
        )
        return AutonomyLevel.READ_ONLY, reasons

    reversible_checks = [
        (
            evidence.sample_count >= thresholds.reversible_min_samples,
            f"sample_count >= {thresholds.reversible_min_samples}",
        ),
        (
            evidence.task_success_rate >= thresholds.reversible_task_success_rate,
            f"task_success_rate >= {thresholds.reversible_task_success_rate:.3f}",
        ),
        (
            evidence.tool_success_rate >= thresholds.reversible_tool_success_rate,
            f"tool_success_rate >= {thresholds.reversible_tool_success_rate:.3f}",
        ),
        (
            evidence.human_override_rate <= thresholds.reversible_max_human_override_rate,
            f"human_override_rate <= {thresholds.reversible_max_human_override_rate:.3f}",
        ),
        (
            evidence.rollback_rate <= thresholds.reversible_max_rollback_rate,
            f"rollback_rate <= {thresholds.reversible_max_rollback_rate:.3f}",
        ),
    ]

    failed_reversible = [message for passed, message in reversible_checks if not passed]
    if failed_reversible:
        reasons.extend(f"Reversible execution requirement failed: {item}" for item in failed_reversible)
        return AutonomyLevel.SUGGEST, reasons

    privileged_checks = [
        (
            evidence.sample_count >= thresholds.privileged_min_samples,
            f"sample_count >= {thresholds.privileged_min_samples}",
        ),
        (
            evidence.task_success_rate >= thresholds.privileged_task_success_rate,
            f"task_success_rate >= {thresholds.privileged_task_success_rate:.3f}",
        ),
        (
            evidence.tool_success_rate >= thresholds.privileged_tool_success_rate,
            f"tool_success_rate >= {thresholds.privileged_tool_success_rate:.3f}",
        ),
        (
            evidence.human_override_rate <= thresholds.privileged_max_human_override_rate,
            f"human_override_rate <= {thresholds.privileged_max_human_override_rate:.3f}",
        ),
        (
            evidence.rollback_rate <= thresholds.privileged_max_rollback_rate,
            f"rollback_rate <= {thresholds.privileged_max_rollback_rate:.3f}",
        ),
    ]

    failed_privileged = [message for passed, message in privileged_checks if not passed]
    if failed_privileged:
        reasons.extend(f"Privileged execution requirement failed: {item}" for item in failed_privileged)
        return AutonomyLevel.EXECUTE_REVERSIBLE, reasons

    reasons.append("All reversible and privileged evidence thresholds passed.")
    return AutonomyLevel.EXECUTE_PRIVILEGED, reasons


def evaluate_request(
    requested: AutonomyLevel,
    evidence: Evidence,
    thresholds: Thresholds = Thresholds(),
) -> Decision:
    maximum, reasons = maximum_allowed_level(evidence, thresholds)
    approved = min(requested, maximum)
    allowed = requested <= maximum

    if allowed:
        reasons.append(f"Requested level {requested.name} is within the evidence-backed limit.")
    else:
        reasons.append(
            f"Requested level {requested.name} exceeds the evidence-backed limit {maximum.name}."
        )

    return Decision(
        requested_level=requested.name,
        maximum_allowed_level=maximum.name,
        approved_level=approved.name,
        allowed=allowed,
        reasons=tuple(reasons),
        evidence=evidence,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gate production agent autonomy using deterministic evaluation evidence."
    )
    parser.add_argument(
        "--evidence",
        required=True,
        type=Path,
        help="Path to JSON evidence file.",
    )
    parser.add_argument(
        "--request",
        required=True,
        help="Requested autonomy level.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write the JSON decision.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        data = json.loads(args.evidence.read_text(encoding="utf-8"))
        evidence = Evidence.from_mapping(data)
        requested = AutonomyLevel.parse(args.request)
        decision = evaluate_request(requested, evidence)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}, indent=2), file=sys.stderr)
        return 2

    output = decision.to_json()
    print(output)

    if args.output:
        args.output.write_text(output + "\n", encoding="utf-8")

    return 0 if decision.allowed else 1


if __name__ == "__main__":
    raise SystemExit(main())
