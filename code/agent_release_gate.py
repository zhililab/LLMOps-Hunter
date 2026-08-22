#!/usr/bin/env python3
"""CI-friendly release gate for LLM/agent changes.

The gate compares a candidate run against a baseline on five dimensions:
quality, latency, cost, tool reliability, and runtime safety.

Input format: JSON Lines (one trace per line). Expected fields:
- trace_id: str
- quality_score: float in [0, 1]
- latency_ms: int or float
- cost_usd: float
- tool_errors: int
- policy_violations: int
- sandbox_violations: int
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

EXIT_OK = 0
EXIT_QUALITY = 10
EXIT_COST = 11
EXIT_LATENCY = 12
EXIT_RELIABILITY = 13
EXIT_INSUFFICIENT_DATA = 14


@dataclass(frozen=True)
class Trace:
    trace_id: str
    quality_score: float
    latency_ms: float
    cost_usd: float
    tool_errors: int = 0
    policy_violations: int = 0
    sandbox_violations: int = 0


@dataclass(frozen=True)
class Summary:
    samples: int
    avg_quality: float
    p95_latency_ms: float
    avg_cost_usd: float
    tool_error_rate: float
    policy_violations: int
    sandbox_violations: int


@dataclass(frozen=True)
class GatePolicy:
    min_samples: int = 3
    max_quality_drop: float = 0.03
    max_cost_increase_ratio: float = 0.15
    max_p95_latency_increase_ratio: float = 0.20
    max_tool_error_rate: float = 0.05
    max_policy_violations: int = 0
    max_sandbox_violations: int = 0


@dataclass(frozen=True)
class GateResult:
    passed: bool
    exit_code: int
    reasons: tuple[str, ...]
    baseline: Summary
    candidate: Summary


def stable_bucket(key: str, seed: str = "llmops-release-gate") -> int:
    """Return a deterministic bucket in [0, 9999]."""
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % 10_000


def deterministic_sample(
    traces: Sequence[Trace],
    sample_rate: float,
    seed: str = "llmops-release-gate",
) -> list[Trace]:
    """Sample the same trace IDs consistently across evaluator runs."""
    if not 0 < sample_rate <= 1:
        raise ValueError("sample_rate must be in (0, 1]")

    threshold = int(sample_rate * 10_000)
    return [trace for trace in traces if stable_bucket(trace.trace_id, seed) < threshold]


def percentile_nearest_rank(values: Sequence[float], percentile: float) -> float:
    """Compute a nearest-rank percentile without external dependencies."""
    if not values:
        raise ValueError("values must not be empty")
    if not 0 < percentile <= 1:
        raise ValueError("percentile must be in (0, 1]")

    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return float(ordered[rank - 1])


def summarize(traces: Sequence[Trace]) -> Summary:
    if not traces:
        raise ValueError("cannot summarize an empty trace set")

    return Summary(
        samples=len(traces),
        avg_quality=sum(t.quality_score for t in traces) / len(traces),
        p95_latency_ms=percentile_nearest_rank(
            [t.latency_ms for t in traces],
            0.95,
        ),
        avg_cost_usd=sum(t.cost_usd for t in traces) / len(traces),
        tool_error_rate=sum(1 for t in traces if t.tool_errors > 0) / len(traces),
        policy_violations=sum(t.policy_violations for t in traces),
        sandbox_violations=sum(t.sandbox_violations for t in traces),
    )


def evaluate_gate(
    baseline: Sequence[Trace],
    candidate: Sequence[Trace],
    policy: GatePolicy = GatePolicy(),
) -> GateResult:
    baseline_summary = summarize(baseline)
    candidate_summary = summarize(candidate)

    reasons: list[str] = []
    exit_codes: list[int] = []

    if baseline_summary.samples < policy.min_samples or candidate_summary.samples < policy.min_samples:
        reasons.append(
            f"insufficient samples: baseline={baseline_summary.samples}, "
            f"candidate={candidate_summary.samples}, required={policy.min_samples}"
        )
        exit_codes.append(EXIT_INSUFFICIENT_DATA)

    quality_drop = baseline_summary.avg_quality - candidate_summary.avg_quality
    if quality_drop > policy.max_quality_drop:
        reasons.append(
            f"quality regression: drop={quality_drop:.4f}, "
            f"allowed={policy.max_quality_drop:.4f}"
        )
        exit_codes.append(EXIT_QUALITY)

    if baseline_summary.avg_cost_usd > 0:
        cost_increase = (
            candidate_summary.avg_cost_usd / baseline_summary.avg_cost_usd
        ) - 1.0
        if cost_increase > policy.max_cost_increase_ratio:
            reasons.append(
                f"cost regression: increase={cost_increase:.2%}, "
                f"allowed={policy.max_cost_increase_ratio:.2%}"
            )
            exit_codes.append(EXIT_COST)

    if baseline_summary.p95_latency_ms > 0:
        latency_increase = (
            candidate_summary.p95_latency_ms / baseline_summary.p95_latency_ms
        ) - 1.0
        if latency_increase > policy.max_p95_latency_increase_ratio:
            reasons.append(
                f"latency regression: p95 increase={latency_increase:.2%}, "
                f"allowed={policy.max_p95_latency_increase_ratio:.2%}"
            )
            exit_codes.append(EXIT_LATENCY)

    if candidate_summary.tool_error_rate > policy.max_tool_error_rate:
        reasons.append(
            f"tool reliability regression: rate={candidate_summary.tool_error_rate:.2%}, "
            f"allowed={policy.max_tool_error_rate:.2%}"
        )
        exit_codes.append(EXIT_RELIABILITY)

    if candidate_summary.policy_violations > policy.max_policy_violations:
        reasons.append(
            f"policy violation count={candidate_summary.policy_violations}, "
            f"allowed={policy.max_policy_violations}"
        )
        exit_codes.append(EXIT_RELIABILITY)

    if candidate_summary.sandbox_violations > policy.max_sandbox_violations:
        reasons.append(
            f"sandbox violation count={candidate_summary.sandbox_violations}, "
            f"allowed={policy.max_sandbox_violations}"
        )
        exit_codes.append(EXIT_RELIABILITY)

    return GateResult(
        passed=not reasons,
        exit_code=max(exit_codes) if exit_codes else EXIT_OK,
        reasons=tuple(reasons),
        baseline=baseline_summary,
        candidate=candidate_summary,
    )


def load_jsonl(path: Path) -> list[Trace]:
    traces: list[Trace] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                traces.append(Trace(**json.loads(line)))
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"{path}:{line_number}: invalid trace: {exc}") from exc
    return traces


def demo_traces() -> tuple[list[Trace], list[Trace]]:
    baseline = [
        Trace("b-1", 0.92, 1200, 0.018),
        Trace("b-2", 0.90, 1400, 0.020),
        Trace("b-3", 0.91, 1300, 0.019),
        Trace("b-4", 0.93, 1250, 0.018),
    ]
    candidate = [
        Trace("c-1", 0.93, 1260, 0.019),
        Trace("c-2", 0.92, 1420, 0.020),
        Trace("c-3", 0.91, 1360, 0.020),
        Trace("c-4", 0.94, 1290, 0.019),
    ]
    return baseline, candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, help="Baseline JSONL traces")
    parser.add_argument("--candidate", type=Path, help="Candidate JSONL traces")
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=1.0,
        help="Deterministic sample rate in (0, 1], default: 1.0",
    )
    parser.add_argument(
        "--seed",
        default="llmops-release-gate",
        help="Stable sampling seed",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if bool(args.baseline) != bool(args.candidate):
        print("Both --baseline and --candidate are required together.", file=sys.stderr)
        return EXIT_INSUFFICIENT_DATA

    if args.baseline:
        baseline = load_jsonl(args.baseline)
        candidate = load_jsonl(args.candidate)
    else:
        baseline, candidate = demo_traces()

    baseline = deterministic_sample(baseline, args.sample_rate, args.seed)
    candidate = deterministic_sample(candidate, args.sample_rate, args.seed)

    if not baseline or not candidate:
        print("Sampling produced an empty trace set.", file=sys.stderr)
        return EXIT_INSUFFICIENT_DATA

    result = evaluate_gate(baseline, candidate)
    payload = {
        "passed": result.passed,
        "exit_code": result.exit_code,
        "reasons": list(result.reasons),
        "baseline": asdict(result.baseline),
        "candidate": asdict(result.candidate),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
