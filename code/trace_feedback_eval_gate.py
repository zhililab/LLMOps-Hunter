"""Production-trace feedback loop and CI quality gate demo.

This example shows a small but practical LLMOps pattern:

production traces -> failure mining -> eval dataset -> candidate gate

The implementation is intentionally dependency-free so it can run in a CI job.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from hashlib import sha256
from statistics import mean
from typing import Iterable
import json
import re


@dataclass(frozen=True)
class AgentTrace:
    trace_id: str
    task: str
    success: bool
    latency_ms: int
    input_tokens: int
    output_tokens: int
    tool_errors: int = 0
    human_score: float | None = None
    model: str = "unknown"
    prompt_version: str = "unknown"

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class QualitySummary:
    success_rate: float
    tool_error_rate: float
    avg_latency_ms: float
    avg_tokens: float


@dataclass(frozen=True)
class GatePolicy:
    min_success_rate: float = 0.95
    max_tool_error_rate: float = 0.02
    max_latency_regression_ratio: float = 1.20
    max_token_regression_ratio: float = 1.15


def normalize_task(task: str) -> str:
    """Normalize task text for stable deduplication."""
    normalized = re.sub(r"\s+", " ", task.strip().lower())
    normalized = re.sub(r"\b\d+\b", "<num>", normalized)
    return normalized


def task_fingerprint(task: str) -> str:
    """Create a short fingerprint for grouping repeated failure shapes."""
    return sha256(normalize_task(task).encode("utf-8")).hexdigest()[:16]


def is_eval_candidate(trace: AgentTrace) -> bool:
    """Select production traces worth converting into regression evals."""
    bad_human_feedback = trace.human_score is not None and trace.human_score < 0.8
    return (not trace.success) or trace.tool_errors > 0 or bad_human_feedback


def mine_eval_cases(traces: Iterable[AgentTrace]) -> list[dict]:
    """Convert representative production failures into deterministic eval cases."""
    cases: dict[str, dict] = {}

    for trace in traces:
        if not is_eval_candidate(trace):
            continue

        fingerprint = task_fingerprint(trace.task)
        if fingerprint in cases:
            continue

        cases[fingerprint] = {
            "case_id": f"prod-{fingerprint}",
            "instruction": trace.task,
            "source_trace_id": trace.trace_id,
            "expected": {
                "success": True,
                "max_tool_errors": 0,
            },
            "metadata": {
                "model": trace.model,
                "prompt_version": trace.prompt_version,
                "human_score": trace.human_score,
            },
        }

    return list(cases.values())


def summarize(traces: Iterable[AgentTrace]) -> QualitySummary:
    """Aggregate operational metrics used by a release gate."""
    items = list(traces)
    if not items:
        raise ValueError("At least one trace is required")

    return QualitySummary(
        success_rate=sum(t.success for t in items) / len(items),
        tool_error_rate=sum(t.tool_errors > 0 for t in items) / len(items),
        avg_latency_ms=mean(t.latency_ms for t in items),
        avg_tokens=mean(t.total_tokens for t in items),
    )


def evaluate_gate(
    baseline: QualitySummary,
    candidate: QualitySummary,
    policy: GatePolicy,
) -> tuple[bool, list[str]]:
    """Compare a candidate agent configuration against a baseline."""
    reasons: list[str] = []

    if candidate.success_rate < policy.min_success_rate:
        reasons.append(
            f"success_rate {candidate.success_rate:.3f} < {policy.min_success_rate:.3f}"
        )

    if candidate.tool_error_rate > policy.max_tool_error_rate:
        reasons.append(
            "tool_error_rate "
            f"{candidate.tool_error_rate:.3f} > {policy.max_tool_error_rate:.3f}"
        )

    if candidate.avg_latency_ms > baseline.avg_latency_ms * policy.max_latency_regression_ratio:
        reasons.append(
            "latency regression "
            f"{candidate.avg_latency_ms:.1f}ms vs {baseline.avg_latency_ms:.1f}ms"
        )

    if candidate.avg_tokens > baseline.avg_tokens * policy.max_token_regression_ratio:
        reasons.append(
            "token regression "
            f"{candidate.avg_tokens:.1f} vs {baseline.avg_tokens:.1f}"
        )

    return not reasons, reasons


def to_otel_like_attributes(trace: AgentTrace) -> dict[str, object]:
    """Map a trace to OpenTelemetry-style GenAI attributes for demonstration."""
    return {
        "gen_ai.request.model": trace.model,
        "gen_ai.usage.input_tokens": trace.input_tokens,
        "gen_ai.usage.output_tokens": trace.output_tokens,
        "llmops.prompt.version": trace.prompt_version,
        "llmops.tool.error_count": trace.tool_errors,
        "llmops.outcome.success": trace.success,
    }


def demo() -> None:
    baseline_runs = [
        AgentTrace("b1", "Analyze Jenkins build failure", True, 1800, 900, 240, model="small"),
        AgentTrace("b2", "Analyze Jenkins build failure", True, 1750, 880, 230, model="small"),
        AgentTrace("b3", "Diagnose Kubernetes OOMKill", True, 2200, 1100, 280, model="large"),
    ]

    candidate_runs = [
        AgentTrace("c1", "Analyze Jenkins build failure", True, 1700, 820, 210, model="small-v2"),
        AgentTrace("c2", "Analyze Jenkins build failure", True, 1680, 800, 205, model="small-v2"),
        AgentTrace("c3", "Diagnose Kubernetes OOMKill", False, 2150, 1050, 270, tool_errors=1, model="large-v2"),
    ]

    production_traces = [
        AgentTrace(
            "p1001",
            "Jenkins pipeline failed after a timeout in package stage",
            False,
            4200,
            1500,
            420,
            tool_errors=1,
            human_score=0.4,
            model="reasoning-model",
            prompt_version="v17",
        ),
        AgentTrace(
            "p1002",
            "Jenkins pipeline failed after a timeout in package stage 26710",
            False,
            4100,
            1480,
            405,
            tool_errors=1,
            human_score=0.5,
            model="reasoning-model",
            prompt_version="v17",
        ),
        AgentTrace(
            "p1003",
            "Explain why a Kubernetes pod is OOMKilled",
            True,
            2100,
            980,
            260,
            human_score=0.95,
            model="reasoning-model",
            prompt_version="v17",
        ),
    ]

    eval_cases = mine_eval_cases(production_traces)
    baseline = summarize(baseline_runs)
    candidate = summarize(candidate_runs)
    passed, reasons = evaluate_gate(baseline, candidate, GatePolicy())

    output = {
        "eval_cases": eval_cases,
        "baseline": asdict(baseline),
        "candidate": asdict(candidate),
        "gate": {"passed": passed, "reasons": reasons},
        "sample_otel_attributes": to_otel_like_attributes(production_traces[0]),
    }

    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    demo()
