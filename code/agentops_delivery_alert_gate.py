"""Deterministic alert gate for AI-assisted CI failure diagnosis."""
from dataclasses import dataclass
from enum import Enum

class Severity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"

@dataclass(frozen=True)
class Signal:
    failed_jobs: int
    total_jobs: int
    queue_wait_p95_s: float
    queue_wait_baseline_s: float
    blocked_train_minutes: float
    evidence_links: int
    stale_source: bool = False

def assess(s: Signal) -> tuple[Severity, tuple[str, ...]]:
    """Fail closed for invalid inputs; require evidence before automated escalation."""
    if s.total_jobs <= 0 or s.failed_jobs < 0 or s.failed_jobs > s.total_jobs:
        raise ValueError("invalid job counts")
    if min(s.queue_wait_p95_s, s.queue_wait_baseline_s,
           s.blocked_train_minutes, s.evidence_links) < 0:
        raise ValueError("negative metric")
    reasons = []
    rate = s.failed_jobs / s.total_jobs
    if s.stale_source:
        reasons.append("stale_source")
    if s.blocked_train_minutes >= 15:
        reasons.append("central_build_blocked")
    if s.total_jobs >= 10 and rate >= 0.4:
        reasons.append("failure_spike")
    if s.queue_wait_baseline_s > 0 and s.queue_wait_p95_s >= 3 * s.queue_wait_baseline_s:
        reasons.append("queue_latency_spike")
    if not reasons:
        return Severity.INFO, ()
    if s.evidence_links == 0:
        return Severity.WARNING, tuple(reasons + ["missing_evidence"])
    if "central_build_blocked" in reasons or "stale_source" in reasons:
        return Severity.CRITICAL, tuple(reasons)
    return Severity.WARNING, tuple(reasons)
