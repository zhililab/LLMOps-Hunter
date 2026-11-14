"""Detect missing telemetry independently of observed CI failures."""
from dataclasses import dataclass
from enum import Enum

class Decision(str, Enum):
    OK = "OK"
    REVIEW = "REVIEW"
    ALERT = "ALERT"

@dataclass(frozen=True)
class Window:
    expected_jobs: int
    observed_jobs: int
    last_event_age_seconds: int
    queue_depth: int
    blocked_minutes: int

def evaluate(w: Window, heartbeat_limit_seconds: int = 300) -> tuple[Decision, list[str]]:
    """Fail closed on absent evidence; distinguish silence from health."""
    if min(w.expected_jobs, w.observed_jobs, w.last_event_age_seconds, w.queue_depth, w.blocked_minutes) < 0:
        raise ValueError("metrics must be non-negative")
    reasons = []
    if w.expected_jobs > 0 and w.observed_jobs == 0:
        reasons.append("telemetry_missing_while_jobs_expected")
    if w.expected_jobs > 0 and w.last_event_age_seconds > heartbeat_limit_seconds:
        reasons.append("telemetry_heartbeat_expired")
    if w.blocked_minutes >= 15 and w.queue_depth > 0:
        reasons.append("central_build_queue_blocked")
    if reasons:
        return Decision.ALERT, reasons
    if w.observed_jobs < w.expected_jobs:
        return Decision.REVIEW, ["partial_telemetry_coverage"]
    return Decision.OK, []
