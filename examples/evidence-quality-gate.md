# Evidence quality gate

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Evidence:
    task_success_rate: float
    evidence_coverage: float
    hard_invariant_failures: int = 0

def evaluate(e: Evidence):
    if e.hard_invariant_failures > 0:
        return "BLOCK"
    if e.task_success_rate < 0.90 or e.evidence_coverage < 0.95:
        return "REVIEW"
    return "PASS"
```

This minimal pattern keeps hard delivery invariants separate from probabilistic quality scores.