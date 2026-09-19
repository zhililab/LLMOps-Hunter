# LLMOps Weekly Report — 2026-09-19

## Executive signal

LLMOps is becoming an evidence-driven control plane. Production traces increasingly feed evaluators; historical traffic can be replayed; evaluation workflows are becoming programmable; and deterministic policy gates remain the right place for hard business invariants.

## Trends

### Production evaluation becomes replayable
Langfuse added evaluator backfills on September 7, allowing recent historical observations to be scored when an evaluator is attached to a rule. Treat production failures as reusable regression assets instead of one-off incidents.

### Evaluation becomes agent-operable
Evaluation systems increasingly expose CLI, API and MCP interfaces. Coding agents can inspect low-scoring traces, build datasets and review regressions. Keep privileged configuration changes policy-gated and auditable.

### Deterministic checks complement LLM judges
Use code for objective properties such as schema validity, required tool arguments and business invariants. Use judge models for semantic properties such as diagnosis quality, with human calibration.

### Just-in-time expensive reasoning
LlamaIndex describes a two-pass document pattern: cheap parsing and search first, then expensive VLM/OCR only on selected pages. Apply the same pattern to DevOps diagnosis: narrow with logs, metrics and metadata before sending a minimal evidence bundle to a stronger model.

### Observability must itself be evaluated
Recent research on observability-aware code generation found a gap between having logging artifacts and exposing useful fault-specific runtime signals. Add regression cases that verify whether the system exposes enough evidence to distinguish stale state, dependency failure, infrastructure failure, policy violation and application failure.

## Code of the week

`code/evidence_replay_gate.py` implements a small CI-friendly gate:

```python
if case.hard_invariant_violations:
    return Decision.BLOCK
if case.evidence_coverage < 0.95:
    return Decision.REVIEW
if case.observed_action != case.expected_action:
    return Decision.REVIEW
return Decision.PASS
```

The design principle is separation of responsibilities: probabilistic AI interprets evidence; deterministic code protects invariants.

## Personalized application

Connect Central Build, Jenkins diagnosis and Kubernetes telemetry into an evidence flywheel:

```text
Production failure -> evidence -> AI diagnosis -> human correction
-> replay case -> regression evaluator -> CI gate -> safer next version
```

Start with 10–20 real Central Build/Jenkins failures, including stale post-dispatch results, concurrent source updates, rebase conflicts, missing build identity and infrastructure failures. Each case should contain input evidence, expected classification, required evidence, forbidden action and a deterministic verifier.

For expensive EDA builds, a costly failure can become a cheap permanent regression asset. The durable asset is therefore not a prompt or model choice, but the accumulated evidence corpus plus executable invariants.

## One-week experiment

1. Select 10 historical incidents.
2. Normalize them into JSON replay cases.
3. Add deterministic merge-safety checks.
4. Require the diagnosis agent to return classification plus evidence references.
5. Run the corpus in CI on every agent change.
6. Measure accuracy, evidence coverage, unsafe actions, latency and model cost.

Success criterion: improve diagnosis quality without weakening deterministic delivery invariants.

## Sources

- Langfuse: Run evaluators on historical observations, 2026-09-07.
- Langfuse: Agentic access to evaluation.
- Langfuse: Code evaluators, 2026-05-28.
- LlamaIndex: Just-in-Time Agentic OCR, 2026-09-11.
- Tao et al.: Can Large Language Models Generate Observability-Aware Code?, arXiv:2607.05785.
