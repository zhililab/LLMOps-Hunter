# LLMOps Weekly | 2026-11-07
## Theme: AgentOps becomes an operational control loop

**Research window:** 2026-11-01 through 2026-11-07 (Asia/Shanghai). Official releases are distinguished from proposed engineering experiments.

### 1. Verified developments
1. **November 5 — OpenTelemetry JavaScript SDK 3.0.** A major SDK upgrade and migration boundary for Node.js-based AgentOps instrumentation. For a TypeScript Central Build service, pin versions, test span continuity, and preserve a stable internal event adapter before upgrading. Source: https://opentelemetry.io/blog/
2. **November 6 — GitHub Copilot code review API management.** Review settings can be managed through REST/GraphQL, enabling policy-as-code and auditability. The same day, Copilot impact dashboards expanded agent/chat adoption coverage. Adoption is a utilization metric, not a productivity outcome. Source: https://github.blog/changelog/month/11-2026/
3. **November 3 — Dependabot support for Agent Package Manager dependencies.** Agent skills/prompts/instructions increasingly require version and dependency governance, just like CI workflows. Source: https://github.blog/changelog/month/11-2026/
4. **November 3–6 — Langfuse: skill management beta, PagerDuty alerts, AI Gateway beta, and explicit missing-token usage.** The emerging control plane combines versioned skills, runtime tracing, alert routing and cost visibility. Features differ in maturity; evaluate gateway beta separately from stable telemetry. Source: https://langfuse.com/changelog

### 2. Interpretation for EDA DevOps
The next bottleneck is not more agents: it is a reliable **signal → evidence → decision → action → verified outcome** chain. Keep merge eligibility, commit identity and policy checks deterministic. AI may explain failure clusters, but should not autonomously merge or restart expensive build trains without policy authorization.

### 3. Reproducible sample: delivery alert gate
Files: `code/agentops_delivery_alert_gate.py`, `code/test_agentops_delivery_alert_gate.py`.

```python
severity, reasons = assess(Signal(
    failed_jobs=8, total_jobs=20,
    queue_wait_p95_s=360, queue_wait_baseline_s=90,
    blocked_train_minutes=18, evidence_links=3,
))
# CRITICAL: blocked train, failure spike, queue latency spike
```

Run: `python -m unittest discover -s code -p 'test_agentops_delivery_alert_gate.py'`. The gate deliberately downgrades alerts without evidence, and checks invalid metrics; production deployment must additionally define event windows, de-duplication, suppression, owner routing and alert recovery.

### 4. One-week minimum experiment (Nov 9–13)
**Scope:** central-build queue + Jenkins precheck x86/ARM, with lyra/amsim UT clusters. Read ELK `jenkins-jobs`, `jenkins-queue`, `jenkins-logs` and Central Build Pino JSON; aggregate in 5-minute windows by module, architecture and train set. Include build URL, source SHA, queue state and evidence links. Compare against the previous 2–4 weeks.

**Deliverables:** 20 historical incident windows (10 real incidents and 10 normal windows), deterministic baseline gate, optional LLM explanation only after evidence is assembled, and a human-readable alert to the existing team channel. No automatic restart or merge.

**Acceptance criteria (targets, not measured results):** recall >= 80% on known severe incidents; false positive <= 10% on normal windows; alert latency < 5 min after threshold; 100% of critical alerts include Jenkins URL and source SHA; human-confirmed root cause tracked separately from model hypotheses. Measure p95 queue wait and build-minutes wasted, not just agent accuracy.

**Risk controls:** no Jenkins tokens in prompts or trace payloads; redact commit messages where necessary; separate local-precheck and pipeline failure domains; prevent alert storms with a per-train deduplication key.

### 5. Personalized recommendation
Your Central Build and Jenkins AI Diagnosis are already producing the ingredients for a durable EDA delivery intelligence platform. Instead of starting another broad agent project, make **one incident ledger** the shared contract between CI, ELK, diagnosis and evaluation. An incident should store `train_set_id`, `build_url`, `dispatch_sha`, `current_sha`, `failure_category`, `evidence_urls`, `human_verdict`, `time_to_recovery` and `wasted_build_minutes`. Feed verified incidents back into regression evaluation. This improves engineering productivity and creates portable platform/product skills without committing your limited learning time to a new framework.

**Decision for next week:** Can a deterministic detector catch expensive delivery failures before an LLM explains them? If yes, use the LLM only where it adds marginal diagnostic value.
