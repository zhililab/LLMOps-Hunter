# LLMOps Weekly | 2026-11-14

## Executive summary
**From observing failures to detecting missing evidence.** An empty CI dashboard is not proof of healthy delivery. Instrument the telemetry pipeline itself and require independent liveness evidence before an AI diagnosis is trusted.

## Verified developments
1. **Nov 10 — OpenTelemetry: When Missing Telemetry Is the Signal.** Missing telemetry can mean collection failure, not absence of incidents. For Jenkins/ELK, alert on expected-vs-observed build events and heartbeat gaps. Source: https://opentelemetry.io/blog/2026/ (Nov 10 entry).
2. **Nov 11 — OpenTelemetry: Deprecating the Collector Batch Processor.** Review Collector pipelines and migration guidance before upgrades; avoid blindly assuming batching configurations remain stable. Source: https://opentelemetry.io/blog/2026/ (Nov 11 entry).
3. **Nov 12 — OpenTelemetry: Agent Harness for GenAI Instrumentation.** Agent-generated instrumentation is checked against GenAI semantic conventions using Weaver and reviewed by engineers. For your Node/Pino Central Build, add schema conformance tests before exporting traces. Source: https://opentelemetry.io/blog/2026/agent-harness-genai-instrumentation/
4. **Nov 11–13 — Langfuse: evaluation and gateway controls.** Changelog lists batch evaluations via MCP, self-hosted code evaluators on an owned executor, model restrictions per API key, and visibility of missing token usage. Treat evaluation and cost provenance as explicit contracts. Source: https://langfuse.com/changelog
5. **Nov 13 — GitHub Copilot model deprecation notice and GHES integration.** Add model inventory, compatibility checks and fallback policy to agent workflows. Source: https://github.blog/changelog/month/11-2026/

## Code: Telemetry Completeness Gate
File: code/telemetry_completeness_gate.py. Pure Python; no external dependencies. Distinguishes ALERT for missing heartbeat, missing expected events or blocked queue; REVIEW for partial coverage; OK for genuinely idle or complete windows. Run: `python -m unittest discover -s code -p 'test_telemetry_completeness_gate.py'`.

```python
from telemetry_completeness_gate import Window, evaluate
status, reasons = evaluate(Window(expected_jobs=10, observed_jobs=0, last_event_age_seconds=420, queue_depth=3, blocked_minutes=0))
print(status, reasons)  # ALERT, missing telemetry reasons
```

## Personalized recommendation for EDA DevOps
Your Central Build, Jenkins precheck, K8s scheduling, and ELK pipeline are already connected conceptually, but a blind spot remains: if Filebeat/Logstash or a Jenkins exporter fails, a drop in failure counts can be misread as recovery. Use a dual signal: (A) build outcomes and (B) telemetry completeness. Tie each event to setId, buildKey, buildNumber, sourceSHA, k8s pod UID, timestamp and traceId. Keep deterministic alerting separate from LLM root-cause reasoning.

**One-week experiment:** replay 20 windows (10 normal, 5 queue incidents, 5 missing-telemetry incidents). Measure missing-event recall, false alerts, time-to-detection, and diagnosis evidence coverage. Start read-only; alert into existing channels only after review. Avoid auto-restarting pods or re-merging PRs.

**Second-order insight:** a faster diagnosis model cannot compensate for absent evidence. Improving observability completeness may deliver higher ROI than adding another agent. Optimize developer wait time and incident blast radius, not token throughput.

## Delivery acceptance
- Source references above are dated and independently reviewable.
- Code includes five unit tests; tests must be executed separately before claiming pass.
- External commits and email delivery must be confirmed from tool results, not assumed.
