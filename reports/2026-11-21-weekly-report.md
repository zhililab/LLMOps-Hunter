# LLMOps / AgentOps 技术周报 — 2026-11-21

覆盖：2026-11-15 至 2026-11-21。主题：Agent 预算治理、评测、审计与 EDA DevOps 结果闭环。

## 本周重要更新（官方来源）

1. **11/19 Langfuse AI Gateway 预算上限**：按 API key、用户和组织设置消费上限；成本治理从事后统计走向运行时约束。https://langfuse.com/changelog
2. **11/17 Langfuse 读取审计日志**：追踪 traces、observations、scores、sessions、experiments 的读取操作。内部代码与故障日志访问同样需要审计。https://langfuse.com/changelog
3. **11/17 GitHub Copilot Evals 公测**：coding agent 评估产品化；应使用真实任务、可执行验收标准，而非调用次数。https://github.blog/changelog/
4. **11/17 GitHub Copilot Code Review Outcomes**：Repository Insights 提供 AI 审查结果指标，值得借鉴“建议→采纳→缺陷结果”的闭环。https://github.blog/changelog/
5. **11/20 Langfuse Trace 视图重设计**：帮助在复杂 Agent trace 中定位具体 observation 和工具执行。https://langfuse.com/changelog/

## 架构判断

AgentOps 从“可观测”走向“预算约束 + 数据访问审计 + 真实结果评测”。OpenTelemetry 11/12 的 Agent Harness + Weaver 实践说明：模型审查可能漏掉 telemetry schema 违规，而确定性校验器可以给出可执行的阻断。https://opentelemetry.io/blog/2026/agent-harness-genai-instrumentation/

## 本周代码：Agent Budget + Evidence Gate

新增 `code/agent_budget_gate.py` 和 `code/test_agent_budget_gate.py`；8/8 单元测试通过。标准库实现，无第三方依赖。

```python
if obs.policy_violations:
    block.append('policy_violation')
if not obs.dispatch_sha or obs.dispatch_sha != obs.current_sha:
    block.append('stale_or_missing_source_sha')
if obs.cost_usd is not None and obs.cost_usd > obs.budget_usd:
    block.append('budget_exceeded')
if not obs.token_usage_known or obs.cost_usd is None:
    review.append('unknown_token_usage_or_cost')
```

返回 PASS / REVIEW / BLOCK。该代码是本地决策组件，不能替代实际 IAM、Gateway 强制限制、人工审批或 Kubernetes 资源隔离。未知 token/cost 返回 REVIEW 而非假定为零。

## 个人建议：EDA Delivery Intelligence

- **Central Build**：用 train_set_id + source_sha + build_number 绑定诊断证据。源码变化则拒绝把旧 Jenkins SUCCESS 用于自动合入。
- **Jenkins Diagnosis / ELK**：补充 diagnosis_trace_id、evidence_coverage、token_usage_known、estimated_cost_usd、verified_outcome；避免将敏感 Prompt 全量入库。
- **Kubernetes / Precheck**：队列阻塞、Pod Pending、失败率异常先由确定性规则检测，再将最小 Evidence Bundle 交给 AI，减少不必要调用。
- **业务指标**：衡量每次被验证正确诊断的成本、真实节约的研发等待分钟数和额外 CI 开销，而不是 Agent 运行次数。

### 一周最小实验

离线回放 20 个历史窗口：5 个 stale、5 个基础设施、5 个 UT、5 个正常或误报。固定 expected classification、source SHA、必要证据与禁止动作；统计准确率、误报率、延迟、成本和可节省的排查时间。暂不开放自动合入/重启权限。

**原则：不要优化 Agent 做了多少事，而要优化每单位风险与成本换来多少被证实的工程价值。**
