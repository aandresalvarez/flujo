# Cookbook: Budget-Aware Workflows

Design pipelines that stop spending when a cost or token budget is reached. This recipe shows
how to supply budgets, how Flujo enforces them with quotas, and how to keep loops and parallel
branches within limits.

## What you'll learn
- Set per-run budgets in code or `flujo.toml`.
- Add stricter budgets to individual steps.
- Understand how quotas protect loops and parallel branches before they execute.

## Choose where the budget comes from

### Programmatic budget (code)
```python
from flujo import Flujo, Pipeline, Step
from flujo.agents import make_agent_async
from flujo.domain.dsl.loop import LoopStep
from flujo.domain.models import UsageLimits

summarize = make_agent_async(
    "openai:gpt-4o-mini",
    "Summarize the latest findings and say whether to stop or continue.",
    dict,
)
research = make_agent_async(
    "openai:gpt-4o-mini",
    "Expand on the summary with one concrete follow-up action.",
    dict,
)

loop_body = Pipeline(
    steps=[
        Step(
            "plan",
            summarize,
            usage_limits=UsageLimits(total_cost_usd_limit=0.10, total_tokens_limit=1500),
        ),
        Step(
            "execute",
            research,
            usage_limits=UsageLimits(total_cost_usd_limit=0.20, total_tokens_limit=2500),
        ),
    ]
)

pipeline = Pipeline(
    steps=[
        LoopStep(
            name="research_loop",
            loop_body_pipeline=loop_body,
            exit_condition_callable=lambda output, ctx: (output or {}).get("action") == "stop",
            max_loops=4,
        )
    ]
)

runner = Flujo(
    pipeline,
    usage_limits=UsageLimits(total_cost_usd_limit=1.00, total_tokens_limit=6000),
)
result = await runner.run_async("Start with a two-sentence summary.")
```

### Central budget (flujo.toml)
```toml
[budgets.default]
total_cost_usd_limit = 5.0
total_tokens_limit = 20000

[budgets.pipeline]
"research-*" = { total_cost_usd_limit = 2.5 }
```
- Exact pipeline names win over wildcards; wildcards win over `budgets.default`.
- Code-provided limits and `flujo.toml` limits are combined using the most restrictive values.
- See: guides/cost_and_budget_governance.md for the complete policy rules.

## Runtime enforcement: Reserve → Execute → Reconcile
- The runner builds a root `Quota` from the resolved `UsageLimits` and shares it with every step.
- Policies estimate step usage, reserve quota **before** an API call, and raise
  `UsageLimitExceededError` if the reservation fails.
- After execution, actual usage is reconciled so unused budget is refunded.
- Loops reuse the same quota across iterations; parallel branches receive split quotas so a
  single branch cannot overspend the run-level budget.
- Pricing must be configured; otherwise `PricingNotConfiguredError` will surface before spending.

## Checklist for budget-aware pipelines
- Provide `usage_limits` when creating `Flujo` and tighten step-level `usage_limits` on the most
  expensive agents.
- Keep loop exit conditions deterministic and review them with quota awareness (see
  cookbook/safe_loop_budgeting.md).
- For YAML pipelines, place budgets in `flujo.toml` so CLI runs inherit them automatically.
- Surface budget failures to operators; the raised `UsageLimitExceededError` includes the
  partial `PipelineResult` for debugging and auditing.

## See also
- cookbook/safe_loop_budgeting.md — sharing quota across iterations without overruns.
- guides/cost_and_budget_governance.md — centralized budgeting rules and CLI helpers.
- advanced/usage_estimation.md — how estimates are produced for reservations.
