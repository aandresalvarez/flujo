## Usage Estimation: Configuration and Tuning

This guide explains how Flujo estimates resource usage (cost/tokens) for quota reservations, how to configure per-provider/model hints via TOML, and how to use telemetry to tune estimates safely.

### Overview

- Estimation is pluggable and policy-driven.
- Selection precedence in agent steps:
  1. Direct estimator injected into `ExecutorCore(usage_estimator=...)`.
  2. Factory selection via registry rules (`ExecutorCore(estimator_factory=...)`).
  3. Local policy fallback heuristic (minimal/defaults).
- Deterministic quota splitting uses the selected estimate to reserve before execution.

### Default behavior

Flujo ships with a conservative default factory:
- Adapter/validation steps → minimal estimate `0 cost / 0 tokens`.
- Heuristic estimator for agents:
  - Honors `step.config.expected_cost_usd` and `step.config.expected_tokens` when present.
  - Provides small conservative bounds for known models (e.g., GPT-4 family).
  - Otherwise returns a minimal estimate.

### Configuring estimates via flujo.toml

You can override estimates per provider/model using the TOML config. Under the `[cost.estimators]` table, add provider and model specific keys:

```toml
[cost.estimators.openai."gpt-4o"]
expected_cost_usd = 0.12
expected_tokens = 600

[cost.estimators.openai."gpt-3.5-turbo"]
expected_cost_usd = 0.02
expected_tokens = 300
```

Notes:
- The provider key is the agent provider identifier (e.g., `openai`).
- The model key is the exact model name exposed by the agent (e.g., `gpt-4o`).
- You may set one or both fields.
- These hints take precedence over built-in heuristics.

### Injecting a custom estimator

Provide a custom estimator at core construction time:

```python
from flujo.application.core.runtime.estimation import UsageEstimator
from flujo.domain.models import UsageEstimate

class MyEstimator(UsageEstimator):
    def estimate(self, step, data, context) -> UsageEstimate:
        # Example: learned or historical estimate
        return UsageEstimate(cost_usd=0.05, tokens=250)

core = ExecutorCore(usage_estimator=MyEstimator())
``;

Or register rules in a factory:

```python
from flujo.application.core.runtime.estimation import EstimatorRegistry, UsageEstimatorFactory

registry = EstimatorRegistry()
registry.register(lambda s: getattr(s, "name", "").endswith("_heavy"), MyEstimator())
factory = UsageEstimatorFactory(registry)

core = ExecutorCore(estimator_factory=factory)
```

### Telemetry: estimate selection and tuning

Flujo emits low-overhead telemetry to help calibration:
- Estimator selection event: `[cost.estimator.selected]` with fields `strategy`, `expected_tokens`, `expected_cost_usd`.
- Reservation attempt event: `[quota.reserve.attempt]` with fields `estimate_tokens`, `estimate_cost_usd`, `remaining_tokens`, `remaining_cost_usd`.
- Reservation denied event: `[quota.reserve.denied]` with fields `reason_code`, and message parity.
- Reconciliation event: `[quota.reconcile]` with fields `actual_tokens`, `actual_cost_usd`, plus implicit deltas.

Counters exposed via optimized telemetry helpers:
- `estimator.usage{strategy}` – increments on selection.
- `quota.denials.total{code}` – increments on denial.
- `estimation.variance.count{type,bucket}` – buckets by absolute delta for cost and tokens (0, <=1, <=10, <=100, >100).

Use these signals to:
- Identify steps/models with large variance between estimate and actual.
- Update TOML hints or plug in a different estimator for high-variance steps.

### Best practices

- Start conservative to avoid false denials; refine with telemetry.
- Prefer per-model TOML hints for widely used models; use custom estimators for specialized steps.
- Keep estimator selection in `ExecutorCore`; keep policies free of model-specific logic per the Team Guide.

### Budget-Aware Execution

Flujo’s first-class `Quota` enables proactive, deterministic budgeting across all control-flow primitives.

- Root quota: The Runner constructs a root `Quota` from `UsageLimits` and injects it into execution frames.
- Reservation-first: Agent steps reserve based on estimate before execution; on denial the policy raises `UsageLimitExceededError` with legacy-compatible messages.
- Reconciliation: After execution, policies reconcile with actual usage via `quota.reclaim` to correct under/over-reservations.
- Parallel determinism: `Quota.split(n)` deterministically allocates sub-quotas to branches and zeroes the parent to prevent double-spend.
- Loop safety: Loops share the same `Quota` across iterations; each iteration reserves and reconciles, ensuring bounded consumption.
- Control-flow safety: `PausedException` and other control-flow exceptions bypass reservation logic and are re-raised unmodified.

Integration checklist
- Provide `UsageLimits` when constructing the Runner to enable budgeting.
- Configure `[cost.estimators]` TOML overrides for common models to reduce denial noise.
- Monitor counters: `estimator.usage{strategy}`, `quota.denials.total{code}`, `estimation.variance.count{type,bucket}`.
- For parallel or loop-heavy pipelines, validate behavior with the cookbook recipes: `cookbook/deterministic_parallel_quota.md` and `cookbook/safe_loop_budgeting.md`.


