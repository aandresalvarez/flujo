## FSD-009: First-Class Quotas – Implementation Journal and Plan

This document tracks the ongoing work to introduce First-Class Quotas to Flujo (depends on FSD-008 Typed Step Outcomes), summarizing what’s done, current challenges, and the detailed task list to reach a production-ready rollout in line with the guidance in `FLUJO_TEAM_GUIDE.md`.

### 1) Scope and Intent (short)
- Replace reactive post-step usage checks with proactive, deterministic budget reservations that flow as a first-class object (`Quota`) through execution.
- Guarantee safety and determinism across complex control flow (Parallel, Loop, Conditional, Dynamic Router) without violating our policy-driven architecture and control-flow exception patterns.

### 2) Work completed so far
- Models and state wiring
  - Implemented `UsageEstimate` and `Quota` (thread-safe, with `reserve`, `reclaim`, `split`, `get_remaining`) in `flujo/domain/models.py`.
  - Added `quota: Optional[Quota]` to `ExecutionFrame` in `flujo/application/core/types.py`.
  - Introduced root `Quota` creation in the Runner (`flujo/application/runner.py`) from `UsageLimits` and injected it via `ExecutionManager`.
  - Propagated `quota` through the backend request path:
    - `flujo/domain/backends.py` (`StepExecutionRequest.quota`)
    - `flujo/infra/backends.py` (frame creation includes `quota`).
  - Added `CURRENT_QUOTA` (ContextVar) to `ExecutorCore` to reliably propagate quota across nested/recursive execution.

- Policy integrations
  - `DefaultAgentStepExecutor` now performs pre-execution reservation:
    - Simple `_estimate_usage` heuristic (uses `step.config.expected_tokens/expected_cost_usd` when present; otherwise minimal defaults to avoid false preemption) and calls `quota.reserve`.
    - On insufficient quota, raises `UsageLimitExceededError` with legacy-friendly messages (e.g., “Cost limit of $X exceeded” / “Token limit of N exceeded”), preserving test expectations.
    - After execution, reconciles differences via `quota.reclaim` using actual usage from `extract_usage_metrics`.
  - `DefaultParallelStepExecutor` uses the shared quota object for branches while keeping the existing `_ParallelUsageGovernor` for aggregate post-usage checks (split quotas implementation planned; see Tasks).
  - `DefaultLoopStepExecutor` relies on the same `Quota` instance across iterations; body steps reserve/reclaim normally.

- Executor and orchestration
  - `ExecutorCore` sets and forwards `quota` on any frame it constructs (via `CURRENT_QUOTA`).
  - No monolithic logic added to `ExecutorCore`; all behavior stays inside policies per `FLUJO_TEAM_GUIDE.md`.

- Estimation layer
  - Introduced pluggable `UsageEstimator` with registry + factory selection in `ExecutorCore` (adapter/validation minimal rule, heuristic default).
  - Added TOML-driven overrides under `[cost.estimators.<provider>.<model>]` for `expected_cost_usd` and `expected_tokens`.
  - Telemetry added for estimator selection, estimate used, and actual vs estimate variance to guide tuning.

- Test run snapshot (fast tests)
  - Current status (post-integration iteration): majority of tests pass; remaining failures cluster around control-flow exception propagation (Paused/HITL), message parity for legacy expectations, and fallback/conditional edge cases that now intersect with quota logic.

### 3) Key challenges observed
- Control-flow exceptions vs. quota errors (architectural)
  - PausedException and other control-flow exceptions must bypass all reservation handlers and be re-raised exactly as before. Some handlers still wrap exceptions incorrectly causing unexpected outcomes.

- Legacy error message parity
  - Several tests assert precise wording from the legacy `UsageGovernor`. Our new reservation path must emit identical messages (cost/token limits) where appropriate without distorting error categorization.

- Fallback/conditional/redirect edge-cases
  - Some branches now encounter quota errors earlier than before. We must ensure fallback and conditional handlers preserve original error semantics and do not mask control-flow exceptions.

- Parallel determinism and fairness
  - Shared-quota approach prevents overruns but does not yet provide deterministic sub-quota allocation. We’ll implement `Quota.split(n)` usage with precise, deterministic distribution and parent zeroing semantics (already implemented in the model, pending policy wiring and tests).

- Strict pricing mode interplay
  - Reservation and post-usage reconciliation must not swallow strict pricing errors. Those must surface identically to legacy flows.

### 4) Detailed task list to reach production-ready

#### A. Policy correctness and control-flow safety
- [x] Ensure `DefaultAgentStepExecutor` reservation block re-raises control-flow exceptions unmodified (PausedException, InfiniteRedirectError, MockDetectionError) before any quota logic.
- [x] Audit try/except blocks across policies (Agent/Loop/Parallel/Conditional/Router/Cache) so control-flow exceptions always bypass transformation, per “Control Flow Exception Pattern”.
- [x] Normalize error imports/usage to avoid scope issues (e.g., UnboundLocalError for `UsageLimitExceededError`).

#### B. Parallel quota splitting (deterministic)
- [x] Replace shared-quota branch execution with deterministic `quota.split(n)`; pass each sub-quota to branch frames.
- [x] Guarantee parent quota is consumed (set to zero) once split; ensure no double-spend.
- [x] Add tests for nested parallel-within-loop and loop-within-parallel to prove composition safety and determinism.

#### C. Estimation strategy (phase 1 – heuristic, phase 2 – learnable)
- [x] Phase 1: Keep safe defaults and step-config hints. Add conservative upper-bounds for known agents where feasible.
- [x] Introduce pluggable estimator interface (registry + factory) with default heuristic and adapter/validation minimal rule.
- [x] Add TOML-driven overrides for provider/model under `[cost.estimators.<provider>.<model>]` with `expected_cost_usd` and `expected_tokens`.
- [x] Telemetry: record selected estimator, estimate values, and actual vs estimate deltas post-step.
- [x] Phase 2: Add learnable/historical estimator variant wiring into the factory; gate via config (`cost.estimation_strategy = "learnable"` or `[cost.learnable] enabled=true`).

#### D. Message compatibility and UX
- [x] Centralize translation of reservation failures into legacy-style messages (cost/token exceeded) for tests that rely on string equality.
- [x] Ensure post-usage governor remains authoritative for detailed breach summaries until full migration.

#### E. Strict pricing and metrics extraction
- [x] Maintain strict pricing errors as first-class, non-masked exceptions (no new wrapping introduced by quota code).
- [x] Add regression tests where reservation succeeds but strict pricing later fails; verify surfaced failure semantics and feedback.

#### F. Conditional, Fallback, Router paths
- [x] Conditional: propagate quota to branch execution; ensure reservation failure in selected branch yields correct failure semantics without masking original feedback.
- [x] Fallback: ensure primary error + fallback error composition remains intact when quota preemption is involved; preserve metadata.
- [x] Dynamic Router: router agent executes with quota and post-selection branches get appropriate sub-quotas or shared quota per design.

#### G. Deprecation and migration of `UsageGovernor`
- [ ] Mark legacy governor as deprecated in docs; remove all reactive checks from non-parallel codepaths.
- [x] Keep `_ParallelUsageGovernor` during transition; retire once deterministic split + reservations exist and tests are updated to assert preemptive safety only.

#### H. Telemetry, diagnostics, and performance
- [x] Add telemetry breadcrumbs for reservation decisions (estimate, reserved, actual, reclaimed) at low overhead.
- [x] Estimation telemetry: record selector choice, estimate used, and actual vs estimate variance.
- [x] Micro-benchmark reservation/reclaim under high contention; validate lock contention is negligible.
- [x] Expose minimal counters for quota denials and estimation variance to support future estimator improvements.

#### I. Documentation and examples
- [x] Update advanced guides (budget aware workflows) with Quota-based flow and examples.
- [x] Add cookbook entries: deterministic parallel budget splitting; safe loop budgeting.
- [x] Migration note: how to opt in, configure estimates, and validate.
- [x] Added advanced guide: Usage Estimation configuration and tuning (TOML overrides, factory, telemetry).

#### 4.a Implementation specs for remaining tasks (aligned with Team Guide)

##### D.1 Reservation failure message formatter (spec)
- **Intent**: Centralize translation from `Quota.reserve` denials into legacy-compatible messages without leaking policy logic into `ExecutorCore`.
- **Module**: `flujo/application/core/usage_messages.py`
- **API**:
```python
from typing import Optional, Tuple
from flujo.domain.models import UsageEstimate, UsageLimits

class ReservationFailureMessage:
    def __init__(self, human: str, code: str) -> None:
        self.human: str = human           # e.g., "Cost limit of $1.00 exceeded"
        self.code: str = code             # e.g., "COST_LIMIT_EXCEEDED" | "TOKEN_LIMIT_EXCEEDED"

def format_reservation_denial(
    estimate: UsageEstimate,
    limits: UsageLimits,
) -> ReservationFailureMessage:
    """Return legacy-compatible message (string-equality stable for tests)."""
    ...
```
- **Rules** (string parity):
  - **Cost-first**: If `estimate.expected_cost_usd` exceeds `limits.max_cost_usd`, return exactly: `"Cost limit of $<limits.max_cost_usd:.2f> exceeded"`.
  - **Token second**: Else if `estimate.expected_tokens` exceeds `limits.max_tokens`, return exactly: `"Token limit of <limits.max_tokens> exceeded"`.
  - **Fallback**: Default to cost-style when both exceed; prefer most constrained resource (minimum remaining ratio).
  - **No wrapping**: The caller (policy) raises `UsageLimitExceededError` using the returned `.human` message unchanged.
- **Policy integration**: Only policies call this utility; `ExecutorCore` remains a dispatcher per `FLUJO_TEAM_GUIDE.md`.
- **Tests**: Unit tests assert exact strings and error codes; regression tests cover adapter/validation steps and control-flow exception bypass.

##### H.1 Telemetry events and metrics (spec)
- **Intent**: Low-overhead breadcrumbs for estimator selection, reservations, and reconciliation.
- **Event names**:
  - `cost.estimator.selected` — fields: `provider`, `model`, `strategy`, `expected_tokens`, `expected_cost_usd`.
  - `quota.reserve.attempt` — fields: `estimate_tokens`, `estimate_cost_usd`, `remaining_tokens`, `remaining_cost_usd`.
  - `quota.reserve.denied` — fields: `reason_code` (matches message code), `limit_tokens`, `limit_cost_usd`.
  - `quota.reconcile` — fields: `actual_tokens`, `actual_cost_usd`, `delta_tokens`, `delta_cost_usd`.
- **Counters**:
  - `quota.denials.total` (label: `reason_code`), `estimation.variance.count` (buckets by |delta|), `estimator.usage` (by `strategy`).
- **Overhead guard**: No synchronous I/O; rely on existing metrics/telemetry adapters. Single function calls per event on the hot path.

##### H.2 Performance micro-benchmark plan
- Add micro-benchmarks for `Quota.reserve/reclaim/split` under contention (N=1,4,16 threads); assert < 2% overhead vs. no-reservation baseline in hot path benchmarks.
- Validate RLock contention via synthetic parallel workloads; document results in `docs/optimization/state_backend_optimization.md` appendix.

##### I.1 Docs and examples update plan
- Update `docs/advanced/usage_estimation.md` with TOML overrides, factory selection, and telemetry fields listed above.
- Add cookbook entries:
  - `docs/cookbook/deterministic_parallel_quota.md`
  - `docs/cookbook/safe_loop_budgeting.md`
- Extend examples:
  - `examples/robust_flujo_pipeline.py`: enable `[cost]` config and demonstrate denial surfacing.
  - `examples/strict_pricing_demo.py`: cover reservation succeed + strict pricing fail scenario.

##### G.1 UsageGovernor deprecation plan
- Phase-out steps:
  1) Mark `UsageGovernor` as deprecated in docs; link to Quota docs.
  2) Remove reactive checks from non-parallel code paths; keep `_ParallelUsageGovernor` until parallel split is fully wired and tested.
  3) After deterministic split rollout, retire `_ParallelUsageGovernor`; update tests to focus on preemptive denial semantics.
- Acceptance: No behavior regressions in existing parallel tests; new tests prove deterministic fairness with `Quota.split`.

##### Configuration flags (rollout toggles)
- `cost.estimation_strategy`: `"heuristic" | "learnable"` (default: `"heuristic"`).
- `[cost.learnable] enabled`: `bool` — gates learnable estimator instantiation.
- `cost.enable_parallel_split`: `bool` — gate `Quota.split(n)` wiring in parallel policies (default: true once stabilized).
- `[cost.estimators.<provider>.<model>] expected_cost_usd, expected_tokens`: precise overrides.

##### Acceptance criteria (cross-cutting)
- String equality for denial messages in tests using the centralized formatter.
- Control-flow exceptions never wrapped or transformed by quota code.
- Deterministic branch budgeting with `Quota.split(n)` in parallel; parent zeroed, no double-spend.
- Telemetry present with negligible overhead and stable schemas.
- Comprehensive unit + integration coverage as outlined in Section 5.

### 5) Test plan (aligned with Team Guide)
- Unit tests
  - `Quota` behavior (reserve/reclaim/split/thread-safety), estimation functions, message formatting utility.
  - Policy-level tests: Agent, Loop, Parallel, Conditional, Router – focus on reservation order, control-flow exceptions, message parity.
- Integration tests
  - End-to-end pipelines with varying `UsageLimits`, strict pricing on/off, nested control-flow.
  - Deterministic parallel with `split(n)`; ensure no budget overrun and expected branches run/abort deterministically.
- Regression tests
  - Fallback semantics, conditional mapping, router selection; ensure previous guarantees remain.
- Performance
  - Measure overhead of reservations in hot paths; keep within acceptable thresholds defined by current benchmarks.

### 6) Rollout strategy
- Feature branch: `feature/FSD-009-first-class-quotas`.
- Phased rollout:
  1) Baseline reservation in Agent, shared-quota parallel (current state).
  2) Deterministic quota splitting for parallel, refine exception boundaries.
  3) Strict pricing/regression hardening, telemetry, documentation.
  4) Deprecate legacy governor paths and finalize migration notes.

### 7) Open questions (to be closed before GA)
- Should estimation enforce different policies per step type (e.g., higher conservative tokens for LLM streaming)?
- How to expose estimator pluggability without coupling policies to a concrete implementation?
- Do we gate `Quota.split` behind a configuration flag initially for safer rollout?

### 8) Decision log (high-level)
- Quota is a mutable, thread-safe object (RLock) passed by reference through frames – chosen for low overhead and composability with loops/parallel.
- Initial parallel uses shared quota to prevent regressions; move to deterministic `split(n)` next to eliminate race conditions fully and improve fairness.
- Reservation failure maps to legacy governor messages for compatibility while migrating tests incrementally.

### 9) Alignment with `FLUJO_TEAM_GUIDE.md`
- Policy-driven architecture: all logic lives in policies; `ExecutorCore` remains a dispatcher.
- Control-flow exception pattern: ensure Paused/Redirect/Mock exceptions bypass transformations.
- Idempotency and isolation: loop and parallel policies keep proper context isolation; quota is orthogonal to context state.
- Type safety and tests first: add unit/integration/regression tests for every policy path touched; maintain strict typing.
