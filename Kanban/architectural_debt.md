# Architectural Debt (Flujo) — Current State

This document is a *practical* backlog of architectural debt that is **actually present today** in
the repo. It is written for maintainers: each item includes symptoms, likely root causes, where to
look, and what “done” means.

Non-goals:
- This is not a roadmap for new features.
- This is not a record of past refactors (only current debt + guardrails).

## How to use this doc

- **P0** items are correctness/stability risks and should block broad refactors.
- **P1** items reduce ongoing maintenance cost and prevent regressions.
- **P2** items are hygiene/ergonomics (do when touching adjacent code).

## Guardrails (must remain true)

- **Policy-driven execution**: no step-specific branching in `ExecutorCore`.
- **Control-flow exception safety**: do not swallow `PausedException`, `PipelineAbortSignal`,
  `InfiniteRedirectError`-class exceptions.
- **Context idempotency**: loops/parallel must isolate via `ContextManager.isolate()`.
- **Quota-only enforcement**: keep proactive `Reserve → Execute → Reconcile`; do not reintroduce reactive
  governor/breach patterns.
- **Centralized config**: use `infra.config_manager` helpers; no direct env/toml reads in domain logic.

## “Previously debt, now solved” (do not re-open)

These were historically painful; they are now implemented and should be preserved:

- **Async generator lifecycle leaks under xdist**: fast suite no longer emits
  `RuntimeWarning: coroutine method 'aclose' ... was never awaited` by ensuring runner/session/manager
  iterators are deterministically closed and by auto-closing iterators on terminal yields.
- **Quota is the only usage enforcement surface**: removed reactive post-exec limit checks in core
  execution + step policies; quota reservation denials now surface a stable legacy message
  (e.g. `Cost limit of $0.1 exceeded`) and include a `PipelineResult` for assertions/inspection.
- **CLI resource ownership**: `flujo run` closes the runner + state backend explicitly and no longer
  relies on broad GC/task/thread cleanup; skill registry is reset between in-process CLI runs for
  test isolation.
- **Nested HITL hard gate** via `_check_hitl_nesting_safety` raising `ConfigurationError`
  (`flujo/application/core/policy_primitives.py`).
- **Strict pricing exception propagation** (`PricingNotConfiguredError`) explicitly re-raised through
  core execution (`flujo/application/core/agent_execution_runner.py` + core stack).
- **Executor DI**: `ExecutorCore` wired via `FlujoRuntimeBuilder` and accepts injected deps
  (`flujo/application/core/executor_core.py`).
- **Scratchpad is removed**: `PipelineContext` rejects legacy `scratchpad` inputs and has typed fields
  (`flujo/domain/models.py`).
- **Unified async↔sync contract**: sync entrypoints bridge via `flujo/utils/async_bridge.py`
  (`run_sync` uses `asyncio.run()` only when no loop is running and otherwise raises with guidance).
- **Pydantic serializer warnings during context updates**: context injection now coerces dict payloads
  into model-typed fields (including PEP604 unions) so `model_dump()` is warning-free.

## P0 — Correctness & Stability

### P0.2 Unclear async↔sync contract (multiple bridges, inconsistent semantics) — Solved

Moved to “Previously debt, now solved” above. Keep sync→async bridging centralized (avoid direct
`asyncio.run()` in library/CLI surfaces).

## P1 — Architecture & Maintainability

### P1.1 Dual usage enforcement surfaces (legacy meters vs quota) — Solved

Moved to “Previously debt, now solved” above. Do not reintroduce reactive checks.

### P1.2 Serialization is fragmented (persistence vs helpers)

#### Current Status
- Serializer-warning noise from context updates is resolved; remaining debt here is consolidation of the
  persistence serialization surface (`state/backends/*` vs `flujo/utils/serialization.py`).

---

### P1.3 CLI “global cleanup” is compensating for unclear ownership — Solved

Moved to “Previously debt, now solved” above. Do not reintroduce broad “global cleanup” in CLI flows.

## P2 — Hygiene & DX

### P2.1 Python 3.13 timezone hygiene (`datetime.utcnow()` deprecations) — Solved

#### Symptoms
- Deprecation warnings from `datetime.utcnow()` in tests and fixtures.

#### Current Status
- No `utcnow()` usage remains; timestamps use `datetime.now(datetime.UTC)` and are timezone-aware.

---

### P2.2 CI micro-timing brittleness (flake detector failures) — Solved

#### Symptoms
- Ratio assertions on sub-millisecond operations fail intermittently under xdist / noisy schedulers.

#### Current Status
- The core micro-timing test uses batched measurements and P95/median ratios (not max/avg), reducing
  scheduler-outlier flakiness under xdist/randomized order.
