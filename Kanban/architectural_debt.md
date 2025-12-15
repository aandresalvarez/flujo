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
- **Nested HITL hard gate** via `_check_hitl_nesting_safety` raising `ConfigurationError`
  (`flujo/application/core/policy_primitives.py`).
- **Strict pricing exception propagation** (`PricingNotConfiguredError`) explicitly re-raised through
  core execution (`flujo/application/core/agent_execution_runner.py` + core stack).
- **Executor DI**: `ExecutorCore` wired via `FlujoRuntimeBuilder` and accepts injected deps
  (`flujo/application/core/executor_core.py`).
- **Scratchpad is removed**: `PipelineContext` rejects legacy `scratchpad` inputs and has typed fields
  (`flujo/domain/models.py`).
- **SQLite sync bridge** uses `anyio` `BlockingPortal` (no ad-hoc loop/thread spawning)
  (`flujo/state/backends/sqlite_core.py`).
- **Pydantic serializer warnings during context updates**: context injection now coerces dict payloads
  into model-typed fields (including PEP604 unions) so `model_dump()` is warning-free.

## P0 — Correctness & Stability

### P0.2 Unclear async↔sync contract (multiple bridges, inconsistent semantics)

**Symptoms**
- Different sync entrypoints behave differently depending on whether an event loop exists.
- Some code forbids sync calls inside a running loop, while other parts spawn threads/portals.

**Why it matters**
- This is a classic cause of deadlocks, blocked event loops, and shutdown weirdness in real apps.

**What exists today (inconsistent)**
- `Flujo.run()` rejects being called inside a running loop and uses `asyncio.run()`
  (`flujo/application/runner_methods.py`).
- SQLite backend runs coroutines from sync via a shared `anyio` portal
  (`flujo/state/backends/sqlite_core.py`).
- A generic helper runs coroutines by spawning a daemon thread
  (`flujo/utils/async_bridge.py`).
- CLI flows sometimes “escape hatch” with `asyncio.to_thread()` plus broad teardown
  (`flujo/cli/architect_command.py`).

**Action plan**
1. Decide one supported contract and enforce it:
   - Option A (recommended for simplicity): “Sync entrypoints are only valid when no loop is running.”
     Remove thread-bridge helpers and ensure callers use async APIs in async environments.
   - Option B: “Sync entrypoints are safe inside a running loop.”
     Standardize on `anyio` portal and remove ad-hoc thread spawning.
2. Consolidate to *one* bridge module; delete the others.
3. Add regression tests:
   - calling sync APIs inside a loop behaves as documented,
   - no threads/portals are leaked after runs.

**Done when**
- There is exactly one supported sync↔async strategy, documented, and tested.

## P1 — Architecture & Maintainability

### P1.1 Dual usage enforcement surfaces (legacy meters vs quota)

**Symptoms**
- Some “usage limit” checks are compatibility no-ops while quota enforcement happens elsewhere.
- Tests or policies can accidentally time/validate the wrong surface (micro-timing flake patterns).

**Why it matters**
- Developers can reintroduce reactive checks or think limits are enforced when they aren’t.

**Action plan**
1. Treat quota as the only enforcement mechanism in core execution.
2. Make `UsageLimits` → quota conversion occur in one boundary layer and document it.
3. Add architecture tests that fail on:
   - new reactive limit checks,
   - new “governor/breach” style plumbing.

**Done when**
- Enforcement is unambiguous: quota is required, legacy surfaces are clearly compatibility-only.

---

### P1.2 Serialization is fragmented (persistence vs helpers)

**Current status**
- Serializer-warning noise from context updates is resolved; remaining debt here is consolidation of the
  persistence serialization surface (`state/backends/*` vs `flujo/utils/serialization.py`).

---

### P1.3 CLI “global cleanup” is compensating for unclear ownership

**Symptoms**
- CLI code uses broad cleanup (GC, task cancellation, thread joins) to avoid hangs.

**Why it matters**
- It masks real leaks and makes behavior timing-dependent.

**Where to look**
- `flujo/cli/architect_command.py` teardown sections.

**Action plan**
1. Identify owned resources (runner/state backend/http clients/portals) and close them explicitly.
2. Remove global cleanup once explicit closure exists.
3. Add a CLI regression test that asserts clean shutdown without broad cleanup.

**Done when**
- CLI commands exit cleanly without enumerating threads/canceling “all tasks” as a safety net.

## P2 — Hygiene & DX

### P2.1 Python 3.13 timezone hygiene (`datetime.utcnow()` deprecations)

**Symptoms**
- Deprecation warnings from `datetime.utcnow()` in tests and fixtures.

**Action plan**
- Migrate to `datetime.now(datetime.UTC)` and keep timestamps timezone-aware end-to-end.

---

### P2.2 CI micro-timing brittleness (flake detector failures)

**Symptoms**
- Ratio assertions on sub-millisecond operations fail intermittently under xdist / noisy schedulers.

**Action plan**
- Prefer batched measurements and per-op timing; keep assertions regression-oriented.

**Done when**
- Timing tests are robust under xdist and different random seeds.
