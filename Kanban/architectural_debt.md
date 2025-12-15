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

## P0 — Correctness & Stability

### P0.1 Async lifecycle leaks (warnings indicate real leaks)

**Symptoms**
- `RuntimeWarning: coroutine ... was never awaited`
- `ResourceWarning: unclosed event loop`
- Intermittent teardown-time warnings and occasional hangs/flakes.

**Why it matters**
- This often means real leaked resources (DB connections, HTTP clients, tasks, generators).
- It makes “green CI” non-deterministic and hides correctness bugs in pause/abort paths.

**Likely root causes**
- Async generators / streams not deterministically closed on early termination.
- Cleanup implemented via “best-effort” GC / thread joining instead of explicit `aclose()`/context managers.

**Where to look**
- Streaming/outcome paths: `Flujo.run_outcomes_async`, `run_session`, coordinator/dispatcher.
- “Global cleanup” code blocks in CLI flows (they’re usually compensating for missing resource ownership).

**Action plan**
1. Add a “no leaks” test mode (or CI job) that runs a tight subset with warnings treated as errors:
   `-W error::ResourceWarning -W error::RuntimeWarning` (xdist disabled).
2. Ensure early-exit paths (`PausedException`, `PipelineAbortSignal`, hard failures) close streams/generators
   in `finally` and call `aclose()` where applicable.
3. Make resource ownership explicit:
   - runners/backends/clients provide `aclose()` (or context manager),
   - runners call `aclose()` exactly once.

**Done when**
- The “no leaks” job passes and stays stable (no warning noise) without relying on GC or thread enumeration.

---

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

### P1.2 Serialization is fragmented (persistence vs helpers vs Pydantic warnings)

**Symptoms**
- Multiple serialization helpers exist (`flujo/utils/serialization.py`, `_serialize_for_json` in state backends).
- Pydantic serializer warnings appear in tests (usually union/type mismatch issues).

**Why it matters**
- Persistence/resume/trace replay depend on deterministic serialization. Warnings often mean “silent shape drift”.

**Action plan**
1. Define a single “persistence serialization” contract and implementation (prefer `state/backends/base.py`).
2. Restrict `flujo/utils/serialization.py` to non-persistence usage (or remove it if redundant).
3. Fix model shapes that produce serializer warnings (prefer discriminated unions / stable schemas for logs).
4. Add round-trip tests for persisted workflow state and trace replay payloads.

**Done when**
- Persistence payloads round-trip deterministically and serializer warnings are eliminated (or explicitly justified).

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

