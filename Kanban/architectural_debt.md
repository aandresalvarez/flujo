# Architectural Debt (Flujo) — Pending Backlog

This file lists **only** unresolved architectural debt. When an item is finished, remove it from
this backlog (do not keep “done” history here).

## Guardrails (must remain true)

- **Policy-driven execution**: no step-specific branching in `ExecutorCore`.
- **Control-flow exception safety**: never swallow `PausedException`, `PipelineAbortSignal`,
  `InfiniteRedirectError`-class exceptions.
- **Context idempotency**: loop/parallel must isolate via `ContextManager.isolate()`.
- **Quota-only enforcement**: keep proactive `Reserve → Execute → Reconcile`; do not reintroduce
  reactive governor/breach patterns.
- **Centralized config**: use `infra.config_manager` helpers; no direct env/toml reads in domain logic.

## P0 — Correctness & Stability

- None currently tracked.

## P1 — Architecture & Maintainability

- None currently tracked.

## P2 — Hygiene & DX

### P2.1 Docs/examples timezone hygiene (`datetime.utcnow()` deprecations)

#### Symptoms
- Docs/examples still use `datetime.utcnow()` (deprecated) and can encourage naive datetimes.

#### Where to look
- `docs/guides/sqlite_backend_guide.md`
- `docs/advanced/sqlite_backend_comprehensive_guide.md`
- `docs/advanced/state_backend_optimization.md`
- `examples/admin_queries_demo.py`

#### Done when
- No `datetime.utcnow()` remains in `docs/` or `examples/`; examples use timezone-aware datetimes
  (`datetime.now(timezone.utc)` or explicit `timezone.utc` conversions) and serialize via ISO8601.

---

### P2.2 Kanban/docs drift (scratchpad removal status)

#### Symptoms
- Kanban docs claim `PipelineContext.scratchpad` still exists even though the runtime rejects it.

#### Where to look
- `Kanban/close_gaps.md` (contains “Remaining: scratchpad field still exists…”)
- `Kanban/index.md` (scratchpad migration status text)

#### Done when
- Kanban docs reflect the current reality: scratchpad is removed and any remaining work is scoped
  to validation/migration guidance (not runtime removal).
