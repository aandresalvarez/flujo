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

- None currently tracked.
