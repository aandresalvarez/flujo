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

### P1.2 Typed-context validation depth (branch/parallel/import routers + mappings)

#### Symptoms
- Step I/O contract validation is not yet consistently enforced across branch/parallel/import routers.
- Legacy scratchpad-era patterns still need guided mappings to typed fields (`status`, `step_outputs`, `import_artifacts`, etc.).

#### Where to look
- `flujo/domain/dsl/pipeline_step_validations.py`
- `flujo/domain/dsl/pipeline_validation_helpers.py`
- `flujo/domain/dsl/import_step.py`

#### Done when
- Branch/parallel/import validation is enforced with explicit, actionable errors.
- Mapping helpers (and tests) cover the remaining legacy patterns without reintroducing scratchpad acceptance.

## P2 — Hygiene & DX

- None currently tracked.
