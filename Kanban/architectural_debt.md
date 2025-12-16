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

- **P1.1 ~~CLI mypy bypasses~~**: ✅ RESOLVED — Dead config removed. 4 non-existent modules were
  listed (helpers_core, helpers_runtime, helpers_extensions, helpers_project). Actual CLI modules
  pass `mypy --strict`.
- **P1.2 Non-CLI mypy bypasses**: 4 modules have `ignore_errors = true` (linters_orchestration,
  linters_extended, execution_manager_finalization, pipeline_mermaid).
- **P1.3 Core mypy relaxations**: 4 modules have 8+ disabled error codes (step_policies, execution_manager,
  step_executor, domain/dsl/step). Blocks strict type enforcement.
- **P1.4 type: ignore comments**: ~78 inline suppressions in `flujo/`. Priority targets: policies (~30),
  blueprint (~10).

## P2 — Hygiene & DX

- **P2.1 Deprecated code pending removal (target v1.0)**:
  - `OptimizationConfig` stub (`application/core/optimization_config_stub.py`)
  - `ImproperStepInvocationError` (`exceptions.py:238-245`)
  - 6 legacy global agents in `agents/recipes.py` (`__getattr__` stubs)
- **P2.2 ~~DSL lazy-import pattern~~**: ✅ REFACTORED — Replaced 70-line if-chain with 45-line lookup
  table pattern. Still uses `__getattr__` (necessary for circular import avoidance) but now more
  maintainable. See `domain/dsl/__init__.py:_LAZY_IMPORTS`.
- **P2.3 ~~Any baseline burn-down~~**: ✅ Baseline maintained per `close_gaps.md`. Current: dsl.Any=166
  (core/runtime/blueprint all at 0). See `scripts/type_safety_baseline.json`.
