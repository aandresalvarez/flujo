# ADR-004: Component Decomposition

- **Status:** Accepted
- **Date:** 2025-12-02
- **Context:** `executor_core.py` and `runner.py` had grown into monoliths, blocking extensibility and testing.
- **Decision:** Decompose into focused components:
  - Executor: core composition root plus managers/handlers for quota, fallback, background tasks, caching, validation, routing, and policy registry.
  - Runner: `runner.py` (facade) delegates to `runner_methods.py`, `runner_execution.py`, and `runner_components/` (tracing manager, state backend manager, resume/replay orchestrators).
- **Consequences:**
  - ✅ Easier testing and targeted replacements.
  - ✅ Policy-driven wiring stays centralized; no per-step branching in core.
  - ⚠️ Contributors must honor module boundaries; new logic goes into the appropriate component, not back into the facade.

