# ADR-003: Exception Hierarchy Design

- **Status:** Accepted
- **Date:** 2025-12-02
- **Context:** Legacy `OrchestratorError`/`FlujoFrameworkError` created inconsistent handling and deprecation churn.
- **Decision:** Unified all framework errors under `FlujoError` with category subclasses (`ConfigurationError`, `ExecutionError`, `ControlFlowError`, `ContextError`, `ValidationError`). Control-flow signals (`PausedException`, `PipelineAbortSignal`, etc.) must be re-raised, not wrapped.
- **Consequences:**
  - ✅ Predictable handling in policies/runners.
  - ✅ Deprecation shims removed; new code imports category types.
  - ❌ Callers must migrate to category exceptions; lint/type checks enforce this.

