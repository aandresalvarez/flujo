# ADR-002: Context Isolation Strategy

- **Status:** Accepted
- **Date:** 2025-12-02
- **Context:** Parallel/looping steps could mutate shared context and poison retries.
- **Decision:** Every complex step (loops, parallel, conditional branches) must run inside an isolated context clone via `ContextManager.isolate()`, merging back only after success. Strict mode (`FLUJO_STRICT_CONTEXT=1`) enforces mutation checks in development.
- **Consequences:**
  - ✅ Retries run against pristine context; failed branches do not leak state.
  - ✅ Context merge errors surface as `ContextMergeError` rather than silent corruption.
  - ⚠️ Slight overhead for cloning; acceptable for correctness.

