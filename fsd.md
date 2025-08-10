FSD-009: Policy-First Decomposition of Ultra Executor

Summary
- Purpose: Decompose `flujo/application/core/ultra_executor.py` into policy-first, single-responsibility modules while preserving behavior and public interfaces.
- Why: Reduce cognitive load, improve maintainability, and align with the policy-driven architecture and control-flow principles described in FLUJO_TEAM_GUIDE.md.

Goals
- Policy-driven execution: `ExecutorCore` acts as a dispatcher; step logic lives in policies.
- Control-flow exceptions: Never swallow; always re-raise (`PausedException`, `PipelineAbortSignal`, `InfiniteRedirectError`).
- Context safety: Use `ContextManager` and `safe_merge_context_updates`; no direct field mutation.
- Config and agents: Access configuration via `ConfigManager`/`get_settings()`; instantiate agents via `flujo.agents.factory.make_agent` or `make_agent_async`.
- Backward compatibility: Maintain existing public signatures and behavior; keep legacy shims where tests expect them.

Non-Goals
- Changing DSL semantics, telemetry schema, or step APIs.
- Adjusting performance thresholds or test expectations.
- Introducing new persistence or network behaviors.

Architecture & Design
- Source of truth: Policies
  - Keep all step-specific behavior in `flujo/application/core/step_policies.py` or narrowly scoped policy modules.
  - `ExecutorCore.execute()` only routes to injected policies and manages cross-cutting concerns (caching, retries, context isolation/merge), per team guide.
- Control-flow exceptions
  - Re-raise `PausedException`, `PipelineAbortSignal`, `InfiniteRedirectError` from policies; classify in `optimized_error_handler.py`; use non-retryable strategies for control-flow.
- Context handling
  - Isolate branch contexts; merge via `safe_merge_context_updates`; never mutate context attributes directly.
- Configuration and agents
  - Resolve configuration via `get_settings()`; do not read `flujo.toml` directly.
  - Create agents using `make_agent`/`make_agent_async`; avoid ad-hoc constructors.
- Module decomposition
  - Protocols: `flujo/application/core/executor_protocols.py` (exists) centralizes `ISerializer`, `IHasher`, `ICacheBackend`, etc.
  - Default components: Add `flujo/application/core/default_components.py` for concrete defaults (e.g., `OrjsonSerializer`, `Blake3Hasher`, `InMemoryLRUBackend`, `ThreadSafeMeter`, `DefaultAgentRunner`, `DefaultValidatorRunner`, `DefaultPluginRunner`, `DefaultTelemetry`, `DefaultCacheKeyGenerator`).
  - Executor core: Extract/minimize `ExecutorCore` to orchestrate DI, routing, cache hooks, and telemetry; no step business logic.
  - Shims: Preserve legacy helper methods in `ExecutorCore` that tests patch/mock (e.g., `_handle_loop_step`, `_execute_complex_step`) but delegate to policies.

API & Compatibility
- Public API: Preserve `ExecutorCore` initialization parameters and `execute()` behavior. Keep compatibility helpers that tests import/patch.
- Import paths: Introduce new imports for default components; re-export if needed to avoid breaking users.
- Runner integration: Ensure `flujo/application/runner.py` composes `ExecutorCore` with defaults using factories/config.

Scope & Work Items
- Create `flujo/application/core/default_components.py` with default concrete implementations wired to `executor_protocols.py`.
- Ensure all protocol types live in `executor_protocols.py` (already present); remove duplicates from `ultra_executor.py`.
- Minimize `ultra_executor.py` by:
  - Keeping routing and compatibility shims only.
  - Moving concrete default classes and helpers into `default_components.py`.
- Update imports across core and runner to use new modules.
- Add deprecation notes or re-exports where necessary to avoid breaking external imports.

Control-Flow & Error Handling Requirements
- Policies must re-raise control-flow exceptions; never convert them to `StepResult` failures.
- Classify new exception types in `optimized_error_handler.py` and map to non-retryable strategies.
- Maintain fallback graph protections and iteration caps; keep infinite-loop detection as specified in the guide.

Context & Data Handling Requirements
- Use `ContextManager` APIs and `safe_merge_context_updates` for all merges.
- Isolate branch/loop contexts and merge back only on success as appropriate.
- Avoid direct attribute writes on context; prefer safe merge/update utilities.

Testing Strategy
- Unit: Target policy behaviors, executor routing, and deprecation shims.
- Integration: Pipelines with loops, conditionals, parallel, dynamic router, cache, and HITL paths.
- E2E: Representative example pipelines to validate behavior parity.
- Quality gates: `make all` must pass (ruff format/check, mypy --strict, pytest). Do not change tests to “make green”; fix root causes.

Performance & Telemetry
- Preserve current performance characteristics and thresholds; do not raise thresholds.
- Keep existing telemetry fields; add new fields only if backward-compatible.

Risks & Mitigations
- Behavior drift: Mitigate via exhaustive tests and maintaining shims that delegate to policies.
- Import churn: Provide re-exports and clear deprecation path.
- Control-flow mishandling: Enforce re-raise pattern in code reviews and tests.

Acceptance Criteria
- Executor remains policy-driven; no step-specific logic lives in `ExecutorCore`.
- `executor_protocols.py` holds all protocol interfaces; no duplicates in `ultra_executor.py`.
- `default_components.py` exists and provides default DI components used by `ExecutorCore`.
- Runner composes `ExecutorCore` via defaults pulled from config and factories.
- All imports build cleanly; `mypy --strict` and `ruff check` pass; full test suite passes.

Rollout Plan
- Phase 1: Introduce `default_components.py`, update imports, keep `ultra_executor.py` routing and shims.
- Phase 2: Remove duplicated concrete classes from `ultra_executor.py`; add re-exports if necessary.
- Phase 3: Optional renames/refactors inside policies with tests guarding behavior.

Open Questions
- Do we want a dedicated `policies/` package to split `step_policies.py` by step type?
- Should `ExecutorCore` be re-homed as `execution_manager.py` for clarity and keep `ultra_executor.py` as a thin compatibility wrapper for one release?
