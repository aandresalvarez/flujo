# Phase 2 Tracking — ExecutorCore Decomposition

Status: **In Progress** (updated 2025-11-26)  
Scope: Code deliverables for FSD.md §6 (ExecutorCore Decomposition, Weeks 4–6)

## Objectives (from FSD)
- Slim `executor_core.py` to a composition root (~500–600 LOC), delegating to extracted components.
- Extract/own these modules: `ExecutionDispatcher`, `QuotaManager`, `FallbackHandler`, `BackgroundTaskManager`, `CacheManager`, `HydrationManager`, `StepHistoryTracker`.
- Route all step execution via dispatcher/policy registry (no `isinstance` logic in `ExecutorCore`).
- Centralize quota handling via `QuotaManager` (reserve → execute → reconcile); legacy shims removed.
- Maintain policy-driven architecture and policy injection.

## Current Code State (verified 2025-11-26)
- `ExecutorCore` is ~3,402 LOC (`wc -l flujo/application/core/executor_core.py`); background launch, cache, and complex routing are delegated but the composition-root target is still ahead.
- Delegated components in place: `execution_dispatcher.py`, `background_task_manager.py`, `cache_manager.py`, `complex_step_router.py`, `pipeline_orchestrator.py`, `agent_orchestrator.py`, `conditional_orchestrator.py`, `hitl_orchestrator.py`, `loop_orchestrator.py`, `import_orchestrator.py`, `validation_orchestrator.py`, `context_update_manager.py`, `failure_builder.py`, `quota_manager.py`, `step_history_tracker.py`.
- Validation orchestration extracted and wired through the agent path; validation fallback now marks `fallback_triggered` and preserves outputs on failure.
- Quota is unified through `QuotaManager` with a per-instance contextvar; the legacy `CURRENT_QUOTA` shim is gone and tests use `_set_current_quota`.
- Complex routing flows through `ComplexStepRouter` into loop/conditional/HITL/import orchestrators; context merges/failure normalization are centralized.
- Latest verification: `make test-fast` (2025-11-26) ✅; loop dispatch and validation fallback targeted tests also ✅.

## Progress Since Last Update
- Added `_make_execution_frame` helper for orchestrators (import/validation) to build frames with the current quota.
- Fixed validation handling: outputs are preserved on validation failure, fallback metadata sets `fallback_triggered`/`validation_failure`, and StepOutcome scoping bug is removed.
- Fully removed legacy `CURRENT_QUOTA` references in favor of `QuotaManager` helpers.

## Gaps vs Objectives
1) `ExecutorCore` still ~3.4k LOC; agent/pipeline/fallback glue needs further extraction to reach the 500–600 LOC goal.
2) Inline agent retry/telemetry/fallback logic still lives in core/policies; consolidate into orchestrators for policy-driven execution.
3) Continue type hardening per `docs/advanced/typing_guide.md` (e.g., prefer `JSONObject`, strict signatures, protocol use) across new orchestrators.
4) Keep architecture/type gates green after each extraction.

## Robust Approach (typing-guide aligned)
- Keep the executor as a composition root: no step-specific branching; delegate to orchestrators/dispatcher/policies and use `ContextManager.isolate()` for retries/loops.
- Control-flow safety: never swallow `PausedException`/`PipelineAbortSignal`; re-raise in orchestrators and policies.
- Quota discipline: always reserve → execute → reconcile via `QuotaManager`; propagate via `ExecutionFrame.quota`; no `CURRENT_QUOTA` shim usage.
- Typing: follow `docs/advanced/typing_guide.md` — prefer `JSONObject` over `dict[str, Any]`, keep concrete type hints, and use protocols/generics for orchestrators and policies.
- Use agent factories (`flujo.agents.*`) and centralized config (`infra.config_manager`) to avoid ad-hoc construction or env reads.

## Iteration Plan & Required Tests (run these per change)
1) **Background/cache delegation cleanup**: finish moving background launch/caching decisions out of `ExecutorCore.execute`; ensure `BackgroundTaskManager`/`CacheManager` own lifecycle.  
   Run: `pytest tests/robustness/test_memory_leak_detection.py` and `pytest tests/application/core/test_executor_core_chokepoint.py -k background`.
2) **Loop/HITL orchestration**: keep `ComplexStepRouter` → `LoopOrchestrator`/`HitlOrchestrator` isolation with context idempotency.  
   Run: `pytest tests/application/core/test_executor_core_loop_step_dispatch.py tests/application/core/test_executor_core_execute_loop.py tests/unit/test_loop_step_policy.py`.
3) **Validation/agent path**: keep `ValidationOrchestrator` authoritative; ensure fallback metadata (`fallback_triggered`, `validation_failure`) and output preservation.  
   Run: `pytest tests/unit/test_ultra_executor.py::TestExecutorCore::test_validation_failure tests/integration/test_strict_validation.py::test_regular_step_keeps_output_on_validation_failure tests/integration/test_executor_core_fallback_integration.py::TestExecutorCoreFallbackIntegration::test_real_fallback_with_validation_failure tests/application/core/test_executor_core_fallback.py -k validation`.
4) **Quota normalization**: ensure only `QuotaManager` is used and parallel splits rely on `Quota.split()`.  
   Run: `pytest tests/unit/test_parallel_step_policy.py tests/unit/test_integration_quota_propagation.py tests/unit/test_agent_step_policy.py tests/unit/test_agent_strict_pricing.py`.
5) **Router/pipeline delegation**: trim remaining conditional/router/pipeline glue into orchestrators/dispatcher.  
   Run: `pytest tests/application/core/test_executor_core_conditional_step_dispatch.py tests/application/core/test_executor_core_conditional_step_logic.py tests/regression/test_executor_core_optimization_regression.py`.
6) **Regression gate after each batch**: run `make test-fast` (fast suite) to validate Phase 2 surfaces; rerun the targeted suites above when their areas are touched.

## Verification Status
- `make test-fast` (2025-11-26) — **PASS**
- Targeted validation/loop dispatch checks — **PASS** (`pytest tests/application/core/test_executor_core_loop_step_dispatch.py` plus validation fallback cases)

## Artifacts/References
- Key components: `execution_dispatcher.py`, `background_task_manager.py`, `cache_manager.py`, `complex_step_router.py`, `pipeline_orchestrator.py`, `agent_orchestrator.py`, `conditional_orchestrator.py`, `hitl_orchestrator.py`, `loop_orchestrator.py`, `import_orchestrator.py`, `validation_orchestrator.py`, `context_update_manager.py`, `failure_builder.py`, `quota_manager.py`, `step_history_tracker.py`.
- Executor size check: `wc -l flujo/application/core/executor_core.py` → ~3,402.
