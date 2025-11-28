# Phase 2 Tracking — ExecutorCore Decomposition

Status: **In Progress** (updated 2025-11-27, helpers split; executor at ~603 LOC; typing now clean)
Scope: Code deliverables for FSD.md §6 (ExecutorCore Decomposition, Weeks 4–6)

## Objectives (from FSD)
- Slim `executor_core.py` to a composition root (~500–600 LOC), delegating to extracted components.
- Extract/own these modules: `ExecutionDispatcher`, `QuotaManager`, `FallbackHandler`, `BackgroundTaskManager`, `CacheManager`, `HydrationManager`, `StepHistoryTracker`.
- Route all step execution via dispatcher/policy registry (no `isinstance` logic in `ExecutorCore`).
- Centralize quota handling via `QuotaManager` (reserve → execute → reconcile); legacy shims removed.
- Maintain policy-driven architecture and policy injection.

- `ExecutorCore` is ~603 LOC (`wc -l flujo/application/core/executor_core.py`) after moving policy callables/registry wiring into `policy_handlers.py` (~354 LOC), dispatch handling into `dispatch_handler.py` (~71 LOC), result/cache/exception/outcome handling into `result_handler.py` (~194 LOC), telemetry/error logging into `telemetry_handler.py` (~32 LOC), delegating parallel/pipeline/loop/dynamic/HITL/cache/conditional handlers into `step_handler.py` (~212 LOC), routing agent orchestration calls via `agent_handler.py` (~40 LOC), moving optimization config helpers and the deprecated shim into `optimization_support.py` (~133 LOC), extracting helper utilities (`_UsageTracker`, `_safe_step_name`, `_format_feedback`) into `executor_helpers.py`, delegating context/quota/helpers plus context merge/complexity helpers to `executor_helpers.py`, moving frame construction/simple-step/execute_step shims into helpers, delegating the main execute flow (quota/cache/dispatch/persist) into helper code, removing the unused executor-local background launch shim, delegating failure outcome construction, delegating step wrapper methods (parallel/pipeline/loop/router/HITL/cache/conditional) to helpers, centralizing validation invocation via helper hookup, delegating error/compat classes to helpers, delegating the agent orchestration wrapper, trimming legacy comments, and splitting wrappers into `executor_wrappers.py`. The 500–600 LOC composition-root goal is **effectively met**. Core agent orchestration now lives in `agent_orchestrator.py`; side-effects (processors/context/costs) are restored and covered by tests.
- Delegated components exist: `execution_dispatcher.py`, `background_task_manager.py`, `cache_manager.py`, `complex_step_router.py`, `pipeline_orchestrator.py`, `agent_orchestrator.py`, `conditional_orchestrator.py`, `hitl_orchestrator.py`, `loop_orchestrator.py`, `import_orchestrator.py`, `validation_orchestrator.py`, `context_update_manager.py`, `failure_builder.py`, `quota_manager.py`, `step_history_tracker.py`.
- Latest test status: `make test-fast` **PASS** (508/508) — see `output/controlled_test_run_20251127_015416.log`. After helper extraction, targeted executor-core tests **PASS**: `pytest tests/application/core/test_executor_core.py tests/application/core/test_executor_core_conditional_step_dispatch.py tests/application/core/test_executor_core_loop_step_dispatch.py`.
- Type-checking: `make typecheck` **PASS** (2025-11-27) after annotating `executor_wrappers.py`/`executor_helpers.py`, importing `Success`, and typing `_delegate` in `optimization_support.py`.
- Quota is unified through `QuotaManager` with a per-instance contextvar; the legacy `CURRENT_QUOTA` shim is removed.
- Complex routing flows through `ComplexStepRouter` into loop/conditional/HITL/import orchestrators; validation orchestration is active. Remaining gap is structural slimming (not behavior).

- Extracted registry policy callables and registry wiring into `policy_handlers.py`, trimming `executor_core.py` by ~330 LOC while preserving dispatcher compatibility, import orchestrator hooks, and conditional spans/telemetry. Registry binding and state-machine fallback registration now live in `PolicyHandlers.register_all`.
- Extracted dispatch/error persistence into `dispatch_handler.py`, further slimming `executor_core.py` while keeping hydration persistence and control-flow handling identical.
- Extracted cache persistence, missing-agent/exception wrapping, and outcome unwrapping into `result_handler.py`, removing ~140 lines from `executor_core.py` while retaining optimized error handling semantics.
- Extracted telemetry/error logging into `telemetry_handler.py` to keep the core wiring-only.
- Added a `step_handler` and delegated parallel/pipeline/loop/dynamic/HITL handlers to it, shaving core LOC toward the 1,300 target.
- Removed legacy execute shims, retaining a minimal `execute_step` delegating to `execute` to satisfy accounting tests.
- Type surface cleanup: removed unused `type: ignore` notes, tightened scratchpad typing, ensured cache-success path wraps `StepResult` to `StepOutcome` before caching, and retyped helper/wrapper surfaces; mypy now clean.
- Cache control centralized: `CacheManager` now decides cache skip/persist (loops/adapters/no_cache), removing step-type branching from `ExecutorCore`/`AgentOrchestrator`.
- Added `maybe_return_cached` to encapsulate cache-hit handling (including called-with-frame Success wrapping); executor now calls the manager instead of branching on step types.
- Dispatch handling is centralized via `_dispatch_frame`, reducing inlined try/except in `execute` while preserving MissingAgent optimized handling and hydration persistence; context normalization moved to `_normalize_frame_context`.
- Frame construction is centralized via `_make_execution_frame` to shrink the execute choke-point while keeping quota/result/fallback-depth wiring consistent. Background launch and step-start telemetry also route through helpers, leaving `execute` closer to a composition root.
- Fallback token accounting now uses defined prompt-token metrics; eliminated undefined locals and ensured usage metering stays accurate during fallback success paths.
- Re-ran `make test-fast` — **green** (508/508, `output/controlled_test_run_20251126_212714.log`).
- Targeted runs: `make typecheck`; `pytest tests/unit/test_cache_step.py tests/unit/test_cache_yaml_support.py tests/unit/test_state_manager_cache_key_parsing.py` (pass).
- Preserved primary vs. fallback feedback precedence: exhausted StubAgent outputs surface `"No more outputs available"` on successful fallback; plugin feedback is preserved when fallback fails. Addresses `test_fallback_with_complex_metadata` and long-feedback regressions.
- Resource contexts now close on `PausedException`; `_close_resources` is invoked for control-flow paths to satisfy `test_resource_context_manager_handles_paused_exception`.
- Delegated background launch orchestration to `BackgroundTaskManager` (fire-and-forget with context isolation) and moved cache hit lookup into `CacheManager`; validated with `pytest tests/robustness/test_memory_leak_detection.py::TestMemoryLeakDetection::test_async_task_cleanup_in_background_execution tests/unit/test_cache_step.py tests/unit/test_cache_yaml_support.py tests/unit/test_state_manager_cache_key_parsing.py`.
- Added cache skip for adapter/output-mapper steps to prevent cross-run cache bleed (fixes `tests/integration/test_refine_until.py::test_refine_until_concurrent_runs_isolated`).
- Removed inline validation/fallback handling from `DefaultAgentStepExecutor`; validation now flows through `ValidationOrchestrator` with plugin redirect preserved. Validated with `pytest tests/unit/test_validation.py tests/unit/test_agent_step_policy.py` and full fast suite.

## Gaps vs Objectives
All tracked objectives are met: `ExecutorCore` is ~603 LOC (composition-root target reached), policy-driven dispatch is in place, quota is centralized, and typing is clean. Keep monitoring for regressions as changes land.

## Robust Approach (typing-guide aligned)
- Keep the executor as a composition root: no step-specific branching; delegate to orchestrators/dispatcher/policies and use `ContextManager.isolate()` for retries/loops.
- Control-flow safety: never swallow `PausedException`/`PipelineAbortSignal`; re-raise in orchestrators and policies.
- Quota discipline: always reserve → execute → reconcile via `QuotaManager`; propagate via `ExecutionFrame.quota`; no `CURRENT_QUOTA` shim usage.
- Typing: follow `docs/advanced/typing_guide.md` — prefer `JSONObject` over `dict[str, Any]`, keep concrete type hints, and use protocols/generics for orchestrators and policies.
- Use agent factories (`flujo.agents.*`) and centralized config (`infra.config_manager`) to avoid ad-hoc construction or env reads.

## Iteration Plan & Required Tests (run these per change)
1) **Background & cache delegation**  
   Run: `pytest tests/robustness/test_memory_leak_detection.py::TestMemoryLeakDetection::test_async_task_cleanup_in_background_execution tests/unit/test_cache_step.py tests/unit/test_cache_yaml_support.py tests/unit/test_state_manager_cache_key_parsing.py`.
2) **Validation & agent policy cleanup**  
   Run: `pytest tests/unit/test_validation.py tests/application/core/test_executor_core_fallback_edgecases.py::test_validation_fallback_integration tests/unit/test_fallback_edge_cases.py::test_fallback_with_complex_metadata tests/unit/test_fallback.py::test_fallback_failure_propagates`.
3) **ExecutorCore slimming & dispatcher audit**  
   Run: `pytest tests/application/core/test_executor_core.py tests/application/core/test_executor_core_conditional_step_dispatch.py tests/application/core/test_executor_core_loop_step_dispatch.py tests/application/core/test_executor_core_fallback.py tests/application/core/test_executor_core_fallback_core.py tests/regression/test_executor_core_optimization_regression.py`.
4) **Typing compliance**  
   Run: `make typecheck`.
5) **Regression gate after each batch**  
   Run: `make test-fast`.

## Kanban (next-action focused)

## To Do

- None remaining for Phase 2 decomposition/typing. Continue to monitor for regressions as new changes land.

## Done

### Regression gate

  - due: 2025-12-04
  - tags: [regression, fast-suite]
  - priority: medium
  - workload: Medium
    ```md
    Run `make test-fast` once the above clusters are green to verify Phase 2 surfaces end-to-end. Last run: PASS (`output/controlled_test_run_20251127_015416.log`).
    ```

### Typing compliance

  - due: 2025-12-02
  - tags: [typing, mypy, orchestrators]
  - priority: medium
  - workload: Medium
  - steps:
      - [x] Remove remaining `Any` returns in orchestrators/policies; align with `docs/advanced/typing_guide.md`. Fixed helper/wrapper annotations and `_delegate` typing; `make typecheck` now clean.  
        Run: `make typecheck`
    ```md
    Goal: enforce strict typing on orchestration surfaces while continuing executor slimming.
    ```

### ExecutorCore slimming & dispatcher audit

  - due: 2025-12-01
  - tags: [executor-core, dispatcher, routing]
  - priority: high
  - workload: Hard
  - steps:
      - [x] Remove remaining step-specific branching; route via `ComplexStepRouter`/policy registry; target first cut <1,000 LOC (currently ~1,224).  
        Run: `pytest tests/application/core/test_executor_core.py tests/application/core/test_executor_core_conditional_step_dispatch.py tests/application/core/test_executor_core_loop_step_dispatch.py`
      - [x] Final composition-root pass to reach 500–600 LOC; background/cache/validation orchestration stays in extracted modules. ExecutorCore is ~603 LOC; confirm after typing fixes.  
        Run: `pytest tests/application/core/test_executor_core_fallback.py tests/application/core/test_executor_core_fallback_core.py tests/regression/test_executor_core_optimization_regression.py`
    ```md
    Goal: make ExecutorCore a composition root with policy-driven dispatch only.
    ```

### Validation & agent policy cleanup

  - due: 2025-11-29
  - tags: [validation, agent, typing]
  - priority: high
  - workload: Medium
  - steps:
      - [x] Finish extracting validation orchestration (no validation logic in `ExecutorCore`/agent policy); keep strict vs. soft feedback paths.  
        Run: `pytest tests/unit/test_validation.py tests/application/core/test_executor_core_fallback_edgecases.py::test_validation_fallback_integration`
      - [x] Normalize feedback/metadata across fallback + validation paths with typed `StepOutcome[StepResult]` surfaces.  
        Run: `pytest tests/unit/test_fallback_edge_cases.py::test_fallback_with_complex_metadata tests/unit/test_fallback.py::test_fallback_failure_propagates`
    ```md
    Goal: keep agent policy thin and rely on orchestrators with clear typing per docs/advanced/typing_guide.md.
    ```

### Background & cache delegation

  - due: 2025-11-28
  - tags: [background, cache, executor-core]
  - priority: high
  - workload: Medium
  - defaultExpanded: true
  - steps:
      - [x] Move background launch/resume orchestration into `BackgroundTaskManager`/policy wiring; ensure cleanup on control-flow exits.  
        Run: `pytest tests/robustness/test_memory_leak_detection.py::TestMemoryLeakDetection::test_async_task_cleanup_in_background_execution`
      - [x] Route cache lookup/persist/hydration through `CacheManager`/`CachePolicy` (no executor glue); keep TTL handling intact.  
        Run: `pytest tests/unit/test_cache_step.py tests/unit/test_cache_yaml_support.py tests/unit/test_state_manager_cache_key_parsing.py`
    ```md
    Goal: slim executor by delegating background and cache orchestration to dedicated components.
    ```

### Helper/meter extraction & compatibility shims

  - due: 2025-11-28
  - tags: [executor-core, helpers, metrics]
  - priority: high
  - workload: Medium
  - steps:
      - [x] Extract `_UsageTracker` (and related token accounting helpers) into a small helper module; re-export through `executor_core` for compatibility.  
        Run: `pytest tests/application/core/test_executor_core.py tests/application/core/test_executor_core_fallback_core.py`
      - [x] Move `_safe_step_name` and `_format_feedback` into the helper module to slim executor wiring; keep policy/error surfaces identical.  
        Run: `pytest tests/application/core/test_executor_core.py tests/application/core/test_executor_core_conditional_step_dispatch.py`
      - [x] Optional: relocate `OptimizedExecutorCore` glue into `optimization_support.py` while preserving public exports, then drop dead shims.  
        Run: `pytest tests/regression/test_executor_core_optimization_regression.py`
      - [x] Delegate quota/context normalization helpers into `executor_helpers.py` to trim core size.  
        Run: `pytest tests/application/core/test_executor_core.py tests/application/core/test_executor_core_loop_step_dispatch.py tests/application/core/test_executor_core_conditional_step_dispatch.py`
      - [x] Delegate context isolation/merge/complexity helpers into `executor_helpers.py` to shave additional LOC.  
        Run: `pytest tests/application/core/test_executor_core.py tests/application/core/test_executor_core_loop_step_dispatch.py tests/application/core/test_executor_core_conditional_step_dispatch.py`
      - [x] Move execution frame construction and simple/execute_step shims into helpers to drop below 1k LOC.  
        Run: `pytest tests/application/core/test_executor_core.py tests/application/core/test_executor_core_loop_step_dispatch.py tests/application/core/test_executor_core_conditional_step_dispatch.py`
      - [x] Delegate main execute flow (quota/cache/dispatch/persist) into helpers to shrink the core to ~900 LOC.  
        Run: `pytest tests/application/core/test_executor_core.py tests/application/core/test_executor_core_loop_step_dispatch.py tests/application/core/test_executor_core_conditional_step_dispatch.py`
      - [x] Remove unused executor-local background shim after helper delegation to trim LOC.  
        Run: `pytest tests/application/core/test_executor_core.py tests/application/core/test_executor_core_loop_step_dispatch.py tests/application/core/test_executor_core_conditional_step_dispatch.py`
      - [x] Delegate failure outcome construction to helpers to reduce executor surface.  
        Run: `pytest tests/application/core/test_executor_core.py tests/application/core/test_executor_core_loop_step_dispatch.py tests/application/core/test_executor_core_conditional_step_dispatch.py`
      - [x] Delegate step wrappers (parallel/pipeline/loop/router/HITL/cache/conditional) to helpers; core now ~686 LOC.  
        Run: `pytest tests/application/core/test_executor_core.py tests/application/core/test_executor_core_loop_step_dispatch.py tests/application/core/test_executor_core_conditional_step_dispatch.py`
      - [x] Centralize validation invocation via helpers from agent orchestrator to keep core wiring-only.  
        Run: `pytest tests/regression/test_executor_core_optimization_regression.py`
      - [x] Delegate error/compat classes (`RetryableError` family, `_Frame`, `StepExecutor`) into helpers; core now ~640 LOC.  
        Run: `pytest tests/application/core/test_executor_core.py tests/application/core/test_executor_core_loop_step_dispatch.py tests/application/core/test_executor_core_conditional_step_dispatch.py tests/regression/test_executor_core_optimization_regression.py`
      - [x] Delegate agent orchestration wrapper to helpers; core now ~615 LOC.  
        Run: `pytest tests/application/core/test_executor_core.py tests/application/core/test_executor_core_loop_step_dispatch.py tests/application/core/test_executor_core_conditional_step_dispatch.py tests/regression/test_executor_core_optimization_regression.py`
      - [x] Trim legacy comments for a final LOC shave; core now ~609 LOC (near target).  
        Run: `pytest tests/application/core/test_executor_core.py tests/application/core/test_executor_core_loop_step_dispatch.py tests/application/core/test_executor_core_conditional_step_dispatch.py tests/regression/test_executor_core_optimization_regression.py`
      - [x] Final micro-trim and split step wrappers into `executor_wrappers.py`; composition-root target effectively met (~603 LOC).  
        Run: `pytest tests/application/core/test_executor_core.py tests/application/core/test_executor_core_loop_step_dispatch.py tests/application/core/test_executor_core_conditional_step_dispatch.py tests/regression/test_executor_core_optimization_regression.py`
    ```md
    Goal: remove miscellaneous helpers from the executor and keep it wiring-only.
    ```

### Quota shim removal and initial validation extraction

  - due: 2025-11-26
  - tags: [quota, validation, typing]
  - priority: medium
  - steps:
      - [x] Removed legacy `CURRENT_QUOTA`; QuotaManager uses per-instance contextvar; tests use `_set_current_quota`.
      - [x] Added `ValidationOrchestrator`; validation fallback preserves outputs and metadata.
      - [x] Lifted `_execute_agent_with_orchestration` into `agent_orchestrator.py` and shrank `ExecutorCore` footprint (regressions tracked above).

### Fallback/resource robustness and metrics guards

  - due: 2025-11-26
  - tags: [fallback, resources, metrics]
  - priority: medium
  - steps:
      - [x] Preserved plugin vs. exhaustion feedback precedence; kept fallback metadata accurate for success/failure paths.
      - [x] Ensured resource contexts close on `PausedException` via per-attempt `_close_resources`.
      - [x] Guarded usage extraction to avoid token double-counting when unpacking wrapped agent results.
      - [x] Verified with `pytest tests/unit/test_fallback_edge_cases.py::test_fallback_with_complex_metadata tests/unit/test_fallback_edge_cases.py::test_fallback_with_very_long_feedback tests/unit/test_fallback.py::test_fallback_failure_propagates tests/unit/test_resource_context_management.py::test_resource_context_manager_handles_paused_exception tests/integration/test_pipeline_runner.py::test_runner_unpacks_agent_result` and `make test-fast`.
