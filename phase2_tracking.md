# Phase 2 Tracking — ExecutorCore Decomposition

Status: **In Progress**  
Scope: Code deliverables for FSD.md §6 (ExecutorCore Decomposition, Weeks 4–6)

## Objectives (from FSD)
- Slim `executor_core.py` to a composition root (~500–600 LOC), delegating to extracted components.
- Extract/own these modules: `ExecutionDispatcher`, `QuotaManager`, `FallbackHandler`, `BackgroundTaskManager`, `CacheManager`, `HydrationManager`, `StepHistoryTracker`.
- Route all step execution via dispatcher/policy registry (no `isinstance` logic in `ExecutorCore`).
- Centralize quota handling via `QuotaManager` (reserve → execute → reconcile); legacy `CURRENT_QUOTA` shim removed from code paths.
- Maintain policy-driven architecture and policy injection.

## Current Code State (verified 2025-11-26)
- Extracted modules present: `quota_manager.py`, `fallback_handler.py`, `background_task_manager.py`, `cache_manager.py`, `hydration_manager.py`, `step_history_tracker.py`, `execution_dispatcher.py`, `loop_executor.py` (thin wrapper). Missing: `agent_orchestrator.py`, `validation_orchestrator.py`, `context_update_manager.py`, `policy_invoker.py`, `complex_step_router.py`, `pipeline_orchestrator.py`, `failure_builder.py`.
- `ExecutorCore` remains a 3,966-line monolith (`wc -l flujo/application/core/executor_core.py`) with background launch, cache lookup/metadata wiring, complex-step branching, and legacy loop logic implemented inline.
- Dispatcher is wired (`ExecutionDispatcher` + `PolicyRegistry`), but `execute` still branches on `ConditionalStep`/`LoopStep`/routers/HITL via `_execute_complex_step` and handles cache/background orchestration directly instead of delegating to extracted managers/policies.
- Quota: `QuotaManager` exists but `ExecutorCore.CURRENT_QUOTA` is still exposed and used across policies (`agent_policy`, `parallel_policy`, `router_policy`, `cache_policy`) and `step_coordinator`; reserve → execute → reconcile is not centralized and the contextvar has not been removed.
- Background/cache: `BackgroundTaskManager`/`CacheManager` exist, yet core performs background task creation and cache get/metadata injection inline; cache hits use `_cache_backend` directly.
- Agent path: no agent/validation orchestrator extraction; `DefaultAgentStepExecutor` still contains full orchestration, retries, and fallbacks.
- Testing/architecture: not re-run in this pass; gate status currently unknown.

## Gaps vs Objectives
1) **Monolith not slimmed**: `ExecutorCore` still ~4k LOC with cache/background/complex-step orchestration inline; far above 500–600 LOC target.
2) **Missing extractions**: Agent/validation/fallback/pipeline/complex-router/context-update/failure-builder components do not exist despite being claimed; step-specific branching remains in core.
3) **Quota integration incomplete**: `CURRENT_QUOTA` persists in executor and policies; reserve → execute → reconcile is not standardized through `QuotaManager`.
4) **Delegation gaps**: Background/cache/context merge handling and complex routing live in core with `isinstance` checks instead of being policy/manager-owned.
5) **Verification unknown**: Architecture/type/test gate outcomes not validated in this check.

## Execution Plan (robust, typing-safe)
- Baseline gates: run `make test-fast` and the architecture suite to capture current failures (keep `PYTEST_DISABLE_PLUGIN_AUTOLOAD` unset). Use results to validate each refactor increment.
- Quota normalization first: remove `CURRENT_QUOTA` usage from executor/policies/step coordinator; pass quotas via `ExecutionFrame.quota` and enforce reserve → execute → reconcile in every policy through `QuotaManager`.
- Core slimming with extractions:
  - Create `agent_orchestrator.py` and `validation_orchestrator.py` to own agent retries/fallback/plugins; keep core delegators for compatibility during migration.
  - Add `pipeline_orchestrator.py` for sequential pipelines and move branch/router/HITL/hydration glue out of core; introduce `complex_step_router.py` to replace `_execute_complex_step` `isinstance` routing.
  - Add `context_update_manager.py` for context merges (incl. HITL pause updates) and `failure_builder.py` for outcome construction; delete inline equivalents.
- Cache/background delegation: move cache hit/miss logic from `ExecutorCore.execute` into `CacheManager`/cache policy; move background task creation/tracking into `BackgroundTaskManager` with a policy hook to avoid recursion.
- Policy routing cleanup: ensure dispatcher registry covers conditional/router/HITL/import/cache/loop; remove remaining `isinstance` branches in `execute`/`_execute_complex_step` after router/policies are wired.
- Typing discipline (see `docs/advanced/typing_guide.md`): avoid `Dict[str, Any]` for JSON (use `JSONObject`); keep protocols for policy interfaces; ensure return/param annotations and 100-col lines; maintain `TContext_w_Scratch` bounds.
- Verification cadence: after each extraction milestone, run the targeted tests covering touched areas (e.g., `pytest tests/application/core/test_executor_core_execute_loop.py` for loop changes; `pytest tests/application/core/test_executor_core.py` and `tests/unit/test_parallel_step_policy.py` for dispatcher/quota/parallel edits; architecture suite for policy/routing changes). Only after milestones are stable, run `make test-fast`; reserve `make all` for pre-PR finalization.

## Immediate Next Steps
- Stand up missing orchestrator modules and map current inline logic (agent/validation/pipeline/complex routing/failure handling) into them.
- Extract background launch and cache handling out of `ExecutorCore.execute` to shrink LOC and reduce choke-point risk.
- Replace `CURRENT_QUOTA` reads/writes with `QuotaManager` flows and propagate quotas through `ExecutionFrame`.
- Re-run `make test-fast` and architecture suite to validate the above moves, then iterate.

## Artifacts/References
- `flujo/application/core/executor_core.py` — 3,966 LOC; contains background/cache/complex-step orchestration and legacy loop logic.
- Present helpers: `execution_dispatcher.py`, `quota_manager.py`, `fallback_handler.py`, `background_task_manager.py`, `cache_manager.py`, `hydration_manager.py`, `step_history_tracker.py`, `loop_executor.py` (wrapper to `_execute_loop`).
- Missing helpers claimed earlier: `agent_orchestrator.py`, `validation_orchestrator.py`, `context_update_manager.py`, `policy_invoker.py`, `complex_step_router.py`, `pipeline_orchestrator.py`, `failure_builder.py`.
- Quota contextvar still exposed: `CURRENT_QUOTA` in `executor_core.py` and used by `agent_policy.py`, `parallel_policy.py`, `router_policy.py`, `cache_policy.py`, `step_coordinator.py`.
- Tests/architecture: not executed in this pass; status unknown.
