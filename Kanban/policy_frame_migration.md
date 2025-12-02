# Policy Frame-Only Migration Plan

Goal: migrate all step policies to a strict `execute(core, frame: ExecutionFrame)` signature, eliminate legacy arg unpacking, and retire the legacy shims/wrappers while keeping tests green.

## Scope
- Policies: agent, simple, cache, parallel, loop, conditional, dynamic router, HITL, import, any custom policies registered by default.
- Callers: `StepHandler`, orchestrators, helpers, dispatcher interactions, and any tests/fixtures invoking `_handle_*`/`_execute_*` shims.
- Compatibility shims: `_handle_*`, `_execute_pipeline*`, `execute_step`, `execute_step_compat`.

## Phase 1: Signature Refactor (Policies)
- Update each `Default*Executor` to accept `(core, frame: ExecutionFrame[Any])` only.
- Inline frame unpacking at the top of each policy and remove legacy arg lists.
- Adjust type hints in `StepPolicy.execute` (protocol) to prefer frame signature; keep temporary overloads only if unavoidable.
- Ensure telemetry/quota/cache key derivation uses frame fields (not legacy args).

## Phase 2: Caller Convergence
- Update `StepHandler` methods to build an `ExecutionFrame` (when not already framed) and call `policy.execute(core, frame)` directly.
- Update orchestrators/helpers that currently call `_handle_*` with arg lists to route through dispatcher or construct frames.
- Normalize cache key handling in handlers to rely on frame data (no ad-hoc params).

## Phase 3: Remove Legacy Shims/Wrappers
- Delete `_handle_*`, `_execute_*`, and `execute_step`/`execute_step_compat` shims from `ExecutorCore` and helpers once all call sites are migrated.
- Remove `executor_wrappers.py` references permanently (already deleted) and scrub residual imports.
- Clean up `__all__`/exports and tracking docs reflecting shim removal.

## Phase 4: Test Updates
- Update tests invoking `_handle_*`, `_execute_*`, or `execute_step_compat` to:
  - Use `executor.execute(frame)` with `make_execution_frame`/`ExecutionFrame`, or
  - Use dispatcher/policy registry directly where appropriate.
- Adjust fixtures/mocks to pass frames instead of unpacked args.
- Verify dynamic router/HITL tests still assert pause/resume semantics through dispatcher spans.

## Phase 5: Validation
- Run `make test-fast`; if failures persist, run targeted suites:
  - `pytest tests/application/core/test_executor_core_* tests/unit/test_*policy.py tests/unit/test_parallel_step_strategies.py`
  - HITL/pause paths: `pytest tests/integration/test_conversational_loop_nested_hitl.py tests/integration/test_hitl_integration.py`
- Run `make typecheck` to confirm signature consistency.

## Risks & Mitigations
- **Breakage in external callers/tests** expecting shims → mitigate by staging refactor: adjust tests first, keep temporary compatibility adapter until final cleanup.
- **Cache/telemetry regressions** if frame fields omitted → audit each policy for cache key/telemetry usage post-refactor.
- **HITL pause/resume semantics** rely on handler orchestration → ensure handler-level orchestration remains intact when calling policies with frames.

## Exit Criteria
- All policies expose only `execute(core, frame)`; no `isinstance(frame, ExecutionFrame)` branches remain.
- Dispatcher has no legacy branching (already done) and no shims required in `ExecutorCore`.
- `make test-fast` and `make typecheck` pass without legacy imports or shims.***
