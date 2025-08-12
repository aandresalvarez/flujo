## Typed Step Outcomes: Hybrid Migration Guide (FSD-008)

This document describes the current hybrid outcomes architecture and how to migrate policies and callers safely.

### Current State (Stabilized)

- Backend/frame path is outcome-first: components exchange `StepOutcome[StepResult]` (`Success`, `Failure`, `Paused`, `Chunk`).
- Legacy path remains `StepResult`-based for backward compatibility.
- Policies support a hybrid return via `return_outcome` flag:
  - `return_outcome=True` → return `StepOutcome`.
  - `return_outcome=False` (default) → return `StepResult`.
- `ExecutorCore` explicitly passes `return_outcome` per path and unwraps for legacy callers.

### Why Hybrid Now

Moving directly to outcome-only broke existing tests and external integrations expecting `StepResult`. Hybrid mode preserves behavior while enabling outcome-first backends.

### Writing Policies in Hybrid Mode

- Accept `*, return_outcome: bool = False` in `execute` signatures.
- Build a `StepResult` internally, then:
  - Return `to_outcome(result)` when `return_outcome` is `True`.
  - Return `result` otherwise.
- Do not raise `PausedException`; return `Paused(message=...)` instead.

### Calling Policies

- Backend path (frame) should pass `return_outcome=True`.
- Legacy path should pass `return_outcome=False` or omit the flag for older executors.

### Runner and Streaming

- `run_outcomes_async` yields strictly `StepOutcome` values.
- `run_async` remains legacy-compatible.

### Migration Plan to Outcome-Only

1. Keep hybrid mode until downstreams are updated.
2. Add deprecation notes in release notes when ready.
3. Remove `return_outcome` and `StepResult` return types in a major release.

### Testing Expectations

- Outcome-first paths covered in integration tests.
- Legacy paths covered in regression tests to ensure no breakage during migration.

## Typed Outcomes (FSD-008)

Flujo steps now support typed outcomes in the backend/runner path. Instead of returning raw `StepResult` directly, policies are adapted to return a `StepOutcome[StepResult]` on the `ExecutionFrame` path.

- Success: `Success(step_result=StepResult)`
- Failure: `Failure(error=Exception, feedback=str | None, step_result=StepResult | None)`
- Paused: `Paused(message=str)` (control flow)

Key points:
- Backward compatibility: Legacy callers continue to receive `StepResult`; the executor unwraps outcomes when not called with an `ExecutionFrame`.
- Adapters: `Default*StepExecutorOutcomes` wrap existing policies without altering logic.
- Utilities: `flujo/domain/outcomes.py` provides `to_outcome(sr)` and `unwrap(outcome, step_name=...)`.

Which paths return typed outcomes?
- Backend/runner calls use `ExecutorCore.execute(frame: ExecutionFrame)` → returns `StepOutcome[StepResult]`.
- Legacy `execute(step, data, ...)` → returns `StepResult` (for tests and backward compatibility).

Extending to new policies:
1. Implement the policy returning `StepResult` as usual.
2. Create an outcomes adapter taking the policy and returning `StepOutcome[StepResult]`.
3. Wire the adapter for the backend/frame path in `ExecutorCore.execute()`.


