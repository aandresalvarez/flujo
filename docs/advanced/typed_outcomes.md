## Typed Step Outcomes: Outcome-First Architecture (FSD-008)

This document describes the outcome-first architecture and the remaining compatibility behavior at the core boundary.

### Current State (Stabilized)

- Backend/frame path is outcome-first: components exchange `StepOutcome[StepResult]` (`Success`, `Failure`, `Paused`, `Chunk`).
- Legacy path remains `StepResult`-based for backward compatibility; the core unwraps outcomes for legacy callers.
- Policies are native outcomes: Agent, Simple, Loop, Parallel, Conditional, Cache.
- No outcomes adapters are used; `ExecutorCore` routes directly to policies.

### Legacy Compatibility Boundary

Legacy callers that invoke `execute(step, data, ...)` (non-frame) receive `StepResult`. The core unwraps `StepOutcome` to `StepResult` uniformly. Backend/frame calls receive `StepOutcome` directly.

### Writing Policies (Outcome-Only)

- Implement `execute(...) -> Awaitable[StepOutcome[StepResult]]`.
- Build `StepResult` internally, then return `Success(step_result=...)` or `Failure(..., step_result=...)`.
- Do not raise `PausedException`; return `Paused(message=...)` instead.

### Calling Policies

- Backend path (frame) receives `StepOutcome` directly.
- Legacy path is supported via core unwrapping; policy signatures are outcome-only.

### Runner and Streaming

- `run_outcomes_async` yields strictly `StepOutcome` values.
- `run_async` remains legacy-compatible.

#### run_outcomes_async usage

```python
from flujo.application.runner import Flujo
from flujo.domain.dsl.step import Step
from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.models import Success, Failure, Paused, Chunk

step = Step(name="echo", agent=MyAgent())
pipe = Pipeline.from_step(step)
runner = Flujo(pipe)

async for event in runner.run_outcomes_async("hi"):
    if isinstance(event, Chunk):
        handle_stream_chunk(event.data)
    elif isinstance(event, Success):
        print("final:", event.step_result.output)
    elif isinstance(event, Failure):
        log_error(event.feedback)
    elif isinstance(event, Paused):
        persist_for_hitl(event.message)
```

#### run_with_events helper (FSD-009)

`run_with_events` streams both lifecycle events **and** the final `PipelineResult`. Use it
when you need to know exactly when a background task was launched or to react to streaming
chunks while still receiving the final result in one pass.

```python
from flujo import Flujo
from flujo.domain.models import BackgroundLaunched, Chunk, PipelineResult

runner = Flujo(pipeline=my_pipeline)

async for event in runner.run_with_events("doc to process"):
    if isinstance(event, BackgroundLaunched):
        print(f"Background task {event.task_id} launched by {event.step_name}")
    elif isinstance(event, Chunk):
        handle_chunk(event.data)
    elif isinstance(event, PipelineResult):
        print(f"Pipeline finished with {len(event.step_history)} steps")
```

Notes:
- `run_async` still yields only the final `PipelineResult` (legacy shape).
- `run_outcomes_async` yields only `StepOutcome` events (no `PipelineResult`).
- `run_with_events` is the ergonomic bridge when you want both streams without juggling
  two APIs.

Policy contract

- All policy `execute(...)` methods must return `StepOutcome[StepResult]`.
- Use `to_outcome(sr)` when normalizing a constructed `StepResult`.
- Prefer returning `Paused(message=...)` over raising inside policies; raising is reserved at core/runner legacy boundaries.

### Migration Notes

- The system has completed migration to native-outcome policies. Any remaining hybrid handling exists only at the core boundary for legacy callers.
- Deprecation warnings for legacy-only entry points can be enabled behind a flag in future releases.

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
- Utilities: `flujo/domain/outcomes.py` provides `to_outcome(sr)` for normalizing legacy results inside policies when needed.

Which paths return typed outcomes?
- Backend/runner calls use `ExecutorCore.execute(frame: ExecutionFrame)` → returns `StepOutcome[StepResult]`.
- Legacy `execute(step, data, ...)` → returns `StepResult` (for tests and backward compatibility); the core unwraps.

Extending to new policies:
1. Implement the policy to return `StepOutcome[StepResult]`.
2. Build internal `StepResult` instances and wrap with `Success` or `Failure`.
3. Do not introduce adapters; route directly through `ExecutorCore`.

