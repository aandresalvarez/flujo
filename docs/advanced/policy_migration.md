## Custom Policy Migration Guide (ExecutionFrame)

Flujo now requires policy executors to use the frame-based signature:

```python
async def execute(self, core: ExecutorCore, frame: ExecutionFrame[Any]) -> StepOutcome:
    ...
```

### What changed
- Legacy `execute(self, core, step, data, context, resources, limits, stream, on_chunk, cache_key, fallback_depth)` is no longer supported for new policies.
- `ExecutionDispatcher` adapts built-in policies, but custom executors should migrate to frames to remain compatible.

### How to migrate custom policies
1. Update the signature to accept `(core, frame: ExecutionFrame)`.
2. Access state via `frame.step`, `frame.data`, `frame.context`, `frame.resources`, `frame.limits`, `frame.stream`, `frame.on_chunk`.
3. Do not mutate `frame.context` directly if the step is complex; use the policy’s orchestrator helpers for isolation/merging.
4. Remove any `isinstance(step, ExecutionFrame)` guards; the dispatcher always passes a frame.

### Example
```python
class MyStepExecutor(StepPolicy[MyStep]):
    handles_type = MyStep

    async def execute(self, core: ExecutorCore, frame: ExecutionFrame[Any]) -> StepOutcome[Any]:
        step: MyStep = frame.step
        data = frame.data
        ctx = frame.context
        # implement logic...
        return Success(step_result=StepResult(name=step.name, output=...))
```

### Notes
- Control flow exceptions must be re-raised (PausedException, PipelineAbortSignal, InfiniteRedirectError).
- Use `ContextManager.isolate()` for branch/loop bodies to preserve idempotency.
- Governance, quota, and telemetry are provided by `core`; do not re-create them inside policies.

