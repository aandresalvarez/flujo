## Safe Loop Budgeting with Quota

This recipe demonstrates how looped executions cooperate with a shared `Quota` without overruns.

### Guidelines
- The loop body steps perform their own reservations; the same `Quota` instance is shared across iterations.
- Validation or fallback paths must not swallow control-flow exceptions.

### Example
```python
from flujo.application.core.executor_core import ExecutorCore
from flujo.domain.dsl.loop import LoopStep

loop = LoopStep(
    name="L",
    max_loops=5,
    loop_body_pipeline=body_pipeline,
)

core = ExecutorCore()
result = await core.execute(step=loop, data={})
```

Each iteration reserves from the shared `Quota` and reconciles with actuals, ensuring deterministic, safe consumption across the loop.

