## Deterministic Parallel Quota Splitting

This cookbook shows how Flujo deterministically splits a parent `Quota` across parallel branches using `Quota.split(n)` and enforces pre-execution reservation.

### Key points
- Parent `Quota` is split into `n` child quotas; parent is effectively consumed to avoid double-spend.
- Each branch receives a dedicated sub-quota and reserves before execution.
- Parallel fairness is deterministic — no branch can starve others by racing for the parent quota.

### Example
```python
from flujo.application.core.executor_core import ExecutorCore
from flujo.domain.dsl.parallel import ParallelStep

parallel = ParallelStep(
    name="p",
    branches={
        "a": some_pipeline_a,
        "b": some_pipeline_b,
        "c": some_pipeline_c,
    },
)

core = ExecutorCore()
result = await core.execute(step=parallel, data={})
```

The executor splits the current `Quota` into three sub-quotas and delegates to branch executions with deterministic budgeting.

