# Cookbook: Controlling LLM Costs

## The Problem

Pipelines, especially those with loops or that call powerful models, can incur unpredictable costs. You need a reliable way to enforce a budget on a pipeline run to prevent unexpected bills.

## The Solution

The `Flujo` engine has a built-in **Usage Governor**. You can enable it by passing a `UsageLimits` object when you create your runner. The engine will then track the cumulative cost and token count after each step and automatically halt the pipeline if a limit is breached.

```python
import pytest
from pydantic import BaseModel
from flujo import Flujo, Step, UsageLimits, UsageLimitExceededError

# An agent that reports a fixed cost of $0.05 per call
class CostlyAgent:
    async def run(self, x: int) -> int:
        class Output(BaseModel):
            value: int
            cost_usd: float = 0.05
            token_counts: int = 50
        return Output(value=x + 1)

# This pipeline runs the same costly step three times
pipeline = Step("step_1", CostlyAgent()) >> Step("step_2", CostlyAgent()) >> Step("step_3", CostlyAgent())

# Set a hard limit of $0.12
limits = UsageLimits(total_cost_usd_limit=0.12)
runner = Flujo(pipeline, usage_limits=limits)

try:
    print("Running pipeline... it should be stopped by the quota limits.")
    runner.run(0)
except UsageLimitExceededError as e:
    print(f"\n✅ Pipeline halted as expected!")
    print(f"   Reason: {e}")
    print(f"   The pipeline ran for {len(e.result.step_history)} steps before stopping.")
    print(f"   Final recorded cost was ${e.result.total_cost_usd:.2f}")
```

### How It Works

1.  We define `UsageLimits` with `total_cost_usd_limit=0.12`.
2.  The `Flujo` runner receives these limits.
3.  **Step 1** runs, costing $0.05. The total cost is $0.05, which is less than $0.12. The pipeline continues.
4.  **Step 2** runs, costing another $0.05. The total cost is now $0.10, which is still less than $0.12. The pipeline continues.
5.  **Step 3** runs, costing $0.05. The total cost becomes $0.15.
6.  *After* Step 3 completes, the engine checks the total cost ($0.15), sees it has breached the limit ($0.12), and immediately raises `UsageLimitExceededError`.
7.  The exception contains the `result` object with the history up to the point of failure, which is useful for debugging.

This mechanism is a critical safety feature for running `Flujo` in production.

## Advanced Usage

For more complex scenarios involving loops, parallel execution, and nested workflows, see the **Safe Loop Budgeting** guide, which demonstrates proactive quota patterns with `LoopStep` and `ParallelStep` (**docs/cookbook/safe_loop_budgeting.md**).
