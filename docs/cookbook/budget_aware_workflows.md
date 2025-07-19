# Cookbook: Running Budget-Aware Workflows with the Usage Governor

## The Problem

Complex AI pipelines with loops, parallel execution, or iterative refinement can quickly spiral out of control in terms of cost and token usage. When you're running production workloads, you need confidence that your pipelines won't create runaway processes that exceed your budget or hit API rate limits.

Traditional approaches often require manual monitoring or external systems, but `flujo` provides a built-in **Usage Governor** that automatically enforces cost and token limits across all control flow primitives.

## The Solution

The `UsageGovernor` is a first-class feature in `flujo` that works seamlessly with all pipeline constructs, including `LoopStep` and `ParallelStep`. It provides:

- **Automatic cost tracking** across all steps and iterations
- **Immediate halting** when limits are breached
- **Task cancellation** of parallel branches to save resources
- **Detailed error reporting** with complete execution history

## Core Concepts

### UsageLimits Model

The `UsageLimits` model defines your budget constraints:

```python
from flujo.domain import UsageLimits

# Set cost and token limits
limits = UsageLimits(
    total_cost_usd_limit=0.50,  # Maximum $0.50 spent
    total_tokens_limit=1000      # Maximum 1000 tokens used
)
```

### UsageLimitExceededError

When limits are breached, `flujo` raises this exception with complete execution context:

```python
from flujo.exceptions import UsageLimitExceededError

try:
    result = await runner.run(input_data)
except UsageLimitExceededError as e:
    print(f"Pipeline halted: {e}")
    print(f"Total cost: ${e.result.total_cost_usd}")
    print(f"Steps completed: {len(e.result.step_history)}")
```

## Example 1: Halting a Runaway Loop

Loops are particularly dangerous for cost control. The `UsageGovernor` checks limits after each iteration, allowing you to safely use `Step.loop_until` without fear of infinite loops.

```python
import asyncio
from pydantic import BaseModel
from flujo import Flujo, Step, Pipeline, UsageLimits, UsageLimitExceededError

class MockAgentOutput(BaseModel):
    """An agent output that includes cost and token metrics."""
    value: int
    cost_usd: float = 0.1
    token_counts: int = 100

class FixedMetricAgent:
    """An agent that returns a fixed cost and token count on each call."""

    async def run(self, data: int | MockAgentOutput) -> MockAgentOutput:
        val = data.value if isinstance(data, MockAgentOutput) else data
        return MockAgentOutput(value=val + 1)

# Create a simple pipeline that increments a value
metric_pipeline = Pipeline.from_step(
    Step.model_validate({"name": "metric_step", "agent": FixedMetricAgent()})
)

# Create a loop that will be halted by the governor
loop_step = Step.loop_until(
    name="governed_loop",
    loop_body_pipeline=metric_pipeline,
    exit_condition_callable=lambda out, ctx: out.value >= 10,  # Could exit naturally, but governor likely halts first
    max_loops=20,  # High max to ensure governor triggers first
)

# Set a cost limit that will be breached after a few iterations
limits = UsageLimits(total_cost_usd_limit=0.25, total_tokens_limit=None)
runner = Flujo(loop_step, usage_limits=limits)

try:
    print("Running loop with cost limit...")
    result = await runner.run(0)
    print("Loop completed successfully!")
except UsageLimitExceededError as e:
    print(f"\n✅ Loop halted by governor as expected!")
            print(f"   Error: {e}")
        print(f"   Iterations completed: {e.result.step_history[0].attempts}")
        if hasattr(e.result.step_history[0].output, 'value'):
            print(f"   Final value: {e.result.step_history[0].output.value}")
        else:
            print(f"   Final value: None (loop halted mid-iteration)")
        print(f"   Total cost: ${e.result.total_cost_usd:.2f}")
```

**Expected Output:**
```
Running loop with cost limit...

✅ Loop halted by governor as expected!
   Error: Cost limit of $0.25 exceeded
   Iterations completed: 3
   Final value: None (loop halted mid-iteration)
   Total cost: $0.30
```

### How the Loop Governor Works

1. **Per-Iteration Checking**: After each loop iteration, the governor checks the cumulative cost
2. **Immediate Halting**: When the limit is breached, the loop stops immediately, even mid-iteration
3. **Complete Context**: The exception contains the full execution history, including how many iterations completed
4. **Safe Output Access**: The code safely handles cases where the output might not be available due to mid-iteration halting

## Example 2: Proactive Cancellation in Parallel Steps

The `UsageGovernor` provides efficient optimization for parallel execution. When one branch breaches the limit, it cancels other in-flight branches to save time and resources.

```python
import asyncio
import time
from pydantic import BaseModel
from flujo import Flujo, Step, UsageLimits, UsageLimitExceededError

class CostlyAgent:
    """An agent that reports high cost and takes time to simulate expensive operations."""

    def __init__(self, cost: float = 0.1, tokens: int = 100, delay: float = 0.1):
        self.cost = cost
        self.tokens = tokens
        self.delay = delay

    async def run(self, data: any) -> any:
        await asyncio.sleep(self.delay)  # Simulate expensive operation

        class Output(BaseModel):
            value: any
            cost_usd: float = self.cost
            token_counts: int = self.tokens

        return Output(value=data)

# Create branches with different costs and execution times
branches = {
    "fast_expensive": Step.model_validate({
        "name": "fast_expensive",
        "agent": CostlyAgent(cost=0.15, delay=0.05)  # Breaches limit quickly
    }),
    "slow_cheap": Step.model_validate({
        "name": "slow_cheap",
        "agent": CostlyAgent(cost=0.01, delay=0.5)   # Takes longer but is cheap
    }),
}

parallel = Step.parallel("parallel_cancellation", branches)
limits = UsageLimits(total_cost_usd_limit=0.10)  # Limit that will be breached by fast_expensive
runner = Flujo(parallel, usage_limits=limits)

start_time = time.monotonic()

try:
    print("Running parallel steps with cost limit...")
    result = await runner.run("input")
    print("Parallel execution completed!")
except UsageLimitExceededError as e:
    execution_time = time.monotonic() - start_time

    print(f"\n✅ Parallel execution halted by governor!")
    print(f"   Error: {e}")
    print(f"   Execution time: {execution_time:.2f}s")
    print(f"   Total cost: ${e.result.total_cost_usd:.2f}")
    print(f"   Note: slow_cheap branch was cancelled")
```

**Expected Output:**
```
Running parallel steps with cost limit...

✅ Parallel execution halted by governor!
   Error: Cost limit of $0.1 exceeded
   Execution time: 0.06s
   Total cost: $0.15
   Note: slow_cheap branch was cancelled
```

### How Parallel Cancellation Works

1. **Concurrent Execution**: Both branches start executing simultaneously
2. **Fast Branch Breaches**: The `fast_expensive` branch completes quickly and breaches the limit
3. **Task Cancellation**: The governor cancels the `slow_cheap` branch that was still running
4. **Time Savings**: Execution completes quickly instead of waiting for the slow branch

This optimization is valuable in production scenarios where you might have expensive API calls running in parallel.

## Example 3: Complex Nested Workflows

The `UsageGovernor` works seamlessly with complex nested structures, providing protection at every level.

```python
from flujo import Flujo, Step, Pipeline, UsageLimits, UsageLimitExceededError

# Create a nested workflow: loop containing parallel steps
inner_branches = {
    "a": Step.model_validate({"name": "a", "agent": FixedMetricAgent()}),
    "b": Step.model_validate({"name": "b", "agent": FixedMetricAgent()}),
}

inner_parallel = Step.parallel("inner_parallel", inner_branches)
outer_loop = Step.loop_until(
    name="outer_loop",
    loop_body_pipeline=Pipeline.from_step(inner_parallel),
    exit_condition_callable=lambda _out, _ctx: False,  # Never exit naturally
    iteration_input_mapper=lambda _out, _ctx, _i: 0,
    max_loops=10,
)

limits = UsageLimits(total_cost_usd_limit=0.5)
runner = Flujo(outer_loop, usage_limits=limits)

try:
    print("Running complex nested workflow...")
    result = await runner.run(0)
    print("Nested workflow completed!")
except UsageLimitExceededError as e:
    print(f"\n✅ Complex workflow halted by governor!")
    print(f"   Error: {e}")
    print(f"   Loop iterations: {e.result.step_history[0].attempts}")
    print(f"   Total cost: ${e.result.total_cost_usd:.2f}")
```

## Best Practices

### 1. Always Set Limits for Production

```python
# Good: Always set limits for production pipelines
limits = UsageLimits(
    total_cost_usd_limit=1.00,  # $1.00 budget
    total_tokens_limit=5000      # 5000 token limit
)
runner = Flujo(pipeline, usage_limits=limits)

# Bad: No limits can lead to runaway costs
runner = Flujo(pipeline)  # No protection!
```

### 2. Use Conservative Limits

```python
# Start with conservative limits and adjust based on monitoring
limits = UsageLimits(
    total_cost_usd_limit=0.10,  # Start small
    total_tokens_limit=1000      # Conservative token limit
)
```

### 3. Handle Exceptions Gracefully

```python
try:
    result = await runner.run(input_data)
    # Process successful result
except UsageLimitExceededError as e:
    # Log the breach for monitoring
    logger.warning(f"Pipeline halted due to limit breach: {e}")

    # Optionally retry with different parameters
    # Or fall back to a simpler pipeline
    result = await fallback_runner.run(input_data)
```

### 4. Monitor and Adjust

```python
# Use the result data to understand usage patterns
except UsageLimitExceededError as e:
    result = e.result
    print(f"Pipeline used ${result.total_cost_usd} in {len(result.step_history)} steps")

    # Adjust limits based on actual usage
    # Consider implementing dynamic limits based on step complexity
```

## Summary

The `UsageGovernor` transforms cost control from a manual, error-prone process into an automatic, reliable safety net. By integrating seamlessly with all `flujo` control flow primitives, it enables you to:

- **Run complex pipelines confidently** without fear of runaway costs
- **Optimize resource usage** through task cancellation
- **Maintain predictable budgets** in production environments
- **Scale safely** with automatic protection at every level

This makes `flujo` uniquely suited for production AI workloads where cost predictability is critical.
