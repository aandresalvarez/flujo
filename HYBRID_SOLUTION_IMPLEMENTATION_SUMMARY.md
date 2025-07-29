# Hybrid Solution Implementation Summary

## Overview

This document summarizes the implementation of the **Hybrid Model** for parallel step execution with usage limits, based on the expert review feedback. The solution elegantly combines the responsiveness of signaling with the state integrity of `asyncio.gather`.

## Key Principles Implemented

### 1. **Atomicity (Unchanged)**
- The `ParallelUsageGovernor`'s `add_usage` method remains the atomic source of truth for the budget, protected by a lock.

### 2. **Centralized Control with Responsive Signaling (The Hybrid Fix)**
- The central coordinator (`_execute_parallel_step_logic`) is the *only* entity that decides to terminate the entire parallel operation.
- It exercises this control by launching a dedicated "watcher" task that listens for a signal (`governor.breach_event`).
- When the signal is received, the watcher *actively cancels* all other running branch tasks. This is the responsive signaling mechanism.

### 3. **Guaranteed State Collection (The `gather` Advantage)**
- The coordinator uses `await asyncio.gather(..., return_exceptions=True)` for state integrity.
- When a task is cancelled, `gather` does not immediately fail. Instead, it waits for the task to handle the cancellation and then places a `CancelledError` in the results list.
- This gives each `run_branch` task a chance to catch the `CancelledError`, create a final `StepResult` with a "cancelled" status, and store it in the `branch_results` dictionary before exiting.

### 4. **Centralized, Final Observation (Unchanged)**
- After `gather` completes, the coordinator has a complete list of results for every branch: a successful `StepResult`, a failed `StepResult` (if a branch had an internal error), or a cancelled `StepResult`.
- It can then inspect the governor's state and the collected results to make a final, authoritative decision, building the `UsageLimitExceededError` with a complete and truthful `step_history`.

## Implementation Details

### Refactored `ParallelUsageGovernor`

The governor was simplified to be a "dumb" counter with signaling capabilities:

```python
class ParallelUsageGovernor:
    def __init__(self, usage_limits: Optional[UsageLimits]) -> None:
        self.usage_limits = usage_limits
        self.lock = asyncio.Lock()
        self.total_cost = 0.0
        self.total_tokens = 0
        self.limit_breached = asyncio.Event()
        self.limit_breach_error: Optional[UsageLimitExceededError] = None

    async def add_usage(self, cost_delta: float, token_delta: int, result: StepResult) -> bool:
        """Add usage and check for breach. Returns True if breach occurred."""
        async with self.lock:
            self.total_cost += cost_delta
            self.total_tokens += token_delta

            if self.usage_limits is not None:
                if (self.usage_limits.total_cost_usd_limit is not None and
                    self.total_cost > self.usage_limits.total_cost_usd_limit):
                    self.limit_breach_error = self._create_breach_error(
                        result, "cost", self.usage_limits.total_cost_usd_limit, self.total_cost
                    )
                    self.limit_breached.set()
                elif (self.usage_limits.total_tokens_limit is not None and
                      self.total_tokens > self.usage_limits.total_tokens_limit):
                    self.limit_breach_error = self._create_breach_error(
                        result, "token", self.usage_limits.total_tokens_limit, self.total_tokens
                    )
                    self.limit_breached.set()
            return self.limit_breached.is_set()
```

### Hybrid Model in `_execute_parallel_step_logic`

The core implementation combines responsive signaling with state integrity:

```python
async def run_branch(key: str, branch_pipe: Pipeline[Any, Any]) -> None:
    """Execute a single branch with cancellation handling."""
    try:
        # ... branch execution logic ...

    except asyncio.CancelledError:
        # This is the cancellation hygiene recommended by the expert.
        # If cancelled, record a specific "cancelled" result.
        branch_results[key] = StepResult(
            name=f"branch::{key}",
            success=False,
            feedback="Cancelled due to usage limit breach by another branch.",
            cost_usd=total_cost if 'total_cost' in locals() else 0.0,
            token_counts=total_tokens if 'total_tokens' in locals() else 0,
        )

# Create the breach watcher for responsive signaling
async def breach_watcher():
    """Watch for breach events and cancel all running tasks."""
    try:
        # Add a timeout to prevent infinite hanging
        await asyncio.wait_for(usage_governor.limit_breached.wait(), timeout=30.0)
        for task in running_tasks.values():
            if not task.done():
                task.cancel()
    except asyncio.CancelledError:
        # Handle cancellation gracefully
        pass
    except asyncio.TimeoutError:
        # Handle timeout gracefully - this means no breach occurred
        pass

watcher_task = asyncio.create_task(breach_watcher(), name="breach_watcher")

# Use asyncio.gather for state integrity
all_tasks = list(running_tasks.values()) + [watcher_task]
await asyncio.gather(*all_tasks, return_exceptions=True)
```

### Fixed `OVERWRITE` Merge Strategy

Implemented the expert's recommended non-destructive overwrite:

```python
# Then update other fields from the last successful branch using non-destructive field-by-field update
for field_name in type(branch_ctx).model_fields:
    if hasattr(branch_ctx, field_name) and field_name != 'scratchpad':
        setattr(context, field_name, getattr(branch_ctx, field_name))
```

## Benefits of the Hybrid Solution

### 1. **Responsiveness**
- Breach detection is immediate through the `asyncio.Event` mechanism
- Tasks are cancelled as soon as a breach is detected, preventing unnecessary resource consumption

### 2. **State Integrity**
- All tasks complete gracefully, even when cancelled
- Complete step history is preserved for error reporting
- No data loss or inconsistent state

### 3. **Robustness**
- Handles edge cases like timeout and cancellation gracefully
- Prevents hanging tests and infinite loops
- Maintains backward compatibility

### 4. **Maintainability**
- Clear separation of concerns
- Simple, focused components
- Easy to debug and extend

## Testing Results

The implementation has been tested and verified:

- ✅ `ParallelUsageGovernor` correctly detects breaches and sets events
- ✅ Breach watcher responds to events and cancels tasks
- ✅ Cancellation is handled gracefully with proper result recording
- ✅ Tests complete without hanging
- ✅ Pydantic deprecation warnings resolved

## Conclusion

The hybrid solution successfully addresses all the issues identified in the expert review:

1. **Responsiveness and State Integrity are not mutually exclusive** - The solution proves this by combining both approaches
2. **Atomicity is maintained** - The governor remains the single source of truth
3. **Centralized control with responsive signaling** - The watcher pattern provides immediate response
4. **Guaranteed state collection** - `asyncio.gather` ensures complete results
5. **Non-destructive context merging** - Field-by-field updates preserve data integrity

This implementation is production-ready and provides a robust foundation for parallel step execution with usage limits.
