# FSD-008: Typed Outcomes Migration - Incremental Implementation Plan

## Overview
Migrate all step execution policies to return typed `StepOutcome[StepResult]` natively while maintaining backward compatibility and test stability.

## Core Principles
- **Incremental**: Each step is small, focused, and testable
- **Safe**: No breaking changes; maintain green tests throughout
- **Testable**: Each step includes validation before proceeding
- **Reversible**: Easy rollback at any point

## Implementation Steps

### Step 0: Outcome conversion utility (test-first)
- **Goal**: Provide a single, reusable helper to wrap `StepResult -> StepOutcome[StepResult]` and vice-versa for tests.
- **Implementation**:
  - Add small helpers (e.g., `to_outcome(sr)`, `unwrap(outcome)`) in a shared module.
  - Preserve all metadata (attempts, latency_ms, feedback, usage, cached flags).
- **Testing**:
  - Add property-based tests asserting `unwrap(to_outcome(sr)) == sr` for a wide variety of generated `StepResult`s.
  - Ensure diagnostics/feedback mapping is preserved.
- **Validation**: This reduces duplication across adapters and guarantees semantics equivalence.

### Step 1: Add Agent Policy Outcomes Adapter ✅ COMPLETED
- **Goal**: Create `DefaultAgentStepExecutorOutcomes` that wraps `DefaultAgentStepExecutor`.
- **Implementation**:
  - Adapter added in `step_policies.py`; `ExecutorCore` uses it on backend/runner path.
  - Returns `Success(step_result=result)` for successful results; `Failure(error=Exception(...), step_result=result)` for failed results.
- **Testing**: `make test-fast` is green (validated).
- **Validation**: Backend path now emits typed `StepOutcome` for Agent steps without breaking legacy tests.

### Step 2: Add Parallel Policy Outcomes Adapter
- **Goal**: Create `DefaultParallelStepExecutorOutcomes` that wraps `DefaultParallelStepExecutor`
- **Implementation**:
  - Add adapter class in `step_policies.py`
  - Handle both success and failure cases from parallel execution
  - Preserve all result metadata (attempts, latency, feedback, usage)
- **Testing**: Run `make test-fast` - should remain green
- **Validation**: Verify parallel step outcomes are properly wrapped

### Step 3: Add Conditional Policy Outcomes Adapter
- **Goal**: Create `DefaultConditionalStepExecutorOutcomes` that wraps `DefaultConditionalStepExecutor`
- **Implementation**:
  - Add adapter class in `step_policies.py`
  - Handle conditional branching outcomes
  - Preserve conditional logic and result metadata
- **Testing**: Run `make test-fast` - should remain green
- **Validation**: Verify conditional step outcomes are properly wrapped

### Step 4: Wire Simple Policy Adapter in ExecutorCore
- **Goal**: Update `ExecutorCore.execute()` to use `DefaultSimpleStepExecutorOutcomes` when called with `ExecutionFrame`
- **Implementation**:
  - Modify the simple step routing logic in `ExecutorCore.execute()`
  - Use outcomes adapter only for backend/runner path (when `called_with_frame=True`)
  - Keep direct policy calls returning `StepResult` for tests
- **Testing**: Run `make test-fast` - should remain green
- **Validation**: Verify simple steps now return `StepOutcome` in backend context

### Step 5: Wire Parallel Policy Adapter in ExecutorCore
- **Goal**: Update `ExecutorCore.execute()` to use `DefaultParallelStepExecutorOutcomes` when called with `ExecutionFrame`
- **Implementation**:
  - Modify the parallel step routing logic in `ExecutorCore.execute()`
  - Use outcomes adapter only for backend/runner path
  - Keep direct policy calls returning `StepResult` for tests
- **Testing**: Run `make test-fast` - should remain green
- **Validation**: Verify parallel steps now return `StepOutcome` in backend context

### Step 6: Wire Conditional Policy Adapter in ExecutorCore
- **Goal**: Update `ExecutorCore.execute()` to use `DefaultConditionalStepExecutorOutcomes` when called with `ExecutionFrame`
- **Implementation**:
  - Modify the conditional step routing logic in `ExecutorCore.execute()`
  - Use outcomes adapter only for backend/runner path
  - Keep direct policy calls returning `StepResult` for tests
- **Testing**: Run `make test-fast` - should remain green
- **Validation**: Verify conditional steps now return `StepOutcome` in backend context

### Step 7: Integration Testing
- **Goal**: Verify all step types now return `StepOutcome` in backend context
- **Implementation**:
  - Create integration test that exercises all step types through backend path
  - Verify `StepOutcome` types are properly propagated
  - Check that legacy consumers still receive unwrapped `StepResult`
- **Testing**: Run `make test-fast` and integration tests - should remain green
- **Validation**: Confirm typed outcomes are working end-to-end

### Step 8: Protocol Interface Updates (Optional)
- **Goal**: Update protocol definitions to reflect `StepOutcome` return types
- **Implementation**:
  - Update `SimpleStepExecutor`, `ParallelStepExecutor`, `ConditionalStepExecutor` protocols
  - Change return type from `StepResult` to `StepOutcome[StepResult]`
  - Note: This is cosmetic - actual behavior unchanged due to adapters
- **Testing**: Run `make test-fast` - should remain green
- **Validation**: Verify type hints are consistent

### Step 9: Performance Validation
- **Goal**: Ensure outcomes adapters don't introduce performance regressions
- **Implementation**:
  - Run performance benchmarks comparing before/after
  - Profile memory usage and execution time
  - Verify no significant overhead from outcome wrapping
- **Testing**: Run performance tests and benchmarks
- **Validation**: Confirm performance impact is minimal (<5%)

### Step 10: Documentation and Cleanup
- **Goal**: Document the new typed outcomes system
- **Implementation**:
  - Update API documentation
  - Add migration guide for consumers
  - Document the adapter pattern used
- **Testing**: Verify documentation is accurate and helpful
- **Validation**: Confirm developers can understand and use the new system

## Success Criteria
- All step types return `StepOutcome[StepResult]` in backend context
- All tests remain green throughout migration
- No breaking changes for existing consumers
- Performance impact is minimal
- Clear documentation for the new system

## Rollback Plan
At any step, if tests fail:
1. Revert the specific change
2. Investigate the issue
3. Fix the underlying problem
4. Retry the step

## Next Steps
Begin with Step 2 (Parallel Policy Outcomes Adapter) since Step 1 is already completed.