# FSD 1: Complete LoopStep Logic Migration to ExecutorCore

## Overview
Complete the migration of LoopStep logic from `step_logic.py` to `ExecutorCore._handle_loop_step` method, replacing the current TODO implementation with a full, optimized implementation.

## Rationale & First Principles
- **Goal**: Eliminate the last remaining legacy dependency in the ExecutorCore by fully migrating LoopStep logic
- **Why**: Achieve complete isolation of orchestration logic in the new architecture
- **Impact**: Remove the TODO comment and legacy import, ensuring all control flow is managed by the new orchestrator

## Scope of Work

### 1. Core Implementation
- **File**: `flujo/application/core/ultra_executor.py`
- **Method**: `_handle_loop_step` (lines 1737-1780)
- **Action**: Replace TODO implementation with complete logic migration

### 2. Logic Migration Requirements
- Copy entire `_execute_loop_step_logic` function from `step_logic.py`
- Refactor internal `step_executor` calls to recursively call `self.execute`
- Maintain all existing functionality:
  - Iteration control with `max_loops`
  - Exit condition evaluation with `exit_condition_callable`
  - Input/output mapping with `initial_input_to_loop_body_mapper`, `iteration_input_mapper`, `loop_output_mapper`
  - Context isolation between iterations
  - Usage limits enforcement
  - Telemetry and logging

### 3. Performance Optimizations
- Pre-allocate result objects for better performance
- Optimize context copying operations
- Reduce function call overhead
- Implement early returns for error conditions

## Testing Strategy (TDD Approach)

### Phase 1: Unit Tests (Write First)
**File**: `tests/application/core/test_executor_core_loop_step_migration.py`

#### 1.1 Basic Functionality Tests
```python
async def test_handle_loop_step_basic_iteration():
    """Test basic loop iteration without mappers."""

async def test_handle_loop_step_with_exit_condition():
    """Test loop termination via exit condition."""

async def test_handle_loop_step_max_iterations():
    """Test loop termination via max_loops limit."""

async def test_handle_loop_step_input_mappers():
    """Test initial_input_to_loop_body_mapper functionality."""

async def test_handle_loop_step_iteration_mappers():
    """Test iteration_input_mapper functionality."""

async def test_handle_loop_step_output_mappers():
    """Test loop_output_mapper functionality."""
```

#### 1.2 Context Isolation Tests
```python
async def test_handle_loop_step_context_isolation():
    """Test that context modifications are isolated between iterations."""

async def test_handle_loop_step_context_merge():
    """Test that context updates are properly merged after iterations."""

async def test_handle_loop_step_context_preservation():
    """Test that original context is preserved after loop completion."""
```

#### 1.3 Error Handling Tests
```python
async def test_handle_loop_step_mapper_errors():
    """Test error handling in input/output mappers."""

async def test_handle_loop_step_exit_condition_errors():
    """Test error handling in exit condition evaluation."""

async def test_handle_loop_step_body_step_failures():
    """Test handling of failures in loop body steps."""
```

#### 1.4 Usage Limits Tests
```python
async def test_handle_loop_step_cost_limits():
    """Test cost limit enforcement during iterations."""

async def test_handle_loop_step_token_limits():
    """Test token limit enforcement during iterations."""

async def test_handle_loop_step_limits_accumulation():
    """Test that usage is properly accumulated across iterations."""
```

### Phase 2: Integration Tests
**File**: `tests/integration/test_loopstep_migration_integration.py`

#### 2.1 End-to-End Scenarios
```python
async def test_loopstep_complex_pipeline_integration():
    """Test LoopStep within a complex pipeline with multiple step types."""

async def test_loopstep_nested_control_flow():
    """Test LoopStep with nested conditional and parallel steps."""

async def test_loopstep_with_caching():
    """Test LoopStep with cache-enabled steps."""

async def test_loopstep_with_telemetry():
    """Test LoopStep telemetry and observability."""
```

#### 2.2 Migration Compatibility Tests
```python
async def test_loopstep_legacy_compatibility():
    """Test that migrated LoopStep produces identical results to legacy."""

async def test_loopstep_backward_compatibility():
    """Test that existing LoopStep configurations continue to work."""

async def test_loopstep_performance_comparison():
    """Test that migrated implementation maintains or improves performance."""
```

### Phase 3: Regression Tests
**File**: `tests/regression/test_loopstep_migration_regression.py`

#### 3.1 Existing Functionality Preservation
```python
async def test_loopstep_existing_behavior_preservation():
    """Test that all existing LoopStep behaviors are preserved."""

async def test_loopstep_edge_cases_regression():
    """Test edge cases that existed before migration."""

async def test_loopstep_error_scenarios_regression():
    """Test error scenarios that existed before migration."""
```

### Phase 4: Performance Tests
**File**: `tests/benchmarks/test_loopstep_migration_performance.py`

#### 4.1 Performance Benchmarks
```python
async def test_loopstep_migration_performance_improvement():
    """Benchmark performance improvement from migration."""

async def test_loopstep_memory_usage():
    """Test memory usage patterns of migrated implementation."""

async def test_loopstep_concurrent_execution():
    """Test performance under concurrent execution scenarios."""
```

## Implementation Details

### Step 1: Write Failing Tests
1. Create all unit tests with expected behavior
2. Ensure tests fail against current TODO implementation
3. Document expected vs actual behavior

### Step 2: Implement Core Logic
```python
async def _handle_loop_step(
    self,
    loop_step: LoopStep[TContext],
    data: Any,
    context: Optional[TContext],
    resources: Optional[Any],
    limits: Optional[UsageLimits],
    context_setter: Optional[Callable[["PipelineResult[Any]", Optional[Any]], None]],
) -> StepResult:
    """Handle LoopStep execution using optimized component-based architecture."""

    # Initialize result with pre-allocated metadata
    loop_overall_result = StepResult(name=loop_step.name)
    loop_overall_result.metadata_ = {}

    # Handle initial input mapping
    if loop_step.initial_input_to_loop_body_mapper:
        try:
            current_body_input = loop_step.initial_input_to_loop_body_mapper(data, context)
        except Exception as e:
            loop_overall_result.success = False
            loop_overall_result.feedback = f"Initial input mapper raised an exception: {e}"
            return loop_overall_result
    else:
        current_body_input = data

    # Initialize loop state
    last_successful_iteration_body_output: Any = None
    final_body_output_of_last_iteration: Any = None
    loop_exited_successfully_by_condition = False

    # Main loop iteration
    for i in range(1, loop_step.max_loops + 1):
        loop_overall_result.attempts = i

        # Create isolated context for this iteration
        iteration_context = copy.deepcopy(context) if context is not None else None

        # Execute loop body pipeline
        iteration_succeeded_fully = True
        current_iteration_data_for_body_step = current_body_input

        # Execute each step in the loop body
        for body_step in loop_step.loop_body_pipeline.steps:
            try:
                body_step_result = await self.execute(
                    body_step,
                    current_iteration_data_for_body_step,
                    context=iteration_context,
                    resources=resources,
                    limits=limits,
                    context_setter=context_setter,
                )

                # Accumulate metrics
                loop_overall_result.latency_s += body_step_result.latency_s
                loop_overall_result.cost_usd += getattr(body_step_result, "cost_usd", 0.0)
                loop_overall_result.token_counts += getattr(body_step_result, "token_counts", 0)

                # Check for step failure
                if not body_step_result.success:
                    iteration_succeeded_fully = False
                    final_body_output_of_last_iteration = body_step_result.output
                    break

                current_iteration_data_for_body_step = body_step_result.output

            except Exception as e:
                iteration_succeeded_fully = False
                final_body_output_of_last_iteration = None
                loop_overall_result.feedback = f"Step execution error: {e}"
                break

        # Update successful iteration output
        if iteration_succeeded_fully:
            final_body_output_of_last_iteration = current_iteration_data_for_body_step
            last_successful_iteration_body_output = current_iteration_data_for_body_step

        # Merge context updates from iteration
        if context is not None and iteration_context is not None:
            safe_merge_context_updates(context, iteration_context)

        # Check exit condition
        try:
            should_exit = loop_step.exit_condition_callable(
                final_body_output_of_last_iteration, iteration_context
            )
        except Exception as e:
            loop_overall_result.success = False
            loop_overall_result.feedback = f"Exit condition callable raised an exception: {e}"
            break

        if should_exit:
            loop_overall_result.success = iteration_succeeded_fully
            if not iteration_succeeded_fully:
                loop_overall_result.feedback = "Loop exited by condition, but last iteration body failed."
            loop_exited_successfully_by_condition = True
            break

        # Prepare input for next iteration
        if i < loop_step.max_loops:
            if loop_step.iteration_input_mapper:
                try:
                    current_body_input = loop_step.iteration_input_mapper(
                        final_body_output_of_last_iteration, context, i
                    )
                except Exception as e:
                    loop_overall_result.success = False
                    loop_overall_result.feedback = f"Iteration input mapper raised an exception: {e}"
                    break
            else:
                current_body_input = final_body_output_of_last_iteration
    else:
        # Loop completed without exit condition
        loop_overall_result.success = False
        loop_overall_result.feedback = f"Reached max_loops ({loop_step.max_loops}) without meeting exit condition."

    # Set final output
    if loop_overall_result.success and loop_exited_successfully_by_condition:
        if loop_step.loop_output_mapper:
            try:
                loop_overall_result.output = loop_step.loop_output_mapper(
                    last_successful_iteration_body_output, context
                )
            except Exception as e:
                loop_overall_result.success = False
                loop_overall_result.feedback = f"Loop output mapper raised an exception: {e}"
                loop_overall_result.output = None
        else:
            loop_overall_result.output = last_successful_iteration_body_output
    else:
        loop_overall_result.output = final_body_output_of_last_iteration
        if not loop_overall_result.feedback:
            loop_overall_result.feedback = "Loop did not complete successfully or exit condition not met positively."

    return loop_overall_result
```

### Step 3: Remove Legacy Dependencies
1. Remove TODO comment
2. Remove legacy import: `from .step_logic import _handle_loop_step as legacy_handle_loop_step`
3. Remove legacy function call
4. Update imports to remove unused dependencies

### Step 4: Update Dispatcher
Ensure the dispatcher in `_execute_complex_step` correctly routes LoopStep to the new implementation.

## Acceptance Criteria

### Functional Requirements
- [ ] All existing LoopStep functionality preserved
- [ ] Context isolation between iterations maintained
- [ ] Usage limits properly enforced and accumulated
- [ ] All mapper functions (input, iteration, output) work correctly
- [ ] Exit conditions evaluated properly
- [ ] Error handling matches legacy behavior
- [ ] Telemetry and logging maintained

### Performance Requirements
- [ ] Performance equal to or better than legacy implementation
- [ ] Memory usage optimized
- [ ] No memory leaks in long-running loops
- [ ] Efficient context copying and merging

### Quality Requirements
- [ ] All unit tests pass (100% coverage)
- [ ] All integration tests pass
- [ ] All regression tests pass
- [ ] All performance benchmarks meet or exceed targets
- [ ] No TODO comments remain
- [ ] No legacy dependencies remain

### Documentation Requirements
- [ ] Method docstring updated with comprehensive documentation
- [ ] Inline comments explain complex logic
- [ ] Migration notes added to changelog
- [ ] Performance impact documented

## Risk Mitigation

### High-Risk Areas
1. **Context Isolation**: Ensure deep copying works correctly
2. **Usage Limits**: Verify accumulation across iterations
3. **Error Propagation**: Maintain proper error handling
4. **Performance**: Monitor for regressions

### Mitigation Strategies
1. **Comprehensive Testing**: TDD approach ensures all scenarios covered
2. **Gradual Migration**: Test against legacy implementation
3. **Performance Monitoring**: Continuous benchmarking
4. **Rollback Plan**: Keep legacy implementation as fallback until fully validated

## Success Metrics

### Quantitative Metrics
- [ ] 100% test coverage for LoopStep functionality
- [ ] 0% performance regression
- [ ] 0% memory usage increase
- [ ] 100% backward compatibility

### Qualitative Metrics
- [ ] Code maintainability improved
- [ ] Architecture consistency achieved
- [ ] Technical debt reduced
- [ ] Developer experience enhanced

## Timeline
- **Phase 1 (Unit Tests)**: 2 days
- **Phase 2 (Implementation)**: 3 days
- **Phase 3 (Integration Tests)**: 2 days
- **Phase 4 (Performance Tests)**: 1 day
- **Phase 5 (Validation & Cleanup)**: 1 day

**Total Estimated Time**: 9 days
