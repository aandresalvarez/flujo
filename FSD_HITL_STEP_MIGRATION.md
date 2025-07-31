# FSD 4: HumanInTheLoopStep Migration to ExecutorCore

## Overview
Migrate HumanInTheLoopStep (HITL) logic from `step_logic.py` to `ExecutorCore._handle_hitl_step` method, completing the migration of all step types to the new architecture.

## Rationale & First Principles
- **Goal**: Complete the migration of all step types to ExecutorCore for unified orchestration
- **Why**: Achieve complete isolation and eliminate the last remaining step type in legacy code
- **Impact**: All step orchestration logic will be managed by the new, optimized ExecutorCore

## Scope of Work

### 1. Core Implementation
- **File**: `flujo/application/core/ultra_executor.py`
- **Method**: `_handle_hitl_step` (new method to be created)
- **Action**: Implement complete HITLStep logic migration

### 2. Logic Migration Requirements
- Copy entire `_handle_hitl_step` function from `step_logic.py`
- Integrate with ExecutorCore's exception handling
- Maintain all existing functionality:
  - Message generation for user
  - Context state management (scratchpad updates)
  - PausedException raising
  - User interaction handling
  - State preservation during pause

### 3. Performance Optimizations
- Optimize message generation
- Improve context state management
- Reduce memory allocations during pause
- Efficient state serialization

## Testing Strategy (TDD Approach)

### Phase 1: Unit Tests (Write First)
**File**: `tests/application/core/test_executor_core_hitl_step_migration.py`

#### 1.1 Basic Functionality Tests
```python
async def test_handle_hitl_step_basic_message():
    """Test basic HITL step with default message."""

async def test_handle_hitl_step_custom_message():
    """Test HITL step with custom message for user."""

async def test_handle_hitl_step_context_scratchpad():
    """Test that context scratchpad is properly updated."""

async def test_handle_hitl_step_paused_exception():
    """Test that PausedException is raised correctly."""

async def test_handle_hitl_step_data_handling():
    """Test handling of various data types in HITL step."""
```

#### 1.2 Context Management Tests
```python
async def test_handle_hitl_step_context_preservation():
    """Test that context state is preserved during pause."""

async def test_handle_hitl_step_scratchpad_updates():
    """Test scratchpad updates in context."""

async def test_handle_hitl_step_context_isolation():
    """Test context isolation during HITL step execution."""

async def test_handle_hitl_step_context_serialization():
    """Test context serialization for pause/resume."""
```

#### 1.3 Error Handling Tests
```python
async def test_handle_hitl_step_message_generation_errors():
    """Test error handling in message generation."""

async def test_handle_hitl_step_context_update_errors():
    """Test error handling during context updates."""

async def test_handle_hitl_step_exception_propagation():
    """Test that exceptions are properly propagated."""

async def test_handle_hitl_step_invalid_context():
    """Test handling of invalid context objects."""
```

#### 1.4 Performance Tests
```python
async def test_handle_hitl_step_message_performance():
    """Test performance of message generation."""

async def test_handle_hitl_step_context_performance():
    """Test performance of context operations."""

async def test_handle_hitl_step_memory_usage():
    """Test memory usage patterns."""

async def test_handle_hitl_step_serialization_performance():
    """Test performance of state serialization."""
```

### Phase 2: Integration Tests
**File**: `tests/integration/test_hitl_step_migration_integration.py`

#### 2.1 End-to-End Scenarios
```python
async def test_hitl_step_complex_pipeline_integration():
    """Test HITL step within complex pipelines."""

async def test_hitl_step_with_different_contexts():
    """Test HITL step with various context types."""

async def test_hitl_step_with_telemetry():
    """Test HITL step telemetry and observability."""

async def test_hitl_step_with_usage_limits():
    """Test HITL step with usage limit enforcement."""
```

#### 2.2 Migration Compatibility Tests
```python
async def test_hitl_step_legacy_compatibility():
    """Test that migrated HITL step produces identical results to legacy."""

async def test_hitl_step_backward_compatibility():
    """Test that existing HITL step configurations continue to work."""

async def test_hitl_step_performance_comparison():
    """Test that migrated implementation maintains or improves performance."""
```

### Phase 3: Regression Tests
**File**: `tests/regression/test_hitl_step_migration_regression.py`

#### 3.1 Existing Functionality Preservation
```python
async def test_hitl_step_existing_behavior_preservation():
    """Test that all existing HITL step behaviors are preserved."""

async def test_hitl_step_edge_cases_regression():
    """Test edge cases that existed before migration."""

async def test_hitl_step_error_scenarios_regression():
    """Test error scenarios that existed before migration."""
```

### Phase 4: Performance Tests
**File**: `tests/benchmarks/test_hitl_step_migration_performance.py`

#### 4.1 Performance Benchmarks
```python
async def test_hitl_step_migration_performance_improvement():
    """Benchmark performance improvement from migration."""

async def test_hitl_step_memory_usage():
    """Test memory usage patterns of migrated implementation."""

async def test_hitl_step_concurrent_execution():
    """Test performance under concurrent execution scenarios."""
```

## Implementation Details

### Step 1: Write Failing Tests
1. Create all unit tests with expected behavior
2. Ensure tests fail against current implementation
3. Document expected vs actual behavior

### Step 2: Implement Core Logic
```python
async def _handle_hitl_step(
    self,
    step: HumanInTheLoopStep,
    data: Any,
    context: Optional[TContext],
    resources: Optional[Any],
    limits: Optional[UsageLimits],
    context_setter: Optional[Callable[["PipelineResult[Any]", Optional[Any]], None]],
) -> StepResult:
    """Handle HumanInTheLoopStep execution using optimized component-based architecture."""

    # Generate message for user
    message = step.message_for_user if step.message_for_user is not None else str(data)

    # Update context scratchpad if available
    if isinstance(context, PipelineContext):
        try:
            context.scratchpad["status"] = "paused"
            context.scratchpad["hitl_message"] = message
            context.scratchpad["hitl_data"] = data
        except Exception as e:
            telemetry.logfire.error(f"Failed to update context scratchpad: {e}")

    # Log HITL step execution
    telemetry.logfire.info(f"HITL step '{step.name}' paused execution with message: {message}")

    # Raise PausedException to pause pipeline execution
    raise PausedException(message)
```

### Step 3: Update Dispatcher
Update the dispatcher in `_execute_complex_step` to route HITLStep to the new implementation:

```python
elif isinstance(step, HumanInTheLoopStep):
    telemetry.logfire.debug("Handling HumanInTheLoopStep")
    result = await self._handle_hitl_step(
        step,
        data,
        context,
        resources,
        limits,
        context_setter,
    )
```

### Step 4: Remove Legacy Dependencies
1. Remove legacy import: `from .step_logic import _handle_hitl_step`
2. Update imports to remove unused dependencies
3. Ensure no legacy function calls remain

## Acceptance Criteria

### Functional Requirements
- [ ] All existing HITLStep functionality preserved
- [ ] Message generation works correctly
- [ ] Context scratchpad updates work properly
- [ ] PausedException raised correctly
- [ ] State preservation during pause maintained
- [ ] Error handling matches legacy behavior
- [ ] User interaction handling preserved

### Performance Requirements
- [ ] Performance equal to or better than legacy implementation
- [ ] Message generation optimized
- [ ] Memory usage optimized
- [ ] Efficient context state management

### Quality Requirements
- [ ] All unit tests pass (100% coverage)
- [ ] All integration tests pass
- [ ] All regression tests pass
- [ ] All performance benchmarks meet or exceed targets
- [ ] No legacy dependencies remain

### Documentation Requirements
- [ ] Method docstring updated with comprehensive documentation
- [ ] Inline comments explain complex logic
- [ ] Migration notes added to changelog
- [ ] Performance impact documented

## Risk Mitigation

### High-Risk Areas
1. **State Preservation**: Ensure context state is properly preserved during pause
2. **Exception Handling**: Verify PausedException is raised correctly
3. **Context Updates**: Maintain proper scratchpad updates
4. **Performance**: Monitor for performance regressions

### Mitigation Strategies
1. **Comprehensive Testing**: TDD approach ensures all scenarios covered
2. **Gradual Migration**: Test against legacy implementation
3. **Performance Monitoring**: Continuous benchmarking
4. **Rollback Plan**: Keep legacy implementation as fallback until fully validated

## Success Metrics

### Quantitative Metrics
- [ ] 100% test coverage for HITLStep functionality
- [ ] 0% performance regression
- [ ] 0% memory usage increase
- [ ] 100% backward compatibility
- [ ] Improved message generation performance

### Qualitative Metrics
- [ ] Code maintainability improved
- [ ] Architecture consistency achieved
- [ ] Technical debt reduced
- [ ] Developer experience enhanced

## Timeline
- **Phase 1 (Unit Tests)**: 1 day
- **Phase 2 (Implementation)**: 1 day
- **Phase 3 (Integration Tests)**: 1 day
- **Phase 4 (Performance Tests)**: 1 day
- **Phase 5 (Validation & Cleanup)**: 1 day

**Total Estimated Time**: 5 days
