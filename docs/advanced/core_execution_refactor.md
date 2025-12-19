# Core Execution Logic Refactor

## Overview

This document describes the refactoring of the monolithic `_execute_steps` method in `flujo/application/runner.py` into smaller, single-responsibility components. This refactor improves maintainability, testability, and makes the codebase more contributor-friendly.

## Problem Statement

The original `_execute_steps` method was a large, monolithic function that handled multiple responsibilities:

1. **Step Execution Coordination** - Main loop logic
2. **State Management** - Loading/saving workflow state
3. **Usage Limit Checking** - Monitoring and enforcing limits
4. **Hook Dispatching** - Pre/post step hooks
5. **Error Handling** - Pause, failure, and type mismatch handling
6. **Telemetry** - Span management and metrics
7. **Type Validation** - Step-to-step type compatibility
8. **Streaming Support** - Handling streaming output

This made the code difficult to:
- Read and understand
- Test in isolation
- Debug when issues arose
- Extend with new features
- Contribute to for new developers

## Solution

The refactor decomposes the monolithic method into dedicated components within the `flujo.application.core` package:

### 1. ExecutionManager

**File**: `flujo/application/core/execution/execution_manager.py`

The main orchestrator that coordinates all execution components. It provides a clean interface for executing pipeline steps while delegating specific responsibilities to specialized components.

**Key Responsibilities**:
- Orchestrates the main execution loop
- Coordinates between all other components
- Manages the overall execution flow

**Key Methods**:
- `execute_steps()` - Main execution loop
- `set_final_context()` - Set final context in result
- `persist_final_state()` - Persist final workflow state

### 2. StateManager

**File**: `flujo/application/core/state/state_manager.py`

Handles all workflow state persistence and loading operations.

**Key Responsibilities**:
- Load workflow state from persistence backend
- Persist current workflow state
- Extract run_id from context
- Handle state reconstruction

**Key Methods**:
- `load_workflow_state()` - Load and reconstruct state
- `persist_workflow_state()` - Save current state
- `get_run_id_from_context()` - Extract run_id

### 3. Quota Enforcement (Pure Quota)

**Files**: Quota model in `flujo/domain/models.py`; proactive enforcement in
`flujo/application/core/policies/agent_policy.py`, `parallel_policy.py`, and related policy
modules.

Enforces usage limits proactively via reserve/execute/reconcile. Legacy governors and
breach_event hooks are removed.

**Key Responsibilities**:
- Estimate usage before agent invocation and reserve quota
- Split quota for parallel branches (`Quota.split`) and reconcile actuals
- Raise `UsageLimitExceededError` proactively when reservations fail or actuals exceed limits
- Surface usage metrics through policy outcomes for telemetry

### 4. StepCoordinator

**File**: `flujo/application/core/orchestration/step_coordinator.py`

Coordinates individual step execution with telemetry and hook management.

**Key Responsibilities**:
- Execute individual steps
- Dispatch pre/post step hooks
- Handle step success/failure
- Manage telemetry spans
- Handle pause exceptions

**Key Methods**:
- `execute_step()` - Execute single step with full coordination
- `update_pipeline_result()` - Update result with step outcome
- `_dispatch_hook()` - Dispatch hooks for events

### 5. TypeValidator

**File**: `flujo/application/core/support/type_validator.py`

Validates type compatibility between pipeline steps.

**Key Responsibilities**:
- Validate step output compatibility with next step
- Handle None values appropriately
- Provide type information utilities

**Key Methods**:
- `validate_step_output()` - Check type compatibility
- `get_step_input_type()` - Get expected input type
- `get_step_output_type()` - Get output type

## Benefits

### 1. Improved Maintainability

- **Single Responsibility**: Each component has a clear, focused purpose
- **Reduced Complexity**: Smaller, more manageable code units
- **Better Organization**: Related functionality is grouped together

### 2. Enhanced Testability

- **Isolated Testing**: Each component can be tested independently
- **Mocking**: Easier to mock dependencies for unit tests
- **Coverage**: Better test coverage with focused test cases

### 3. Reduced Risk of Regressions

- **Clear Interfaces**: Well-defined boundaries between components
- **Easier Debugging**: Issues can be isolated to specific components
- **Safer Changes**: Modifications to one component don't affect others

### 4. Contributor-Friendly

- **Clear Structure**: New contributors can understand the codebase more easily
- **Focused Learning**: Can learn one component at a time
- **Better Documentation**: Each component has clear responsibilities

## Migration Guide

### For Existing Code

The refactor maintains full backward compatibility. Existing code using the `Flujo` class will continue to work without changes.

### For New Features

When adding new execution-related features:

1. **State Management**: Extend `StateManager` for new state operations
2. **Quota Enforcement**: Extend quota estimation/reservation in policies (agent/loop/parallel) or
   evolve `Quota` for new limit types
3. **Step Coordination**: Extend `StepCoordinator` for new step behaviors
4. **Type Validation**: Extend `TypeValidator` for new type rules
5. **Execution Flow**: Extend `ExecutionManager` for new coordination patterns

### Testing Strategy

Each component has comprehensive unit tests in `tests/unit/test_execution_manager.py`:

- **StateManager Tests**: Test state loading, persistence, and run_id extraction
- **Quota/Usage Tests**: Test proactive reservation, limit enforcement, and telemetry updates
- **TypeValidator Tests**: Test type compatibility validation
- **StepCoordinator Tests**: Test step execution and hook dispatching
- **ExecutionManager Tests**: Test overall execution coordination

## Performance Impact

The refactor has minimal performance impact:

- **No Runtime Overhead**: Components are lightweight and efficient
- **Same Execution Path**: Core execution logic remains the same
- **Optimized Coordination**: Minimal overhead from component coordination

## Future Enhancements

The new architecture enables several future enhancements:

1. **Advanced State Management**: Easier to add new state backends
2. **Flexible Usage Limits**: New limit types can be added easily
3. **Enhanced Telemetry**: Better observability and monitoring
4. **Custom Step Coordination**: Pluggable step execution strategies
5. **Advanced Type Validation**: More sophisticated type checking

## Conclusion

The core execution logic refactor successfully addresses the maintainability and contributor-friendliness goals while maintaining full backward compatibility. The new architecture provides a solid foundation for future enhancements and makes the codebase more accessible to new contributors.
