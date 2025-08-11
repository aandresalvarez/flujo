# Requirements Document: FSD-12 Step Logic Final Migration

## Introduction

This feature addresses the final incomplete migration from `step_logic.py` to the new `ExecutorCore` architecture. Following Flujo's architectural philosophy, this migration must embody **production readiness** and **algebraic closure**—ensuring that every step, regardless of complexity, is a first-class citizen in the execution graph while maintaining the framework's dual architecture of declarative shell and execution core.

## Requirements

### Requirement 1

**User Story:** As a developer, I want the StepExecutor type alias migrated from step_logic.py to ultra_executor.py so that type definitions are co-located with their primary usage, improving maintainability and reducing dependencies.

#### Acceptance Criteria

1. WHEN the StepExecutor type alias is migrated THEN it SHALL be defined in ultra_executor.py
2. WHEN the type alias is migrated THEN all existing imports SHALL continue to work
3. WHEN the type alias is migrated THEN it SHALL maintain the same signature and behavior
4. WHEN the migration is complete THEN step_logic.py SHALL no longer export StepExecutor
5. WHEN the migration is complete THEN all tests SHALL continue to pass

### Requirement 2

**User Story:** As a developer, I want the ParallelUsageGovernor function migrated to ExecutorCore so that usage governance is encapsulated within the executor, improving separation of concerns and reducing coupling.

#### Acceptance Criteria

1. WHEN ParallelUsageGovernor is migrated THEN it SHALL be implemented as a private method in ExecutorCore
2. WHEN the function is migrated THEN it SHALL maintain identical behavior and performance
3. WHEN the function is migrated THEN all parallel step execution SHALL continue to work
4. WHEN the migration is complete THEN step_logic.py SHALL no longer export ParallelUsageGovernor
5. WHEN the migration is complete THEN all tests SHALL continue to pass

### Requirement 3

**User Story:** As a developer, I want the _should_pass_context function migrated to ExecutorCore so that context passing logic is encapsulated within the executor, improving maintainability and reducing external dependencies.

#### Acceptance Criteria

1. WHEN _should_pass_context is migrated THEN it SHALL be implemented as a private method in ExecutorCore
2. WHEN the function is migrated THEN it SHALL maintain identical behavior for all context passing scenarios
3. WHEN the function is migrated THEN all plugin and processor execution SHALL continue to work
4. WHEN the migration is complete THEN step_logic.py SHALL no longer export _should_pass_context
5. WHEN the migration is complete THEN all tests SHALL continue to pass

### Requirement 4

**User Story:** As a developer, I want the _run_step_logic function integrated into ExecutorCore so that core step execution logic is fully encapsulated within the executor, eliminating external dependencies and improving performance.

#### Acceptance Criteria

1. WHEN _run_step_logic is integrated THEN it SHALL be implemented as part of ExecutorCore's step execution logic
2. WHEN the function is integrated THEN it SHALL maintain identical behavior for all step types
3. WHEN the function is integrated THEN all step execution scenarios SHALL continue to work
4. WHEN the integration is complete THEN step_logic.py SHALL no longer export _run_step_logic
5. WHEN the integration is complete THEN all tests SHALL continue to pass

### Requirement 5

**User Story:** As a developer, I want the _default_set_final_context function migrated to ExecutorCore so that context management is encapsulated within the executor, improving consistency and reducing external dependencies.

#### Acceptance Criteria

1. WHEN _default_set_final_context is migrated THEN it SHALL be implemented as a private method in ExecutorCore
2. WHEN the function is migrated THEN it SHALL maintain identical behavior for all context setting scenarios
3. WHEN the function is migrated THEN all context management SHALL continue to work
4. WHEN the migration is complete THEN step_logic.py SHALL no longer export _default_set_final_context
5. WHEN the migration is complete THEN all tests SHALL continue to pass

### Requirement 6

**User Story:** As a developer, I want all import statements updated to remove step_logic.py dependencies so that the codebase is clean and free of legacy dependencies.

#### Acceptance Criteria

1. WHEN import statements are updated THEN all step_logic.py imports SHALL be removed
2. WHEN import statements are updated THEN all functionality SHALL continue to work
3. WHEN import statements are updated THEN no broken imports SHALL remain
4. WHEN the cleanup is complete THEN step_logic.py SHALL be safe to delete
5. WHEN the cleanup is complete THEN all tests SHALL continue to pass

### Requirement 7

**User Story:** As a developer, I want comprehensive testing to ensure the migration maintains functional equivalence and doesn't introduce regressions.

#### Acceptance Criteria

1. WHEN all migrations are complete THEN all existing tests SHALL continue to pass
2. WHEN all migrations are complete THEN no behavioral changes SHALL be introduced
3. WHEN all migrations are complete THEN performance characteristics SHALL be maintained or improved
4. WHEN all migrations are complete THEN all edge cases SHALL be handled correctly
5. WHEN all migrations are complete THEN error handling SHALL work as expected

### Requirement 8

**User Story:** As a developer, I want the migration to maintain backward compatibility so that existing code continues to work without changes.

#### Acceptance Criteria

1. WHEN the migration is complete THEN all existing step types SHALL continue to work unchanged
2. WHEN the migration is complete THEN all existing pipeline configurations SHALL continue to work
3. WHEN the migration is complete THEN all existing API contracts SHALL be preserved
4. WHEN the migration is complete THEN no breaking changes SHALL be introduced
5. WHEN the migration is complete THEN all existing documentation SHALL remain accurate

## Technical Requirements

### Functional Requirements

1. **Type System Migration**: Migrate StepExecutor type alias to ultra_executor.py
2. **Utility Function Migration**: Migrate ParallelUsageGovernor and _should_pass_context to ExecutorCore
3. **Core Logic Integration**: Integrate _run_step_logic into ExecutorCore's step execution logic
4. **Context Management Migration**: Migrate _default_set_final_context to ExecutorCore
5. **Import Cleanup**: Remove all step_logic.py import statements
6. **Testing**: Comprehensive test coverage for all migrated functionality

### Non-Functional Requirements

1. **Dual Architecture**: Strengthen execution core while preserving DSL elegance
2. **Extensibility**: New step types should not require core changes
3. **Reliability**: All existing functionality should continue to work in production environments
4. **Performance**: No degradation in step execution performance, with potential improvements
5. **Observability**: Maintain clean execution traces and debugging capabilities
6. **Memory Efficiency**: Optimize for high-frequency execution paths

## Constraints

### Backward Compatibility
- All existing step types must continue to work unchanged
- No breaking changes to public APIs
- Existing tests must pass without modification

### Performance
- Step execution performance must not degrade
- Memory usage should remain similar or improve
- CPU usage should not increase significantly

### Code Quality
- The migrated code should be more maintainable
- Code complexity should be reduced
- Dependencies should be minimized

## Success Criteria

1. **Algebraic Closure**: Every step type, current and future, is a first-class citizen in the execution graph
2. **Production Readiness**: The migrated elements maintain resilience, performance, and observability characteristics
3. **Recursive Execution**: Seamless integration with Flujo's recursive execution model
4. **Dual Architecture**: Strengthens the execution core while preserving DSL elegance
5. **Extensibility**: New step types can be added without core changes
6. **Functional Equivalence**: The migrated elements produce identical results to the current implementation
7. **Performance**: No degradation in performance characteristics, with potential improvements
8. **Testing**: All existing tests continue to pass, including complex nested workflow scenarios
9. **Clean Architecture**: Complete removal of step_logic.py dependencies
10. **Backward Compatibility**: No breaking changes to existing functionality
