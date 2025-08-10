# Requirements Document

## Introduction

This feature addresses the final incomplete task from FSD 10: "Hardening Contracts & Finalizing `ExecutorCore`" from first principles. The remaining work focuses on refactoring the `ExecutorCore._is_complex_step` method to use the new object-oriented approach with the `is_complex` property instead of the current procedural `isinstance` checks.

Following Flujo's architectural philosophy, this refactoring must embody **production readiness** and **algebraic closure**—ensuring that every step, regardless of complexity, is a first-class citizen in the execution graph while maintaining the framework's dual architecture of declarative shell and execution core.

## Requirements

### Requirement 1

**User Story:** As a developer, I want the `ExecutorCore._is_complex_step` method to use an object-oriented approach so that new complex step types can be added without modifying core dispatch logic, maintaining Flujo's algebraic closure principle.

#### Acceptance Criteria

1. WHEN the `_is_complex_step` method is called THEN it SHALL use `getattr(step, 'is_complex', False)` instead of `isinstance` checks
2. WHEN a step has the `is_complex` property THEN it SHALL be used for complexity detection
3. WHEN a step does not have the `is_complex` property THEN it SHALL default to `False`
4. WHEN the method is refactored THEN all existing tests SHALL continue to pass
5. WHEN new step types are added THEN they SHALL be first-class citizens in the execution graph

### Requirement 2

**User Story:** As a developer, I want the refactored method to maintain backward compatibility so that existing step types continue to work without changes.

#### Acceptance Criteria

1. WHEN existing step types are processed THEN they SHALL be classified correctly
2. WHEN steps with plugins are processed THEN they SHALL be classified as complex
3. WHEN validation steps are processed THEN they SHALL be classified as complex
4. WHEN basic steps are processed THEN they SHALL be classified as simple

### Requirement 3

**User Story:** As a developer, I want the new implementation to be more extensible so that adding new complex step types doesn't require core changes.

#### Acceptance Criteria

1. WHEN a new step type implements `is_complex = True` THEN it SHALL be automatically recognized as complex
2. WHEN a new step type doesn't implement `is_complex` THEN it SHALL default to simple
3. WHEN the `ExecutorCore` is extended THEN no changes SHALL be required to `_is_complex_step`
4. WHEN new complex step types are added THEN they SHALL work without core modifications

### Requirement 4

**User Story:** As a developer, I want the refactored method to maintain production-ready performance characteristics so that step dispatch remains efficient in high-frequency execution paths.

#### Acceptance Criteria

1. WHEN the refactored method is called THEN it SHALL have similar or better performance than the current implementation
2. WHEN multiple step types are processed THEN the method SHALL scale efficiently with constant time complexity
3. WHEN the method is called frequently in recursive execution THEN it SHALL not cause performance bottlenecks
4. WHEN profiling is performed THEN the method SHALL show acceptable performance metrics for production workloads
5. WHEN the method is used in complex nested workflows THEN it SHALL maintain optimal memory and CPU characteristics

### Requirement 5

**User Story:** As a developer, I want comprehensive testing to ensure the refactored method works correctly in all scenarios.

#### Acceptance Criteria

1. WHEN all existing step types are tested THEN they SHALL be classified identically to the current implementation
2. WHEN edge cases are tested THEN the method SHALL handle them gracefully
3. WHEN integration tests are run THEN no behavioral changes SHALL be observed
4. WHEN regression tests are executed THEN all existing functionality SHALL be preserved

### Requirement 6

**User Story:** As a developer, I want the refactored method to be well-documented so that future developers understand the object-oriented approach.

#### Acceptance Criteria

1. WHEN the method is documented THEN it SHALL explain the object-oriented approach
2. WHEN examples are provided THEN they SHALL show how to extend complex step types
3. WHEN the documentation is updated THEN it SHALL reflect the new implementation
4. WHEN the method signature is changed THEN the documentation SHALL be updated accordingly

## Technical Requirements

### Functional Requirements

1. **Method Refactoring**: Replace `isinstance` checks with `getattr(step, 'is_complex', False)`
2. **Algebraic Closure**: Ensure every step type is a first-class citizen in the execution graph
3. **Recursive Execution**: Maintain seamless integration with Flujo's recursive execution model
4. **Production Readiness**: Ensure resilience, performance, and observability characteristics
5. **Testing**: Comprehensive test coverage for all scenarios including complex nested workflows

### Non-Functional Requirements

1. **Dual Architecture**: Strengthen execution core while preserving DSL elegance
2. **Extensibility**: New complex step types should not require core changes
3. **Reliability**: All existing functionality should continue to work in production environments
4. **Performance**: No degradation in step dispatch performance, with potential improvements
5. **Observability**: Maintain clean execution traces and debugging capabilities
6. **Memory Efficiency**: Optimize for high-frequency execution paths

## Constraints

### Backward Compatibility
- All existing step types must continue to work unchanged
- No breaking changes to public APIs
- Existing tests must pass without modification

### Performance
- Step dispatch performance must not degrade
- Memory usage should remain similar
- CPU usage should not increase significantly

### Code Quality
- The refactored method should be more readable
- Code complexity should be reduced
- Maintainability should be improved

## Success Criteria

1. **Algebraic Closure**: Every step type, current and future, is a first-class citizen in the execution graph
2. **Production Readiness**: The refactored method maintains resilience, performance, and observability characteristics
3. **Recursive Execution**: Seamless integration with Flujo's recursive execution model
4. **Dual Architecture**: Strengthens the execution core while preserving DSL elegance
5. **Extensibility**: New complex step types can be added without core changes
6. **Functional Equivalence**: The refactored method produces identical results to the current implementation
7. **Performance**: No degradation in performance characteristics, with potential improvements
8. **Testing**: All existing tests continue to pass, including complex nested workflow scenarios
9. **Documentation**: Method documentation is updated to reflect the new approach and architectural principles
