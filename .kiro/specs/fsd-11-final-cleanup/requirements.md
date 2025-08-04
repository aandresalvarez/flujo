# Requirements Document: FSD-11 Final Deprecation and Cleanup

## Introduction

This feature addresses the final cleanup phase of Flujo's architectural refactoring from first principles. The goal is to eliminate all obsolete code that has been made redundant by the new `ExecutorCore`, thereby simplifying the codebase, reducing cognitive load, and solidifying the new architecture as the single source of truth.

Following Flujo's architectural philosophy, this cleanup must embody **Maintainability** and **Production Readiness** principles—ensuring the codebase is clean, lean, and reflects only the correct, production-ready architecture while maintaining all functionality.

## Requirements

### Requirement 1

**User Story:** As a developer, I want the obsolete `step_logic.py` file to be completely removed so that the codebase reflects only the correct, production-ready architecture and eliminates potential confusion.

#### Acceptance Criteria

1. WHEN the `flujo/application/core/step_logic.py` file is deleted THEN it SHALL be completely removed from the repository
2. WHEN a project-wide search is performed THEN no references to `step_logic` SHALL remain in the active codebase
3. WHEN imports are checked THEN no broken imports SHALL exist due to the file deletion
4. WHEN the deletion is complete THEN the `ExecutorCore` SHALL be the single source of truth for step execution logic
5. WHEN the file is removed THEN all tests SHALL continue to pass

### Requirement 2

**User Story:** As a developer, I want the backward compatibility wrapper classes to be removed so that the codebase is clean and only contains the correct implementation.

#### Acceptance Criteria

1. WHEN the legacy classes are removed THEN `UltraStepExecutor` SHALL be deleted from `ultra_executor.py`
2. WHEN the legacy classes are removed THEN `_Frame` SHALL be deleted from `ultra_executor.py`
3. WHEN the legacy classes are removed THEN `_LRUCache` SHALL be deleted from `ultra_executor.py`
4. WHEN the legacy classes are removed THEN `_UsageTracker` SHALL be deleted from `ultra_executor.py`
5. WHEN the classes are removed THEN no import errors SHALL occur
6. WHEN tests that targeted these classes are updated THEN they SHALL test the new components they were wrapping

### Requirement 3

**User Story:** As a developer, I want the `_run_step` method to be removed from `runner.py` so that the execution flow is simplified and unnecessary indirection is eliminated.

#### Acceptance Criteria

1. WHEN the `_run_step` method is removed THEN the method SHALL be completely deleted from `runner.py`
2. WHEN the `_execute_steps` method is updated THEN it SHALL use the backend directly without the wrapper
3. WHEN the execution flow is simplified THEN the flow SHALL be: `Flujo._execute_steps() -> ExecutionManager.execute_steps() -> backend.execute_step()`
4. WHEN the wrapper is removed THEN no functionality SHALL be lost
5. WHEN the simplification is complete THEN all tests SHALL continue to pass

### Requirement 4

**User Story:** As a developer, I want the `step_executor` parameter to be removed from `ExecutionManager.execute_steps` so that the parameter passing is simplified and the backend is used directly.

#### Acceptance Criteria

1. WHEN the `step_executor` parameter is removed THEN it SHALL be deleted from the method signature
2. WHEN the `StepCoordinator.execute_step` is updated THEN it SHALL receive the backend directly
3. WHEN the parameter is removed THEN the execution manager SHALL use `self.backend` directly
4. WHEN the simplification is complete THEN all functionality SHALL work identically
5. WHEN the parameter is removed THEN all tests SHALL continue to pass

### Requirement 5

**User Story:** As a developer, I want comprehensive static analysis to be performed so that all code quality issues are caught and resolved.

#### Acceptance Criteria

1. WHEN linters are run THEN `ruff` SHALL pass with no errors
2. WHEN linters are run THEN `flake8` SHALL pass with no errors
3. WHEN type checkers are run THEN `mypy` SHALL pass with no errors
4. WHEN import validation is performed THEN no broken imports SHALL exist
5. WHEN static analysis is complete THEN all code quality issues SHALL be resolved

### Requirement 6

**User Story:** As a developer, I want the entire test suite to pass so that I can be confident no regressions were introduced by the cleanup.

#### Acceptance Criteria

1. WHEN the full test suite is run THEN 100% of tests SHALL pass
2. WHEN integration tests are run THEN all complex workflows SHALL work correctly
3. WHEN performance tests are run THEN no performance degradation SHALL be observed
4. WHEN edge case tests are run THEN all edge cases SHALL be handled correctly
5. WHEN the test suite passes THEN the cleanup SHALL be considered complete and successful

## Technical Requirements

### Functional Requirements

1. **File Deletion**: Remove `flujo/application/core/step_logic.py` completely
2. **Class Removal**: Delete legacy wrapper classes from `ultra_executor.py`
3. **Method Removal**: Remove `_run_step` method from `runner.py`
4. **Parameter Removal**: Remove `step_executor` parameter from `ExecutionManager.execute_steps`
5. **Import Cleanup**: Remove all imports from `step_logic.py`
6. **Flow Simplification**: Simplify execution flow to use backend directly
7. **Static Analysis**: Ensure all linters and type checkers pass
8. **Test Validation**: Ensure all tests pass after cleanup

### Non-Functional Requirements

1. **Maintainability**: Codebase should be cleaner and easier to understand
2. **Performance**: No performance degradation should be introduced
3. **Reliability**: All functionality should work identically to before
4. **Code Quality**: All static analysis tools should pass
5. **Cognitive Load**: Reduced complexity and easier navigation
6. **Technical Debt**: Eliminate dead code and legacy components

## Constraints

### Backward Compatibility
- All public APIs must remain unchanged
- No breaking changes to external interfaces
- All existing functionality must work identically
- Users should not notice any changes

### Performance
- No performance degradation should be introduced
- Execution flow should be at least as fast as before
- Memory usage should not increase
- CPU usage should remain similar

### Code Quality
- All static analysis tools must pass
- No code quality issues should be introduced
- Code should be cleaner and more maintainable
- Documentation should be updated if needed

## Success Criteria

1. **Complete Cleanup**: All obsolete code has been removed
2. **Simplified Architecture**: Execution flow is streamlined and direct
3. **No Regressions**: All functionality works identically to before
4. **Static Analysis**: All linters and type checkers pass
5. **Full Test Suite**: 100% of tests pass
6. **Performance Maintained**: No performance degradation
7. **Clean Codebase**: Reduced cognitive load and technical debt
8. **Production Ready**: Architecture is clean, lean, and maintainable

## Risk Assessment

### Low Risk
- **File Deletion**: `step_logic.py` is completely unused
- **Class Removal**: Legacy classes are not used by any active code
- **Method Removal**: `_run_step` is a simple wrapper that can be eliminated
- **Parameter Removal**: `step_executor` parameter is redundant

### Mitigation Strategies
- **Comprehensive Testing**: Run full test suite after each change
- **Static Analysis**: Use linters and type checkers to catch issues
- **Incremental Changes**: Make changes one at a time and validate
- **Rollback Plan**: Keep git history for easy rollback if needed

## Testing Strategy

### Static Analysis
- **Linters**: `ruff` and `flake8` for code quality
- **Type Checkers**: `mypy` for type safety
- **Import Validation**: Check for broken imports
- **Dependency Analysis**: Verify all dependencies are resolved

### Regression Testing
- **Full Test Suite**: Run all tests to ensure no regressions
- **Integration Tests**: Verify complex workflows still work
- **Performance Tests**: Confirm no performance degradation
- **Edge Case Tests**: Validate error handling and edge cases

### Cleanup Validation
- **File Deletion**: Verify deleted files are completely removed
- **Reference Check**: Confirm no remaining references to deleted components
- **Functionality Test**: Ensure all features work with simplified architecture

## Migration Strategy

### Phase 1: File Deletion
1. Delete `flujo/application/core/step_logic.py`
2. Remove any remaining imports from `step_logic.py`
3. Run static analysis to catch any issues
4. Run tests to ensure no regressions

### Phase 2: Class Removal
1. Remove `UltraStepExecutor` class from `ultra_executor.py`
2. Remove `_Frame` class from `ultra_executor.py`
3. Remove `_LRUCache` class from `ultra_executor.py`
4. Remove `_UsageTracker` class from `ultra_executor.py`
5. Update any tests that targeted these classes
6. Run static analysis and tests

### Phase 3: Method Removal
1. Remove `_run_step` method from `runner.py`
2. Simplify `_execute_steps` to use backend directly
3. Update any references to `_run_step`
4. Run static analysis and tests

### Phase 4: Parameter Removal
1. Remove `step_executor` parameter from `ExecutionManager.execute_steps`
2. Update `StepCoordinator.execute_step` to receive backend directly
3. Simplify parameter passing throughout the chain
4. Run static analysis and tests

### Phase 5: Final Validation
1. Run comprehensive static analysis
2. Execute full test suite
3. Verify performance characteristics
4. Confirm all functionality works correctly

## Conclusion

This cleanup phase is essential to complete the architectural refactoring and establish the new `ExecutorCore` as the single source of truth. By removing all obsolete code and simplifying the execution flow, we create a clean, maintainable codebase that reflects only the correct, production-ready architecture.

The requirements ensure that:
- **All obsolete code is removed** without breaking functionality
- **Execution flow is simplified** by eliminating unnecessary indirection
- **Code quality is maintained** through comprehensive static analysis
- **No regressions are introduced** through thorough testing
- **Performance is preserved** while reducing complexity
- **Maintainability is improved** by eliminating technical debt

This final step ensures Flujo's architecture is clean, lean, and ready for future development while maintaining all production characteristics and functionality. 