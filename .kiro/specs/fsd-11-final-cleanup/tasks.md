# Task List: FSD-11 Final Deprecation and Cleanup

## Overview

This document provides a detailed breakdown of tasks for FSD-11: Final Deprecation and Cleanup. Each task is designed to be small and focused, with a clear completion test that runs `make test-fast` to ensure no regressions are introduced.

## Task 1: Analyze Current State and Dependencies

### **Task 1.1: Verify step_logic.py is Unused**

**Objective:** Confirm that `flujo/application/core/step_logic.py` is completely unused and safe to delete.

**Actions:**
1. Search for all imports of `step_logic` in the codebase
2. Verify no active code references the file
3. Check git history to confirm it's legacy code
4. Document any remaining references found

**Completion Test:**
```bash
# Search for any remaining references
grep -r "step_logic" flujo/ --exclude-dir=.git
# Should return no results or only git history references
make test-fast
# Should pass 100% - no functionality depends on step_logic.py
```

**Acceptance Criteria:**
- [ ] No active code imports from `step_logic.py`
- [ ] No references to `step_logic` in the codebase
- [ ] All tests pass with `make test-fast`
- [ ] File is confirmed to be legacy and unused

---

### **Task 1.2: Identify Legacy Classes in ultra_executor.py**

**Objective:** Locate and document all legacy wrapper classes that need to be removed.

**Actions:**
1. Find `UltraStepExecutor` class in `ultra_executor.py`
2. Find `_Frame` class in `ultra_executor.py`
3. Find `_LRUCache` class in `ultra_executor.py`
4. Find `_UsageTracker` class in `ultra_executor.py`
5. Document their current usage and dependencies

**Completion Test:**
```bash
# Verify current state before removal
grep -n "class UltraStepExecutor\|class _Frame\|class _LRUCache\|class _UsageTracker" flujo/application/core/ultra_executor.py
# Should show the class definitions
make test-fast
# Should pass 100% - current state is working
```

**Acceptance Criteria:**
- [ ] All 4 legacy classes are identified
- [ ] Their locations in the file are documented
- [ ] Current usage patterns are understood
- [ ] All tests pass with `make test-fast`

---

### **Task 1.3: Analyze _run_step Method Dependencies**

**Objective:** Understand the `_run_step` method in `runner.py` and its dependencies.

**Actions:**
1. Locate the `_run_step` method in `flujo/application/runner.py`
2. Analyze its current implementation
3. Identify what it wraps and how it's used
4. Document the execution flow it creates

**Completion Test:**
```bash
# Verify current _run_step method exists
grep -n "def _run_step" flujo/application/runner.py
# Should show the method definition
make test-fast
# Should pass 100% - current state is working
```

**Acceptance Criteria:**
- [ ] `_run_step` method is located and documented
- [ ] Its current implementation is understood
- [ ] Dependencies and usage patterns are clear
- [ ] All tests pass with `make test-fast`

---

### **Task 1.4: Analyze step_executor Parameter Usage**

**Objective:** Understand the `step_executor` parameter in `ExecutionManager.execute_steps`.

**Actions:**
1. Locate the `execute_steps` method in `flujo/application/core/execution_manager.py`
2. Find the `step_executor` parameter in the method signature
3. Analyze how it's used throughout the method
4. Document the parameter passing flow

**Completion Test:**
```bash
# Verify current step_executor parameter exists
grep -n "step_executor" flujo/application/core/execution_manager.py
# Should show the parameter usage
make test-fast
# Should pass 100% - current state is working
```

**Acceptance Criteria:**
- [ ] `step_executor` parameter is located and documented
- [ ] Its usage throughout the method is understood
- [ ] Parameter passing flow is clear
- [ ] All tests pass with `make test-fast`

## Task 2: Remove step_logic.py File

### **Task 2.1: Delete step_logic.py File**

**Objective:** Completely remove the unused `step_logic.py` file from the repository.

**Actions:**
1. Delete `flujo/application/core/step_logic.py`
2. Verify the file is completely removed
3. Check that no broken imports result from the deletion

**Completion Test:**
```bash
# Verify file is deleted
ls flujo/application/core/step_logic.py
# Should return "No such file or directory"
# Verify no broken imports
python -c "import flujo.application.core.step_logic" 2>&1 | grep "No module named"
# Should show import error (expected)
make test-fast
# Should pass 100% - no functionality depends on this file
```

**Acceptance Criteria:**
- [ ] `step_logic.py` file is completely removed
- [ ] No broken imports result from the deletion
- [ ] All tests pass with `make test-fast`
- [ ] No functionality is broken by the removal

---

### **Task 2.2: Clean Up Any Remaining step_logic Imports**

**Objective:** Remove any remaining import statements that reference `step_logic.py`.

**Actions:**
1. Search for any remaining `import` statements referencing `step_logic`
2. Remove any found import statements
3. Verify no broken imports remain

**Completion Test:**
```bash
# Search for any remaining step_logic imports
grep -r "from.*step_logic\|import.*step_logic" flujo/ --exclude-dir=.git
# Should return no results
make test-fast
# Should pass 100% - no broken imports
```

**Acceptance Criteria:**
- [ ] No import statements reference `step_logic`
- [ ] All imports resolve correctly
- [ ] All tests pass with `make test-fast`
- [ ] No import errors occur

---

### **Task 2.3: Verify step_logic References Are Gone**

**Objective:** Confirm that no references to `step_logic` remain in the active codebase.

**Actions:**
1. Perform comprehensive search for `step_logic` references
2. Verify only git history references remain
3. Confirm no active code references the deleted file

**Completion Test:**
```bash
# Comprehensive search for step_logic references
grep -r "step_logic" flujo/ --exclude-dir=.git
# Should return no results or only git history references
make test-fast
# Should pass 100% - no active code references step_logic
```

**Acceptance Criteria:**
- [ ] No active code references `step_logic`
- [ ] Only git history references remain (if any)
- [ ] All tests pass with `make test-fast`
- [ ] Codebase is clean of step_logic references

## Task 3: Remove Legacy Classes from ultra_executor.py

### **Task 3.1: Remove UltraStepExecutor Class**

**Objective:** Delete the `UltraStepExecutor` class from `ultra_executor.py`.

**Actions:**
1. Locate the `UltraStepExecutor` class definition
2. Remove the entire class definition
3. Verify no syntax errors result from the removal

**Completion Test:**
```bash
# Verify class is removed
grep -n "class UltraStepExecutor" flujo/application/core/ultra_executor.py
# Should return no results
# Verify no syntax errors
python -m py_compile flujo/application/core/ultra_executor.py
# Should compile without errors
make test-fast
# Should pass 100% - no functionality depends on this class
```

**Acceptance Criteria:**
- [ ] `UltraStepExecutor` class is completely removed
- [ ] No syntax errors result from the removal
- [ ] All tests pass with `make test-fast`
- [ ] No functionality is broken by the removal

---

### **Task 3.2: Remove _Frame Class**

**Objective:** Delete the `_Frame` class from `ultra_executor.py`.

**Actions:**
1. Locate the `_Frame` class definition
2. Remove the entire class definition
3. Verify no syntax errors result from the removal

**Completion Test:**
```bash
# Verify class is removed
grep -n "class _Frame" flujo/application/core/ultra_executor.py
# Should return no results
# Verify no syntax errors
python -m py_compile flujo/application/core/ultra_executor.py
# Should compile without errors
make test-fast
# Should pass 100% - no functionality depends on this class
```

**Acceptance Criteria:**
- [ ] `_Frame` class is completely removed
- [ ] No syntax errors result from the removal
- [ ] All tests pass with `make test-fast`
- [ ] No functionality is broken by the removal

---

### **Task 3.3: Remove _LRUCache Class**

**Objective:** Delete the `_LRUCache` class from `ultra_executor.py`.

**Actions:**
1. Locate the `_LRUCache` class definition
2. Remove the entire class definition
3. Verify no syntax errors result from the removal

**Completion Test:**
```bash
# Verify class is removed
grep -n "class _LRUCache" flujo/application/core/ultra_executor.py
# Should return no results
# Verify no syntax errors
python -m py_compile flujo/application/core/ultra_executor.py
# Should compile without errors
make test-fast
# Should pass 100% - no functionality depends on this class
```

**Acceptance Criteria:**
- [ ] `_LRUCache` class is completely removed
- [ ] No syntax errors result from the removal
- [ ] All tests pass with `make test-fast`
- [ ] No functionality is broken by the removal

---

### **Task 3.4: Remove _UsageTracker Class**

**Objective:** Delete the `_UsageTracker` class from `ultra_executor.py`.

**Actions:**
1. Locate the `_UsageTracker` class definition
2. Remove the entire class definition
3. Verify no syntax errors result from the removal

**Completion Test:**
```bash
# Verify class is removed
grep -n "class _UsageTracker" flujo/application/core/ultra_executor.py
# Should return no results
# Verify no syntax errors
python -m py_compile flujo/application/core/ultra_executor.py
# Should compile without errors
make test-fast
# Should pass 100% - no functionality depends on this class
```

**Acceptance Criteria:**
- [ ] `_UsageTracker` class is completely removed
- [ ] No syntax errors result from the removal
- [ ] All tests pass with `make test-fast`
- [ ] No functionality is broken by the removal

---

### **Task 3.5: Update Tests That Targeted Legacy Classes**

**Objective:** Update any tests that specifically targeted the removed legacy classes.

**Actions:**
1. Search for tests that import or test the removed classes
2. Update tests to test the new components they were wrapping
3. Remove tests that are no longer relevant

**Completion Test:**
```bash
# Search for tests that reference removed classes
grep -r "UltraStepExecutor\|_Frame\|_LRUCache\|_UsageTracker" tests/ --exclude-dir=__pycache__
# Should return no results or only updated tests
make test-fast
# Should pass 100% - all tests updated and working
```

**Acceptance Criteria:**
- [ ] No tests reference the removed classes
- [ ] Tests are updated to test new components
- [ ] All tests pass with `make test-fast`
- [ ] Test coverage is maintained

## Task 4: Remove _run_step Method from runner.py

### **Task 4.1: Remove _run_step Method Definition**

**Objective:** Delete the `_run_step` method from `flujo/application/runner.py`.

**Actions:**
1. Locate the `_run_step` method definition
2. Remove the entire method definition
3. Verify no syntax errors result from the removal

**Completion Test:**
```bash
# Verify method is removed
grep -n "def _run_step" flujo/application/runner.py
# Should return no results
# Verify no syntax errors
python -m py_compile flujo/application/runner.py
# Should compile without errors
make test-fast
# Should pass 100% - no functionality depends on this method
```

**Acceptance Criteria:**
- [ ] `_run_step` method is completely removed
- [ ] No syntax errors result from the removal
- [ ] All tests pass with `make test-fast`
- [ ] No functionality is broken by the removal

---

### **Task 4.2: Update _execute_steps to Use Backend Directly**

**Objective:** Simplify the `_execute_steps` method to use the backend directly without the wrapper.

**Actions:**
1. Locate the `_execute_steps` method in `runner.py`
2. Remove the call to `self._run_step`
3. Update the method to pass the backend directly to `ExecutionManager`
4. Simplify the parameter passing

**Completion Test:**
```bash
# Verify _execute_steps method is updated
grep -n "self._run_step" flujo/application/runner.py
# Should return no results
# Verify method uses backend directly
grep -n "backend=self.backend" flujo/application/runner.py
# Should show the direct backend usage
make test-fast
# Should pass 100% - simplified execution flow works
```

**Acceptance Criteria:**
- [ ] `_execute_steps` no longer calls `self._run_step`
- [ ] Method uses `backend=self.backend` directly
- [ ] All tests pass with `make test-fast`
- [ ] Execution flow is simplified

---

### **Task 4.3: Remove Any Remaining _run_step References**

**Objective:** Clean up any remaining references to the removed `_run_step` method.

**Actions:**
1. Search for any remaining references to `_run_step`
2. Remove any found references
3. Verify no broken references remain

**Completion Test:**
```bash
# Search for any remaining _run_step references
grep -r "_run_step" flujo/ --exclude-dir=.git
# Should return no results
make test-fast
# Should pass 100% - no broken references
```

**Acceptance Criteria:**
- [ ] No references to `_run_step` remain
- [ ] All imports and calls are cleaned up
- [ ] All tests pass with `make test-fast`
- [ ] No broken references exist

## Task 5: Remove step_executor Parameter from ExecutionManager

### **Task 5.1: Remove step_executor Parameter from Method Signature**

**Objective:** Remove the `step_executor` parameter from `ExecutionManager.execute_steps`.

**Actions:**
1. Locate the `execute_steps` method in `execution_manager.py`
2. Remove the `step_executor` parameter from the method signature
3. Verify no syntax errors result from the removal

**Completion Test:**
```bash
# Verify parameter is removed from signature
grep -n "step_executor" flujo/application/core/execution_manager.py
# Should return no results or only in comments
# Verify no syntax errors
python -m py_compile flujo/application/core/execution_manager.py
# Should compile without errors
make test-fast
# Should pass 100% - parameter removal doesn't break functionality
```

**Acceptance Criteria:**
- [ ] `step_executor` parameter is removed from method signature
- [ ] No syntax errors result from the removal
- [ ] All tests pass with `make test-fast`
- [ ] No functionality is broken by the removal

---

### **Task 5.2: Update StepCoordinator to Receive Backend Directly**

**Objective:** Update `StepCoordinator.execute_step` to receive the backend directly.

**Actions:**
1. Locate the `execute_step` method in `StepCoordinator`
2. Update the method signature to receive `backend` parameter
3. Update the method implementation to use the backend directly
4. Remove any `step_executor` parameter usage

**Completion Test:**
```bash
# Verify StepCoordinator uses backend directly
grep -n "backend=" flujo/application/core/execution_manager.py
# Should show backend parameter usage
# Verify no step_executor references
grep -n "step_executor" flujo/application/core/execution_manager.py
# Should return no results
make test-fast
# Should pass 100% - StepCoordinator uses backend directly
```

**Acceptance Criteria:**
- [ ] `StepCoordinator.execute_step` receives `backend` parameter
- [ ] Method uses backend directly instead of step_executor
- [ ] All tests pass with `make test-fast`
- [ ] Execution flow is simplified

---

### **Task 5.3: Update ExecutionManager to Use self.backend**

**Objective:** Update `ExecutionManager.execute_steps` to use `self.backend` directly.

**Actions:**
1. Locate where `step_executor` was used in `execute_steps`
2. Replace with `self.backend`
3. Update any parameter passing to use backend directly
4. Simplify the execution flow

**Completion Test:**
```bash
# Verify ExecutionManager uses self.backend
grep -n "self.backend" flujo/application/core/execution_manager.py
# Should show backend usage
# Verify no step_executor usage
grep -n "step_executor" flujo/application/core/execution_manager.py
# Should return no results
make test-fast
# Should pass 100% - ExecutionManager uses backend directly
```

**Acceptance Criteria:**
- [ ] `ExecutionManager.execute_steps` uses `self.backend` directly
- [ ] No `step_executor` parameter usage remains
- [ ] All tests pass with `make test-fast`
- [ ] Execution flow is simplified

---

### **Task 5.4: Clean Up Any Remaining step_executor References**

**Objective:** Remove any remaining references to the `step_executor` parameter.

**Actions:**
1. Search for any remaining references to `step_executor`
2. Remove any found references
3. Verify no broken references remain

**Completion Test:**
```bash
# Search for any remaining step_executor references
grep -r "step_executor" flujo/ --exclude-dir=.git
# Should return no results
make test-fast
# Should pass 100% - no broken references
```

**Acceptance Criteria:**
- [ ] No references to `step_executor` remain
- [ ] All parameter passing is simplified
- [ ] All tests pass with `make test-fast`
- [ ] No broken references exist

## Task 6: Static Analysis and Code Quality

### **Task 6.1: Run Ruff Linter**

**Objective:** Ensure code quality with the `ruff` linter.

**Actions:**
1. Run `ruff` on the entire codebase
2. Fix any code quality issues found
3. Ensure all issues are resolved

**Completion Test:**
```bash
# Run ruff linter
ruff check flujo/
# Should pass with no errors
make test-fast
# Should pass 100% - code quality maintained
```

**Acceptance Criteria:**
- [ ] `ruff` passes with no errors
- [ ] All code quality issues are resolved
- [ ] All tests pass with `make test-fast`
- [ ] Code quality is maintained

---

### **Task 6.2: Run Flake8 Linter**

**Objective:** Ensure code quality with the `flake8` linter.

**Actions:**
1. Run `flake8` on the entire codebase
2. Fix any code quality issues found
3. Ensure all issues are resolved

**Completion Test:**
```bash
# Run flake8 linter
flake8 flujo/
# Should pass with no errors
make test-fast
# Should pass 100% - code quality maintained
```

**Acceptance Criteria:**
- [ ] `flake8` passes with no errors
- [ ] All code quality issues are resolved
- [ ] All tests pass with `make test-fast`
- [ ] Code quality is maintained

---

### **Task 6.3: Run MyPy Type Checker**

**Objective:** Ensure type safety with the `mypy` type checker.

**Actions:**
1. Run `mypy` on the entire codebase
2. Fix any type issues found
3. Ensure all issues are resolved

**Completion Test:**
```bash
# Run mypy type checker
mypy flujo/
# Should pass with no errors
make test-fast
# Should pass 100% - type safety maintained
```

**Acceptance Criteria:**
- [ ] `mypy` passes with no errors
- [ ] All type issues are resolved
- [ ] All tests pass with `make test-fast`
- [ ] Type safety is maintained

---

### **Task 6.4: Verify Import Resolution**

**Objective:** Ensure all imports resolve correctly after the cleanup.

**Actions:**
1. Check for any broken imports
2. Verify all imports resolve correctly
3. Fix any import issues found

**Completion Test:**
```bash
# Check for import issues
python -c "import flujo; print('All imports resolve correctly')"
# Should print success message
make test-fast
# Should pass 100% - all imports work correctly
```

**Acceptance Criteria:**
- [ ] All imports resolve correctly
- [ ] No import errors occur
- [ ] All tests pass with `make test-fast`
- [ ] Import system is clean

## Task 7: Comprehensive Testing

### **Task 7.1: Run Full Test Suite**

**Objective:** Ensure all tests pass after the cleanup.

**Actions:**
1. Run the complete test suite
2. Verify all tests pass
3. Document any failures and fix them

**Completion Test:**
```bash
# Run full test suite
make test-fast
# Should pass 100% - no regressions introduced
```

**Acceptance Criteria:**
- [ ] All tests pass (100% success rate)
- [ ] No regressions are introduced
- [ ] All functionality works correctly
- [ ] Test suite validates the cleanup

---

### **Task 7.2: Run Integration Tests**

**Objective:** Verify complex workflows still work correctly.

**Actions:**
1. Run integration tests specifically
2. Verify complex workflows function correctly
3. Ensure no integration issues

**Completion Test:**
```bash
# Run integration tests
python -m pytest tests/integration/ -v
# Should pass 100% - complex workflows work correctly
make test-fast
# Should pass 100% - no integration regressions
```

**Acceptance Criteria:**
- [ ] All integration tests pass
- [ ] Complex workflows function correctly
- [ ] No integration regressions
- [ ] All tests pass with `make test-fast`

---

### **Task 7.3: Run Performance Tests**

**Objective:** Verify no performance degradation was introduced.

**Actions:**
1. Run performance tests
2. Verify no performance regression
3. Document performance characteristics

**Completion Test:**
```bash
# Run performance tests
python -m pytest tests/benchmarks/ -v
# Should pass with no performance regression
make test-fast
# Should pass 100% - performance maintained
```

**Acceptance Criteria:**
- [ ] Performance tests pass
- [ ] No performance degradation
- [ ] Performance characteristics maintained
- [ ] All tests pass with `make test-fast`

---

### **Task 7.4: Run Edge Case Tests**

**Objective:** Verify edge cases and error handling still work correctly.

**Actions:**
1. Run edge case tests
2. Verify error handling works correctly
3. Ensure no edge case regressions

**Completion Test:**
```bash
# Run edge case tests
python -m pytest tests/unit/ -k "error\|exception\|edge" -v
# Should pass 100% - edge cases handled correctly
make test-fast
# Should pass 100% - no edge case regressions
```

**Acceptance Criteria:**
- [ ] Edge case tests pass
- [ ] Error handling works correctly
- [ ] No edge case regressions
- [ ] All tests pass with `make test-fast`

## Task 8: Final Validation

### **Task 8.1: Verify Clean Architecture**

**Objective:** Confirm the architecture is clean and simplified.

**Actions:**
1. Verify execution flow is simplified
2. Confirm no legacy code remains
3. Validate the new architecture is the single source of truth

**Completion Test:**
```bash
# Verify simplified execution flow
grep -n "backend.execute_step" flujo/application/runner.py
# Should show direct backend usage
# Verify no legacy references
grep -r "step_logic\|UltraStepExecutor\|_Frame\|_LRUCache\|_UsageTracker\|_run_step\|step_executor" flujo/ --exclude-dir=.git
# Should return no results
make test-fast
# Should pass 100% - clean architecture validated
```

**Acceptance Criteria:**
- [ ] Execution flow is simplified and direct
- [ ] No legacy code remains
- [ ] New architecture is the single source of truth
- [ ] All tests pass with `make test-fast`

---

### **Task 8.2: Final Comprehensive Test**

**Objective:** Run the final comprehensive test to validate the complete cleanup.

**Actions:**
1. Run all static analysis tools
2. Execute the complete test suite
3. Verify all functionality works correctly
4. Document the successful completion

**Completion Test:**
```bash
# Run all static analysis
ruff check flujo/
flake8 flujo/
mypy flujo/
# All should pass with no errors

# Run complete test suite
make test-fast
# Should pass 100% - complete cleanup validated

# Verify no legacy references
grep -r "step_logic\|UltraStepExecutor\|_Frame\|_LRUCache\|_UsageTracker\|_run_step\|step_executor" flujo/ --exclude-dir=.git
# Should return no results
```

**Acceptance Criteria:**
- [ ] All static analysis tools pass
- [ ] Complete test suite passes (100%)
- [ ] No legacy references remain
- [ ] Cleanup is complete and successful

---

### **Task 8.3: Documentation Update**

**Objective:** Update documentation to reflect the completed cleanup.

**Actions:**
1. Update any documentation that referenced removed components
2. Document the simplified architecture
3. Update any examples or references

**Completion Test:**
```bash
# Verify documentation is updated
grep -r "step_logic\|UltraStepExecutor\|_Frame\|_LRUCache\|_UsageTracker\|_run_step\|step_executor" docs/ --exclude-dir=__pycache__
# Should return no results or only updated references
make test-fast
# Should pass 100% - documentation updated
```

**Acceptance Criteria:**
- [ ] Documentation is updated and accurate
- [ ] No references to removed components remain
- [ ] Architecture documentation reflects the cleanup
- [ ] All tests pass with `make test-fast`

## Summary

This task list provides a comprehensive, step-by-step approach to completing FSD-11: Final Deprecation and Cleanup. Each task is:

1. **Small and Focused**: Each task has a single, clear objective
2. **Testable**: Every task ends with a completion test using `make test-fast`
3. **Incremental**: Changes are made step by step to minimize risk
4. **Validated**: Each step is verified before proceeding to the next

The tasks follow first principles by:
- **Eliminating Technical Debt**: Remove dead code systematically
- **Establishing Single Source of Truth**: Only the correct implementation remains
- **Simplifying Architecture**: Remove unnecessary layers and indirection
- **Improving Maintainability**: Create a clean, focused codebase

This approach ensures that the cleanup is thorough, safe, and maintains all functionality while achieving the goal of a clean, lean codebase that reflects only the correct, production-ready architecture.
