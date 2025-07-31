# FSD 2: Legacy Step Logic Cleanup and Deprecation

## Overview
After completing the LoopStep and ConditionalStep migrations, remove unused legacy functions from `step_logic.py` and establish a deprecation strategy for remaining legacy code.

## Rationale & First Principles
- **Goal**: Eliminate technical debt by removing unused legacy code
- **Why**: Reduce maintenance burden and improve code clarity
- **Impact**: Cleaner codebase with clear separation between new and legacy implementations

## Scope of Work

### 1. Legacy Function Analysis
- **File**: `flujo/application/core/step_logic.py`
- **Action**: Identify and categorize all legacy functions

### 2. Migration Status Assessment
- **LoopStep**: ✅ Migrated to `ExecutorCore._handle_loop_step`
- **ConditionalStep**: ✅ Migrated to `ExecutorCore._handle_conditional_step`
- **ParallelStep**: ✅ Migrated to `ExecutorCore._handle_parallel_step`
- **DynamicParallelRouterStep**: ✅ Migrated to `ExecutorCore._handle_dynamic_router_step`
- **CacheStep**: ❌ Still used by legacy path
- **HITLStep**: ❌ Still used by legacy path

### 3. Cleanup Strategy
- Remove unused migrated functions
- Deprecate remaining legacy functions
- Update imports and references
- Maintain backward compatibility for remaining functions

## Testing Strategy (TDD Approach)

### Phase 1: Impact Analysis Tests
**File**: `tests/regression/test_legacy_cleanup_impact.py`

#### 1.1 Function Usage Analysis
```python
async def test_legacy_function_usage_analysis():
    """Analyze which legacy functions are still in use."""

async def test_import_dependency_analysis():
    """Analyze import dependencies on legacy functions."""

async def test_backward_compatibility_verification():
    """Verify that remaining legacy functions work correctly."""
```

#### 1.2 Migration Completeness Tests
```python
async def test_migrated_functions_removal():
    """Test that migrated functions can be safely removed."""

async def test_legacy_function_deprecation():
    """Test deprecation warnings for remaining legacy functions."""

async def test_import_path_updates():
    """Test that import paths are updated correctly."""
```

### Phase 2: Cleanup Validation Tests
**File**: `tests/integration/test_legacy_cleanup_validation.py`

#### 2.1 Function Removal Tests
```python
async def test_loop_step_logic_removal():
    """Test that _execute_loop_step_logic can be removed."""

async def test_conditional_step_logic_removal():
    """Test that _execute_conditional_step_logic can be removed."""

async def test_parallel_step_logic_removal():
    """Test that _execute_parallel_step_logic can be removed."""

async def test_dynamic_router_logic_removal():
    """Test that _execute_dynamic_router_step_logic can be removed."""
```

#### 2.2 Remaining Function Tests
```python
async def test_cache_step_logic_preservation():
    """Test that _handle_cache_step continues to work."""

async def test_hitl_step_logic_preservation():
    """Test that _handle_hitl_step continues to work."""

async def test_run_step_logic_preservation():
    """Test that _run_step_logic continues to work."""
```

### Phase 3: Performance Impact Tests
**File**: `tests/benchmarks/test_legacy_cleanup_performance.py`

#### 3.1 Performance Impact Analysis
```python
async def test_cleanup_performance_impact():
    """Test performance impact of removing legacy code."""

async def test_import_performance_improvement():
    """Test import performance improvement from cleanup."""

async def test_memory_usage_improvement():
    """Test memory usage improvement from cleanup."""
```

## Implementation Details

### Step 1: Create Function Usage Analysis
```python
# tools/analyze_legacy_usage.py
import ast
import os
from typing import Dict, Set, List

def analyze_function_usage(codebase_path: str) -> Dict[str, Set[str]]:
    """Analyze which legacy functions are still being used."""

def find_import_dependencies() -> Dict[str, List[str]]:
    """Find all import dependencies on legacy functions."""

def generate_cleanup_report() -> str:
    """Generate a comprehensive cleanup report."""
```

### Step 2: Remove Migrated Functions
**File**: `flujo/application/core/step_logic.py`

#### 2.1 Functions to Remove
```python
# Remove these functions (already migrated to ExecutorCore)
async def _execute_loop_step_logic(...):  # ❌ REMOVE
async def _execute_conditional_step_logic(...):  # ❌ REMOVE
async def _execute_parallel_step_logic(...):  # ❌ REMOVE
async def _execute_dynamic_router_step_logic(...):  # ❌ REMOVE

# Keep these functions (still used by legacy path)
async def _handle_cache_step(...):  # ✅ KEEP
async def _handle_hitl_step(...):  # ✅ KEEP
async def _run_step_logic(...):  # ✅ KEEP
```

#### 2.2 Update Imports
```python
# Remove unused imports
from ...domain.dsl.loop import LoopStep  # ❌ REMOVE
from ...domain.dsl.conditional import ConditionalStep  # ❌ REMOVE
from ...domain.dsl.parallel import ParallelStep  # ❌ REMOVE
from ...domain.dsl.dynamic_router import DynamicParallelRouterStep  # ❌ REMOVE

# Keep necessary imports
from ...domain.dsl.step import HumanInTheLoopStep  # ✅ KEEP
from flujo.steps.cache_step import CacheStep  # ✅ KEEP
```

### Step 3: Add Deprecation Warnings
```python
import warnings
from typing import Any

def deprecated_function(func):
    """Decorator to mark functions as deprecated."""
    def wrapper(*args, **kwargs):
        warnings.warn(
            f"{func.__name__} is deprecated and will be removed in a future version. "
            "Use the new ExecutorCore implementation instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return func(*args, **kwargs)
    return wrapper

@deprecated_function
async def _handle_cache_step(...):
    """Legacy cache step handler (deprecated)."""

@deprecated_function
async def _handle_hitl_step(...):
    """Legacy HITL step handler (deprecated)."""

@deprecated_function
async def _run_step_logic(...):
    """Legacy step logic runner (deprecated)."""
```

### Step 4: Update Documentation
```python
# Add deprecation notices to docstrings
async def _handle_cache_step(...):
    """
    Legacy cache step handler (DEPRECATED).

    This function is deprecated and will be removed in a future version.
    Use ExecutorCore._handle_cache_step instead.
    """
```

## Acceptance Criteria

### Functional Requirements
- [ ] All migrated functions removed from `step_logic.py`
- [ ] Remaining functions marked as deprecated
- [ ] Deprecation warnings displayed when legacy functions are used
- [ ] No breaking changes to existing functionality
- [ ] Import paths updated throughout codebase

### Quality Requirements
- [ ] All tests pass after cleanup
- [ ] No unused imports remain
- [ ] Code coverage maintained or improved
- [ ] Documentation updated with deprecation notices
- [ ] Migration guide created for remaining legacy functions

### Performance Requirements
- [ ] Import time improved
- [ ] Memory usage reduced
- [ ] No performance regressions
- [ ] Cleanup doesn't introduce new technical debt

## Risk Mitigation

### High-Risk Areas
1. **Breaking Changes**: Ensure no functionality is lost
2. **Import Dependencies**: Verify all imports are updated
3. **Test Coverage**: Maintain comprehensive test coverage
4. **Backward Compatibility**: Preserve existing APIs

### Mitigation Strategies
1. **Gradual Deprecation**: Use warnings instead of immediate removal
2. **Comprehensive Testing**: Test all affected code paths
3. **Documentation**: Clear migration guides for remaining functions
4. **Rollback Plan**: Ability to restore removed functions if needed

## Success Metrics

### Quantitative Metrics
- [ ] 0% test failures after cleanup
- [ ] 0% performance regression
- [ ] Reduced codebase size (lines of code)
- [ ] Reduced import time
- [ ] Reduced memory usage

### Qualitative Metrics
- [ ] Improved code maintainability
- [ ] Clearer separation of concerns
- [ ] Reduced technical debt
- [ ] Better developer experience

## Timeline
- **Phase 1 (Analysis)**: 1 day
- **Phase 2 (Removal)**: 2 days
- **Phase 3 (Deprecation)**: 1 day
- **Phase 4 (Documentation)**: 1 day
- **Phase 5 (Validation)**: 1 day

**Total Estimated Time**: 6 days
