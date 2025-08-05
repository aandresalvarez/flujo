# Design Document

## Overview

The test suite restoration will follow a systematic, first-principles approach to diagnose and fix the 241 failing tests. The design focuses on identifying root causes rather than symptoms, ensuring that fixes address fundamental issues while maintaining backward compatibility and test integrity.

## Architecture

### Diagnostic Framework

The restoration process will use a layered diagnostic approach:

1. **Pattern Analysis Layer**: Group failures by common patterns and root causes
2. **Dependency Analysis Layer**: Identify cascading failures and fix order dependencies  
3. **Validation Layer**: Ensure fixes don't introduce regressions
4. **Integration Layer**: Verify fixes work together harmoniously

### Core Problem Categories

Based on the failure analysis, the issues fall into these categories:

1. **Contract Violations**: Tests expect specific behavior that has changed
2. **State Management Issues**: Improper handling of test isolation and cleanup
3. **Type System Mismatches**: Serialization and validation inconsistencies
4. **Infrastructure Drift**: Database schema and CLI interface changes

## Components and Interfaces

### 1. Usage Governor Fixes

**Problem**: Format inconsistencies and token aggregation issues

**Solution Design**:
- Fix cost formatting to match test expectations (preserve decimal places when needed)
- Implement proper token aggregation from step history to `total_tokens`
- Ensure error messages match expected patterns exactly

**Interface Changes**:
```python
# In UsageGovernor.check_usage_limits()
# Before: format_cost(10.0) -> "10" 
# After: format_cost(10.0) -> "10.0" (when test expects it)

# Before: Check pipeline_result.total_tokens (may be 0)
# After: Aggregate from step_history if total_tokens is 0
```

### 2. Parallel Step Context Handling

**Problem**: Context validation failures and attribute access issues

**Solution Design**:
- Implement proper context validation for parallel branches
- Add missing attributes to mock context objects
- Fix context merging logic to handle edge cases

**Interface Changes**:
```python
# Add missing attributes to test contexts
class MockContext:
    def __init__(self):
        self.initial_prompt = "test"  # Add missing attribute
        self.scratchpad = {}
        # ... other required attributes
```

### 3. Serialization System Hardening

**Problem**: Unregistered types and edge case handling

**Solution Design**:
- Extend custom serializer registry for all test mock objects
- Implement robust fallback serialization for unknown types
- Add proper handling for circular references and edge cases

**Interface Changes**:
```python
# In conftest.py - extend serializer registration
register_custom_serializer(MockEnum, lambda obj: obj.value)
register_custom_serializer(OrderedDict, lambda obj: dict(obj))
# Add fallback for unknown types
```

### 4. Database Schema Alignment

**Problem**: Column name mismatches and schema drift

**Solution Design**:
- Identify schema differences between test expectations and actual schema
- Implement migration logic or update tests to match current schema
- Add proper error handling for schema mismatches

**Interface Changes**:
```python
# Update column references
# Before: SELECT start_time FROM runs
# After: SELECT created_at FROM runs (or add migration)
```

### 5. Test Infrastructure Improvements

**Problem**: Inadequate test isolation and mock object handling

**Solution Design**:
- Enhance NoOpStateBackend to better simulate real backends
- Improve test fixture consistency
- Add proper cleanup mechanisms

## Data Models

### Test Execution Context
```python
@dataclass
class TestExecutionContext:
    """Context for tracking test execution and fixes"""
    test_name: str
    failure_category: str
    root_cause: str
    fix_applied: bool
    regression_risk: str
```

### Fix Tracking
```python
@dataclass
class FixRecord:
    """Record of applied fixes for regression tracking"""
    component: str
    issue_type: str
    fix_description: str
    affected_tests: List[str]
    validation_status: str
```

## Error Handling

### Regression Prevention Strategy

1. **Incremental Fixing**: Fix one category at a time to isolate impacts
2. **Validation Gates**: Run test subsets after each fix to catch regressions
3. **Rollback Capability**: Maintain ability to revert changes if needed
4. **Impact Analysis**: Document which tests are affected by each fix

### Error Classification

- **Critical**: Breaks core functionality (usage limits, parallel execution)
- **High**: Affects user-facing features (CLI, serialization)
- **Medium**: Infrastructure issues (database, test setup)
- **Low**: Edge cases and minor inconsistencies

## Testing Strategy

### Fix Validation Process

1. **Unit Test Validation**: Ensure individual components work correctly
2. **Integration Test Validation**: Verify components work together
3. **Regression Test Suite**: Run full test suite after each major fix
4. **Performance Impact Assessment**: Ensure fixes don't degrade performance

### Test Categories for Validation

- **Fast Tests**: Core functionality that must always pass
- **Integration Tests**: Cross-component interactions
- **Edge Case Tests**: Boundary conditions and error scenarios
- **Performance Tests**: Ensure no degradation

### Validation Checkpoints

1. After Usage Governor fixes: Validate cost/token limit enforcement
2. After Parallel Step fixes: Validate concurrent execution
3. After Serialization fixes: Validate object persistence
4. After Database fixes: Validate state management
5. After CLI fixes: Validate command-line tools
6. Final validation: Full test suite pass

## Implementation Phases

### Phase 1: Critical Infrastructure (Usage Governor, Parallel Steps)
- Fix usage limit enforcement
- Resolve parallel execution context issues
- Validate core pipeline functionality

### Phase 2: Data Layer (Serialization, Database)
- Fix serialization edge cases
- Resolve database schema issues
- Ensure proper state persistence

### Phase 3: User Interface (CLI, Error Messages)
- Fix CLI command failures
- Standardize error message formats
- Improve user-facing diagnostics

### Phase 4: Edge Cases and Polish
- Handle remaining edge cases
- Optimize test performance
- Document fixes and prevention strategies

## Success Criteria

- All 241 failing tests pass
- No new test failures introduced
- Test execution time remains acceptable
- Core functionality verified through manual testing
- Comprehensive documentation of fixes applied