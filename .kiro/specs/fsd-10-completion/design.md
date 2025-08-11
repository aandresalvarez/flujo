# Design Document

## Overview

This design addresses the final incomplete task from FSD 10: "Hardening Contracts & Finalizing `ExecutorCore`" from first principles. The remaining work focuses on refactoring the `ExecutorCore._is_complex_step` method to use the new object-oriented approach with the `is_complex` property instead of the current procedural `isinstance` checks.

Following Flujo's architectural philosophy, this refactoring embodies the **dual architecture principle**: maintaining the elegant, composable DSL while strengthening the production-ready execution core. The solution must be **algebraically closed**—any step, regardless of complexity, should be a first-class citizen in the execution graph.

## Architecture

The solution follows Flujo's architectural principles and the established patterns in FSD 10:

1. **Algebraic Closure**: Every step, whether simple or complex, must be a first-class citizen in the execution graph
2. **Dual Architecture**: Strengthen the execution core while preserving the declarative shell's elegance
3. **Production Readiness**: Ensure the refactored method maintains resilience, performance, and observability
4. **Recursive Execution**: The object-oriented approach must work seamlessly with the recursive execution model
5. **Dependency Injection**: Maintain the modular, pluggable architecture that enables extensibility

## Components and Interfaces

### Current Implementation Analysis

The current `_is_complex_step` method in `flujo/application/core/ultra_executor.py` uses procedural checks:

```python
def _is_complex_step(self, step: Any) -> bool:
    """Check if step needs complex handling."""
    # Check for specific step types
    if isinstance(step, (
        CacheStep,
        LoopStep,
        ConditionalStep,
        DynamicParallelRouterStep,
        ParallelStep,
        HumanInTheLoopStep,
    )):
        return True

    # Check for validation steps
    if hasattr(step, "meta") and step.meta and step.meta.get("is_validation_step", False):
        return True

    # Check for steps with plugins
    if hasattr(step, "plugins") and step.plugins:
        return True

    return False
```

### Target Implementation

The refactored method should use the object-oriented approach:

```python
def _is_complex_step(self, step: Any) -> bool:
    """Check if step needs complex handling using an object-oriented approach."""
    return getattr(step, 'is_complex', False)
```

### Property-Based Detection System

All complex step types now implement the `is_complex` property:

- **Base Step**: `is_complex = False` (default)
- **LoopStep**: `is_complex = True`
- **ParallelStep**: `is_complex = True`
- **ConditionalStep**: `is_complex = True`
- **CacheStep**: `is_complex = True`
- **HumanInTheLoopStep**: `is_complex = True`
- **DynamicParallelRouterStep**: `is_complex = True`

## Data Models

### Step Complexity Classification

```python
class StepComplexity(Enum):
    SIMPLE = "simple"
    COMPLEX = "complex"

class StepType(Enum):
    BASIC = "basic"
    LOOP = "loop"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    CACHE = "cache"
    HUMAN_IN_THE_LOOP = "human_in_the_loop"
    DYNAMIC_ROUTER = "dynamic_router"
    VALIDATION = "validation"
    PLUGIN_ENABLED = "plugin_enabled"
```

### Detection Logic Mapping

| Step Type | Current Detection | New Detection | Rationale |
|-----------|------------------|---------------|-----------|
| Basic Step | `isinstance` checks | `is_complex = False` | Default behavior |
| LoopStep | `isinstance(LoopStep)` | `is_complex = True` | Object-oriented |
| ParallelStep | `isinstance(ParallelStep)` | `is_complex = True` | Object-oriented |
| ConditionalStep | `isinstance(ConditionalStep)` | `is_complex = True` | Object-oriented |
| CacheStep | `isinstance(CacheStep)` | `is_complex = True` | Object-oriented |
| HumanInTheLoopStep | `isinstance(HumanInTheLoopStep)` | `is_complex = True` | Object-oriented |
| DynamicParallelRouterStep | `isinstance(DynamicParallelRouterStep)` | `is_complex = True` | Object-oriented |
| Validation Steps | `meta.get("is_validation_step")` | `is_complex = True` | Keep existing logic |
| Plugin Steps | `hasattr(step, "plugins") and step.plugins` | `is_complex = True` | Keep existing logic |

## Error Handling

### Graceful Degradation
- Use `getattr(step, 'is_complex', False)` to handle steps without the property
- Maintain backward compatibility with existing step types
- Preserve existing behavior for validation and plugin steps

### Validation Strategy
- Ensure all existing tests continue to pass
- Verify that complex step detection remains accurate
- Confirm that new step types can be added without core changes

## Testing Strategy

### Unit Tests
- Test the refactored `_is_complex_step` method with all step types
- Verify that the new implementation produces identical results
- Test edge cases with steps that don't have the `is_complex` property

### Integration Tests
- Run existing complex step classification tests
- Verify that step dispatch logic remains unchanged
- Test with real pipeline execution scenarios

### Regression Tests
- Ensure no functional changes to step execution
- Verify that performance characteristics remain similar
- Confirm that error handling behavior is preserved

## Migration Strategy

### Phase 1: Implementation
1. Refactor `_is_complex_step` method to use object-oriented approach
2. Add fallback logic for steps without `is_complex` property
3. Maintain existing validation and plugin step detection

### Phase 2: Validation
1. Run comprehensive test suite
2. Verify no behavioral changes
3. Confirm performance characteristics

### Phase 3: Documentation
1. Update method documentation
2. Add examples of extending complex step types
3. Document the object-oriented approach

## Performance Considerations

### Current Performance
- `isinstance` checks are O(1) operations
- Multiple conditional checks per method call
- Linear time complexity with number of step types
- Potential performance bottlenecks in high-frequency execution paths

### New Performance
- `getattr` with default is O(1) operation
- Single property access per method call
- Constant time complexity regardless of step types
- Optimized for the recursive execution model

### Expected Impact
- **Production-Ready Performance**: Reduced branching improves execution efficiency
- **Scalability**: Constant time complexity regardless of step type proliferation
- **Memory Efficiency**: Reduced method complexity and improved cache locality
- **Observability**: Cleaner execution traces with simplified dispatch logic

## Backward Compatibility

### Existing Step Types
- All existing step types continue to work unchanged
- No changes required to step implementations
- Existing tests should pass without modification

### Plugin and Validation Steps
- Maintain existing detection logic for backward compatibility
- Preserve current behavior for steps with plugins or validation metadata
- Ensure no breaking changes to existing workflows

## Future Extensibility

### Adding New Complex Step Types
With the object-oriented approach, adding new complex step types becomes trivial:

```python
class NewComplexStep(Step[Any, Any]):
    @property
    def is_complex(self) -> bool:
        return True
```

### No Core Changes Required
- New complex step types don't require changes to `ExecutorCore`
- The dispatch logic automatically recognizes new complex steps
- Follows the Open-Closed Principle (open for extension, closed for modification)

## Success Criteria

1. **Algebraic Closure**: Every step type, current and future, is a first-class citizen in the execution graph
2. **Production Readiness**: The refactored method maintains resilience, performance, and observability characteristics
3. **Recursive Execution**: Seamless integration with Flujo's recursive execution model
4. **Dual Architecture**: Strengthens the execution core while preserving DSL elegance
5. **Extensibility**: New complex step types can be added without core changes
6. **Functional Equivalence**: The refactored method produces identical results to the current implementation
7. **Performance**: No degradation in performance characteristics, with potential improvements
8. **Testing**: All existing tests continue to pass
