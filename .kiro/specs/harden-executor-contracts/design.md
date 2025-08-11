# Design Document

## Overview

This design implements FSD 10's contract hardening approach by introducing explicit type contracts, making plugin failures explicit, and simplifying complex step detection through object-oriented design. The design focuses on static verifiability and eliminating runtime assumptions while maintaining full backward compatibility.

## Architecture

The hardening approach follows three main architectural changes:

1. **Type Contract Hardening**: Introduction of Protocol-based contracts with bounded TypeVars
2. **Explicit Failure Handling**: Plugin failures become explicit exceptions rather than silent continuations
3. **Object-Oriented Dispatch**: Complex step detection moves from procedural checks to object properties

## Components and Interfaces

### 1. Type Contract System

#### ContextWithScratchpad Protocol
```python
# flujo/application/core/types.py
from typing import Protocol, TypeVar, Dict, Any

class ContextWithScratchpad(Protocol):
    """A contract ensuring a context object has a scratchpad attribute."""
    scratchpad: Dict[str, Any]

TContext_w_Scratch = TypeVar("TContext_w_Scratch", bound=ContextWithScratchpad)
```

This protocol ensures that any context used with ParallelStep operations has the required scratchpad attribute, making the contract statically verifiable.

#### Updated ParallelStep Signature
```python
# flujo/domain/dsl/parallel.py
class ParallelStep(Step[Any, Any], Generic[TContext]):
    merge_strategy: Union[MergeStrategy, Callable[[TContext_w_Scratch, Dict[str, StepResult]], None]]
```

#### Updated ExecutorCore Method
```python
# flujo/application/core/ultra_executor.py
async def _handle_parallel_step(
    self,
    parallel_step: ParallelStep[TContext_w_Scratch],
    data: Any,
    context: Optional[TContext_w_Scratch],
    # ... other params
) -> StepResult:
```

### 2. Explicit Plugin Failure System

#### Updated DefaultPluginRunner
```python
# flujo/application/core/ultra_executor.py
class DefaultPluginRunner:
    async def run_plugins(self, plugins: List[tuple[Any, int]], data: Any, *, context: Any) -> Any:
        processed_data = data
        for plugin, priority in sorted(plugins, key=lambda x: x[1], reverse=True):
            try:
                # ... existing logic to call the plugin ...
                result = await func(...)

                if isinstance(result, PluginOutcome):
                    if not result.success:
                        # NEW: Raise an exception to fail the step
                        raise ValueError(f"Plugin validation failed: {result.feedback}")
                    # ...
            except Exception as e:
                # NEW: Re-raise the exception to ensure the step fails
                telemetry.logfire.error(f"Plugin {type(plugin).__name__} failed: {e}")
                raise e
        return processed_data
```

### 3. Object-Oriented Complex Step Detection

#### Base Step Class Enhancement
```python
# flujo/domain/dsl/step.py
class Step(BaseModel, Generic[StepInT, StepOutT]):
    # ... existing fields ...

    @property
    def is_complex(self) -> bool:
        """Indicates whether this step requires complex handling."""
        return False
```

#### Complex Step Subclasses
Each complex step type (LoopStep, ParallelStep, ConditionalStep, etc.) overrides the property:
```python
@property
def is_complex(self) -> bool:
    return True
```

#### Simplified ExecutorCore Detection
```python
# flujo/application/core/ultra_executor.py
def _is_complex_step(self, step: Any) -> bool:
    """Check if step needs complex handling using an object-oriented approach."""
    return getattr(step, 'is_complex', False)
```

## Data Models

### Protocol Definition
- **ContextWithScratchpad**: Protocol ensuring scratchpad attribute exists
- **TContext_w_Scratch**: Bounded TypeVar for type-safe context handling

### Property Addition
- **Step.is_complex**: Boolean property indicating complex step handling requirement

## Error Handling

### Plugin Error Propagation
- Plugin exceptions are re-raised rather than caught and ignored
- PluginOutcome failures become ValueError exceptions
- All plugin failures are logged to telemetry before re-raising

### Type Safety Errors
- mypy will catch type violations at static analysis time
- Runtime AttributeError for missing scratchpad becomes compile-time error

## Testing Strategy

### Static Analysis Testing
- New test file: `tests/static_analysis/test_contracts.py`
- Tests that mypy catches type violations for contexts without scratchpad
- Verifies Protocol enforcement works correctly

### Plugin Failure Testing
- Tests in `tests/application/core/test_executor_core.py`
- Verifies plugin exceptions cause step failure
- Confirms error messages are preserved in StepResult

### Complex Step Detection Testing
- Unit tests for `ExecutorCore._is_complex_step`
- Verifies base steps return False, complex steps return True
- Tests extensibility by creating mock complex step types

### Regression Testing
- Full existing test suite must pass at 100%
- Particular attention to plugin-related tests that may have relied on forgiving behavior
- Any test failures must be investigated and resolved

## Migration Strategy

### Phase 1: Type Contract Introduction
1. Create `flujo/application/core/types.py` with Protocol and TypeVar
2. Update ParallelStep class definition
3. Update ExecutorCore._handle_parallel_step signature
4. Run mypy to verify no new type errors

### Phase 2: Plugin Failure Hardening
1. Modify DefaultPluginRunner.run_plugins to re-raise exceptions
2. Add telemetry logging before re-raising
3. Run existing plugin tests to identify any that need updating

### Phase 3: Complex Step Detection Refactoring
1. Add is_complex property to base Step class
2. Override property in all complex step subclasses
3. Simplify ExecutorCore._is_complex_step implementation
4. Verify all step types are correctly classified

### Phase 4: Testing and Validation
1. Add new static analysis tests
2. Add plugin failure tests
3. Add complex step detection tests
4. Run full test suite and achieve 100% pass rate

## Backward Compatibility

### API Compatibility
- All existing public APIs remain unchanged
- Internal contract changes are transparent to users
- Existing step definitions continue to work

### Behavioral Compatibility
- Step execution behavior remains identical
- Error handling becomes more explicit but maintains same failure modes
- Performance characteristics are preserved

### Migration Path
- No user code changes required
- Existing pipelines work without modification
- New type safety benefits are opt-in through static analysis
