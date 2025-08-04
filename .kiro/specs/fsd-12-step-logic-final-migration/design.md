# Design Document: FSD-12 Step Logic Final Migration

## Overview

This design addresses the final incomplete migration from `step_logic.py` to the new `ExecutorCore` architecture. Following Flujo's architectural philosophy, this migration embodies the **dual architecture principle**: maintaining the elegant, composable DSL while strengthening the production-ready execution core. The solution must be **algebraically closed**—every step, regardless of complexity, should be a first-class citizen in the execution graph.

## Current State Analysis

### **Remaining Elements in step_logic.py**

Based on comprehensive analysis, the following elements still need migration:

#### **1. StepExecutor Type Alias** (Line 1142 in step_logic.py)
```python
StepExecutor = Callable[[Any, Any, Optional[Any], Optional[Any], Optional[Any]], Awaitable[StepResult]]
```
**Current Usage:**
- `flujo/application/parallel.py` imports and exports it
- Used as a type alias for step execution function signatures

#### **2. ParallelUsageGovernor Function** (Line 98 import)
**Current Usage:**
- `flujo/application/core/ultra_executor.py` imports it
- Used in parallel step execution for usage tracking and limits enforcement

#### **3. _should_pass_context Function** (Line 98 import)
**Current Usage:**
- `flujo/application/core/ultra_executor.py` imports it
- Used to determine if context should be passed to plugins/processors

#### **4. _run_step_logic Function** (Line 1266 import)
**Current Usage:**
- `flujo/application/core/ultra_executor.py` calls it at line 1387
- This is the main step execution logic that was supposed to be migrated to ExecutorCore

#### **5. _default_set_final_context Function** (Line 1266 import)
**Current Usage:**
- `flujo/application/core/ultra_executor.py` imports it
- Used as a default context setter function

## Architecture

The solution follows Flujo's architectural principles and the established patterns in previous FSDs:

1. **Algebraic Closure**: Every step, whether simple or complex, must be a first-class citizen in the execution graph
2. **Dual Architecture**: Strengthen the execution core while preserving the declarative shell's elegance
3. **Production Readiness**: Ensure the migrated elements maintain resilience, performance, and observability
4. **Recursive Execution**: The migrated elements must work seamlessly with Flujo's recursive execution model
5. **Dependency Injection**: Maintain the modular, pluggable architecture that enables extensibility

## Migration Strategy

### **Phase 1: Type System Migration**

#### **StepExecutor Type Alias Migration**
**Current Location:** `flujo/application/core/step_logic.py` (Line 1142)
**Target Location:** `flujo/application/core/ultra_executor.py` or dedicated types module

**Rationale:**
- Type aliases should be co-located with their primary usage
- ExecutorCore is the primary consumer of this type
- Maintains type safety and IDE support

**Implementation:**
```python
# In flujo/application/core/ultra_executor.py
StepExecutor = Callable[[Any, Any, Optional[Any], Optional[Any], Optional[Any]], Awaitable[StepResult]]
```

### **Phase 2: Utility Function Migration**

#### **ParallelUsageGovernor Migration**
**Current Location:** `flujo/application/core/step_logic.py`
**Target Location:** `flujo/application/core/ultra_executor.py` (as private method)

**Rationale:**
- Parallel step execution is handled by ExecutorCore
- Usage governance is a core responsibility of the executor
- Maintains encapsulation and reduces coupling

**Implementation:**
```python
# In ExecutorCore class
async def _parallel_usage_governor(self, ...) -> None:
    """Govern usage limits for parallel step execution."""
```

#### **_should_pass_context Migration**
**Current Location:** `flujo/application/core/step_logic.py`
**Target Location:** `flujo/application/core/ultra_executor.py` (as private method)

**Rationale:**
- Context passing logic is specific to step execution
- ExecutorCore manages all step execution aspects
- Reduces external dependencies

**Implementation:**
```python
# In ExecutorCore class
def _should_pass_context(self, context: Optional[Any], func: Callable[..., Any]) -> bool:
    """Determine if context should be passed to plugins/processors."""
```

### **Phase 3: Core Logic Migration**

#### **_run_step_logic Migration**
**Current Location:** `flujo/application/core/step_logic.py`
**Target Location:** `flujo/application/core/ultra_executor.py` (integrated into ExecutorCore)

**Rationale:**
- This is the core step execution logic
- Should be fully integrated into ExecutorCore
- Eliminates the need for external step_logic dependencies

**Implementation:**
```python
# In ExecutorCore class
async def _execute_step_logic(self, step: Any, data: Any, ...) -> StepResult:
    """Core logic for executing a single step."""
    # Migrated logic from _run_step_logic
```

#### **_default_set_final_context Migration**
**Current Location:** `flujo/application/core/step_logic.py`
**Target Location:** `flujo/application/core/ultra_executor.py` (as private method)

**Rationale:**
- Context management is a core responsibility of ExecutorCore
- Default behavior should be encapsulated within the executor
- Maintains consistency with other context operations

**Implementation:**
```python
# In ExecutorCore class
def _default_set_final_context(self, result: PipelineResult[Any], context: Optional[Any]) -> None:
    """Default context setter function."""
```

## Data Models

### **StepExecutor Type Definition**
```python
from typing import Any, Optional, Callable, Awaitable
from ...domain.models import StepResult

StepExecutor = Callable[[Any, Any, Optional[Any], Optional[Any], Optional[Any]], Awaitable[StepResult]]
```

### **Usage Governance Interface**
```python
class IUsageGovernor(Protocol):
    @abstractmethod
    async def govern_parallel_execution(
        self, 
        limits: UsageLimits, 
        step_history: List[Any]
    ) -> None:
        """Govern usage limits for parallel step execution."""
```

### **Context Management Interface**
```python
class IContextManager(Protocol):
    @abstractmethod
    def should_pass_context(self, context: Optional[Any], func: Callable[..., Any]) -> bool:
        """Determine if context should be passed to plugins/processors."""
    
    @abstractmethod
    def set_final_context(self, result: PipelineResult[Any], context: Optional[Any]) -> None:
        """Set final context after step execution."""
```

## Error Handling

### **Graceful Migration**
- Maintain backward compatibility during migration
- Preserve existing behavior for all migrated functions
- Ensure no breaking changes to public APIs

### **Validation Strategy**
- Ensure all existing tests continue to pass
- Verify that migrated functions produce identical results
- Confirm that new step types can be added without core changes

## Testing Strategy

### **Unit Tests**
- Test each migrated function in isolation
- Verify that the new implementation produces identical results
- Test edge cases and error conditions

### **Integration Tests**
- Run existing step execution tests
- Verify that step dispatch logic remains unchanged
- Test with real pipeline execution scenarios

### **Regression Tests**
- Ensure no functional changes to step execution
- Verify that performance characteristics remain similar
- Confirm that error handling behavior is preserved

## Migration Strategy

### **Phase 1: Implementation**
1. Migrate StepExecutor type alias to ultra_executor.py
2. Migrate ParallelUsageGovernor to ExecutorCore
3. Migrate _should_pass_context to ExecutorCore
4. Integrate _run_step_logic into ExecutorCore
5. Migrate _default_set_final_context to ExecutorCore

### **Phase 2: Validation**
1. Run comprehensive test suite
2. Verify no behavioral changes
3. Confirm performance characteristics

### **Phase 3: Cleanup**
1. Update import statements
2. Remove step_logic.py dependencies
3. Update documentation

## Performance Considerations

### **Current Performance**
- Direct function calls to step_logic functions
- Minimal overhead for type aliases
- Efficient context passing logic

### **New Performance**
- Inline methods in ExecutorCore
- Reduced import overhead
- Optimized context management

### **Expected Impact**
- **Production-Ready Performance**: Reduced import overhead improves execution efficiency
- **Scalability**: Inline methods reduce function call overhead
- **Memory Efficiency**: Reduced module dependencies and improved cache locality
- **Observability**: Cleaner execution traces with integrated logic

## Backward Compatibility

### **Existing Step Types**
- All existing step types continue to work unchanged
- No changes required to step implementations
- Existing tests should pass without modification

### **Import Compatibility**
- Maintain existing import patterns where possible
- Provide migration path for external consumers
- Ensure no breaking changes to public APIs

## Future Extensibility

### **Adding New Step Types**
With the complete migration, adding new step types becomes trivial:

```python
class NewStepType(Step[Any, Any]):
    @property
    def is_complex(self) -> bool:
        return True  # or False based on complexity
```

### **No Core Changes Required**
- New step types don't require changes to ExecutorCore
- The dispatch logic automatically recognizes new step types
- Follows the Open-Closed Principle (open for extension, closed for modification)

## Success Criteria

1. **Algebraic Closure**: Every step type, current and future, is a first-class citizen in the execution graph
2. **Production Readiness**: The migrated elements maintain resilience, performance, and observability characteristics
3. **Recursive Execution**: Seamless integration with Flujo's recursive execution model
4. **Dual Architecture**: Strengthens the execution core while preserving DSL elegance
5. **Extensibility**: New step types can be added without core changes
6. **Functional Equivalence**: The migrated elements produce identical results to the current implementation
7. **Performance**: No degradation in performance characteristics, with potential improvements
8. **Testing**: All existing tests continue to pass
9. **Clean Architecture**: Complete removal of step_logic.py dependencies 