# Functional Specification Document: Expose `state_providers` in Flujo Runner

**Document Version:** 1.0  
**Date:** 2025-11-24  
**Status:** ✅ IMPLEMENTED  
**Labels:** `enhancement`, `api-design`, `state-management`, `context-reference`

---

## 1. Executive Summary

This FSD proposes exposing the `state_providers` parameter in the public `Flujo` runner API to enable proper usage of the `ContextReference` feature for managed state hydration. Currently, while the core execution layer supports `state_providers`, there is no way to pass them through the public API, forcing users to access private attributes or preventing them from using `ContextReference` entirely.

**Impact:** Enables production-ready usage of `ContextReference` for large-scale state management without serializing entire databases into the context object.

---

## 2. Problem Statement

### 2.1 Current State

The `ContextReference` feature was introduced to support managed state hydration, allowing context objects to contain lightweight pointers to external data stores (databases, knowledge graphs, etc.) rather than serializing entire datasets. The architecture supports this at multiple layers:

- ✅ **Domain Layer**: `ContextReference` class exists in `flujo/domain/models.py`
- ✅ **Core Execution Layer**: `ExecutorCore` accepts `state_providers: Dict[str, StateProvider]` parameter
- ❌ **Application Layer**: `Flujo` runner and factory chain do not expose this parameter

### 2.2 User Impact

Users attempting to use `ContextReference` encounter a critical gap:

```python
from flujo import Flujo
from flujo.domain.interfaces import StateProvider

class MyKGProvider(StateProvider):
    async def load(self, key: str): return []
    async def save(self, key: str, data): pass

# This fails:
runner = Flujo(
    pipeline=my_pipeline,
    context_model=MyContext,
    state_providers={"kg_provider": MyKGProvider()}  # ❌ TypeError
)
```

**Error:** `TypeError: __init__() got an unexpected keyword argument 'state_providers'`

### 2.3 Current Workaround

Users are forced to use brittle workarounds that access private attributes:

```python
runner = Flujo(pipeline, ...)
# Manual injection into private attributes
if hasattr(runner.backend, "_executor"):
    runner.backend._executor._state_providers["kg_provider"] = my_provider
```

This approach:
- Violates encapsulation
- May break with internal refactorings
- Is not documented or supported
- Fails if backend structure changes

---

## 3. Root Cause Analysis

### 3.1 Architecture Flow

The execution stack follows this factory pattern:

```
Flujo.__init__()
  ├─> Creates ExecutorFactory() (line 342)
  ├─> Creates BackendFactory(executor_factory) (line 343)
  └─> Calls _create_default_backend()
      └─> Calls backend_factory.create_execution_backend()
          └─> Calls executor_factory.create_executor()
              └─> Creates ExecutorCore(...) ❌ Missing state_providers
```

### 3.2 Gap Locations

1. **`flujo/application/runner.py`** (Line 230-250)
   - `Flujo.__init__()` does not accept `state_providers` parameter
   - Cannot pass providers to factory chain

2. **`flujo/application/core/factories.py`** (Line 28-56)
   - `ExecutorFactory.__init__()` does not accept `state_providers`
   - `ExecutorFactory.create_executor()` does not pass `state_providers` to `ExecutorCore`

3. **`flujo/application/core/executor_core.py`** (Line 257)
   - ✅ Already supports `state_providers: Optional[Dict[str, StateProvider]] = None`
   - ✅ Already implements hydration/persistence logic (lines 517-560)

### 3.3 Why This Matters

The `ContextReference` pattern is essential for:
- **Performance**: Avoid serializing large datasets (knowledge graphs, conversation history)
- **Scalability**: Keep context objects small and serializable
- **Separation of Concerns**: External state lives in databases, not in-memory objects
- **Production Readiness**: Enable real-world workloads with persistent state

---

## 4. Proposed Solution

### 4.1 High-Level Design

Plumb `state_providers` through the factory chain:

1. Add `state_providers` parameter to `Flujo.__init__()`
2. Store providers in `ExecutorFactory` during initialization
3. Pass providers from `ExecutorFactory` to `ExecutorCore` during executor creation
4. Maintain backward compatibility (optional parameter, defaults to `None`)

### 4.2 Detailed Code Changes

#### 4.2.1 Update `flujo/application/core/factories.py`

**File:** `flujo/application/core/factories.py`

**Changes:**

1. **Add imports** (after line 4):
```python
from typing import Any, Optional, Dict  # Add Dict to imports
```

2. **Add StateProvider import** (after line 25):
```python
from flujo.domain.interfaces import StateProvider
```

3. **Update `ExecutorFactory.__init__`** (modify lines 31-37):
```python
def __init__(
    self,
    *,
    telemetry: ITelemetry | None = None,
    cache_backend: ICacheBackend | None = None,
    optimization_config: OptimizationConfig | None = None,
    state_providers: Optional[Dict[str, StateProvider]] = None,  # ADD THIS
) -> None:
    self._telemetry = telemetry
    self._cache_backend = cache_backend
    self._optimization_config = optimization_config
    self._state_providers = state_providers or {}  # STORE IT
```

4. **Update `ExecutorFactory.create_executor`** (modify line 44-56):
```python
def create_executor(self) -> ExecutorCore[Any]:
    """Return a configured ExecutorCore."""
    return ExecutorCore(
        serializer=OrjsonSerializer(),
        hasher=Blake3Hasher(),
        cache_backend=self._cache_backend or InMemoryLRUBackend(),
        usage_meter=ThreadSafeMeter(),
        agent_runner=DefaultAgentRunner(),
        processor_pipeline=DefaultProcessorPipeline(),
        validator_runner=DefaultValidatorRunner(),
        plugin_runner=DefaultPluginRunner(),
        telemetry=self._telemetry or DefaultTelemetry(),
        optimization_config=self._optimization_config,
        estimator_factory=build_default_estimator_factory(),
        state_providers=self._state_providers,  # ADD THIS
    )
```

#### 4.2.2 Update `flujo/application/runner.py`

**File:** `flujo/application/runner.py`

**Changes:**

1. **Add StateProvider import** (in imports section, around line 1-30):
```python
from flujo.domain.interfaces import StateProvider
```

2. **Update `Flujo.__init__` signature** (modify line 230-250):
```python
def __init__(
    self,
    pipeline: Pipeline[RunnerInT, RunnerOutT] | Step[RunnerInT, RunnerOutT] | None = None,
    *,
    context_model: Optional[Type[ContextT]] = None,
    initial_context_data: Optional[Dict[str, Any]] = None,
    resources: Optional[AppResources] = None,
    usage_limits: Optional[UsageLimits] = None,
    hooks: Optional[list[HookCallable]] = None,
    backend: Optional[ExecutionBackend] = None,
    state_backend: Optional[StateBackend] = None,
    delete_on_completion: bool = False,
    executor_factory: Optional[ExecutorFactory] = None,
    backend_factory: Optional[BackendFactory] = None,
    pipeline_version: str = "latest",
    local_tracer: Union[str, Any, None] = None,
    registry: Optional[PipelineRegistry] = None,
    pipeline_name: Optional[str] = None,
    enable_tracing: bool = True,
    pipeline_id: Optional[str] = None,
    state_providers: Optional[Dict[str, StateProvider]] = None,  # ADD THIS
) -> None:
```

3. **Update factory creation** (modify line 342-343):
```python
# Store state_providers for potential later use
self._state_providers = state_providers or {}

# Pass to factory creation
self._executor_factory = executor_factory or ExecutorFactory(
    state_providers=self._state_providers  # ADD THIS
)
self._backend_factory = backend_factory or BackendFactory(self._executor_factory)
```

### 4.3 Backward Compatibility

✅ **Fully backward compatible:**
- `state_providers` is an optional parameter (defaults to `None`)
- Existing code continues to work without changes
- No breaking changes to existing APIs
- Default behavior (empty dict) matches current behavior

### 4.4 Type Safety

All changes maintain strict type safety:
- Uses `Optional[Dict[str, StateProvider]]` type hints
- Leverages existing `StateProvider` Protocol
- No `Any` types introduced
- Full mypy compliance

---

## 5. Implementation Details

### 5.1 Execution Flow (After Changes)

```
User Code:
  Flujo(pipeline, state_providers={"kg": provider})

Flujo.__init__():
  ├─> Stores state_providers in self._state_providers
  ├─> Creates ExecutorFactory(state_providers=self._state_providers)
  └─> Creates BackendFactory(executor_factory)

BackendFactory.create_execution_backend():
  └─> Calls executor_factory.create_executor()

ExecutorFactory.create_executor():
  └─> Creates ExecutorCore(state_providers=self._state_providers) ✅

ExecutorCore.__init__():
  └─> Stores in self._state_providers (line 319)

During Step Execution:
  └─> ExecutorCore._hydrate_context() (line 517)
      └─> Loads from state_providers using ContextReference.provider_id
  └─> ExecutorCore._persist_context() (line 542)
      └─> Saves to state_providers using ContextReference.provider_id
```

### 5.2 Integration Points

The `state_providers` are used by `ExecutorCore` in two methods:

1. **`_hydrate_context()`** (line 517-541):
   - Iterates over context fields
   - Finds `ContextReference` instances
   - Calls `provider.load(key)` to hydrate data
   - Sets hydrated data via `field_value.set(data)`

2. **`_persist_context()`** (line 542-560):
   - Iterates over context fields
   - Finds `ContextReference` instances
   - Gets current data via `field_value.get()`
   - Calls `provider.save(key, data)` to persist

### 5.3 Error Handling

The existing error handling in `ExecutorCore` is preserved:
- Missing providers log warnings but don't crash (line 533-540)
- Hydration failures are logged to telemetry
- Persistence failures are handled gracefully

---

## 6. Usage Examples

### 6.1 Basic Usage

```python
from flujo import Flujo, Step, step
from flujo.domain.models import PipelineContext, ContextReference
from flujo.domain.interfaces import StateProvider
from typing import List

# Define a StateProvider
class KnowledgeGraphProvider(StateProvider):
    def __init__(self):
        self._db = {}
    
    async def load(self, key: str) -> List[dict]:
        return self._db.get(key, [])
    
    async def save(self, key: str, data: List[dict]) -> None:
        self._db[key] = data

# Define context with ContextReference
class ResearchContext(PipelineContext):
    knowledge: ContextReference[List[dict]] = ContextReference(
        provider_id="kg_provider",
        key="research_graph"
    )

# Create provider instance
kg_provider = KnowledgeGraphProvider()

# Initialize runner with state_providers
runner = Flujo(
    pipeline=my_pipeline,
    context_model=ResearchContext,
    state_providers={"kg_provider": kg_provider}  # ✅ Now works!
)

# Run pipeline - ContextReference will be automatically hydrated/persisted
result = await runner.run_async("research query")
```

### 6.2 Advanced: Multiple Providers

```python
class DatabaseProvider(StateProvider):
    async def load(self, key: str): ...
    async def save(self, key: str, data): ...

class CacheProvider(StateProvider):
    async def load(self, key: str): ...
    async def save(self, key: str, data): ...

runner = Flujo(
    pipeline=my_pipeline,
    context_model=MyContext,
    state_providers={
        "db_provider": DatabaseProvider(),
        "cache_provider": CacheProvider(),
    }
)
```

### 6.3 With Custom ExecutorFactory

```python
# Users can still provide custom factories
custom_factory = ExecutorFactory(
    telemetry=my_telemetry,
    state_providers={"kg": my_provider}
)

runner = Flujo(
    pipeline=my_pipeline,
    executor_factory=custom_factory  # state_providers already configured
)
```

---

## 7. Testing Strategy

### 7.1 Unit Tests

**File:** `tests/unit/test_state_providers_integration.py` (new file)

```python
import pytest
from flujo import Flujo, Step
from flujo.domain.models import PipelineContext, ContextReference
from flujo.domain.interfaces import StateProvider

class MockProvider(StateProvider):
    def __init__(self):
        self._data = {}
    
    async def load(self, key: str):
        return self._data.get(key, [])
    
    async def save(self, key: str, data):
        self._data[key] = data

class TestContext(PipelineContext):
    ref: ContextReference[list] = ContextReference(
        provider_id="test_provider",
        key="test_key"
    )

@pytest.mark.asyncio
async def test_state_providers_parameter_accepted():
    """Verify Flujo accepts state_providers parameter."""
    provider = MockProvider()
    step = Step.from_callable(lambda x: x, name="test")
    
    runner = Flujo(
        pipeline=step,
        context_model=TestContext,
        state_providers={"test_provider": provider}
    )
    
    assert runner is not None

@pytest.mark.asyncio
async def test_state_providers_hydration():
    """Verify ContextReference is hydrated from state_providers."""
    provider = MockProvider()
    provider._data["test_key"] = [1, 2, 3]
    
    async def check_context(data: str, *, context: TestContext) -> str:
        assert context.ref.get() == [1, 2, 3]
        return "ok"
    
    step = Step.from_callable(check_context, name="check")
    runner = Flujo(
        pipeline=step,
        context_model=TestContext,
        state_providers={"test_provider": provider}
    )
    
    result = await runner.run_async("test")
    assert result.success

@pytest.mark.asyncio
async def test_state_providers_persistence():
    """Verify ContextReference changes are persisted."""
    provider = MockProvider()
    
    async def modify_context(data: str, *, context: TestContext) -> str:
        current = context.ref.get()
        current.append(4)
        context.ref.set(current)
        return "modified"
    
    step = Step.from_callable(modify_context, name="modify")
    runner = Flujo(
        pipeline=step,
        context_model=TestContext,
        state_providers={"test_provider": provider}
    )
    
    # Initialize data
    provider._data["test_key"] = [1, 2, 3]
    
    await runner.run_async("test")
    
    # Verify persistence
    assert provider._data["test_key"] == [1, 2, 3, 4]

@pytest.mark.asyncio
async def test_state_providers_backward_compatibility():
    """Verify existing code without state_providers still works."""
    step = Step.from_callable(lambda x: x, name="test")
    
    # Should work without state_providers
    runner = Flujo(
        pipeline=step,
        context_model=PipelineContext
    )
    
    result = await runner.run_async("test")
    assert result.success
```

### 7.2 Integration Tests

Update existing tests in `tests/test_managed_state.py` to use the public API:

```python
# Before (using private attributes):
executor = ExecutorCore(state_providers=providers)

# After (using public API):
runner = Flujo(
    pipeline=step,
    state_providers=providers
)
```

### 7.3 Regression Tests

- Verify all existing tests pass without modification
- Ensure no performance degradation
- Confirm type checking passes (`make typecheck`)

---

## 8. Migration Path

### 8.1 For Existing Users

**No migration required** - this is a purely additive change. Existing code continues to work.

### 8.2 For Users Using Workarounds

Users currently using the private attribute workaround should migrate:

**Before:**
```python
runner = Flujo(pipeline, ...)
runner.backend._executor._state_providers["kg"] = provider  # ❌ Brittle
```

**After:**
```python
runner = Flujo(
    pipeline=pipeline,
    state_providers={"kg": provider}  # ✅ Public API
)
```

### 8.3 Documentation Updates

Update the following documentation:

1. **`docs/user_guide/context-and-resources.md`**
   - Add section on `state_providers` parameter
   - Show `ContextReference` usage example

2. **`docs/advanced/extending.md`**
   - Update StateProvider examples to use public API
   - Remove workaround examples

3. **API Reference**
   - Document `Flujo.__init__()` `state_providers` parameter
   - Document `ExecutorFactory.__init__()` `state_providers` parameter

---

## 9. Performance Considerations

### 9.1 Impact Analysis

- **Memory**: No additional memory overhead (providers are stored once)
- **CPU**: Negligible (dictionary lookup is O(1))
- **Startup Time**: No impact (providers are passed at initialization)
- **Runtime**: No impact (hydration/persistence logic already exists)

### 9.2 Optimization Opportunities

Future optimizations (out of scope):
- Lazy provider loading
- Provider caching
- Batch hydration/persistence

---

## 10. Security Considerations

### 10.1 Access Control

- Providers are stored in executor instance (not globally)
- No cross-runner state leakage
- Each runner instance has isolated providers

### 10.2 Validation

- Type checking ensures `StateProvider` protocol compliance
- Runtime validation in `ExecutorCore` handles missing providers gracefully

---

## 11. Open Questions & Future Enhancements

### 11.1 Open Questions

None - the solution is straightforward and complete.

### 11.2 Future Enhancements (Out of Scope)

1. **Provider Lifecycle Management**
   - `async def close()` method for cleanup
   - Automatic resource management

2. **Provider Registry**
   - Global provider registry
   - Provider discovery mechanism

3. **Provider Middleware**
   - Caching layer
   - Transformation pipeline
   - Validation hooks

---

## 12. Acceptance Criteria

✅ **Implementation Complete When:**

1. `Flujo.__init__()` accepts `state_providers` parameter
2. `ExecutorFactory` stores and passes `state_providers` to `ExecutorCore`
3. All existing tests pass without modification
4. New tests verify hydration and persistence work correctly
5. Documentation updated with usage examples
6. Type checking passes (`make typecheck`)
7. Linting passes (`make lint`)

---

## 13. Implementation Checklist

- [x] Add `Dict` import to `factories.py`
- [x] Add `StateProvider` import to `factories.py`
- [x] Update `ExecutorFactory.__init__()` signature
- [x] Store `state_providers` in `ExecutorFactory`
- [x] Pass `state_providers` in `ExecutorFactory.create_executor()`
- [x] Add `StateProvider` import to `runner.py`
- [x] Update `Flujo.__init__()` signature
- [x] Store `state_providers` in `Flujo`
- [x] Pass `state_providers` to `ExecutorFactory` creation
- [x] Write unit tests (14 tests in `tests/unit/test_state_providers_integration.py`)
- [x] Run full test suite (`make test-fast` - 448 tests passed)
- [x] Verify type checking (`mypy --strict` - no issues)
- [x] Update documentation:
  - `docs/user_guide/context-and-resources.md` - comprehensive guide
  - `docs/advanced/extending.md` - StateProvider section
  - `examples/state_providers_demo.py` - working example with 4 demos

---

## 14. References

- **ContextReference Implementation**: `flujo/domain/models.py` (line 22-41)
- **ExecutorCore Hydration**: `flujo/application/core/executor_core.py` (line 517-560)
- **StateProvider Protocol**: `flujo/domain/interfaces.py` (line 81-90)
- **Existing Tests**: `tests/test_managed_state.py`

---

## 15. Appendix: Complete Code Diff Summary

### Files Modified

1. **`flujo/application/core/factories.py`**
   - Add imports: `Dict`, `StateProvider`
   - Update `ExecutorFactory.__init__()`: add `state_providers` parameter
   - Update `ExecutorFactory.create_executor()`: pass `state_providers` to `ExecutorCore`

2. **`flujo/application/runner.py`**
   - Add import: `StateProvider`
   - Update `Flujo.__init__()`: add `state_providers` parameter
   - Update factory creation: pass `state_providers` to `ExecutorFactory`

### Lines Changed

- `factories.py`: ~5 lines modified, ~2 lines added (imports)
- `runner.py`: ~3 lines modified, ~2 lines added (imports + storage)

**Total Impact**: Minimal, focused changes with zero breaking changes.

---

**End of Document**

