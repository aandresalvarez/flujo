Of course. With all logic now migrated into the `ExecutorCore`, we can proceed with the final hardening and cleanup phases. This FSD focuses on making the architectural contracts explicit and provably safe, just as you outlined in your "Philosophical Shift" document.

---

### **FSD 10 of 11: Hardening Contracts & Finalizing `ExecutorCore`**

**1. Rationale & First Principles**

*   **Goal:** To harden the internal contracts of the `ExecutorCore` and its related components, making them statically verifiable and more robust. This involves improving type safety and making failure modes more explicit.
*   **Why:** This FSD directly implements the core of your "Philosophical Shift" document.
    *   **Reliability & Maintainability:** By moving from runtime assumptions (e.g., "context probably has a scratchpad") to explicit, type-checked contracts, we eliminate an entire class of potential `AttributeError` and `TypeError` bugs. This makes the system more predictable and easier for future developers to reason about.
    *   **Exhaustive Accounting:** Making plugin failures explicit by re-raising exceptions ensures that a step's failure is never silently ignored, upholding the principle that all outcomes must be accounted for.
    *   **Open-Closed Principle:** Refactoring the `_is_complex_step` check to be an object-oriented property (`step.is_complex`) means we can add new complex step types in the future without modifying the `ExecutorCore`'s dispatch logic.

**2. Scope of Work**

1.  **Harden `ParallelStep` Type Contracts:**
    *   In a suitable location (e.g., a new `flujo/application/core/types.py` or at the top of `ultra_executor.py`), define a `ContextWithScratchpad` protocol and a bounded `TypeVar` (`TContext_w_Scratch`).
    *   Update the signature of `ExecutorCore._handle_parallel_step` to use this new bounded `TypeVar` for its `context` parameter.
    *   Update the `merge_strategy` callable signature in the `ParallelStep` class definition (`flujo/domain/dsl/parallel.py`) to also use this new type.

2.  **Make Plugin Failures Explicit:**
    *   Modify `DefaultPluginRunner.run_plugins` in `flujo/application/core/ultra_executor.py`.
    *   It will be changed to **re-raise** any exception that occurs during a plugin's execution, instead of catching it and continuing.

3.  **Simplify Complex Step Detection:**
    *   Add a new property `is_complex: bool = False` to the base `Step` class in `flujo/domain/dsl/step.py`.
    *   Override this property in each of the complex step subclasses (`LoopStep`, `ParallelStep`, `ConditionalStep`, `DynamicParallelRouterStep`, `CacheStep`, `HumanInTheLoopStep`) to return `True`.
    *   Refactor `ExecutorCore._is_complex_step` to a simple one-line implementation: `return getattr(step, 'is_complex', False)`.

**3. Implementation Details**

#### **Task 1: Harden `ParallelStep` Contracts**

```python
# Create a new file: flujo/application/core/types.py
from typing import Protocol, TypeVar, Dict, Any

class ContextWithScratchpad(Protocol):
    """A contract ensuring a context object has a scratchpad attribute."""
    scratchpad: Dict[str, Any]

TContext_w_Scratch = TypeVar("TContext_w_Scratch", bound=ContextWithScratchpad)
```

```python
# In flujo/application/core/ultra_executor.py
from .types import TContext_w_Scratch # Import the new type

class ExecutorCore(...):
    async def _handle_parallel_step(
        self,
        parallel_step: ParallelStep[TContext_w_Scratch], # ✅ Use the bounded TypeVar
        data: Any,
        context: Optional[TContext_w_Scratch], # ✅ Use the bounded TypeVar
        # ... other params
    ) -> StepResult:
        # ... implementation ...
```

```python
# In flujo/domain/dsl/parallel.py
from ...application.core.types import TContext_w_Scratch # Or wherever you define it

class ParallelStep(Step[Any, Any], Generic[TContext]):
    # ...
    # ✅ UPDATE: The signature is now explicit and type-safe.
    merge_strategy: Union[MergeStrategy, Callable[[TContext_w_Scratch, Dict[str, StepResult]], None]] = Field(...)
    # ...
```

#### **Task 2: Make Plugin Failures Explicit**

```python
# In flujo/application/core/ultra_executor.py

class DefaultPluginRunner:
    async def run_plugins(self, plugins: List[tuple[Any, int]], data: Any, *, context: Any) -> Any:
        processed_data = data
        for plugin, priority in sorted(plugins, key=lambda x: x[1], reverse=True):
            try:
                # ... existing logic to call the plugin ...
                result = await func(...) # The plugin call

                if isinstance(result, PluginOutcome):
                    if not result.success:
                        # ✅ NEW: Raise an exception to fail the step.
                        raise ValueError(f"Plugin validation failed: {result.feedback}")
                    # ...
            except Exception as e:
                # ✅ NEW: Re-raise the exception to ensure the step fails.
                telemetry.logfire.error(f"Plugin {type(plugin).__name__} failed: {e}")
                raise e
        return processed_data
```

#### **Task 3: Simplify Complex Step Detection**

```python
# In flujo/domain/dsl/step.py

class Step(BaseModel, Generic[StepInT, StepOutT]):
    # ... existing fields ...
    
    @property
    def is_complex(self) -> bool:
        # ✅ Base steps are not complex by default.
        return False

# In flujo/domain/dsl/loop.py
class LoopStep(Step[Any, Any], Generic[TContext]):
    # ... existing fields ...
    
    @property
    def is_complex(self) -> bool:
        # ✅ Override to mark as complex.
        return True

# ... Do the same for ParallelStep, ConditionalStep, CacheStep, etc. ...
```

```python
# In flujo/application/core/ultra_executor.py

class ExecutorCore(Generic[TContext]):
    # ...
    def _is_complex_step(self, step: Any) -> bool:
        """Check if step needs complex handling using an object-oriented approach."""
        # ✅ NEW: Simple, object-oriented, and extensible check.
        return getattr(step, 'is_complex', False)
```

**4. Testing Strategy**

*   **Unit Tests:**
    *   **Static Analysis Test (for Task 1):** Create a new test file, `tests/static_analysis/test_contracts.py`. In this file, define a simple `BaseModel` context *without* a `scratchpad` attribute. Then, define a `ParallelStep` that uses a `merge_strategy` callable. In a test function, attempt to pass an instance of your simple context to a mock `_handle_parallel_step` function that is typed with `TContext_w_Scratch`. The test itself will do nothing but `pass`. The *real* test is running `mypy` over the codebase. The test passes if `mypy` reports a type error for that line, proving the contract is enforced.
    *   **Plugin Failure Test (for Task 2):** In `tests/application/core/test_executor_core.py`, create a test for `_execute_simple_step`. Configure a step with a mock plugin that always raises an exception. Assert that the `StepResult` has `success=False` and that its `feedback` contains the exception message from the plugin.
    *   **Complex Step Detection Test (for Task 3):** Write a simple unit test for `ExecutorCore._is_complex_step`. Create instances of a basic `Step`, a `LoopStep`, and a `ParallelStep`. Assert that the method returns `False` for the basic step and `True` for the complex ones.

*   **Regression Tests (Existing):**
    *   Run the entire existing test suite.
    *   The change to plugin failure handling is the most likely to cause regressions if any existing tests relied on the old, forgiving behavior. Any failures here must be investigated. It's possible some tests may need to be updated to correctly `pytest.raises` an exception where they previously expected success.
    *   A 100% pass rate is required.

**5. Acceptance Criteria**

*   [ ] The `ParallelStep`'s `merge_strategy` contract is now statically verifiable via a `Protocol`.
*   [ ] The `DefaultPluginRunner` correctly propagates failures by raising exceptions.
*   [ ] The `is_complex` property is implemented on `Step` and its subclasses and is used by `ExecutorCore` for dispatching.
*   [ ] All new unit tests, including the static analysis test, pass.
*   [ ] **100% of the existing test suite passes**, confirming that the hardening has not introduced regressions.