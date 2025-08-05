 
## FSD-13: Fortifying the Execution Core

**Status:** Proposed  
**Author:** Flujo Architecture Team  
**Date:** 2023-10-27

---

### 1. Context & Problem Statement

The FSD-12 migration successfully consolidated the majority of step execution logic from the legacy `step_logic.py` module into the modern `ExecutorCore`. This effort validated Flujo's core recursive execution architecture and resolved approximately 35% of existing test failures, proving the foundational soundness of the new design.

However, the remaining test failures highlight critical second-order integration issues that prevent the Execution Core from being fully production-ready. These are not architectural flaws but rather friction points at the seams of the modular components. The core problems are:

1.  **Inconsistent Context Lifecycle:** The `PipelineContext` is not reliably initialized, particularly in nested or complex pipeline structures, leading to errors like the missing `initial_prompt`.
2.  **Brittle Serialization:** The system fails to serialize or deserialize non-standard objects (e.g., `UsageResponse`, test mocks), compromising the durability and caching layers.
3.  **Non-Atomic Usage Governance:** The `UsageGovernor` is not updated correctly or atomically, especially during parallel execution, failing to enforce cost and token limits reliably.
4.  **Fragile Recursive State Passing:** Operational parameters (like `context_setter`) are not consistently propagated through recursive calls to `ExecutorCore.execute`, causing failures in nested control-flow steps.

This design document proposes a set of targeted solutions to resolve these issues, completing the FSD-12 migration and hardening the Execution Core for production use.

### 2. Goals

*   **Achieve 100% Test Pass Rate:** Resolve all remaining test failures related to the FSD-12 migration.
*   **Ensure Production-Grade Resilience:** Guarantee that core resilience features like usage limits and state persistence are atomic, reliable, and correct under all conditions, including parallel execution.
*   **Formalize Internal Data Contracts:** Eliminate ambiguity in the internal execution flow by introducing a formal data contract (`ExecutionFrame`) for recursive calls, ensuring state is passed consistently.
*   **Implement an Extensible Serialization Protocol:** Decouple the execution core from specific data types by creating a pluggable serialization registry, upholding the framework's extensibility principle.
*   **Complete the Migration:** Fully deprecate and remove all dependencies on `flujo/application/core/step_logic.py`.

### 3. Non-Goals

*   **No New User-Facing DSL Features:** This design focuses exclusively on hardening the internal execution engine. The public DSL (`flujo.domain.dsl`) will not be modified.
*   **No Major Architectural Changes:** The proposed solutions build upon and reinforce the existing dual architecture, recursive execution, and dependency injection principles, rather than replacing them.

### 4. Proposed Design

The solution is a multi-faceted approach that addresses each identified problem by strengthening the contracts and interactions between the core execution components.

#### 4.1. The `ExecutionFrame`: A Formal Data Contract for Recursion

To solve the inconsistent passing of operational parameters, we will introduce a dedicated `ExecutionFrame` dataclass. This object will encapsulate the entire state of a single `execute` call, replacing the current, error-prone reliance on a long list of keyword arguments.

```python
# To be defined within flujo/application/core/types.py or similar

from dataclasses import dataclass

@dataclass
class ExecutionFrame(Generic[ContextT]):
    """Encapsulates all state for a single step execution call."""
    step: Step[Any, Any]
    data: Any
    context: Optional[ContextT]
    resources: Optional[AppResources]
    limits: Optional[UsageLimits]
    stream: bool
    on_chunk: Optional[Callable[[Any], Awaitable[None]]]
    breach_event: Optional[asyncio.Event]
    context_setter: Callable[[PipelineResult[Any], Optional[Any]], None]
    # ... any other operational parameters
```

The `ExecutorCore.execute` method signature will be refactored to accept this single `ExecutionFrame` object. All internal recursive calls (e.g., within `_handle_parallel_step`) will construct and pass a new `ExecutionFrame`, ensuring that all necessary state is explicitly and correctly propagated.

**Benefit:** This enforces a strict, type-safe contract for internal execution calls, eliminating an entire class of bugs related to missing parameters and making the recursive logic easier to reason about and debug.

#### 4.2. Robust Context Lifecycle Management

To fix initialization failures, we will establish a clear and inviolable chain of responsibility for the `PipelineContext`.

1.  **Creation (Responsibility: `Flujo` Runner):** The top-level `Flujo` runner (`flujo/application/runner.py`) will be the *sole* component responsible for instantiating the `PipelineContext`. It will validate the `initial_context_data` and correctly populate the `initial_prompt` from the user's input.
2.  **Stewardship (Responsibility: `ExecutionManager`):** The `ExecutionManager` receives the fully-formed context. It is responsible for passing it down through `ExecutionFrame`s and orchestrating updates via `_inject_context` after a step completes.
3.  **Finalization (Responsibility: `ExecutionManager`):** Upon completion or failure of the entire pipeline, the `ExecutionManager` will set the final, mutated context onto the `PipelineResult.final_pipeline_context` field.

**Benefit:** This establishes a predictable, top-down data flow for the context, ensuring it is always in a valid state at every stage of execution.

#### 4.3. An Extensible Serialization Protocol

To solve the brittle serialization of custom objects, we will enhance `flujo/utils/serialization.py` to act as a central, pluggable registry.

1.  **Registration API:** We will expose two public functions:
    ```python
    def register_custom_serializer(type_class: Type, serializer_func: Callable[[Any], Any]) -> None:
        ...

    def register_custom_deserializer(type_class: Type, deserializer_func: Callable[[Any], Any]) -> None:
        ...
    ```
2.  **Integration:** The `ISerializer` implementation (e.g., `OrjsonSerializer`) and the `StateManager` will be refactored to consult this registry. When encountering an unknown type, they will look up a registered custom serializer/deserializer before falling back to default behavior.
3.  **Test Integration:** Test fixtures will use this registry to register serializers for mock objects like `UsageResponse`, ensuring they can be correctly handled by the persistence and caching layers.

**Benefit:** This decouples the core engine from specific data types, fully embracing the "Pluggable Architecture" pillar. It makes the system robust and allows users to integrate Flujo with any custom data model.

#### 4.4. Atomic and Reliable Usage Governance

To ensure `UsageGovernor` functions correctly, we will enforce a strict, atomic sequence of operations within the `ExecutionManager`'s main execution loop.

1.  **Execute Step:** The `StepCoordinator` executes the step and returns a `StepResult` containing cost and token information.
2.  **Check Limits:** The `ExecutionManager` will *immediately* pass the new usage metrics to the `UsageGovernor`. The governor will calculate the *prospective* total (`current_total + step_delta`) and check it against the limits.
3.  **Update State or Raise Exception:**
    *   If no limit is breached, the `ExecutionManager` will update the `PipelineResult`'s running totals and append the `StepResult` to its history.
    *   If a limit is breached, the `UsageGovernor` will raise a `UsageLimitExceededError`. Crucially, the `ExecutionManager` will first add the breaching `StepResult` to the `PipelineResult`'s history *before* raising the exception, ensuring the final result accurately reflects all work performed up to the breach.

For `ParallelStep`, the `_ParallelUsageGovernor` helper class will be refined. It will be instantiated once per `ParallelStep` execution and passed into every concurrent branch. Its internal `add_usage` method will use an `asyncio.Lock` to guarantee that updates from all branches are atomic, preventing race conditions.

**Benefit:** This guarantees that usage limits are checked immediately and atomically after every unit of work, providing the production-grade financial and resource safety that users expect.

### 5. Detailed Implementation Plan

1.  **Phase 1: Foundational Refactoring**
    *   [ ] Define the `ExecutionFrame` dataclass.
    *   [ ] Refactor the `ExecutorCore.execute` method and all its internal callers (`_handle_..._step`) to use the `ExecutionFrame`.
    *   [ ] Run the test suite to ensure no new regressions are introduced by this refactoring.

2.  **Phase 2: Component-Level Fixes**
    *   [ ] **Context:** Modify the `Flujo` runner to be the sole creator of the `PipelineContext`, ensuring `initial_prompt` is always set.
    *   [ ] **Serialization:** Implement the `register_custom_serializer` and `register_custom_deserializer` functions and integrate them into the `StateManager` and `ICacheBackend`. Update tests to use this new registry for mock objects.
    *   [ ] **Usage Governor:** Refactor the `ExecutionManager` loop to follow the strict atomic update sequence described in 4.4. Refine and thoroughly test the `_ParallelUsageGovernor` for thread safety.

3.  **Phase 3: Final Cleanup and Deprecation**
    *   [ ] Remove all remaining import statements and function calls to `flujo/application/core/step_logic.py`.
    *   [ ] Delete the `flujo/application/core/step_logic.py` file.
    *   [ ] Run the full test suite and fix any remaining failures. The goal is a 100% pass rate.

### 6. Testing Strategy

*   **Unit Tests:** New unit tests will be created for the `ExecutionFrame`, the serialization registry, and the atomic logic of the `_ParallelUsageGovernor`.
*   **Integration Tests:** Specific integration tests will be written to replicate the exact failure conditions observed:
    *   A test with a nested pipeline that fails if `context_setter` is not propagated.
    *   A test that runs a `ParallelStep` designed to breach a usage limit precisely mid-execution.
    *   A test that persists and reloads a pipeline state containing a custom object registered with the new serialization protocol.
*   **Regression Tests:** The entire existing test suite will be run after each phase of the implementation plan to guarantee no existing functionality is broken.

### 7. Backward Compatibility

This design exclusively targets the **internal** Execution Core. All proposed changes are confined to `flujo/application/core/` and `flujo/utils/`.

*   **Public API:** The user-facing API (`Flujo` runner, DSL) remains **100% backward compatible**. No user code will need to be changed.
*   **Internal API:** The signature of `ExecutorCore.execute` will change, but this is a private, internal API.

### 8. Risks and Mitigation

*   **Risk:** The refactoring of `ExecutorCore.execute` to use `ExecutionFrame` is complex and could introduce subtle bugs.
    *   **Mitigation:** The phased implementation plan and a heavy reliance on the comprehensive regression test suite will catch any deviations from expected behavior early.

*   **Risk:** The serialization registry could add performance overhead if not implemented carefully.
    *   **Mitigation:** The registry lookup will be implemented with efficient dictionary lookups and caching. Performance benchmarks will be run before and after implementation.

### 9. Conclusion

This design provides a clear and robust path to completing the FSD-12 migration. By formalizing internal contracts, strengthening state management, and making core systems more extensible, these changes will not only fix the remaining bugs but will also make the entire Execution Core more resilient, predictable, and maintainable. This work is the final step in solidifying Flujo's dual architecture and establishing it as a truly production-ready framework.