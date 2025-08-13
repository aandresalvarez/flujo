  

## **Functional Specification Document: Policy Registry for Step Dispatch (FSD-010)**

**Author:** Alvaro
**Date:** 2023-10-27
**Status:** Proposed
**JIRA/Ticket:** FLUJO-125 (Example Ticket ID)
**Depends On:** FSD-008 (Typed Step Outcomes)

### **1. Overview**

This document specifies the refactoring of the step execution dispatch mechanism within `ExecutorCore`. The current implementation uses a long, sequential `if/elif isinstance(...)` chain to route a `Step` object to the appropriate execution policy (e.g., `DefaultLoopStepExecutor`, `DefaultParallelStepExecutor`).

This pattern is brittle, inefficient, and violates the Open/Closed Principle. Adding a new `Step` type requires modifying the central `ExecutorCore`, increasing the risk of regressions.

This FSD proposes replacing the `isinstance` chain with a **Policy Registry**. This registry will maintain a mapping from `Step` subclasses to their corresponding execution policy instances. The `ExecutorCore`'s dispatch logic will be simplified to a single dictionary lookup, making the system more modular, extensible, and easier to maintain.

### **2. Rationale & Goals**

#### **2.1. Problems with the Current Approach**

*   **Brittleness:** The `ExecutorCore.execute` method must be modified every time a new complex step type is added. This centralizes logic that should be decentralized.
*   **Inefficiency:** The sequential `isinstance` checks introduce a small but measurable overhead for each step execution, especially for steps checked later in the chain.
*   **Poor Extensibility:** Adding custom, user-defined `Step` types with unique execution logic is not possible without modifying the core framework code.
*   **Code Clutter:** The dispatch logic in `ExecutorCore` is long and hard to read, obscuring its primary responsibility of orchestrating the execution frame.

#### **2.2. Goals of this Refactor**

*   **Decouple Dispatcher from Policies:** The `ExecutorCore` should not need to know about concrete `Step` types. Its only job should be to look up the correct policy and invoke it.
*   **Improve Modularity and Extensibility:** Adding a new `Step` type should only require creating a new policy class and registering it, with no changes to `ExecutorCore`.
*   **Simplify Core Logic:** Radically simplify the `ExecutorCore.execute` method, making it shorter, more readable, and focused on its core responsibilities.
*   **Unify Naming:** As part of this cleanup, ensure all executor-related classes are consistently named (e.g., `ExecutionCore` is not a goal of this FSD, but we should standardize on `ExecutorCore` internally).

### **3. Functional Requirements & Design**

#### **Task 3.1: Create the `PolicyRegistry` Class**

A new class will be created to manage the mapping of step types to policy instances.

*   **Location:** `flujo/application/core/step_policies.py`
*   **Implementation Details:**
    *   Create a `PolicyRegistry` class.
    *   It will contain a private dictionary: `_registry: Dict[Type[Step], Any] = {}`.
    *   Implement a `register(self, step_type: Type[Step], policy: Any)` method. This method will add an entry to the `_registry`. It should raise a `TypeError` if `step_type` is not a subclass of `Step`.
    *   Implement a `get(self, step_type: Type[Step]) -> Optional[Any]` method. This method will perform a lookup in the `_registry`.
*   **Acceptance Criteria & Testing (`make test-fast`)**
    *   **Unit Tests:** Create `tests/application/core/test_step_policies.py` (or add to it).
        *   Test that `register` successfully adds a policy for a valid `Step` subclass.
        *   Test that `register` raises a `TypeError` if a non-`Step` class is provided.
        *   Test that `get` returns the correct policy for a registered type.
        *   Test that `get` returns `None` for an unregistered type.

#### **Task 3.2: Instantiate and Populate the Registry**

The `ExecutorCore` will now own an instance of the `PolicyRegistry` and populate it with the default policies.

*   **Location:** `flujo/application/core/executor_core.py`
*   **Implementation Details:**
    *   In the `ExecutorCore.__init__` method, create an instance of `PolicyRegistry`.
    *   Register all the default policies. This involves importing the `Step` types (`LoopStep`, `ParallelStep`, etc.) and the policy classes (`DefaultLoopStepExecutor`, etc.) and calling `self.policy_registry.register(...)` for each one.
    *   The `DefaultAgentStepExecutor` should be registered as the policy for the base `Step` class, making it the default for any step that doesn't have a more specific policy.
*   **Acceptance Criteria & Testing (`make test-fast`)**
    *   **Unit Tests:** In `tests/application/core/test_executor_core.py`:
        *   Test that after `ExecutorCore` is initialized, its `policy_registry` attribute contains mappings for all standard complex step types (`LoopStep`, `ParallelStep`, `ConditionalStep`, `CacheStep`, `HumanInTheLoopStep`, etc.).
        *   Assert that the policy registered for the base `Step` type is `DefaultAgentStepExecutor`.

#### **Task 3.3: Refactor `ExecutorCore.execute` to Use the Registry**

This is the central part of the refactor, where the `isinstance` chain is removed.

*   **Location:** `flujo/application/core/executor_core.py`
*   **Implementation Details:**
    *   Delete the entire `if isinstance(step, ParallelStep): ... elif isinstance(step, LoopStep): ...` chain.
    *   The new dispatch logic will be:
        1.  `policy = self.policy_registry.get(type(frame.step))`
        2.  If `policy` is `None` (for a custom step type without a registered policy), fall back to the policy for the base `Step` class: `policy = self.policy_registry.get(Step)`.
        3.  If no policy is found (which should be impossible if Task 3.2 is done correctly), raise a `NotImplementedError`.
        4.  `return await policy.execute(self, frame)`.
*   **Acceptance Criteria & Testing (`make all`)**
    *   **Integration Tests:**
        *   Create `tests/application/core/test_dispatch_logic.py`.
        *   Create a simple pipeline with one of each `Step` type (`Step`, `LoopStep`, `ParallelStep`, etc.).
        *   Run the pipeline and mock the `execute` method of each corresponding policy (`DefaultLoopStepExecutor`, etc.).
        *   Assert that each mock was called exactly once. This proves the registry dispatch is working correctly for all step types.
    *   **Regression Tests:**
        *   Run the *entire existing test suite* (`make all`). All tests for loops, parallel execution, conditionals, etc., must pass without modification. This is the most critical acceptance criterion, as it proves the refactor did not change any existing behavior.

### **4. Rollout and Regression Plan**

1.  **Branching:** This work will be done on a dedicated feature branch (e.g., `feature/FSD-010-policy-registry`).
2.  **Implementation Order:**
    *   Complete Task 3.1 and its unit tests.
    *   Complete Task 3.2 and its unit tests. This ensures the registry is correctly populated before it's used.
    *   Complete Task 3.3. This is the main change.
3.  **Testing Strategy:**
    *   `make test-fast`: This command will be used to run the unit tests for Task 3.1 and 3.2 as they are developed.
    *   `make all`: This command will be run after Task 3.3 is complete. The full regression suite is the ultimate gate for this refactor. The new integration tests will provide targeted validation of the dispatch logic itself.
4.  **Code Review:** Required. The review should focus on the simplicity of the new `ExecutorCore.execute` method and the correctness of the registry population.
5.  **Merge:** Merge to the main branch after all unit, integration, and regression tests pass.

### **5. Risks and Mitigation**

*   **Risk:** A `Step` type is missed during registration, causing a `NotImplementedError` at runtime.
    *   **Mitigation:** The unit tests for Task 3.2 are designed to prevent this. They will explicitly check that every known `Step` subclass is present in the registry after initialization.
*   **Risk:** Performance degradation due to dictionary lookups.
    *   **Mitigation:** This risk is extremely low. A dictionary lookup is significantly faster (O(1) on average) than a sequential `isinstance` chain (O(N)). This change is expected to be a net performance *improvement*. No specific performance tests are required unless a noticeable slowdown is observed during regression testing.
*   **Risk:** Breaking backward compatibility for users who might have been monkey-patching the old `ExecutorCore.execute` method.
    *   **Mitigation:** This is an internal refactor. The public API of `Flujo.run()` remains unchanged. We will accept this as a low-risk breaking change for advanced users who were modifying internal framework machinery.