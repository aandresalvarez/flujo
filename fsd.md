 
### **Next Task: Formalize and Enforce Context Merge Strategies (FSD-012)**

This task directly addresses **Gap #4** from the original review: "Context merge semantics underspecified."

#### **Why This is the Next Logical Step:**

1.  **It's a Correctness and Safety Issue:** The current default behavior for merging contexts from parallel branches can lead to silent, non-deterministic bugs (race conditions) if two branches modify the same context field. This violates **Axiom #3 (Composability without foot-guns)**.
2.  **It Builds on Existing Work:** Your previous refactors have cleaned up the `ParallelStep` policy and the `ContextManager`, making it much easier to implement this change now.
3.  **It Improves the DSL:** This change will make the `ParallelStep` DSL more explicit and force users to think about how their concurrent operations interact, which is a hallmark of a well-designed framework.

---

### **Functional Specification Document: Explicit Context Merge Strategies (FSD-012)**

**Author:** Alvaro
**Date:** 2023-10-27
**Status:** Proposed
**JIRA/Ticket:** FLUJO-127 (Example Ticket ID)
**Depends On:** FSD-010 (Policy Registry)

#### **1. Overview**

This document specifies the design for enhancing the context merging logic within `ParallelStep` to prevent non-deterministic behavior and race conditions. Currently, when multiple parallel branches modify the same context fields, the final state of the context depends on which branch finishes last. This "last writer wins" behavior is an implicit and unsafe default.

This FSD proposes changing the default merge strategy to a safer option and introducing an explicit conflict-handling mechanism. The `ParallelStep` DSL will be updated to require developers to specify their intent when potential context conflicts exist, making the behavior of parallel pipelines explicit and deterministic.

#### **2. Rationale & Goals**

*   **Goal:** Eliminate race conditions and non-determinism in parallel context merging.
*   **Goal:** Force developers to be explicit about their intent when designing parallel steps that modify shared context.
*   **Goal:** Make the `ParallelStep` construct safer and less error-prone by default.
*   **Goal:** Provide clear, actionable error messages when a context merge conflict is detected without a resolution strategy.

#### **3. Functional Requirements & Design**

**Task 3.1: Enhance `MergeStrategy` Enum**

We need to add a new, safer strategy and adjust the default.

*   **Location:** `flujo/domain/dsl/step.py`
*   **Implementation Details:**
    1.  Add a new member to the `MergeStrategy` enum: `ERROR_ON_CONFLICT`.
    2.  Change the default `merge_strategy` in the `ParallelStep` model from `MergeStrategy.NO_MERGE` to `MergeStrategy.CONTEXT_UPDATE`. This makes merging the default, but the next step will make it safe.
*   **Acceptance Criteria & Testing (`make test-fast`)**:
    *   **Unit Test:** Verify that a `ParallelStep` created without a `merge_strategy` now defaults to `CONTEXT_UPDATE`.

**Task 3.2: Implement Conflict Detection in `safe_merge_context_updates`**

The core merge logic must be enhanced to detect and handle conflicts according to the selected strategy.

*   **Location:** `flujo/utils/context.py`
*   **Implementation Details:**
    1.  Modify the `safe_merge_context_updates` function to accept the `MergeStrategy` as an argument.
    2.  When merging, the function must detect if a field in the `source_context` has a different value than the same field in the `target_context`.
    3.  Implement the following logic based on the strategy:
        *   **If `CONTEXT_UPDATE` (the new default):** This will now implicitly act like `ERROR_ON_CONFLICT` unless a `field_mapping` is provided for the conflicting fields in the `ParallelStep`. If a conflict occurs on an unmapped field, raise a `ConfigurationError`.
        *   **If `OVERWRITE`:** Maintain the current "last writer wins" behavior (no change needed here, but its risk should be documented).
        *   **If `ERROR_ON_CONFLICT`:** If a key exists in both contexts with a different value, raise a `ConfigurationError` with a clear message: `f"Merge conflict for key '{key}'. Set an explicit merge strategy or field_mapping in your ParallelStep."`
*   **Acceptance Criteria & Testing (`make test-fast`)**:
    *   **Unit Tests:** In `tests/utils/test_context.py`:
        *   Test `safe_merge_context_updates` with two contexts that have a conflicting key. Assert that it raises a `ConfigurationError` when the strategy is `CONTEXT_UPDATE` or `ERROR_ON_CONFLICT`.
        *   Test that it succeeds when the strategy is `OVERWRITE`.
        *   Test that it succeeds if the values for the key are the same.

**Task 3.3: Update `DefaultParallelStepExecutor` to Enforce the Strategy**

The policy for `ParallelStep` must pass the strategy down to the merge logic and handle potential errors.

*   **Location:** `flujo/application/core/step_policies.py`
*   **Implementation Details:**
    1.  In `DefaultParallelStepExecutor.execute`, after all branches have completed, iterate through the successful branch contexts to merge them into the main context.
    2.  Pass the `parallel_step.merge_strategy` to the `safe_merge_context_updates` function (or the `ContextManager.merge` wrapper).
    3.  Wrap the merge loop in a `try...except ConfigurationError as e:` block. If a conflict error is caught, the entire `ParallelStep` should fail, returning a `Failure` outcome with the clear error message from the exception.
*   **Acceptance Criteria & Testing (`make all`)**:
    *   **Integration Tests:** In `tests/application/core/test_step_policies.py`:
        *   Create a `ParallelStep` where two branches modify the same context field. Run it with the default `CONTEXT_UPDATE` strategy. Assert that the step fails with a clear error about a merge conflict.
        *   Run the same step but configure it with `MergeStrategy.OVERWRITE`. Assert that the step succeeds.
        *   Run the same step but provide a `field_mapping` to resolve the conflict. Assert that the step succeeds and the context is merged correctly.

### **4. Rollout and Regression Plan**

1.  **Branching:** This work will be done on a dedicated feature branch (e.g., `feature/FSD-012-safe-merging`).
2.  **Implementation Order:** The tasks should be completed sequentially: 3.1, 3.2, then 3.3.
3.  **Testing Strategy:** Unit tests for the `MergeStrategy` enum and `safe_merge_context_updates` will be run with `make test-fast`. The full integration and regression suite (`make all`) is required to validate the changes in the `DefaultParallelStepExecutor`.
4.  **Documentation:** The docstrings for `ParallelStep` and `MergeStrategy` must be updated to clearly explain the new default behavior and how to resolve conflicts using `field_mapping` or by choosing a different strategy.
5.  **Merge:** Merge after all tests pass and documentation is updated.

### **5. Risks and Mitigation**

*   **Risk:** This is a **breaking change** for users who were relying on the old, unsafe default merge behavior.
    *   **Mitigation:** This is an acceptable and necessary breaking change for framework safety. The error messages must be extremely clear, guiding the user to the exact `ParallelStep` and the conflicting key, and explicitly telling them how to fix it (by setting a `merge_strategy` or `field_mapping`). This will be documented as a breaking change in the release notes.