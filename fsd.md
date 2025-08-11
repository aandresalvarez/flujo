## **Functional Specification Document: Typed Step Outcomes (FSD-008)**

**Author:** Alvaro
**Date:** 2023-10-27
**Status:** Partially Implemented (60%) – In Progress
**JIRA/Ticket:** FLUJO-123 (Example Ticket ID)

### Implementation Status

- **Last Updated:** 2025-08-11
- **Completed Tasks:** 3.1 (Models), 3.4 (Exception Choke Point), 3.5 (ExecutionManager Consumption), 3.6 (Unified Streaming API)
- **Remaining Tasks:** 3.2 (Policy Contracts), 3.3 (Policy Implementations)
- **Current State:** Core outcome types and orchestration paths are complete and in use. The policy layer still returns a mix of `StepResult` and `StepOutcome`; contracts need to be updated and implementations migrated to consistently return typed outcomes (`Success`, `Failure`, `Paused`, `Aborted`).

### Implementation Progress Summary (2025-08-11)

- Work completed
  - Defined `StepOutcome` models (`Success`, `Failure`, `Paused`, `Aborted`, `Chunk`) and added unit tests.
  - Implemented exception choke point in `ExecutorCore` that converts unexpected exceptions into typed `Failure` outcomes; re-raises control/config errors (`UsageLimitExceededError`, `PricingNotConfiguredError`, `MissingAgentError`, `MockDetectionError`, `InfiniteRedirectError`).
  - Updated `ExecutionManager` to consume `StepOutcome` and handle `Success`/`Failure`/`Paused`/`Aborted`/`Chunk` correctly.
  - Unified streaming API: added `Flujo.run_outcomes_async` returning `AsyncIterator[StepOutcome[StepResult]]` and normalized legacy chunks to `Chunk`.
  - HITL: `DefaultHitlStepExecutor` now returns typed `Paused` (no longer raises for control flow), and the protocol returns `StepOutcome[StepResult]`.
  - Normalization: Dynamic Router and Cache policies unwrap `StepOutcome` to `StepResult` where legacy expectations apply; `DummyRemoteBackend` now unwraps to `StepResult` to satisfy tests.
  - Strict pricing propagation: added guards to re-raise `PricingNotConfiguredError` from agent execution and simple-step usage-metrics extraction.

- Current status (delta)
  - Overall migration at ~60%. Orchestration is typed; policy layer is partially migrated and still returns a mix of `StepResult` and `StepOutcome` in some paths.
  - Remaining gaps observed in tests: strict pricing still converted to failures in some simple-step fallback paths; fallback semantics (attempt aggregation, usage metering, feedback content) need alignment; runner should dispatch an on-step-failure hook for failure visibility.

- Proposed next steps
  1. Complete Task 3.2/3.3 across policy layer
     - Update remaining executor protocols and implementations to return `StepOutcome[StepResult]` consistently.
     - Remove residual `PausedException` usage in policies in favor of `Paused` outcomes (keep legacy raise at boundaries only where required).
  2. Strict pricing and control-flow propagation
     - Ensure `PricingNotConfiguredError` and `UsageLimitExceededError` re-raise through simple-step fallback/validation branches (no conversion to failed results).
  3. Fallback semantics alignment
     - Aggregate attempts: primary attempts + fallback attempts in final result.
     - Usage metering: guarantee primary usage `add(...)` call before fallback where tests expect it.
     - Feedback: include primary error substring verbatim and preserve detailed text for very-long feedback tests.
  4. Hooks and runner integration
     - In `Flujo.run_async`, dispatch `on_step_failure` when a step fails (before final result) with a typed payload.
  5. Outcome normalization boundaries
     - Keep normalization at backend/runner edges only; policies return typed outcomes; coordinator/legacy helpers unwrap only where required by legacy tests.
  6. Validation
     - Run test-fast and full suite; then run type-checking (mypy) and update docs where signatures changed.


### **1. Overview**

This document outlines the design and implementation plan to refactor Flujo's core execution engine from an exception-based control flow model to a typed, value-based `StepOutcome` model. This change is foundational for improving the framework's robustness, testability, and readiness for distributed backends.

Currently, control flow for events like pausing for human input (`PausedException`) or graceful termination (`PipelineAbortSignal`) is handled by raising and catching specific exceptions. This couples the execution logic to Python's exception system, making it brittle and difficult to adapt for non-local execution backends.

This FSD proposes replacing this pattern with an explicit, serializable `StepOutcome` system, where policies return a typed result (`Success`, `Failure`, `Paused`, `Aborted`) instead of raising exceptions for control flow.

### **2. Rationale & Goals**

#### **2.1. Problems with the Current Approach**

*   **Architectural Coupling:** Tightly couples the orchestration logic to Python's exception handling, hindering the future implementation of distributed backends (e.g., Ray, Temporal) which rely on message passing.
*   **Ambiguous Contracts:** The return type of an execution policy is not fully descriptive of all possible outcomes. A caller must inspect both the return value (`StepResult`) and handle multiple potential exceptions.
*   **Brittle Control Flow:** Using exceptions for non-exceptional events like pausing makes the code harder to reason about and maintain.
*   **Confusing Streaming API:** The current streaming API yields both data chunks and a final `PipelineResult`, creating an inconsistent and difficult-to-use contract.

#### **2.2. Goals of this Refactor**

*   **Decouple Control Flow:** Make the execution engine's control flow explicit through typed return values, independent of the underlying execution runtime.
*   **Improve Architectural Clarity:** Establish a clear, single contract for what a step execution policy can return.
*   **Enhance Safety and Testability:** Make it easier to test for specific outcomes without relying on `pytest.raises`.
*   **Unify the Streaming Contract:** Provide a single, consistent `AsyncIterator[StepOutcome]` return type for all step executions.
*   **Lay the Foundation for Distributed Execution:** The new `StepOutcome` objects will be serializable, enabling them to be passed between processes or machines.

### **3. Functional Requirements & Design**

The core of this change is the introduction of a `StepOutcome` algebraic data type and the refactoring of the execution stack to produce and consume it.

#### **Task 3.1: Define `StepOutcome` Data Models**

The `StepOutcome` will be a set of Pydantic models representing all possible terminal states of a single step's execution.

*   **Location:** `flujo/domain/models.py`
*   **Implementation Details:**
    *   Create a generic `StepOutcome(BaseModel, Generic[T])` base class.
    *   Define the following subclasses:
        *   `Success(StepOutcome[T])`: Contains a single field `step_result: StepResult`.
        *   `Failure(StepOutcome[T])`: Contains `error: Exception`, `feedback: str`, and `step_result: StepResult` (the partial result at the time of failure).
        *   `Paused(StepOutcome[T])`: Contains `message: str` and an optional `state_token: Any` for resumption context.
        *   `Aborted(StepOutcome[T])`: Contains a `reason: str` for why execution was halted (e.g., by a circuit breaker).
*   **Acceptance Criteria & Testing:**
    *   **Unit Tests:** Create `tests/domain/test_models.py`.
        *   Verify that each `StepOutcome` subclass can be instantiated correctly.
        *   Verify that each model is serializable (using `model_dump_json`).

#### **Task 3.2: Update Core Policy Contracts**

All step execution policies must be updated to adhere to the new `StepOutcome` contract.

*   **Location:** `flujo/application/core/executor_protocols.py` and `flujo/application/core/step_policies.py`.
*   **Implementation Details:**
    *   Change the return type signature of all `execute` methods in the policy protocols and their default implementations from `Awaitable[StepResult]` to `Awaitable[StepOutcome[StepResult]]`.
*   **Acceptance Criteria & Testing:**
    *   **Static Analysis:** The type checker (mypy) should pass, confirming all signatures have been updated. No specific runtime tests are needed for this signature-only change.

#### **Task 3.3: Refactor Policy Implementations to Return Outcomes**

This is the core implementation task. Each policy's logic must be modified.

*   **Location:** `flujo/application/core/step_policies.py`.
*   **Implementation Details:**
    *   **Success Path:** Replace all instances of `return StepResult(...)` with `return Success(step_result=StepResult(...))`.
    *   **Failure Path:**
        *   Catch specific, expected exceptions (e.g., from agent calls, validators).
        *   Create a partial `StepResult` with `success=False` and relevant feedback.
        *   Return a `Failure(error=e, feedback=..., step_result=...)` object.
    *   **Pause Path (`DefaultHitlStepExecutor`):** Replace `raise PausedException(...)` with `return Paused(message=...)`.
    *   **Unhandled Exceptions:** True programming errors or unexpected system failures should *still raise exceptions*. They will be handled by the choke point in the next task.
*   **Acceptance Criteria & Testing:**
    *   **Unit Tests:** For each policy in `tests/application/core/test_step_policies.py`:
        *   Test the success path, asserting the returned value is an instance of `Success` and contains the correct `StepResult`.
        *   Test failure paths (e.g., a failing validator), asserting the returned value is `Failure` with the correct error and feedback.
        *   Specifically for `DefaultHitlStepExecutor`, test that it returns a `Paused` outcome.

#### **Task 3.4: Implement the Exception Choke Point in `ExecutorCore`**

The central dispatcher will be the single point where unexpected exceptions are converted into `Failure` outcomes.

*   **Location:** `flujo/application/core/executor_core.py`.
*   **Implementation Details:**
    *   In the `execute` method, wrap the call to the policy (`policy.execute(frame)`) in a `try...except Exception as e:` block.
    *   The `except` block should catch all exceptions *except* for specific ones that the `ExecutionManager` needs to handle directly (like `UsageLimitExceededError`).
    *   Inside the `except` block, create a `StepResult` and return a `Failure` outcome.
*   **Acceptance Criteria & Testing:**
    *   **Integration Test:** Create `tests/application/core/test_executor_core.py`.
        *   Create a mock policy that raises an unexpected `ValueError`.
        *   Call `ExecutorCore.execute` with this policy.
        *   Assert that the returned value is an instance of `Failure` and that its `error` attribute is the original `ValueError`.

#### **Task 3.5: Refactor `ExecutionManager` to Consume `StepOutcome`**

The main execution loop must be updated to handle the new return types.

*   **Location:** `flujo/application/core/execution_manager.py`.
*   **Implementation Details:**
    *   Modify the `execute_steps` method's main loop.
    *   The loop now receives `StepOutcome` objects from `step_coordinator.execute_step`.
    *   Use a `match/case` statement or `if/isinstance` chain to handle each outcome:
        *   `Success`: Extract the `StepResult` and proceed as normal.
        *   `Failure`: Append the partial `StepResult` to the history and terminate the loop.
        *   `Paused`: Update the context's `scratchpad` with the paused status and message, then `raise PipelineAbortSignal` to stop execution and signal the `Flujo` runner.
        *   `Aborted`: Terminate the loop immediately.
*   **Acceptance Criteria & Testing:**
    *   **Integration Tests:** In `tests/application/core/test_execution_manager.py`:
        *   Test a simple successful pipeline, ensuring it completes correctly.
        *   Test a pipeline where a step returns a `Failure` outcome, asserting that the pipeline stops at that step and the final `PipelineResult` is marked as failed.
        *   Test a pipeline where a step returns a `Paused` outcome, asserting that `PipelineAbortSignal` is raised and the final context reflects the paused state.

#### **Task 3.6: Unify the Streaming API**

The streaming contract will be simplified by the `StepOutcome` model.

*   **Location:** `flujo/application/runner.py`.
*   **Implementation Details:**
    *   Define a new `StepOutcome` subclass, `Chunk(StepOutcome[T])`, in `flujo/domain/models.py`.
    *   The `on_chunk` callback in `ExecutionFrame` will now wrap incoming data chunks in a `Chunk` outcome and yield them.
    *   The `run_async` method in the `Flujo` runner will now have a clear return type: `AsyncIterator[StepOutcome[StepResult]]`.
    *   Non-streaming steps will yield a single `Success` or `Failure` outcome.
    *   Streaming steps will yield zero or more `Chunk` outcomes, followed by a final `Success` or `Failure` outcome containing the complete `StepResult`.
*   **Acceptance Criteria & Testing:**
    *   **Regression Tests:** All existing tests for `run` and `run_async` should be updated and pass.
    *   **Integration Tests:** In `tests/application/test_runner.py`:
        *   Test a streaming pipeline. The test should iterate through the async generator, assert that it receives `Chunk` instances, and finally a `Success` instance containing the final `PipelineResult`.
        *   Test a non-streaming pipeline, asserting that the async generator yields exactly one `Success` instance.

### **4. Rollout and Regression Plan**

1.  **Branching:** This work will be done on a dedicated feature branch (e.g., `feature/FSD-008-typed-outcomes`).
2.  **Implementation Order:** Tasks should be completed in the order listed above (3.1 through 3.6).
3.  **Testing Strategy:** Each task must be accompanied by its specified unit and integration tests. After all tasks are complete, a full regression test suite (`pytest`) must be run to catch any unintended side effects.
4.  **Code Review:** Due to the high architectural impact, this change requires at least two senior reviewers.
5.  **Merge:** Once all tests pass and the code is approved, the feature branch will be merged into the main development branch.

### **5. Risks and Mitigation**

*   **Risk:** High blast radius could introduce subtle regressions in complex control flows (e.g., nested loops with fallbacks).
    *   **Mitigation:** The task-by-task testing strategy is designed to mitigate this. A full regression run is mandatory. Particular attention should be paid to tests involving `LoopStep`, `ParallelStep`, and `fallback_step`.
*   **Risk:** The refactor could be time-consuming due to the number of files affected.
    *   **Mitigation:** The changes are highly mechanical. Once the pattern is established in one policy, applying it to others should be straightforward. This FSD provides a clear guide to minimize discovery time.