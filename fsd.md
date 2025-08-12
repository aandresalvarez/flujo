## Remaining Work Plan: Typed Step Outcomes (FSD-008)

Author: Alvaro
Status: Hybrid stabilized; remaining work staged for safe rollout
Scope: This document lists ONLY the outstanding work to complete FSD-008. Existing, already-implemented parts are considered baseline and not repeated here.

### 1. Constraints and Principles

- Enforce explicit, typed control flow using `StepOutcome[T]` across the execution stack.
- Maintain strict typing and SRP; avoid `Any` and monolithic functions.
- Preserve legacy behavior temporarily via adapters only where necessary; eliminate adapters when native outcomes are implemented.
- Robust, test-driven changes with unit and integration tests for each step.
- Backward compatibility: legacy entry points keep returning `StepResult` until deprecation, but outcome-first paths must be canonical.

### 2. Current State Snapshot (for context)

- Outcome models exist: `Success`, `Failure`, `Paused`, `Aborted`, `Chunk`.
- `ExecutorCore.execute(...)` is outcome-first on backend/frame path, and unwraps to `StepResult` on legacy path. Calls to policies now explicitly pass `return_outcome` to minimize ambiguity.
- `DefaultHitlStepExecutor` already returns `Paused` as an outcome.
- Outcome adapters have been removed from code paths; policies are hybrid and accept `return_outcome`.
- `ExecutionManager`, `Runner` have partial outcome support; `run_outcomes_async` is available.

The remaining work is to make outcomes native end-to-end and remove compatibility scaffolding.

### 3. Work Breakdown (Small, Testable Steps)

#### 3.1 Refactor Policies to Return StepOutcome Natively (Deferred)

Policies currently support a hybrid mode via `return_outcome` to maintain backward compatibility. The full native conversion will be staged after doc updates and broader test migration.

- `DefaultAgentStepExecutor`
- `DefaultSimpleStepExecutor`
- `DefaultLoopStepExecutor`
- `DefaultParallelStepExecutor`
- `DefaultConditionalStepExecutor`
- `DefaultCacheStepExecutor`

For each policy, implement the same structured changes:

1) Signature and typing
- Change return type to `Awaitable[StepOutcome[StepResult]]`.

2) Success path
- Wrap final `StepResult` as `Success(step_result=...)`.

3) Failure path (expected, non-programming errors)
- Catch expected, domain-level errors (agent errors, validator failures, pricing errors, etc.).
- Create a partial `StepResult` with `success=False`, `feedback`, and minimal metrics.
- Return `Failure(error=e, feedback=..., step_result=partial_result)`.

4) Pause path
- Only policies that manage HITL-like behavior should return `Paused(message=..., state_token=...)`.
- Policies must NOT raise `PausedException` internally.

5) Streaming support (where applicable)
- When emitting stream chunks, yield/forward them as `Chunk(data=..., step_name=...)` through the existing callbacks.

6) Unexpected exceptions
- Re-raise unexpected programming errors. These will be converted by the central choke point.

Acceptance Criteria (per policy, staged):
- When `return_outcome=True`, all code paths return `StepOutcome[StepResult]`.
- When `return_outcome=False`, policies return `StepResult` preserving legacy expectations.
- No policy raises `PausedException` directly.

Testing (per policy):
- Unit tests covering: success, failure (expected), pause (if applicable), streaming (if applicable).
- File locations: extend/add under `tests/application/core/` with one test module per policy (e.g., `test_step_policies_agent_outcomes.py`).


#### 3.2 Remove Outcome Adapters and Wire Natively (Done)

Objective: Eliminate adapter classes and wire policies directly with explicit `return_outcome`.

Steps:
1) Delete adapter classes in `flujo/application/core/step_policies.py`:
- `DefaultAgentStepExecutorOutcomes`
- `DefaultSimpleStepExecutorOutcomes`
- `DefaultParallelStepExecutorOutcomes`
- `DefaultConditionalStepExecutorOutcomes`

2) Replace all usages in `ExecutorCore.execute(...)` where these adapters are conditionally created.
- Call policies directly; expect `StepOutcome` in frame/backend paths.
- Remove redundant normalization of `StepOutcome -> StepResult` except where legacy signature requires it.

3) Simplify `ExecutorCore` outcome handling
- Where `called_with_frame` is true, return `StepOutcome` directly (no unwrap).
- For legacy callers, unwrap with existing `_unwrap_outcome_to_step_result`.

Acceptance Criteria:
- No references to outcome adapter classes remain. (Met)
- `ExecutorCore.execute(...)` paths explicitly pass `return_outcome` and are consistent. (Met)

Testing:
- Extend existing `tests/application/core/test_executor_core.py` to assert that frame/backend calls receive `StepOutcome` directly and legacy calls receive `StepResult`.


#### 3.3 ExecutionManager: Outcome-First Main Loop (Done)

Objective: `ExecutionManager.execute_steps(...)` consumes `StepOutcome` values from the coordinator/backend and applies explicit control flow.

Steps:
1) Normalize input items to `StepOutcome` using a small helper (or reuse `to_outcome`), then use `isinstance` branching:
- `Success`: append `step_result` to history; continue.
- `Failure`: append partial `step_result` (synthesizing minimal one if absent); stop loop and yield final `PipelineResult`.
- `Paused`: update context scratchpad with status/message; raise `PipelineAbortSignal`.
- `Aborted`: stop loop immediately; yield final `PipelineResult`.
- `Chunk`: forward upstream (streaming path), maintain last known step for attribution if needed.

2) Ensure state persistence order remains correct (persist after successful steps only; not after failure/abort/pause).

Acceptance Criteria:
- All five outcome types handled explicitly.
- Correct termination semantics for `Failure`, `Paused`, `Aborted`.
- Streaming path forwards `Chunk` transparently.

Testing:
- Add/extend `tests/application/core/test_execution_manager.py` with cases for each outcome type and state persistence checks.


#### 3.4 StepCoordinator: Yield StepOutcome Consistently (Done)

Objective: Coordinator yields `StepOutcome` across both streaming and non-streaming calls.

Steps:
1) Ensure coordinator calls backend/executor paths that return `StepOutcome`.
2) If any policy/backend still returns `StepResult`, convert via `to_outcome(sr)`.
3) Streaming: yield `Chunk` outcomes for chunks, and a final `Success` for the completed step.

Acceptance Criteria:
- Coordinator never yields raw `StepResult` in the backend path.
- Streaming behavior emits zero or more `Chunk` then a terminal outcome (usually `Success`).

Testing:
- Add `tests/application/core/test_step_coordinator_outcomes.py` covering non-streaming and streaming flows.


#### 3.5 Runner Alignment and Public API (Done)

Objective: Make `run_outcomes_async` the canonical typed-streaming API; keep `run_async` for legacy.

Steps:
1) Ensure `run_outcomes_async` simply relays `StepOutcome` events from `ExecutionManager`:
- Pass through `Chunk`, `Success`, `Failure`.
- On `Paused`, yield `Paused` and return.
2) Confirm `run_async` remains compatible (yields chunks and final `PipelineResult` for legacy consumers), but internal wiring should use the typed path.

Acceptance Criteria:
- `run_outcomes_async` returns `AsyncIterator[StepOutcome[StepResult]]` with the documented sequence rules.
- Existing `tests/application/test_runner.py` remain valid; extend with a failure-path test.

Testing:
- Add `test_runner_outcomes_failure_stops.py` ensuring a `Failure` outcome short-circuits and no `Success` follows.


#### 3.6 Deprecation and Cleanup Pass (In Progress)

Objective: Remove transitional code and guide users.

Steps:
1) Ensure explicit `return_outcome` wiring across executor paths. (Done)
2) Plan staged migration timeline to remove hybrid flags in a major release.
3) Update docs: `docs/advanced/typed_outcomes.md` and migration notes to describe hybrid behavior and migration guide. (Next)

Acceptance Criteria:
- No adapter usage remains; warnings only appear in pre-migration branches.
- Docs reflect native outcomes across the stack.

Testing:
- Lint pass and static analysis to ensure no references to removed classes remain.


### 4. Test Plan Summary

- Unit tests per policy: success/failure/pause/stream.
- Coordinator tests: outcome yield semantics.
- Execution manager tests: control-flow handling for all outcome types and state persistence order.
- Runner tests: streaming and failure-path correctness.
- Static typing: mypy clean. Run `make install` then `make mypy` prior to CI `pytest` runs.
- Fast regression: `make test-fast`. Full: CI pipeline.

### 5. Rollout Order

1) 3.1 Refactor policies (one class per PR if needed; smallest viable increments).
2) 3.2 Remove adapters and simplify `ExecutorCore` (once at least Agent + Simple are migrated).
3) 3.4 Coordinator outcome consistency.
4) 3.3 ExecutionManager outcome-first loop.
5) 3.5 Runner alignment and public API confirmation.
6) 3.6 Deprecation and cleanup.

Each step must land with green unit/integration tests and no new linter/type errors.

### 6. Acceptance for FSD-008 Completion

- All policies return `StepOutcome[StepResult]` natively.
- No adapter classes remain; executor/coordinator/manager/runner are outcome-first.
- Legacy callers still receive `StepResult` where explicitly supported; deprecation path documented.
- All tests pass; mypy and linters clean; docs updated.

 
## **Functional Specification Document: Typed Step Outcomes (FSD-008)**

**Author:** Alvaro
**Date:** 2023-10-27
**Status:** Proposed
**JIRA/Ticket:** FLUJO-123 (Example Ticket ID)

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