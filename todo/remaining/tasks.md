## FSD-13: Fortifying the Execution Core - Granular Task List

**Legend:**
- **Status:** ⬜️ To Do | 🟧 In Progress | ✅ Done
- **AC:** Acceptance Criteria

---

### **Epic 1: Foundational Refactoring - The `ExecutionFrame`**

**Goal:** Establish a robust, type-safe data contract for all internal execution calls to eliminate parameter-passing bugs.

---

#### **Task 1.1: Define the `ExecutionFrame` Data Structure**
- **Status:** ⬜️
- **Description:** Create the core `ExecutionFrame` dataclass. This is a non-functional change that establishes the new data structure.
- **Details:**
  - Create a new file: `flujo/application/core/types.py`.
  - Define the `ExecutionFrame(Generic[ContextT])` dataclass.
  - Fields must include: `step`, `data`, `context`, `resources`, `limits`, `stream`, `on_chunk`, `breach_event`, `context_setter`.
  - Ensure all fields are correctly typed using `typing` primitives.
- **AC:**
  - [ ] The file `flujo/application/core/types.py` exists and contains the `ExecutionFrame` definition.
  - [ ] The code is fully type-annotated and passes `mypy flujo/`.
  - [ ] `make test-fast` passes without regressions.
- **Requirements Met:** REQ-CORE-001 (Partial)

---

#### **Task 1.2: Refactor `ExecutorCore.execute` Signature**
- **Status:** ⬜️
- **Description:** Modify the primary execution method to use the new `ExecutionFrame`. This is a significant internal API change.
- **Details:**
  - In `flujo/application/core/ultra_executor.py`, import `ExecutionFrame`.
  - Change `ExecutorCore.execute(self, **kwargs)` to `execute(self, frame: ExecutionFrame[ContextT])`.
  - In the method body, replace all `kwargs.get(...)` calls with direct attribute access from the `frame` object (e.g., `step = frame.step`).
- **AC:**
  - [ ] The method signature is updated as specified.
  - [ ] All internal variable assignments are updated to use `frame`.
  - [ ] `make test-fast` is expected to fail in multiple places, which is the correct outcome for this isolated change.
- **Requirements Met:** REQ-CORE-001 (Partial)

---

#### **Task 1.3: Update `ExecutionManager` to Use `ExecutionFrame`**
- **Status:** ⬜️
- **Description:** Update the highest-level caller of `ExecutorCore.execute` to use the new `ExecutionFrame` contract. This will fix a large number of test failures from the previous task.
- **Details:**
  - In `ExecutionManager.execute_steps`, locate the call to `self.backend.execute_step`.
  - Before the call, instantiate `ExecutionFrame` with all the necessary local state (`step`, `data`, `context`, `limits`, etc.).
  - Pass the single `frame` object to the `execute` method.
- **AC:**
  - [ ] The `ExecutionManager` now correctly constructs and passes an `ExecutionFrame`.
  - [ ] A significant portion of test failures from Task 1.2 should now be resolved.
  - [ ] `make test-fast` passes.
- **Requirements Met:** REQ-CORE-001 (Partial)

---

#### **Task 1.4: Update Recursive Calls for Control-Flow Steps**
- **Status:** ⬜️
- **Description:** Update all internal, recursive calls within `ExecutorCore` to use the `ExecutionFrame`, ensuring state is propagated correctly to nested steps.
- **Details:**
  - **For `_handle_parallel_step`:** Modify the `run_branch` inner function. It must construct a new `ExecutionFrame` for the branch's execution, ensuring `context_setter` and other operational parameters are passed down.
  - **For `_handle_loop_step`:** Modify the `step_executor` inner function. It must construct a new `ExecutionFrame` for each iteration of the loop body.
  - **For `_handle_conditional_step`:** Modify the `step_executor` inner function. It must construct a new `ExecutionFrame` for the selected branch.
- **AC:**
  - [ ] All recursive `self.execute` or `step_executor` calls within `ExecutorCore` are updated to use `ExecutionFrame`.
  - [ ] All tests related to `ParallelStep`, `LoopStep`, and `ConditionalStep` pass.
  - [ ] `make test-fast` passes.
- **Requirements Met:** REQ-CORE-001 (Complete)

---

### **Epic 2: Component-Level Fixes**

**Goal:** Address specific bugs in context, serialization, and usage governance.

---

#### **Task 2.1: Centralize `PipelineContext` Creation**
- **Status:** ⬜️
- **Description:** Enforce the architectural rule that only the top-level `Flujo` runner can create a `PipelineContext`.
- **Details:**
  - In `Flujo.run_async`, add the logic to instantiate the `PipelineContext` (or a user-provided `context_model`).
  - Ensure the `initial_prompt` field is populated from the `initial_input` argument.
  - Search for and remove any fallback `PipelineContext()` creation logic within `ExecutionManager` or `ExecutorCore`.
- **AC:**
  - [ ] A new unit test for `Flujo.run_async` asserts that a valid `PipelineContext` with a correct `initial_prompt` is created.
  - [ ] An integration test with a nested pipeline (using `as_step`) verifies the sub-pipeline receives a correctly initialized context.
  - [ ] `make test-fast` passes.
- **Requirements Met:** REQ-CTX-001, REQ-CTX-002

---

#### **Task 2.2: Implement Serialization Registry**
- **Status:** ⬜️
- **Description:** Implement the public API and internal logic for the custom serialization system.
- **Details:**
  - In `flujo/utils/serialization.py`, implement `register_custom_serializer` and `register_custom_deserializer`.
  - Use a `threading.Lock` to ensure the global registry dictionaries are thread-safe.
  - Create `tests/utils/test_serialization.py` and add unit tests specifically for registering a custom class and verifying that both serialization and deserialization work through the registry.
- **AC:**
  - [ ] The new unit tests for the serialization registry pass.
  - [ ] `make test-fast` passes.
- **Requirements Met:** REQ-SER-001

---

#### **Task 2.3: Integrate Serialization Registry into Core Systems**
- **Status:** ⬜️
- **Description:** Connect the `StateManager` and `ICacheBackend` to the new serialization registry.
- **Details:**
  - Modify `SQLiteBackend.save_state` to use `safe_serialize`, which should internally use the registry.
  - Modify `SQLiteBackend.load_state` to use `safe_deserialize`, which should internally use the registry.
  - Modify `InMemoryLRUBackend.put` and `get` methods to use the same serialization/deserialization functions.
- **AC:**
  - [ ] All existing tests for state persistence and caching continue to pass.
  - [ ] `make test-fast` passes.
- **Requirements Met:** REQ-SER-002, REQ-SER-003

---

#### **Task 2.4: Create Serialization Fixture for Tests**
- **Status:** ⬜️
- **Description:** Resolve test failures related to mock objects by teaching the framework how to serialize them.
- **Details:**
  - In `tests/conftest.py`, define a new `pytest` fixture (e.g., `register_mock_serializers`).
  - Within this fixture, call `register_custom_serializer` for mock types like `UsageResponse` and `MockImageResult`. The serializer can be a simple `lambda obj: obj.__dict__`.
  - Apply this fixture to the test modules that were previously failing due to serialization errors.
- **AC:**
  - [ ] All tests that previously failed with `TypeError: Object of type ... is not JSON serializable` now pass.
  - [ ] `make test-fast` passes.
- **Requirements Met:** REQ-SER-004

---
 
#### **Task 2.5.1: Re-sequence the `UsageGovernor` Check**
- **Status:** ✅
- **Description:** Modify the `ExecutionManager` to perform the usage limit check at the correct point in the execution sequence—immediately after a step result is obtained.
- **Details:**
    - In `ExecutionManager.execute_steps`, locate the block of code where `step_result` is received from the `StepCoordinator`.
    - Move the call to `self.usage_governor.check_usage_limits_efficient` to be the *very next* operation after the `step_result` is finalized.
    - At this stage, the `if` block for a breach will be simple: just `raise UsageLimitExceededError`. The goal here is only to change the *timing* of the check. The content of the exception will be fixed in a later task.
- **AC:**
    - [x] A code review confirms the call to `check_usage_limits_efficient` has been moved to the correct location.
    - [x] All existing usage limit tests continue to pass. The behavior should not have changed yet, only the code location.
    - [x] `make test-fast` passes.
- **Requirements Met:** REQ-GOV-001 (Partial)

---

#### **Task 2.5.2: Create a Precise Breach Integration Test**
- **Status:** ✅
- **Description:** Create a new, targeted integration test that reliably fails by precisely breaching a usage limit on its final step. This test is expected to fail initially.
- **Details:**
    - Create a new test file, e.g., `tests/integration/test_usage_governor.py`.
    - Define a simple two-step pipeline.
    - Use `StubAgent`s that return `StepResult`s with known, hardcoded costs (e.g., Step 1 costs $0.50, Step 2 costs $0.75).
    - Set a `UsageLimits` on the `Flujo` runner with a cost limit of exactly $1.00.
    - Run the pipeline and wrap the call in `pytest.raises(UsageLimitExceededError)`.
    - Initially, this test might fail because the exception's `step_history` is incorrect. The goal is to create the failing test case first.
- **AC:**
    - [x] The new integration test is created and committed.
    - [x] The test reliably fails, specifically because the `step_history` in the caught exception is missing the final, breaching step.
    - [x] `make test-fast` shows one new failing test.
- **Requirements Met:** REQ-GOV-002 (Test Case Creation)

---

#### **Task 2.5.3: Ensure Correct State in `UsageLimitExceededError`**
- **Status:** ✅
- **Depends On:** Task 2.5.1, Task 2.5.2
- **Description:** Modify the `ExecutionManager` to ensure that when a usage limit is breached, the `PipelineResult` object passed to the `UsageLimitExceededError` constructor contains the complete and correct `step_history`.
- **Details:**
    - In `ExecutionManager.execute_steps`, inside the `if` block where a breach is detected (from Task 2.5.1):
    1.  First, append the current `step_result` to the `pipeline_result.step_history`.
    2.  Then, instantiate and `raise` the `UsageLimitExceededError`, passing the now-updated `pipeline_result` to its constructor.
- **AC:**
    - [x] The integration test created in Task 2.5.2 now passes. The assertion that the exception's `result.step_history` contains the final step is now met.
    - [x] All other existing usage limit tests continue to pass.
    - [x] `make test-fast` passes.
- **Requirements Met:** REQ-GOV-001 (Complete), REQ-GOV-002 (Implementation)

---

#### **Task 2.5.4: Verify and Refactor `check_usage_limits_efficient`**
- **Status:** ✅
- **Description:** Review and confirm the logic of the `check_usage_limits_efficient` method itself is correct and perform any minor refactoring needed for clarity.
- **Details:**
    - In `flujo/application/core/usage_governor.py`, review the `check_usage_limits_efficient` method.
    - Ensure it correctly handles `None` for limits.
    - Ensure it correctly sums the current totals with the new step's deltas before comparison.
    - Add comments to clarify the logic if needed.
- **AC:**
    - [x] The logic is confirmed to be correct via code review.
    - [x] Any necessary refactoring for clarity is completed.
    - [x] No functional changes are expected, so all tests must continue to pass.
    - [x] `make test-fast` passes.
- **Requirements Met:** Final verification for REQ-GOV-001 & REQ-GOV-002.


---

#### **Task 2.6: Ensure Thread-Safe Parallel Usage Governance**
- **Status:** ✅
- **Description:** Harden the `_ParallelUsageGovernor` to prevent race conditions.
- **Details:**
  - In `ExecutorCore._ParallelUsageGovernor`, ensure the `add_usage` method is an `async` method and that its body is entirely protected by an `asyncio.Lock`.
  - Create a new stress test for `ParallelStep` with at least 10 concurrent branches, each returning a small, known cost. Assert that the final aggregated cost in the `PipelineResult` is exactly the sum of the individual costs.
- **AC:**
  - [x] The new stress test for parallel usage governance passes consistently.
  - [x] `make test-fast` passes.
- **Requirements Met:** REQ-GOV-003

---

### **Epic 3: Finalization and Cleanup**

**Goal:** Remove all legacy code, update documentation, and perform final validation.

---

#### **Task 3.1: Finalize Type Alias Migration and Cleanup**
- **Status:** ✅
- **Description:** Complete the final code migration and remove the legacy module.
- **Details:**
  - Move the `StepExecutor` type alias from `step_logic.py` to `ultra_executor.py`.
  - Globally find and replace all imports of `StepExecutor` to point to the new location.
  - Delete the file `flujo/application/core/step_logic.py`.
- **AC:**
  - [x] The `step_logic.py` file no longer exists.
  - [x] The project compiles and runs without any import errors related to the moved type or deleted file.
  - [x] `make test-fast` passes.
- **Requirements Met:** REQ-CORE-002, REQ-CORE-003

---

#### **Task 3.2: Update Documentation and Final Verification**
- **Status:** ✅
- **Description:** Update documentation to reflect the finalized architecture and perform a final, comprehensive validation.
- **Details:**
  - Edit the main architecture document to describe the `ExecutionFrame` pattern.
  - Remove all mentions of `step_logic.py` from documentation.
  - Run the full test suite, including linters and type checkers.
- **AC:**
  - [x] The architecture documentation is updated.
  - [x] `make lint` passes.
  - [x] `mypy flujo/` passes with no new errors.
  - [x] `make test` (the full suite, not just fast) passes with 100% success.
  - [x] Performance benchmarks show no significant regressions (>5%).
- **Requirements Met:** NFR-DOCS-001, NFR-TEST-001, NFR-PERF-001


### **Epic 4: Final Cleanup and Architectural Polish (New)**

**Goal:** Address minor inconsistencies and remove all remaining legacy artifacts to complete the migration and improve code clarity.

---

#### **Task 4.1: Deprecate and Remove `step_logic.py`**
- **Status:** ✅
- **Description:** The `step_logic.py` file was successfully refactored into `ultra_executor.py`, but the original file was not deleted. This task is to complete the final removal.
- **Details:**
  - Perform a final global search to ensure no lingering imports from `flujo/application/core/step_logic.py`.
  - Delete the file `flujo/application/core/step_logic.py`.
- **AC:**
  - [x] The file `flujo/application/core/step_logic.py` is deleted from the repository.
  - [x] The application compiles and runs without any `ModuleNotFoundError`.
  - [x] `make test-fast` passes.
- **Requirements Met:** REQ-CORE-002 (Complete)

---

#### **Task 4.2: Consolidate `parallel.py` Module**
- **Status:** ✅
- **Description:** The file `flujo/application/parallel.py` now only contains re-exports from `ultra_executor.py` and is redundant. This task is to remove it and update imports.
- **Details:**
  - Identify all files that `import ... from flujo.application.parallel`.
  - Update these import statements to point directly to `flujo.application.core.ultra_executor`.
  - Delete the file `flujo/application/parallel.py`.
- **AC:**
  - [x] The file `flujo/application/parallel.py` is deleted from the repository.
  - [x] All import statements are updated to the new location.
  - [x] `make test-fast` passes.
- **Requirements Met:** Architectural Simplification

---

#### **Task 4.3: Refactor and Clarify Internal Executor Methods**
- **Status:** ✅
- **Description:** The `ExecutorCore` currently has `_execute_simple_step` and `_execute_step_logic` with overlapping responsibilities. This task is to refactor them for clarity.
- **Details:**
  - Analyze the logic within both methods.
  - Merge the logic into a single, private method, for example: `_execute_agent_step`. This method will be responsible for the full lifecycle of a non-control-flow step (retries, fallbacks, agent execution, etc.).
  - The `_execute_complex_step` method will remain as the dispatcher for control-flow steps.
- **AC:**
  - [x] The `_execute_simple_step` and `_execute_step_logic` methods are merged into a single, well-named private method.
  - [x] The `execute` method's main `if/else` block now cleanly dispatches between `_execute_agent_step` and `_execute_complex_step`.
  - [x] `make test-fast` passes.
- **Requirements Met:** Code Clarity and Maintainability

---

#### **Task 4.4: Encapsulate Retry Payload Logic**
- **Status:** ✅
- **Description:** The logic for cloning a payload for retries inside `_execute_agent_step` is complex and can be encapsulated into a dedicated helper function.
- **Details:**
  - Enhanced the existing `_clone_payload_for_retry` method with more robust type handling and better documentation.
  - Added support for dataclasses, lists/tuples, and improved error handling for deep copy operations.
  - Enhanced the method with comprehensive documentation explaining the cloning strategy.
- **AC:**
  - [x] The enhanced `_clone_payload_for_retry` method exists and contains improved cloning logic.
  - [x] The retry loop in the main agent execution method uses the encapsulated method.
  - [x] `make test-fast` passes.
- **Requirements Met:** Code Readability and Encapsulation

### Tasks for Execution Core Simple Step Refactoring

* Task ID: TASK-EXEC-001
  - **Status:** ✅
  - Title: Enable Validator Retrying in Simple Steps
  - Description: Refactor `_execute_simple_step` in `ultra_executor.py` to run `validator_runner.validate` inside the retry loop, so `ValueError` from validators triggers retries up to `step.config.max_retries`.
  - Acceptance Criteria:
    1. In `test_validator_failure_triggers_retry`, a failing validator will be called `1 + step.config.max_retries` times.
    2. The final `result.attempts` matches `1 + step.config.max_retries`.
    3. Feedback contains "Validation failed after max retries".

* Task ID: TASK-EXEC-002
  - **Status:** ✅
  - Title: Standardize Plugin Failure Feedback
  - Description: Adjust `_execute_simple_step` to run plugin runner prior to agent execution with correct error handling and to format plugin failures as "Plugin execution failed after max retries: {message}".
  - Acceptance Criteria:
    1. In `test_plugin_validation_failure_with_feedback`, plugin failures produce feedback matching expectations.
    2. In `test_plugin_failure_propagates`, plugin errors respect retry settings and preserve success semantics.

* Task ID: TASK-EXEC-003
  - **Status:** ✅
  - Title: Restore Caching and Usage Tracking in Simple Steps
  - Description: Ensure `_execute_simple_step` calls `cache_backend.put` on cache miss and `usage_meter.add` on each successful invocation, matching tests in `test_caching_behavior` and `test_usage_tracking`.
  - Acceptance Criteria:
    1. `test_caching_behavior` sees `cache_backend.put` invoked exactly once.
    2. `test_usage_tracking` observes `step_history` populated and usage metrics recorded.

### Tasks for Dynamic Router Context Propagation

* Task ID: TASK-CONTEXT-003
  - **Status:** ✅
  - Title: Fix Context Propagation from DynamicParallelRouterStep
  - Description: Ensure that the merged `branch_context` from the internal parallel execution is correctly assigned as the `branch_context` of the `DynamicParallelRouterStep` and that `context_setter` receives the updated context.
  - Acceptance Criteria:
    1. `test_golden_transcript_dynamic_parallel` and `test_golden_transcript_dynamic_parallel_selective` pass with correct `executed_branches` in the final context.
    2. Failures in branch execution properly populate `executed_branches` with failed branch keys.

* Task ID: TASK-STATE-003
  - **Status:** ✅
  - **Note:** Implemented direct simple-step bypass for single-step loops to preserve context updates in place