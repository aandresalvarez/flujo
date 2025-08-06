---

### `tasks.md`

# Engineering Tasks for Flujo Architectural Refinement (Phase 2)

## Introduction

This document details the specific tasks needed to implement the Phase 2 architectural requirements. These tasks focus on refining the `ExecutorCore`'s handling of failure semantics, state propagation at recursive boundaries, and ensuring the completion of the observability pipeline.

---

### **Tasks for REQ-STATE-002: Consistent State Propagation**

*   **Task ID:** `TASK-STATE-003`
*   **Title:** Ensure `LoopStep` Output Mapper is Always Invoked on Termination
*   **Requirement:** `REQ-STATE-002`
*   **Description:** Refactor the `_handle_loop_step` method in `ultra_executor.py`. Add logic to ensure that the `loop_output_mapper` callable is invoked on the loop's final internal output, *especially* when the loop terminates by reaching its `max_loops` limit.
*   **Acceptance Criteria:**
    1.  When a `refine_until` loop (a specialized `LoopStep`) terminates due to `max_refinements`, its final output must be an instance of `RefinementCheck`, as produced by its output mapper.
    2.  The test `test_golden_transcript_refine_max_iterations` must pass.

---

### **Tasks for REQ-CONTEXT-002: Context Integrity**

*   **Task ID:** `TASK-CONTEXT-003`
*   **Title:** Fix Context Propagation from `DynamicParallelRouterStep`
*   **Requirement:** `REQ-CONTEXT-002`
*   **Description:** In `_handle_dynamic_router_step` within `ultra_executor.py`, after the internal, temporary `ParallelStep` completes, its final `branch_context` (which contains the merged results) must be correctly assigned as the `branch_context` of the `DynamicParallelRouterStep`'s own `StepResult`.
*   **Acceptance Criteria:**
    1.  Context modifications made within the dynamically selected parallel branches must be reflected in the final pipeline context.
    2.  The `executed_branches` list in the final context must be correctly populated after a `DynamicParallelRouterStep` run.
    3.  The tests `test_golden_transcript_dynamic_parallel` and `test_golden_transcript_dynamic_parallel_selective` must pass.

*   **Task ID:** `TASK-CONTEXT-004`
*   **Title:** Optimize High-Load Context Merging
*   **Requirement:** `REQ-CONTEXT-002`
*   **Description:** Analyze and optimize the performance of the `safe_merge_context_updates` function in `flujo/utils/context.py`. Investigate replacing expensive deep-copy or full model validation operations with more efficient, delta-based updates, especially within loops. The goal is to reduce the overhead of context merging in high-iteration scenarios.
*   **Acceptance Criteria:**
    1.  The `test_regression_performance_under_load` integration test, which simulates high-frequency context updates in a loop, must pass without timing out or failing its final assertion.

---

### **Tasks for REQ-FAILURE-002: Failure Domain Semantics**

*   **Task ID:** `TASK-FAILURE-003`
*   **Title:** Correct Retry Logic for Validator Failures
*   **Requirement:** `REQ-FAILURE-002`
*   **Description:** The test `test_validator_failure_triggers_retry` incorrectly assumes that a validation failure should trigger a retry. This assumption is architecturally incorrect. Modify the test to assert that a validation failure results in *exactly one* attempt and an immediate failure of the step. The `ExecutorCore`'s current behavior of failing fast is correct and should be preserved.
*   **Acceptance Criteria:**
    1.  The test `test_validator_failure_triggers_retry` is updated to assert that `result.attempts == 1`.
    2.  The test `test_validator_failure_triggers_retry` must pass with the corrected assertion.

*   **Task ID:** `TASK-FAILURE-004`
*   **Title:** Improve Feedback Propagation from Failed Loop Body
*   **Requirement:** `REQ-FAILURE-002`
*   **Description:** In `_handle_loop_step`, when a loop terminates because its body fails, the feedback message for the parent `LoopStep`'s `StepResult` should be standardized. It should clearly state that the loop failed and include the specific feedback from the failed inner step.
*   **Acceptance Criteria:**
    1.  The final `feedback` string for a failed `LoopStep` must be in the format: `"Loop body failed: [Original Feedback from Inner Step]"`.
    2.  The tests `test_loop_step_body_failure_with_robust_exit_condition` and `test_loop_step_body_failure_causing_exit_condition_error` must pass.

---

### **Tasks for REQ-OBSERVABILITY-001: Trace Persistence**

*   **Task ID:** `TASK-OBSERVABILITY-001`
*   **Title:** Implement End-to-End Trace Persistence and Retrieval
*   **Requirement:** `REQ-OBSERVABILITY-001`
*   **Description:** In `ExecutionManager`, at the end of a pipeline run, extract the `trace_tree` from the `PipelineResult` object. If the trace exists, call the `StateManager`'s `record_run_end` method, which should in turn call the `StateBackend`'s `save_trace` method to persist it. Ensure `SQLiteBackend.get_trace` can correctly retrieve and reconstruct the trace.
*   **Acceptance Criteria:**
    1.  After a pipeline run with tracing enabled, a call to `backend.get_trace(run_id)` must return a non-None, valid trace tree structure.
    2.  The `flujo lens trace <run_id>` CLI command must successfully display the trace.
    3.  All tests in `tests/integration/test_fsd_12_tracing_complete.py` must pass.