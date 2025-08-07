---

### `tasks.md`

# Engineering Tasks for Flujo Architectural Refinement (Phase 2)

## Introduction

This document details the specific tasks needed to implement the Phase 2 architectural requirements. These tasks focus on refining the `ExecutorCore`'s handling of failure semantics, state propagation at recursive boundaries, and ensuring the completion of the observability pipeline.

---

### **Tasks for REQ-STATE-002: Consistent State Propagation**

*   **Task ID:** `TASK-STATE-003`
*   **Status:** ✅
*   **Title:** Ensure `LoopStep` Output Mapper is Always Invoked on Termination
*   **Requirement:** `REQ-STATE-002`
*   **Description:** Refactor the `_handle_loop_step` method in `ultra_executor.py`. Add logic to ensure that the `loop_output_mapper` callable is invoked on the loop's final internal output, *especially* when the loop terminates by reaching its `max_loops` limit.
*   **Acceptance Criteria:**
    1.  When a `refine_until` loop (a specialized `LoopStep`) terminates due to `max_refinements`, its final output must be an instance of `RefinementCheck`, as produced by its output mapper.
    2.  The test `test_golden_transcript_refine_max_iterations` must pass.

*   **Task ID:** `TASK-STATE-004`
*   **Status:** ✅
*   **Title:** Full simple-step bypass for LoopStep
*   **Requirement:** `REQ-STATE-002`
*   **Description:** Refactor `_handle_loop_step` to detect single-step loop bodies and directly invoke `_execute_simple_step` for each iteration—ensuring all simple-step policies (context mutations, caching, plugins, validators, fallback, telemetry) apply on the same `PipelineContext` instance without isolation/merge overhead.
*   **Acceptance Criteria:**
    1.  All tests in `tests/application/core/test_executor_core_loop_step_migration.py` and `tests/integration/test_loop_step_execution.py` pass without further changes.
    2.  Loop performance under load remains within acceptable thresholds.
    3.  No merge or deep-copy operations occur during loop iterations (verified by profiling).

    #### Subtasks for `TASK-STATE-004`
    1. **TASK-STATE-004a**: Integrate prompt and output processors & agent invocation
       * Extract the apply_prompt/apply_output and agent_runner.run logic from `_execute_simple_step` into `_execute_simple_loop_body`, ensuring correct context passing.
       * **Status:** ✅
    2. **TASK-STATE-004b**: Add plugin runner & validator runner support
       * Copy the plugin_runner.run_plugins and validator_runner.validate loops with retry logic into the helper.
       * **Status:** ✅
    3. **TASK-STATE-004c**: Add caching & usage limit enforcement
       * Integrate cache_backend.get/put and usage_meter.snapshot/guard into iterative execution, accumulating cost/tokens.
       * **Status:** ✅
    4. **TASK-STATE-004d**: Embed fallback semantics & retry logic
       * Support fallback_step recursion, infinite fallback detection, and proper feedback formatting within loops.
       * **Status:** ✅
    5. **TASK-STATE-004e**: Wire iteration-input and initial-input mappers
       * Integrate `initial_input_to_loop_body_mapper` and `iteration_input_mapper` from `loop_step` config.
       * **Status:** ✅
    6. **TASK-STATE-004f**: Support loop_output_mapper & finalize result
       * Ensure `loop_output_mapper` is applied after all iterations, matching DSL semantics.
       * **Status:** ✅
    7. **TASK-STATE-004g**: Maintain context_setter & telemetry hooks
       * Invoke `context_setter` if provided, and preserve instrumentation spans to match full pipeline behavior.
       * **Status:** ✅
    8. **TASK-STATE-004h**: Validate against migration and execution tests
       * Run `test_executor_core_loop_step_migration.py` and `test_loop_step_execution.py` to certify correctness and performance.
       * **Status:** ✅q34d1     
    9. **TASK-STATE-004i**: Restore multi-step loop handler
       * Re-inject the original multi-step loop handler implementation below the single-step bypass logic.
       * **Status:** ✅

---

### **Tasks for REQ-CONTEXT-002: Context Integrity**

*   **Task ID:** `TASK-CONTEXT-003`
*   **Status:** ✅
*   **Title:** Fix Context Propagation from `DynamicParallelRouterStep`
*   **Requirement:** `REQ-CONTEXT-002`
*   **Description:** In `_handle_dynamic_router_step` within `ultra_executor.py`, after the internal, temporary `ParallelStep` completes, its final `branch_context` (which contains the merged results) must be correctly assigned as the `branch_context` of the `DynamicParallelRouterStep`'s own `StepResult`.
*   **Acceptance Criteria:**
    1.  Context modifications made within the dynamically selected parallel branches must be reflected in the final pipeline context.
    2.  The `executed_branches` list in the final context must be correctly populated after a `DynamicParallelRouterStep` run.
    3.  The tests `test_golden_transcript_dynamic_parallel` and `test_golden_transcript_dynamic_parallel_selective` must pass.

*   **Task ID:** `TASK-CONTEXT-004`
*   **Status:** ✅
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

---

## Current Status Summary

### ✅ Completed Tasks
- **TASK-STATE-003**: LoopStep output mapper invocation ✅
- **TASK-CONTEXT-003**: DynamicParallelRouterStep context propagation ✅
- **TASK-CONTEXT-004**: High-load context merging optimization ✅
- **TASK-STATE-004**: Full simple-step bypass for LoopStep ✅
  - Single-step bypass implemented and working ✅
  - Multi-step handler restored ✅
- **TASK-FAILURE-003**: Correct retry logic for validator failures ✅
- **TASK-FAILURE-004**: Improve feedback propagation from failed loop body ✅
- **TASK-OBSERVABILITY-001**: Implement end-to-end trace persistence and retrieval ✅

### 📋 Pending Tasks
- **TASK-STABILIZE-001**: Stabilize Remaining Test Suite Failures
  * **Subtasks:**
    - TASK-STABILIZE-001a: Fix `ExecutorCore._execute_simple_step` retry and plugin validation semantics ✅
    - TASK-STABILIZE-001b: Correct `ExecutorCore` fallback metrics and feedback propagation ✅
    - TASK-STABILIZE-001c: Fix CLI runner end-of-run persistence tests
    - TASK-STABILIZE-001d: Repair LoopStep multi-step scenarios in CLI runner
    - TASK-STABILIZE-001e: Extract unified loop helper `_execute_loop` to centralize all loop logic from first principles ⏳ In Progress
    - TASK-STABILIZE-001f: Refactor `_handle_loop_step` to delegate to the unified `_execute_loop` helper ✅ Completed
    - TASK-STABILIZE-001g: Write unit tests for `_execute_loop` validating iteration history, mappers, and metadata ✅ Completed
    - TASK-STABILIZE-001h: Update CLI `run` command JSON mode to serialize and output the full `PipelineResult` including nested loops ⏳ In Progress
    - TASK-STABILIZE-001i: Re-run the full integration suite and iteratively resolve any regressions introduced by the refactor ⏳ In Progress
    - TASK-STABILIZE-001j: Wire `initial_input_to_loop_body_mapper` semantics ⏳ In Progress
    - TASK-STABILIZE-001k: Wire `iteration_input_mapper` semantics ✅ Completed

## Next Steps

1. **Immediate**: Complete all pending subtasks under **TASK-STABILIZE-001** to stabilize the test suite failures
2. **Secondary**: Re-run the full integration suite and iteratively resolve any remaining regressions (TASK-STABILIZE-001i)
3. **Final**: Implement **TASK-OBSERVABILITY-001** for end-to-end trace persistence and retrieval

## Notes

- The single-step bypass optimization is working correctly and provides significant performance improvements
- Multi-step loops were accidentally removed during the refactoring and need to be restored
- The overall test suite shows 196 failed tests, mostly related to multi-step loop functionality
- Context merging optimizations are working well for single-step loops