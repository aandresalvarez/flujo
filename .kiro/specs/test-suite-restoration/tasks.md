
### `tasks.md`

# Engineering Tasks for Flujo Architectural Stabilization

## Introduction

This document lists the specific, actionable engineering tasks required to implement the architectural requirements outlined in `requirements.md`. Each task includes a clear description, a link to the corresponding requirement, and precise acceptance criteria that are directly verifiable through the test suite.

---

### **Prerequisite Task**

*   **Task ID:** `TASK-ENV-001`
*   **Title:** Resolve Test Collection Error for `test_serialization.py`
*   **Status:** Pending
*   **Description:** The test runner is failing to collect tests due to a filename collision between `tests/benchmarks/test_serialization.py` and `tests/utils/test_serialization.py`. This must be resolved to ensure the full test suite runs.
*   **Acceptance Criteria:**
    1.  The `import file mismatch` error is eliminated from the `pytest` output.
    2.  All tests in both `test_serialization.py` files are collected and executed by the test runner.

---

### **Tasks for REQ-STATE-001: State Accumulation and Propagation**

*   **Task ID:** `TASK-STATE-001`
*   **Title:** Fix State Accumulation in Fallback Step Execution
*   **Status:** Completed
*   **Requirement:** `REQ-STATE-001`
*   **Description:** Modify the fallback logic within `_execute_simple_step` in `ultra_executor.py`. When a primary step fails and its fallback is executed, the metrics from the final attempt of the primary step must be added to the metrics of the fallback step's result.
*   **Acceptance Criteria:**
    1.  When a fallback succeeds, the final `StepResult`'s `cost_usd`, `token_counts`, and `latency_s` must be the *sum* of the primary step's final failed attempt and the fallback step's execution.
    2.  The final `attempts` count must be the sum of attempts from the primary and fallback steps.
    3.  The final `StepResult` metadata must contain a `fallback_triggered: True` flag and an `original_error` field preserving the primary step's failure feedback.
    4.  The tests `test_successful_fallback_preserves_metrics` and `test_failed_fallback_accumulates_metrics` must pass.

*   **Task ID:** `TASK-STATE-002`
*   **Title:** Correct Iteration Counting and Termination in `LoopStep` Handler
*   **Status:** Completed
*   **Requirement:** `REQ-STATE-001`
*   **Description:** The `while` loop condition in `_handle_loop_step` is incorrect, causing one more iteration than specified by `max_loops`. Adjust the loop termination logic to strictly adhere to the `max_loops` parameter.
*   **Acceptance Criteria:**
    1.  A `LoopStep` with `max_loops=N` must execute its body *at most* `N` times.
    2.  If the loop terminates because `max_loops` is reached, the final `StepResult.success` must be `False` and its `feedback` must be `"max_loops exceeded"`.
    3.  The `attempts` field of the final `StepResult` must equal the number of iterations performed.
    4.  The tests `test_loop_context_updates_max_loops` and `test_loop_max_loops_reached` must pass.

---

### **Tasks for REQ-CONTEXT-001: Context Integrity**

*   **Task ID:** `TASK-CONTEXT-001`
*   **Title:** Implement Context Merging for `ParallelStep` and `DynamicParallelRouterStep`
*   **Status:** Completed
*   **Requirement:** `REQ-CONTEXT-001`
*   **Description:** Refactor `_handle_parallel_step` and `_handle_dynamic_router_step`. After executing branches in parallel with isolated (deep-copied) contexts, the modifications from *successful* branch contexts must be merged back into the main context according to the step's `merge_strategy`.
*   **Acceptance Criteria:**
    1.  The `branch_context` field of the `ParallelStep`'s final `StepResult` must reflect the merged state from all successful branches.
    2.  Context modifications from failed branches must be discarded and not affect the final merged context.
    3.  The `executed_branches` list in the final context must be correctly populated.
    4.  The tests `test_golden_transcript_dynamic_parallel` and `test_golden_transcript_dynamic_parallel_selective` must pass.

*   **Task ID:** `TASK-CONTEXT-002`
*   **Title:** Implement Context Merging for `ConditionalStep`
*   **Status:** Completed
*   **Requirement:** `REQ-CONTEXT-001`
*   **Description:** Refactor `_handle_conditional_step`. After the selected branch pipeline is executed with an isolated context, its final context state must be correctly merged back into the main pipeline's context.
*   **Acceptance Criteria:**
    1.  The `branch_context` of the `ConditionalStep`'s `StepResult` must contain the modifications made within the executed branch.
    2.  The test `test_regression_conditional_step_context_updates` must pass.

---

### **Tasks for REQ-FAILURE-001: Failure Handling**

*   **Task ID:** `TASK-FAILURE-001`
*   **Title:** Isolate Failure Domains within Step Execution Logic
*   **Status:** Completed
*   **Requirement:** `REQ-FAILURE-001`
*   **Description:** Refactor the `try...except` blocks in `_execute_agent_step` and `_execute_simple_step` to create separate, granular error handling for processors, plugins, validators, and the agent itself. A failure in a processor, plugin, or validator should not be retried and should immediately result in a failed step.
*   **Acceptance Criteria:**
    1.  A failed `ValidationPlugin` must produce a `StepResult` where the `feedback` string is derived from the `PluginOutcome.feedback`.
    2.  A failed `Validator` must produce a `StepResult` where the `feedback` string is derived from the `ValidationResult.feedback`.
    3.  A direct agent execution failure should only be retried up to `max_retries`.
    4.  The tests `test_plugin_validation_failure_with_feedback`, `test_plugin_failure_propagates`, and `test_hybrid_validation.py` failures must pass.

*   **Task ID:** `TASK-FAILURE-002`
*   **Title:** Ensure Failure Propagation from Nested Executions in `LoopStep`
*   **Status:** Pending
*   **Requirement:** `REQ-FAILURE-001`
*   **Description:** Modify `_handle_loop_step` to inspect the `success` flag of the `StepResult` returned from its recursive execution of the loop body. If the body execution fails, the loop must terminate immediately and the parent `LoopStep` must be marked as failed.
*   **Acceptance Criteria:**
    1.  If any step in the `loop_body_pipeline` fails, the `LoopStep` must immediately terminate.
    2.  The final `StepResult` for the `LoopStep` must have `success=False`.
    3.  The `feedback` from the failed inner step must be propagated as the `feedback` for the `LoopStep`.
    4.  The test `test_handle_loop_step_body_step_failures` must pass.