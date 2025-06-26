
Of course. Here is a detailed proposal document for implementing robust `parallel` and `fallback` utilities in the `flujo` library. This document is designed to be added to a project roadmap and outlines the technical design, requirements, and impact of these features.

---

# Proposal: Advanced Pipeline Control Flow for `flujo`

**Date:** June 24, 2025  
**Author:** AI Assistant  
**Status:** Proposed

## 1. Introduction

The `flujo` library currently provides a powerful sequential Pipeline DSL. However, to build truly resilient, performant, and cost-effective AI workflows, the DSL needs to support more advanced control flow patterns. This proposal outlines the robust implementation of two critical features:

1.  **`parallel` Execution:** To run multiple pipeline steps concurrently, drastically reducing latency for I/O-bound tasks.
2.  **`fallback` Logic:** To provide declarative error handling and recovery, making pipelines more resilient to transient failures or model issues.

Implementing these features will significantly enhance the library's capabilities and align it with best practices for production-grade workflow orchestration.

## 2. The `parallel` Utility: Concurrent Step Execution

The `parallel` utility will enable running a set of independent steps at the same time, collecting their results before proceeding.

### 2.1. User-Facing API

The API should be simple and intuitive, integrated directly into the DSL. We propose a `parallel()` function that wraps a series of steps.

```python
from flujo import Flujo, Step, parallel

# Example: Generate three different summaries in parallel
pipeline = (
    Step("fetch_document", fetch_agent)
    >> parallel(
        Step("summary_gpt4o", gpt4o_summary_agent),
        Step("summary_claude", claude_summary_agent),
        Step("summary_gemini", gemini_summary_agent)
    )
    >> Step("select_best", best_summary_selection_agent)
)

runner = Flujo(pipeline)
result = runner.run("URL to a long article")
```

### 2.2. Core Requirements for a Robust Implementation

1.  **True Concurrency:** Steps must execute in parallel, not sequentially.
2.  **Context Safety:** Must prevent race conditions when parallel steps access the shared, mutable `PipelineContext`.
3.  **Defined Data Flow:** The input/output contract must be clear. A single input should be broadcast to all parallel steps, and their multiple outputs should be aggregated into a single, predictable structure for the next step.
4.  **Structured Error Handling:** If one parallel step fails, the entire block should fail gracefully, providing clear diagnostics about which branch caused the issue.
5.  **Metric Aggregation:** `cost_usd` and `token_counts` from all parallel branches must be correctly summed and reflected in the final result.

### 2.3. Proposed Robust Implementation

#### A. Concurrency with `asyncio.TaskGroup`

We will use `asyncio.TaskGroup` (available in Python 3.11+) for managing concurrent execution. This is superior to `asyncio.gather` because it provides **structured concurrency**: if any task within the group fails, all other tasks in the group are immediately cancelled. This prevents orphaned tasks and wasted resources, which is critical for robustness.

#### B. Data Flow: Broadcast-Collect

*   **Input:** The input to the `parallel` block will be **broadcast** to every step within it. Each step receives the same input data.
*   **Output:** The `parallel` block will produce a single output: a `list`, where each element is the `StepResult` from the corresponding parallel step, in the order they were defined. The next step in the pipeline will receive this list of results as its input.

#### C. Context Management: Isolate-and-Merge (The Safest Approach)

To prevent race conditions on the shared `PipelineContext`, we will implement an "isolate-and-merge" strategy. This is the most robust approach to ensure data integrity.

1.  **Isolate:** Before executing the parallel branches, the `flujo` engine will create a **deep copy** of the current `PipelineContext` for each branch. Each step within a parallel branch will operate on its own isolated copy of the context.
2.  **Execute:** The parallel steps run, potentially modifying their own isolated context copies.
3.  **Merge:** After all branches have completed successfully, the engine will merge the changes from each isolated context back into the original `PipelineContext`.
    *   **Default Merge Strategy:** For simple data types (numbers, strings), a "last write wins" strategy will be the default.
    *   **Custom Merge Strategy (Advanced):** For complex conflicts (e.g., merging two lists), the `parallel` utility will accept an optional `on_context_conflict` callable. This function would receive the original context value and the values from each branch, allowing the user to define custom merge logic.

This strategy guarantees that parallel steps cannot interfere with each other, providing deterministic behavior.

#### D. Error Handling and Result Structure

*   A new `ParallelStep` class will be introduced. Its `StepResult` will be a composite object.
*   If any branch fails, the `asyncio.TaskGroup` will cancel the others. The `ParallelStep` itself will be marked as `success=False`.
*   The `output` of the failed `ParallelStep` will be the list of results from branches that completed or were cancelled, and the `feedback` attribute will contain the exception from the branch that failed. This provides complete traceability for debugging.

## 3. The `fallback` Utility: Resilient Error Handling

The `fallback` utility provides a declarative way to define a recovery path if a step fails.

### 3.1. User-Facing API

We propose a chainable method on the `Step` object, which is expressive and easy to read.

```python
from flujo import Step

# Try the expensive model first, fall back to a cheaper one on failure.
main_agent = Step("main_attempt", gpt4_agent)
fallback_agent = Step("backup_attempt", gemini_flash_agent)

resilient_step = main_agent.fallback(fallback_agent)

pipeline = resilient_step >> Step("final_processing", processing_agent)
```

### 3.2. Core Requirements for a Robust Implementation

1.  **Conditional Execution:** The fallback step must *only* execute if the primary step fails (i.e., its `StepResult.success` is `False`).
2.  **Data Integrity:** The fallback step must receive the *original input* that was passed to the failed primary step, not the failed output.
3.  **Transparent Output:** If the fallback succeeds, its output should become the output of the entire composite step, allowing the pipeline to continue seamlessly.
4.  **Full Traceability:** The final `StepResult` must clearly indicate that a fallback occurred and should contain the result of the failed primary attempt for debugging and analysis.

### 3.3. Proposed Robust Implementation

#### A. DSL and Engine Integration

1.  The `Step` class will be modified to include an optional `fallback_step: Optional[Step]` attribute. The `.fallback()` method will set this attribute and return `self` to allow chaining.
2.  The `flujo` engine's step execution logic will be wrapped in a `try...except` block.
3.  If the primary step succeeds, the engine proceeds as normal.
4.  If the primary step fails, the engine will:
    a. Log the failure of the primary step.
    b. Check if a `fallback_step` is defined.
    c. If so, execute the `fallback_step`, passing it the **original input data and context** of the primary step.

#### B. Result and History Management

To ensure full traceability, the `StepResult` for a step that successfully used a fallback will be a composite object:

*   `success`: `True` (because the fallback succeeded).
*   `output`: The output from the successful fallback step.
*   `name`: The name of the primary step.
*   `metadata_`: This dictionary will be populated with crucial information:
    *   `fallback_triggered`: `True`
    *   `primary_attempt_result`: The full `StepResult` object from the failed primary attempt.
    *   `fallback_attempt_result`: The full `StepResult` from the successful fallback attempt.

This structure allows the pipeline to continue as if it succeeded, while preserving the complete history of the failure and recovery for logging, telemetry, and debugging.

## 4. Impact Analysis

*   **`flujo.domain`:** Requires adding a `ParallelStep` class and modifying the `Step` class to support fallbacks. `StepResult` may need a `metadata_` field for fallback history.
*   **`flujo.application.flujo_engine`:** This will see the most significant changes to accommodate the execution logic for both `parallel` (with context management) and `fallback` steps.
*   **`flujo.documentation`:** New guides and cookbook recipes will be essential to explain these powerful features and their nuances (especially context management in parallel blocks).
*   **Backward Compatibility:** These changes are purely additive. Existing pipelines will continue to function without any modification.

## 5. Conclusion

Implementing `parallel` and `fallback` utilities with the robust designs proposed here will elevate `flujo` from a sequential task runner to a mature and resilient workflow orchestration framework. These features are essential for building high-performance, production-ready AI systems and should be prioritized for inclusion in the project roadmap.