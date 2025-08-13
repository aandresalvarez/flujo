  Here is the Functional Specification Document for the next high-priority item on our roadmap: **"Formalize and Test the Trace Contract (FSD-011)"**.

This FSD directly addresses **Gap #8** from the architectural review, moving Flujo's observability from an implementation detail to a formal, guaranteed contract. This is essential for building reliable debugging, monitoring, and auditing tools on top of the framework.

---

## **Functional Specification Document: Formal Trace Contract (FSD-011)**

**Author:** Alvaro
**Date:** 2023-10-27
**Status:** Proposed
**JIRA/Ticket:** FLUJO-126 (Example Ticket ID)
**Depends On:** FSD-008 (Typed Step Outcomes), FSD-009 (First-Class Quotas)

### **1. Overview**

This document specifies the design for a canonical trace model for Flujo executions. Currently, telemetry spans are emitted by various components (`TraceManager`, `OTelHook`), but the structure, attributes, and events within these spans are an implicit, undocumented contract. This makes it difficult to build reliable debugging tools, dashboards, and automated analyses, as the telemetry data may change without notice.

This FSD proposes defining a formal **Trace Contract** that specifies the required structure and metadata for spans and events generated during a pipeline run. This contract will be enforced through a new suite of "golden trace" tests, which will validate that the telemetry output of key pipeline patterns matches a known-good, versioned schema.

This change promotes the **Axiom #4 (Auditability)** by making the observability data a reliable, first-class output of the system.

### **2. Rationale & Goals**

#### **2.1. Problems with the Current Approach**

*   **Implicit Contract:** The structure of telemetry spans is an implementation detail. It can change unintentionally, breaking any downstream tooling that depends on it.
*   **Poor Debuggability:** Key decision-making context (e.g., why a fallback was triggered, what the quota was before a step) is not consistently captured in traces, making it difficult to debug complex failures.
*   **Aspirational Auditability:** Without a guaranteed trace structure, the claim of being "auditable" is weak. It's impossible to build automated audit tools if the data format is unstable.
*   **Inconsistent Tooling:** Building tools like `flujo lens trace` is difficult because the code has to make assumptions about the shape of the trace data stored in the backend.

#### **2.2. Goals of this Refactor**

*   **Establish a Canonical Trace Schema:** Define a stable, versioned contract for the structure of spans and events.
*   **Enrich Trace Data:** Ensure that all critical execution decisions (retries, fallbacks, quota reservations, policy choices) are recorded as structured attributes or events in the trace.
*   **Guarantee Auditability:** Make the trace a reliable, append-only ledger of execution that can be used for automated analysis and replay.
*   **Enable Robust Tooling:** Provide a stable foundation for building and maintaining debugging tools like `flujo lens` and external monitoring dashboards.
*   **Enforce the Contract with Tests:** Create a new category of "golden trace" tests that prevent regressions in telemetry output.

### **3. Functional Requirements & Design**

#### **Task 3.1: Define the Trace Contract Schema**

A formal schema will be documented, defining the required spans, attributes, and events.

*   **Location:** This will be a new documentation file: `docs/reference/trace_contract.md`.
*   **Implementation Details:** The document will specify the following:
    *   **Root Span (`pipeline_run`):**
        *   **Attributes:** `flujo.run_id`, `flujo.pipeline.name`, `flujo.pipeline.version`, `flujo.input`, `flujo.budget.initial_cost_usd`, `flujo.budget.initial_tokens`.
    *   **Step Span (one per step execution):**
        *   **Span Name:** Must be the `step.name`.
        *   **Attributes:**
            *   `flujo.step.id`: A unique ID for this specific execution of the step.
            *   `flujo.step.type`: The class name of the step (e.g., `ParallelStep`, `LoopStep`).
            *   `flujo.step.policy`: The name of the execution policy used (e.g., `DefaultParallelStepExecutor`).
            *   `flujo.attempt_number`: The current retry/attempt number for this step.
            *   `flujo.cache.hit`: `true` or `false`.
            *   `flujo.budget.quota_before_usd`, `flujo.budget.quota_before_tokens`: Quota *before* reservation.
            *   `flujo.budget.estimate_cost_usd`, `flujo.budget.estimate_tokens`: The estimated usage.
            *   `flujo.budget.actual_cost_usd`, `flujo.budget.actual_tokens`: The final actual usage.
    *   **Standard Events (within a Step Span):**
        *   `flujo.retry`: Triggered on a retry attempt. Attributes: `reason`, `delay_seconds`.
        *   `flujo.fallback.triggered`: Triggered when a fallback step is executed. Attributes: `original_error`.
        *   `flujo.paused`: Triggered when a step pauses for HITL. Attributes: `message`.
        *   `flujo.resumed`: Triggered on resumption. Attributes: `human_input`.
        *   `flujo.budget.violation`: Triggered on a `UsageLimitExceededError`. Attributes: `limit_type` (`cost` or `tokens`), `limit_value`, `actual_value`.
*   **Acceptance Criteria & Testing:**
    *   This task is documentation-only. Completion is marked by the creation and approval of the `trace_contract.md` file.

#### **Task 3.2: Update Telemetry Emitters to Conform to the Contract**

The `TraceManager` and `OTelHook` must be updated to emit spans and events that match the new contract.

*   **Location:** `flujo/tracing/manager.py` and `flujo/telemetry/otel_hook.py`.
*   **Implementation Details:**
    *   Modify the `_handle_*` methods in both classes.
    *   These methods will now need to access more detailed information from the `ExecutionFrame` or the `StepResult`. This information (like quota, attempt number) must be added to the hook payloads in `flujo/domain/events.py`.
    *   For example, the `PreStepPayload` should be augmented with `attempt_number` and `quota_before`. The `PostStepPayload` should be augmented with `actual_cost` and `actual_tokens`.
*   **Acceptance Criteria & Testing (`make test-fast`)**
    *   **Unit Tests:** In `tests/tracing/test_manager.py`:
        *   Create mock `HookPayload` objects containing the new, richer data.
        *   Call the `TraceManager.hook` method.
        *   Inspect the generated `Span` objects to assert that all required attributes from the Trace Contract have been correctly set.

#### **Task 3.3: Create a "Golden Trace" Testing Framework**

A new testing harness will be created to validate the end-to-end trace output against a known-good "golden" file.

*   **Location:** A new test directory: `tests/golden_traces/`.
*   **Implementation Details:**
    1.  **Create a Test Pipeline:** Define a complex pipeline in `tests/golden_traces/test_pipeline.py` that uses a `LoopStep`, a `ParallelStep`, a `ConditionalStep`, a step that fails and triggers a `fallback`, and a step that gets retried.
    2.  **Create a Trace Capturing Hook:** In `tests/golden_traces/utils.py`, create a simple hook that appends a serializable dictionary representation of every emitted span to a list.
    3.  **Generate the Golden File:**
        *   Run the test pipeline once with the capturing hook.
        *   Serialize the captured list of spans to a JSON file: `tests/golden_traces/golden_trace_v1.json`.
        *   Manually review this file to ensure it is correct and complete according to the contract. This file is now the "golden" standard.
*   **Acceptance Criteria & Testing (`make test-fast`)**
    *   **Unit Tests:** The utilities for capturing and serializing traces should have their own unit tests.

#### **Task 3.4: Implement and Run Golden Trace Tests**

The final step is to create a test that enforces the contract.

*   **Location:** `tests/golden_traces/test_golden_traces.py`.
*   **Implementation Details:**
    *   Create a test function `test_pipeline_trace_matches_golden_file`.
    *   This test will run the same test pipeline from Task 3.3 with the trace capturing hook.
    *   It will load the `golden_trace_v1.json` file.
    *   It will then perform a deep comparison between the newly generated trace and the golden trace.
        *   **Important:** The comparison logic must ignore dynamic values like `span_id`, `run_id`, and timestamps. It should only compare the structure, span names, parent-child relationships, and key attributes.
*   **Acceptance Criteria & Testing (`make all`)**
    *   **E2E Test:** The `test_pipeline_trace_matches_golden_file` test must pass.
    *   **Regression:** If a developer makes a change that unintentionally alters the telemetry output, this test will fail, forcing them to either fix their change or consciously update the golden file and the Trace Contract version.

### **4. Rollout and Regression Plan**

1.  **Branching:** Work will be done on a dedicated feature branch (e.g., `feature/FSD-011-trace-contract`).
2.  **Implementation Order:**
    *   Task 3.1 (Documentation) should be done first to establish the target.
    *   Task 3.2 (Emitter Refactoring) and its unit tests.
    *   Task 3.3 and 3.4 (Golden Trace Framework and Tests) should be done last, as they validate the entire implementation.
3.  **Testing Strategy:**
    *   `make test-fast`: Will run the unit tests for the updated emitters and the golden trace utilities.
    *   `make all`: Will run the final end-to-end golden trace test, providing the ultimate validation for this FSD. The full existing regression suite must also pass.
4.  **Code Review:** Required. Reviewers should check the implementation against the `trace_contract.md` file to ensure full compliance.
5.  **Merge:** Merge to the main branch after all tests pass.

### **5. Risks and Mitigation**

*   **Risk:** The Trace Contract is too rigid and creates friction for future development.
    *   **Mitigation:** The contract should versioned. Breaking changes to the trace output will require incrementing the version and generating a new golden file, making the change explicit and intentional.
*   **Risk:** The "golden trace" tests could be flaky due to non-deterministic ordering in parallel execution.
    *   **Mitigation:** The comparison logic in the golden trace test must be robust. It should not rely on the order of sibling spans (e.g., in a `ParallelStep`). It can convert the list of spans into a dictionary keyed by `span_id` and validate parent-child links and attributes without enforcing a specific sibling order.
*   **Risk:** Adding all the required attributes to the hook payloads could make them bloated.
    *   **Mitigation:** This is an acceptable trade-off. The purpose of telemetry is to provide rich context, and the payloads are internal to the framework. The benefits of complete auditability outweigh the cost of slightly larger in-memory objects during hook dispatch.