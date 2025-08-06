 

### `requirements.md`

# Architectural Stabilization Requirements for Flujo

## Introduction

This document outlines the high-level architectural requirements necessary to stabilize the Flujo framework. The recent suite of test failures indicates a systematic drift between the intended design of the **Declarative Shell** and the implementation details of the **Execution Core**. These requirements aim to realign the implementation with the core architectural principles of robust state management, context integrity, and granular failure handling.

---

### **REQ-STATE-001: Guaranteed State Accumulation and Propagation**

*   **Statement:** The state and metrics (e.g., `cost_usd`, `token_counts`, `attempts`, `latency_s`) of any nested or sequential execution must be correctly accumulated and propagated upwards through the recursive execution call stack. State from a subsequent execution (e.g., a fallback step) must augment, not replace, the state of the preceding execution.

*   **Rationale:** The current implementation loses the state of failed primary steps when a fallback is triggered and incorrectly manages iteration counts in loops. This violates the principle of exhaustive accounting and leads to incorrect metrics and flawed control flow. This requirement ensures that the final `StepResult` of any composite step provides a complete and accurate representation of all computational work performed within it.

### **REQ-CONTEXT-001: Absolute Context Integrity Across Control Flow Boundaries**

*   **Statement:** Modifications made to a context object within an isolated execution branch (e.g., in a `ParallelStep` or `ConditionalStep`) must be reliably merged back into the main pipeline context according to the step's configured `merge_strategy`. Contexts for parallel branches must be completely isolated to prevent race conditions and unintended side effects.

*   **Rationale:** The architectural principle of algebraic closure depends on `Step` objects being self-contained yet capable of predictably interacting with a shared state. Test failures show that context modifications within branches are being lost, breaking this contract. This requirement enforces that the `ExecutorCore` correctly manages the lifecycle of contexts: isolate, execute, and merge.

### **REQ-FAILURE-001: Granular Failure Domain Handling and Propagation**

*   **Statement:** The `ExecutorCore` must distinguish between failures originating from different domains (Agent, Plugin, Validator, Processor). A failure in any nested execution (e.g., a step within a `LoopStep`'s body) must cause the parent composite step to be marked as failed, and the specific feedback from the original failure source must be preserved and propagated.

*   **Rationale:** Test failures demonstrate that the execution engine is conflating failure domains, leading to generic, unhelpful feedback. Furthermore, failures in nested pipelines are not correctly causing their parent containers to fail. This violates the principles of observability and predictable failure propagation. This requirement ensures that failures are handled with precision, providing clear diagnostics and ensuring that the state of the pipeline accurately reflects the outcome of its constituent parts.

---
