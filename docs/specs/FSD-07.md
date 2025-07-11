### **Functional Specification Document: FSD-07**

**Title:** Enhanced State Management API and Serialization
* **Author:** AI Assistant
* **Status:** Proposed
* **Priority:** P1 - High
* **Date:** 2024-10-27
* **Version:** 1.0

---

### **1. Overview**

This document specifies critical improvements to Flujo's state management and data serialization APIs to improve robustness and developer experience. Currently, initiating resumable pipeline runs is unintuitive, and the state persistence mechanism fails to serialize common application data structures like custom Pydantic models, leading to runtime errors.

This initiative will deliver a more powerful and developer-friendly state management system by implementing three core enhancements:

1. **Introduce a first-class `run_id` parameter** to the `Flujo.run_async()` and `Flujo.run()` methods, creating an explicit and intuitive API for controlling stateful executions.
2. **Enhance Flujo's `orjson` serialization layer** to automatically handle nested, user-defined Pydantic models and other common Python types (e.g., `datetime`), resolving `TypeError` exceptions during state persistence.
3. **Provide an extensible configuration point** for advanced users to supply custom serialization logic.

These changes will address major friction points in the framework, making the creation of durable, long-running agentic workflows significantly more reliable and straightforward for all users.

### **2. Problem Statement**

Developers building stateful applications with Flujo currently face two significant hurdles:

1. **Implicit and Unintuitive `run_id` Management:** To make a pipeline run resumable, a developer must discover and use an implicit convention: passing a `run_id` within the `initial_context_data` dictionary, which then must be correctly mapped to a `run_id` field in a custom `PipelineContext` model. The primary execution methods, `run()` and `run_async()`, offer no direct parameter for this purpose. This lack of an explicit API makes a critical feature difficult to use and prone to implementation errors.
2. **Brittle State Serialization:** Flujo's state backends use `orjson` for serialization but lack a comprehensive `default` handler. Consequently, when a developer stores a standard application object—like an instance of a custom Pydantic model—in the `PipelineContext` (e.g., in the `scratchpad`), the serialization process fails with a `TypeError`. This forces developers to manually convert all complex objects to JSON-native types before they can be persisted, adding significant boilerplate and complexity to their application logic.

### **3. Functional Requirements (FR)**

| ID | Requirement | Justification |
| :--- | :--- | :--- |
| FR-17 | The `Flujo.run()` and `Flujo.run_async()` methods **SHALL** accept an optional `run_id: str` keyword argument. | Provides an explicit, discoverable API for initiating or resuming workflows. |
| FR-17a | If a `run_id` is provided, the Flujo runner **SHALL** automatically attempt to load and resume the workflow state associated with that ID. | Automates resumable execution. |
| FR-17b | If a `run_id` is provided, it **SHALL** populate the `run_id` field in the `PipelineContext` object at the start of the run. | Simplifies context management. |
| FR-18 | Flujo's `StateBackend` implementations **SHALL** automatically serialize nested, user-defined Pydantic models and other common Python data types when persisting state. | Fixes serialization failures. |
| FR-18a | The serialization process **SHALL** use `.model_dump(mode='json')` for Pydantic models and ISO format for `datetime` objects. | Ensures fidelity and compatibility. |
| FR-19 | `StateBackend` initializers **SHALL** accept an optional `serializer_default: Callable` argument. | Provides an extension point for custom serialization logic. |

### **4. Technical Design & Implementation Plan**

1. **Update Runner API** – Add `run_id` parameters to `Flujo.run` and `run_async`. When provided, this ID is used to load and save workflow state through `StateManager`.
2. **Default Serialization Handler** – Implement `flujo.state.serialization.flujo_default_serializer` that handles Pydantic models, `datetime`, and `Enum` values. All backends use this handler by default when calling `orjson.dumps`.
3. **Custom Serializer Hook** – `StateBackend.__init__` accepts a `serializer_default` callable allowing users to override the default handler.

### **5. Acceptance Criteria**

* **AC-17** – `Flujo.run` and `Flujo.run_async` accept `run_id`. Resuming a run with the same ID correctly continues execution.
* **AC-18** – Nested Pydantic models placed in the pipeline context can be persisted to the SQLite backend without errors.
* **AC-19** – Passing a custom `serializer_default` to a backend is used during serialization.

### **6. Risks & Mitigation**

* **State Schema Changes** – Persisted state schemas may evolve; migrations must be documented.
* **Serialization Coverage** – Exotic types may still raise `TypeError`. Users can provide a custom handler via `serializer_default`.
