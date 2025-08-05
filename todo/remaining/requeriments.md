 
## FSD-13: Fortifying the Execution Core - Requirements

This document outlines the concrete functional and non-functional requirements for the FSD-13 initiative, which aims to complete the migration of execution logic into the `ExecutorCore` and harden it for production use. All requirements must be met to consider this initiative complete.

---

### **1. Core Execution Logic Requirements**

These requirements focus on ensuring the internal execution flow is robust, consistent, and correct.

| ID | Requirement | Rationale | Acceptance Criteria |
| :-- | :--- | :--- | :--- |
| **REQ-CORE-001** | The `ExecutorCore.execute` method and all its internal recursive callers MUST use a single, unified `ExecutionFrame` data object to pass operational state. | To eliminate bugs from inconsistent parameter passing in recursive calls and to create a formal, type-safe internal contract. | 1. An `ExecutionFrame` dataclass is defined. <br> 2. The `ExecutorCore.execute` signature is refactored to accept only an `ExecutionFrame`. <br> 3. All recursive calls within `_handle_..._step` methods construct and pass a valid `ExecutionFrame`. |
| **REQ-CORE-002** | The `flujo/application/core/step_logic.py` module MUST be completely removed from the codebase. | To complete the FSD-12 migration and establish `ExecutorCore` as the single source of truth for execution logic. | 1. The file `flujo/application/core/step_logic.py` is deleted. <br> 2. A global search for "step_logic" yields zero import statements or references in the codebase. |
| **REQ-CORE-003** | The `StepExecutor` type alias MUST be migrated from `step_logic.py` to `ultra_executor.py`. | To co-locate the type definition with its primary consumer, the `ExecutorCore`. | 1. The `StepExecutor` type alias is defined in `flujo/application/core/ultra_executor.py`. <br> 2. All files that previously imported it from `step_logic` now import it from `ultra_executor`. |

---

### **2. Context Management Requirements**

These requirements ensure the `PipelineContext` is handled reliably throughout its lifecycle.

| ID | Requirement | Rationale | Acceptance Criteria |
| :-- | :--- | :--- | :--- |
| **REQ-CTX-001** | The top-level `Flujo` runner (`flujo/application/runner.py`) MUST be the sole component responsible for the initial creation of the `PipelineContext`. | To establish a clear chain of responsibility and ensure the context is always created in a valid state. | 1. `ExecutionManager` and `ExecutorCore` no longer contain logic to create a default `PipelineContext`. <br> 2. The `Flujo` runner's `run` and `run_async` methods instantiate the context. |
| **REQ-CTX-002** | The `Flujo` runner MUST correctly populate the `initial_prompt` field of the `PipelineContext` upon creation, using the input provided to the `run`/`run_async` method. | To fix bugs where nested or complex pipelines fail due to a missing `initial_prompt`. | 1. A unit test confirms that a `PipelineContext` created by the `Flujo` runner has its `initial_prompt` field correctly set. <br> 2. An integration test running a nested pipeline (via `as_step`) confirms the sub-pipeline's context also has a valid `initial_prompt`. |

---

### **3. Serialization and State Management Requirements**

These requirements focus on making the state persistence layer robust and extensible.

| ID | Requirement | Rationale | Acceptance Criteria |
| :-- | :--- | :--- | :--- |
| **REQ-SER-001** | The framework MUST provide a public API for registering custom serializers and deserializers for user-defined types. | To decouple the core engine from specific data types and allow users to integrate any custom object with Flujo's caching and state persistence layers. | 1. `register_custom_serializer` and `register_custom_deserializer` functions are available in `flujo.utils.serialization`. <br> 2. The `StateManager` and `ICacheBackend` implementations correctly use these registered functions when handling unknown types. |
| **REQ-SER-002** | Test suites MUST be updated to use the custom serialization registry for mock objects (e.g., `UsageResponse`, `MockImageResult`). | To ensure test objects can be correctly persisted and cached, fixing related test failures and validating the new serialization system. | 1. A test fixture registers serializers for all custom mock objects used in tests. <br> 2. All tests involving caching or state persistence of these mock objects pass. |

---

### **4. Resilience and Usage Governance Requirements**

These requirements ensure Flujo's production-readiness features function correctly.

| ID | Requirement | Rationale | Acceptance Criteria |
| :-- | :--- | :--- | :--- |
| **REQ-GOV-001** | The `ExecutionManager` MUST check usage limits immediately after every individual step execution completes. | To ensure that runaway costs or token usage are stopped as early as possible. | 1. The `ExecutionManager`'s execution loop calls the `UsageGovernor` immediately after a `StepResult` is received from the `StepCoordinator`. |
| **REQ-GOV-002** | When a usage limit is breached, the `UsageLimitExceededError` exception MUST contain a `PipelineResult` whose `step_history` includes the step that caused the breach. | To provide complete and accurate observability data, ensuring that the final state reflects all work performed, including the final, breaching action. | 1. An integration test is created where a step is designed to precisely exceed a usage limit. <br> 2. The test asserts that the caught `UsageLimitExceededError`'s `result.step_history` contains the `StepResult` of the breaching step. |
| **REQ-GOV-003** | The `_ParallelUsageGovernor` MUST perform atomic updates to its internal cost and token counters. | To prevent race conditions and ensure accurate usage tracking when multiple pipeline branches are executing concurrently. | 1. The `_ParallelUsageGovernor.add_usage` method uses an `asyncio.Lock` to protect its internal state. <br> 2. A stress test running a `ParallelStep` with many branches and small, rapid cost updates confirms that the final total cost is correct and no data is lost. |

---

### **5. Non-Functional Requirements (NFRs)**

These requirements define the quality attributes the final implementation must possess.

| ID | Requirement | Rationale | Acceptance Criteria |
| :-- | :--- | :--- | :--- |
| **NFR-TEST-001** | The entire Flujo test suite MUST pass with 100% success after all functional requirements are implemented. | To validate the correctness of the implementation and ensure no regressions have been introduced. | 1. The command `pytest` completes with zero failures and zero errors. |
| **NFR-PERF-001** | The changes MUST NOT introduce a performance regression of more than 5% on standard benchmark tests. | To ensure that the refactoring for correctness and clarity does not negatively impact the performance of the execution core. | 1. Key performance benchmarks (e.g., `flujo bench`) are run before and after the changes. <br> 2. The average latency in the "after" run is no more than 105% of the "before" run. |
| **NFR-DOCS-001** | The architectural documentation, specifically the "Execution Core" section, MUST be updated to reflect the new `ExecutionFrame` pattern and the finalized role of the `ExecutorCore`. | To ensure the documentation remains an accurate and useful resource for developers contributing to the framework. | 1. The design document's description of the `ExecutorCore`'s recursive execution is updated to mention the `ExecutionFrame`. <br> 2. All references to `step_logic.py` are removed. |