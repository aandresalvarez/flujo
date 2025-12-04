This improvement plan is prioritized by the **Impact/Effort Ratio**. We start with critical fixes that prevent data loss or crashes (High Impact / Low-Medium Effort), move to architectural decoupling (High Impact / High Effort), and finish with code hygiene (Medium Impact / Medium Effort).

---

### 🔴 Phase 1: Critical Stability & Correctness
**Focus:** Preventing data loss, financial risk, and runtime crashes.

#### 1. Hard Gate for Nested HITL (Human-in-the-Loop)
*   **Problem:** HITL steps inside conditional branches within loops are silently skipped or fail to resume, causing data loss (`flujo/application/core/policy_primitives.py`).
*   **Impact:** **Critical.** Users lose data/state without warning in complex pipelines.
*   **Effort:** **Low.**
*   **Action Plan:**
    1.  Modify `_check_hitl_nesting_safety`. instead of a soft check or debug log, raise a `ConfigurationError` immediately during pipeline construction or validation.
    2.  Do not allow the pipeline to run if this structure is detected.
    3.  **Long term:** Refactor the resume orchestrator to handle the stack unwinding for nested loops, but for now, fail fast.

#### 2. Enforce Strict Pricing Exceptions
*   **Problem:** `AgentExecutionRunner` catches `PricingNotConfiguredError` and sometimes wraps it in a generic `Failure` object. In a production loop, this might allow an agent to spin indefinitely without cost tracking.
*   **Impact:** **High.** Financial risk.
*   **Effort:** **Low.**
*   **Action Plan:**
    1.  In `flujo/application/core/agent_execution_runner.py`, locate the `except Exception as e` block.
    2.  Explicitly catch `PricingNotConfiguredError` *before* the generic catch block and re-raise it immediately.
    3.  Ensure `ExecutorCore` does not swallow this specific exception type.

#### 3. Stabilize Async/Sync Bridging
*   **Problem:** `_run_coro_sync` spawns ad-hoc threads to run asyncio loops. This causes "Event loop is closed" errors during shutdown/tests and makes signal handling flaky.
*   **Impact:** **High.** System stability and clean shutdowns.
*   **Effort:** **Medium.**
*   **Action Plan:**
    1.  Replace ad-hoc thread spawning with a battle-tested library like `asgiref.sync.async_to_sync` or `anyio`.
    2.  Alternatively, enforce that `Flujo` must be run in an async context (`run_async`) and deprecate `run_sync` for anything other than simple scripts.

---

### 🟡 Phase 2: Architectural Refactoring
**Focus:** Testability, maintainability, and reducing the "God Object."

#### 4. Decompose `ExecutorCore` via Dependency Injection
*   **Problem:** `ExecutorCore` (`flujo/application/core/executor_core.py`) manually instantiates 10+ helpers (`QuotaManager`, `AgentOrchestrator`, etc.) inside its `__init__`.
*   **Impact:** **High.** Hard to test in isolation; tight coupling.
*   **Effort:** **High.**
*   **Action Plan:**
    1.  Change `ExecutorCore.__init__` to accept these managers as optional arguments.
    2.  Create a `Builder` or `Factory` class (e.g., `FlujoRuntimeBuilder`) responsible for wiring these dependencies together.
    3.  Remove the hardcoded instantiation logic from `ExecutorCore`.
    4.  **Benefit:** You can now mock the `QuotaManager` completely when testing the `AgentOrchestrator`.

#### 5. Resolve Circular Dependency Import Hell
*   **Problem:** Extensive use of `import inside function` and `TYPE_CHECKING` hacks (e.g., `flujo/domain/dsl/step.py`).
*   **Impact:** **Medium.** Fragile refactoring; hard to read code; import errors at runtime.
*   **Effort:** **High.**
*   **Action Plan:**
    1.  Create a dedicated `flujo.types` or `flujo.interfaces` module that holds **only** Protocols and ABCs.
    2.  Move `Step`, `Pipeline`, and `Context` definitions to separate files that do not import execution logic.
    3.  Refactor `application/core` to depend on `interfaces`, not concrete implementations in `domain`.

---

### 🟢 Phase 3: Code Hygiene & Developer Experience
**Focus:** Type safety and performance.

#### 6. Formalize Context Typing (Kill the `scratchpad`)
*   **Problem:** `PipelineContext.scratchpad` is a `Dict[str, Any]`. It is a "bag of state" that hides dependencies between steps.
*   **Impact:** **Medium.** Developer experience and bug prevention.
*   **Effort:** **Medium.**
*   **Action Plan:**
    1.  Encourage users to define specific Pydantic models for their context (e.g., `ResearchContext`).
    2.  Update `Step` definition to allow specifying `input_keys` and `output_keys`.
    3.  Add a validation pass that checks: "Step B requires key 'summary', does Step A produce 'summary'?"
    4.  Keep `scratchpad` only for internal framework metadata, not user data.

#### 7. Standardize Serialization
*   **Problem:** `flujo/utils/serialization.py` manually handles Pydantic models, dataclasses, and primitives. It is redundant given Pydantic v2's capabilities.
*   **Impact:** **Medium.** Performance and maintenance debt.
*   **Effort:** **Medium.**
*   **Action Plan:**
    1.  Deprecate `safe_serialize`.
    2.  Adopt `model.model_dump(mode='json')` (Pydantic v2) for all domain objects.
    3.  For custom types, register Pydantic serializers (`@field_serializer`) on the models themselves rather than in a global registry function.

### Summary Roadmap

| Order | Task | Impact | Effort | Goal |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **Fail Fast on Nested HITL** | 🟥 Critical | 🟩 Low | Prevent data loss immediately. |
| **2** | **Fix Pricing Exception Handling** | 🟥 Critical | 🟩 Low | Prevent unmonitored financial spend. |
| **3** | **Refactor Sync/Async Bridge** | 🟥 High | 🟨 Med | Prevent process hangs/crashes. |
| **4** | **Dependency Injection for Core** | 🟥 High | 🟥 High | Enable unit testing of core logic. |
| **5** | **Fix Circular Imports** | 🟨 Med | 🟥 High | Stabilize codebase structure. |
| **6** | **Typed Contexts** | 🟨 Med | 🟨 Med | Improve DX and safety. |
| **7** | **Pydantic V2 Native Serialization** | 🟨 Med | 🟨 Med | Remove maintenance debt. |