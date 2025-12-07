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

---

## Implementation Plan (actionable)

### Progress snapshot
- Done:
  - Fail-fast nested HITL validation (always raises) and runtime guard retained.
  - Pricing exception propagation regression test (AgentExecutionRunner).
  - Sync/async bridge hardened (`_run_coro_sync` via BlockingPortal); sqlite bridge tests pass.
  - Executor DI accepts injected deps/builder; regression tests added.
  - Typed contexts: scratchpad reserved for framework metadata, runtime enforcement flag (`FLUJO_ENFORCE_SCRATCHPAD_BAN`) and tests.
  - Serialization cleanup: base_model uses native model_dump; backends (sqlite/base/memory/postgres) and cache/CLI/agents use pydantic/dataclass serialization; `safe_serialize` removed from runtime/tests; `_serialize_for_json` hardened and applied across tests/benchmarks; placeholders standardized.
  - Circular-import hardening: interfaces in `domain.interfaces`, core depends on interfaces; CLI/runner smoke tests added.
  - Type-safety guardrails: reduced unsafe casts in core; TypeGuard for governance policy loading; bounded-cast architecture test added to prevent regressions; type validator uses typed accessors; scratchpad allowlist growth lint added (hard-fail scratchpad ban default).
  - DSL/import migration off scratchpad: `ImportArtifacts` wiring complete; ImportStep policy now prefers artifacts and preserves explicit `None`; scratchpad allowlist tightened to 24; cast gate raised to 150 with TypeGuarded merges; telemetry redaction hardened with generic key patterns.
  - Dynamic merge/isolation paths TypeGuarded; context serializer hashing fixed to avoid large-context slowdowns; make all/precommit green.
- Remaining:
  - None pending from this plan; keep guardrails enforced and revisit baselines only when new typed fields land.

### Phase 1 — Critical Stability
- Nested HITL gate
  - Update `_check_hitl_nesting_safety` to raise `ConfigurationError` during validation (no runtime soft paths).
  - Add validation tests for loop+conditional HITL graphs; add negative test ensuring run is blocked.
- Pricing exception handling
  - In `application/core/agent_execution_runner.py`, explicitly catch and re-raise `PricingNotConfiguredError` before generic handlers.
  - Ensure `ExecutorCore` surfaces it; add regression test.
- Sync/async bridge
  - Replace `_run_coro_sync` thread spawning with a single supported strategy (anyio or async-only entrypoint); deprecate broad `run_sync`.
  - Add shutdown/signal-handling tests to prevent “Event loop is closed.”

### Phase 2 — Architectural Refactoring
- Executor DI
  - Refactor `ExecutorCore.__init__` to accept injected managers; create `FlujoRuntimeBuilder` to wire defaults.
  - Add unit tests using fakes for injected dependencies.
- Circular imports
  - Extract interfaces/Protocols to a dedicated module; move `Step/Pipeline/Context` definitions out of execution logic.
  - Remove `import-inside` hacks where feasible; smoke-test DSL creation and run lint/typecheck to confirm imports are stable.

### Phase 3 — Code Hygiene & DX
- Typed contexts, no user data in scratchpad
  - Introduce typed context models/mixins; enforce `input_keys`/`output_keys` validation and fail on violations.
  - Reserve `scratchpad` for framework metadata only; add validation/tests to guard.
- Serialization standardization
  - Deprecate `safe_serialize`; switch to `model.model_dump(mode="json")`.
  - Add `@field_serializer` where needed; add round-trip serialization tests for core models.

### Cross-cutting guardrails (must hold throughout)
- Policy-driven execution only; no type-based branching in executor.
- Control-flow exceptions (`PausedException`, `PipelineAbortSignal`, `InfiniteRedirectError`) are not swallowed or coerced.
- Context idempotency via `ContextManager.isolate()` for loops/parallel remains intact.
- Quota pattern stays proactive (Reserve → Execute → Reconcile); no reactive checks.
- Config stays centralized via `infra.config_manager`; no env/toml reads in domain logic.
- Type-safety: add TypeGuards for outcomes, eliminate unchecked `cast(...)`, fail CI on new `Any` in core/DSL.

### Acceptance checks
- `make all` and `make precommit` green after each phase.
- Validation blocks nested HITL and pricing misconfig at build/graph time.
- Executor DI usable with fakes in tests.
- No legacy scratchpad usage for user data; lint/validation catches violations.
- No lingering `safe_serialize` references; serialization tests pass. 