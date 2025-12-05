Here is the **Flujo Engineering Kanban Board**, organized by the priorities established in our architectural review.

---

# 📋 Flujo Architecture Kanban Board

**Board Strategy:** Stability First $\rightarrow$ Core Refactor $\rightarrow$ Features.
**Sprint Goal:** Eliminate deployment blockers (`psutil`) and data loss risks (Nested HITL).

---

## 🚨 Column 1: Critical / Ready for Dev (Phase 1)
*Focus: Immediate stability fixes and deployment safety. These must be done before any feature work.*

### [TASK-001] Fail Fast on Nested HITL
**Priority:** 🔴 Critical | **Effort:** Low | **Tags:** `Stability`, `Data-Loss`
*   **Description:** Human-in-the-Loop steps nested inside conditional branches within loops are silently skipped, causing data loss.
*   **Requirements:**
    *   Modify `flujo/application/core/policy_primitives.py`.
    *   Implement a runtime check that detects `HITL` inside `Conditional` inside `Loop`.
    *   Raise `ConfigurationError` immediately if detected.
*   **DoD:** Unit test confirms pipeline raises error on construction/validation of this pattern.

### [TASK-002] Prune Optimization Layer & Remove `psutil`
**Priority:** 🔴 Critical | **Effort:** Medium | **Tags:** `Cleanup`, `Deployment`
*   **Description:** The `optimization/` folder is dead code, but imports `psutil` which breaks Lambda/Alpine builds.
*   **Requirements:**
    1.  Create `tests/architecture/test_module_isolation.py` (subprocess based) to prove `optimization` isn't loaded.
    2.  Refactor `ExecutorCore` and `BackgroundTaskManager` to remove imports.
    3.  **Keep** `optimization_config_stub.py` to maintain import compatibility.
    4.  Delete `flujo/application/core/optimization/` folder.
    5.  Remove `psutil` from `pyproject.toml`.
*   **DoD:** `make test` passes; `pip install` works without C-compiler.

### [TASK-003] Enforce Strict Pricing Exceptions
**Priority:** 🔴 Critical | **Effort:** Low | **Tags:** `Safety`, `Finance`
*   **Description:** `AgentExecutionRunner` currently swallows pricing configuration errors, potentially allowing infinite unmonitored spend.
*   **Requirements:**
    *   Locate error handling in `flujo/application/core/agent_execution_runner.py`.
    *   Explicitly catch `PricingNotConfiguredError` before generic handling.
    *   Re-raise immediately.
*   **DoD:** Integration test confirms pipeline halts immediately if pricing is missing in strict mode.

---

## 🚧 Column 2: Queued for Refactoring (Phase 2)
*Focus: Paying down technical debt to enable future features. Blocked by Phase 1.*

### [TASK-004] Fix Sync/Async Bridge
**Priority:** 🟠 High | **Effort:** Medium | **Tags:** `Tech-Debt`, `Async`
*   **Description:** `_run_coro_sync` spawns ad-hoc threads, causing "Event Loop Closed" errors during shutdown.
*   **Requirements:**
    *   Replace custom thread spawning in `runner_methods.py` and `sqlite_core.py`.
    *   Use `anyio.from_thread.run` or `asgiref.sync.async_to_sync`.
*   **DoD:** No `RuntimeError: Event loop is closed` in test logs.

### [TASK-005] Policy Frame-Only Migration
**Priority:** 🟠 High | **Effort:** High | **Tags:** `Refactor`, `DX`
*   **Description:** Convert all Policy `execute` signatures to accept a single `ExecutionFrame` object.
*   **Requirements:**
    *   **Strategy:** Vertical Slice (migrate one policy at a time to keep build green).
    *   Update `StepPolicy` protocol.
    *   Implement `ExecutionDispatcher` adapter to support legacy custom policies during transition. ✅
    *   Phase 2/3 complete: Agent, Simple, Cache, Conditional, Parallel, Loop, DynamicRouter, HITL, Import policies now take `(core, frame: ExecutionFrame)`; adapter warns on legacy signatures when `FLUJO_WARN_LEGACY=1`.
*   **DoD:** All internal policies use `frame.context` instead of `*args`.

### [TASK-006] Dependency Injection for `ExecutorCore`
**Priority:** 🟠 High | **Effort:** High | **Tags:** `Architecture`, `Testing`
*   **Description:** Decompose the "God Object". `ExecutorCore` currently instantiates 10+ managers internally.
*   **Requirements:**
    *   Refactor `ExecutorCore.__init__` to accept instances of `QuotaManager`, `AgentOrchestrator`, etc.
    *   Create a `FlujoRuntimeBuilder` factory to handle wiring.
*   **DoD:** `ExecutorCore` can be instantiated in a unit test with Mock dependencies.

---

## 🔮 Column 3: Feature Backlog (Phase 3 & 4)
*Focus: Enterprise capabilities. Blocked by Phase 2.*

### [TASK-007] Governance Policy Layer
**Priority:** 🟡 Medium | **Effort:** Low | **Tags:** `Security`, `Enterprise`
*   **Description:** Middleware to intercept/block inputs before agent execution (e.g., PII redaction).
*   **Implementation:** Add `GovernancePolicy` protocol hooks in `StepCoordinator`.

### [TASK-008] OpenAPI Skill Generator
**Priority:** 🟡 Medium | **Effort:** Low | **Tags:** `DX`, `Tooling`
*   **Description:** CLI command to generate Pydantic-typed Agent Tools from Swagger/OpenAPI specs.
*   **Implementation:** Wrap `datamodel-code-generator` in `flujo dev import-openapi`.

### [TASK-009] Shadow Evaluations
**Priority:** 🟡 Medium | **Effort:** Medium | **Tags:** `Observability`
*   **Description:** Async LLM-as-a-judge scoring on production runs.
*   **Implementation:** Use `BackgroundTaskManager` to sample % of runs and score via Evaluator agent.

### [TASK-010] Abstracted Memory Interface
**Priority:** 🔵 Low | **Effort:** High | **Tags:** `RAG`, `Architecture`
*   **Description:** Interface for long-term memory.
*   **Implementation:** Define `VectorStoreProtocol`. Do **not** hardcode Postgres/pgvector in the core.

### [TASK-011] Sandbox Execution Interface
**Priority:** 🔵 Low | **Effort:** High | **Tags:** `Security`
*   **Description:** Safe execution of generated code.
*   **Implementation:** Define `CodeInterpreter` protocol. Implement `RemoteSandbox` (API-based) before Docker-based.

### [TASK-012] Formalize Context Typing
**Priority:** 🔵 Low | **Effort:** Medium | **Tags:** `Type-Safety`
*   **Description:** Move away from `scratchpad: Dict[str, Any]`.
*   **Implementation:** Force users to define Pydantic models for Context and validate Step I/O against them.