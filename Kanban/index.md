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
*   **Plan (current):**
    *   Minimal injectable set tracked: runners/pipelines, usage/telemetry/quota, cache & serialization, policy executors/orchestrators, and core-bound registry/dispatcher/handlers (via factories).
    1. Define an `ExecutorCoreDeps` dataclass to capture injectables (runners, managers, cache/serialization, policy executors, orchestrators, handlers, registry/dispatcher).
    2. Add `FlujoRuntimeBuilder` that produces `ExecutorCoreDeps` with current defaults; allow targeted overrides (telemetry, quota, cache backend, runners, policy executors).
    3. Refactor `ExecutorCore.__init__` to accept `deps: ExecutorCoreDeps | None` plus legacy kwargs as thin overrides to keep compatibility.
    4. Add tests proving `ExecutorCore` can be built with mocks via the builder and that legacy kwargs still work.
    5. Run `make test-fast` and `make precommit`.

---

## 🔮 Column 3: Feature Backlog (Phase 3 & 4)
*Focus: Enterprise capabilities. Blocked by Phase 2.*

### [TASK-007] Governance Policy Layer
**Priority:** 🟡 Medium | **Effort:** Low | **Tags:** `Security`, `Enterprise`
*   **Description:** Middleware to intercept/block inputs before agent execution (e.g., PII redaction).
*   **Implementation:** Add `GovernancePolicy` protocol hooks in `StepCoordinator`.
*   **Plan (current):**
    1. Add a `GovernancePolicy` protocol + registry with a default allow-all policy; surface a config hook in settings.
    2. Insert interception before agent execution (agent handler/orchestrator entry) with fail-fast deny and telemetry (allow/deny decision, reason).
    3. Provide typed policy input/output models to avoid `Any`; keep frame/context available but immutable.
    4. Tests: allow path, deny path (raises/halts), telemetry emitted, and no-op when no policy configured.
*   **Status:** Complete (engine + telemetry + settings). Optional PIIScrubbingPolicy not started (non-blocking).

### [TASK-008] OpenAPI Skill Generator
**Priority:** 🟡 Medium | **Effort:** Low | **Tags:** `DX`, `Tooling`
*   **Description:** CLI command to generate Pydantic-typed Agent Tools from Swagger/OpenAPI specs.
*   **Implementation:** Wrap `datamodel-code-generator` in `flujo dev import-openapi`.
*   **Status:** Complete (wrappers + agent generation + typed response inference). Optional: richer endpoint-to-tool mapping and a full integration test for generated agents.

### [TASK-009] Shadow Evaluations
**Priority:** 🟡 Medium | **Effort:** Medium | **Tags:** `Observability`
*   **Description:** Async LLM-as-a-judge scoring on production runs.
*   **Implementation:** Use `BackgroundTaskManager` to sample % of runs and score via Evaluator agent.
*   **Plan (draft):**
    1. Add `shadow_eval` settings: enabled flag, sample_rate (0-1), judge model/tool config, timeout, sink choice (telemetry-only or persisted).
    2. Hook after step/pipeline completion (ResultHandler or PipelineOrchestrator) to enqueue background eval with immutable snapshot of input/output/metadata; isolate from user quota.
    3. Background worker runs judge agent/tool, records score/reason, emits telemetry counters (sampled/queued/succeeded/failed, latency) and does not affect user path on failure.
    4. Tests: sampling logic (probabilistic mock), no-op when disabled, enqueue when enabled, judge failure is non-fatal, telemetry metrics emitted.
*   **Status:** Complete. Sampling + scheduling + telemetry shipped; judge agent implemented with `make_agent_async` + `EvaluationScore` schema; optional database sink added (best-effort).

### [TASK-010] Abstracted Memory Interface
**Priority:** 🔵 Low | **Effort:** High | **Tags:** `RAG`, `Architecture`
*   **Description:** Interface for long-term memory.
*   **Implementation:** Define `VectorStoreProtocol`. Do **not** hardcode Postgres/pgvector in the core.
*   **Plan (current):**
    1. Define vector memory primitives (`MemoryRecord`, `VectorQuery`, `ScoredMemory`) and `VectorStoreProtocol` in `flujo/domain/memory.py`; keep async add/query/delete/close and avoid pgvector coupling.
    2. Provide defaults: `NullVectorStore` (no-op) and `InMemoryVectorStore` (cosine similarity) in `flujo/infra/memory/`; no external dependencies.
    3. Wire into DI: expose optional `memory_store`/`memory_manager` via `ExecutorCoreDeps` + `FlujoRuntimeBuilder`; default to Null store; consider a non-serialized handle on `PipelineContext`.
    4. Tests: protocol conformance, in-memory add/query/delete determinism, DI wiring defaults/null, and mypy strictness.
    5. Docs/Kanban: document interface intent and defaults; do not bake in pgvector.
*   **Status:** Complete. VectorStoreProtocol + primitives + Null/InMemory stores + DI done; MemoryManager indexing wired with context, and `PipelineContext.retrieve()` shipped. Optional: production stores (pgvector/chroma).

### [TASK-013] Durable Vector Stores (RAG Persistence)
**Priority:** 🟡 Medium | **Effort:** High | **Tags:** `RAG`, `Persistence`
*   **Description:** Add durable vector stores so RAG survives restarts.
*   **Plan:** 
    1. Implement `SQLiteVectorStore` (no C-ext) storing embeddings as BLOB and cosine search in Python. ✅
    2. Implement `PostgresVectorStore` using `pgvector` (`embedding <=>` queries) with migration to enable extension/table. ✅ (migration added)
    3. Update `FlujoRuntimeBuilder` to select store by `state_uri`/settings (`sqlite`→SQLite, `postgres`→pgvector) when memory indexing is enabled. ✅
*   **Status:** Complete (pending adoption in deployments).

### [TASK-014] Governance Policy Module from Config
**Priority:** 🟡 Medium | **Effort:** Low | **Tags:** `Security`, `DX`
*   **Description:** Allow loading a custom `GovernancePolicy` from `flujo.toml` without Python launcher code.
*   **Plan:** Add `[governance] policy_module="pkg.mod:PolicyCls"` parsing in ConfigManager and inject via `FlujoRuntimeBuilder` using dynamic import.
*   **Status:** Not started.

### [TASK-015] Shadow Eval Persistence & CLI
**Priority:** 🔵 Low | **Effort:** Medium | **Tags:** `Observability`
*   **Description:** Make shadow evaluation scores queryable beyond telemetry.
*   **Plan:** Add `persist_evaluation` to state backend with `evaluations` table (SQLite/Postgres); have `ShadowEvaluator` write when `sink="database"`; add `flujo lens evals` to list/avg scores.
*   **Status:** Complete.

### [TASK-016] Context Generator CLI (Optional)
**Priority:** 🔵 Low | **Effort:** Low | **Tags:** `DX`, `Type-Safety`
*   **Description:** CLI to scaffold a Pydantic context model from pipeline YAML.
*   **Plan:** `flujo dev gen-context pipeline.yaml` parses template usage and input/output keys to emit a `context.py` model stub.
*   **Status:** Not started.

### [TASK-017] Docs: Policy Migration & RAG Recipe
**Priority:** 🔵 Low | **Effort:** Low | **Tags:** `Docs`
*   **Description:** Documentation to support new architecture and RAG.
*   **Plan:** Write migration guide for custom policies (`ExecutionFrame` signature) and a RAG recipe using the new vector stores and MemoryManager.
*   **Status:** Complete (policy migration guide, RAG recipe added).

### [TASK-011] Sandbox Execution Interface
**Priority:** 🔵 Low | **Effort:** High | **Tags:** `Security`
*   **Description:** Safe execution of generated code.
*   **Implementation:** Define `CodeInterpreter` protocol. Implement `RemoteSandbox` (API-based) before Docker-based.
*   **Plan (current):**
    1. Define sandbox primitives (`SandboxExecution`, `SandboxResult`, `SandboxProtocol`) in domain; keep async `exec_code`.
    2. Provide defaults in infra: `NullSandbox` (safe no-op) as the default; keep remote/docker as future optional add-ons.
    3. Wire into DI: add `sandbox` to `ExecutorCoreDeps` + `FlujoRuntimeBuilder`, expose via `core.sandbox`.
    4. Tests: default null sandbox, custom injection via builder, and core exposure; ensure type safety.
    5. Builtin `code_interpreter` skill wired to the sandbox; returns structured stdout/stderr/exit-code.
*   **Status:** Complete. Protocol + NullSandbox + DI + `code_interpreter` skill shipped. RemoteSandbox implemented (API-based) with artifact support; DockerSandbox implemented for python workloads. Optional future: multi-language docker images.

### [TASK-012] Formalize Context Typing
**Priority:** 🔵 Low | **Effort:** Medium | **Tags:** `Type-Safety`
*   **Description:** Move away from `scratchpad: Dict[str, Any]`.
*   **Implementation:** Force users to define Pydantic models for Context and validate Step I/O against them.
*   **Plan (current):**
    1. Introduce typed context enforcement flag (`FLUJO_ENFORCE_TYPED_CONTEXT`) and helper to require Pydantic BaseModel contexts.
    2. Default is advisory (warning + pass-through); strict mode raises on plain dict contexts.
    3. Tests: enforcement-on rejects dict, accepts BaseModel; enforcement-off allows dict.
    4. Next slices: add step input/output key validation and migrate scratchpad usage toward typed fields.
*   **Status:** In progress. Typed context enforcement toggle + tests shipped. Step I/O key validation added with missing-key errors (V-CTX1) and root-only warnings (V-CTX2). Pending: deeper scratchpad-to-typed mappings and branch/parallel/import-aware validation.