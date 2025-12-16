## Strategic Remediation Plan (Pending Items) — Zero-Shim Edition

Purpose: address the remaining gaps highlighted in the architectural debt review with a
sequenced, low-risk rollout. Optimizes for correctness first, then DX/maintainability.

### Current Status (per track)
- Serialization: ✅ **COMPLETE**. Runtime, tests, and docs migrated to `model_dump(mode="json")` / `_serialize_for_json`. Function deleted; `rg serialize_jsonable flujo/ tests/ docs/` returns zero.
- Async/Sync Bridge: Prometheus path migrated to shared bridge; further verification ongoing.
- Circular Imports: DSL now uses interfaces; continue tightening architecture checks.
- Typed Context: Scratchpad removed (legacy payloads rejected); mapping helpers in progress.

### Prioritized Tracks

1) Serialization Standardization (Highest Impact/Medium Effort, Zero Shim)
- Goal: remove `serialize_jsonable` entirely (no shim); standardize on Pydantic v2 `model_dump(mode="json")` and `_serialize_for_json`.
- Risks mitigated: inconsistent serialization, hidden circular refs, performance drift.
- Approach:
  - Inventory: `rg serialize_jsonable flujo/ tests/` (expect only tests/defs remaining).
  - Migration slices:
    1) Runtime ✅ (done).
    2) Tests/benchmarks/fixtures → replace with `model_dump(mode="json")` (models) or `_serialize_for_json` (mixed/primitives); delete serializer-spec tests.
  - Hardening: architecture/lint tests forbid any `serialize_jsonable` in repo.
  - Exit: delete function and exports; update cookbook.

2) Async/Sync Bridge Unification (High Impact/Low-Med Effort)
- Goal: single, battle-tested strategy (anyio BlockingPortal) for running async from sync.
- Approach:
  - Replace ad-hoc thread spawning with shared portal utility; enforce async-only entrypoints where appropriate.
  - Add unit tests: running under existing loop, shutdown paths, double-invoke safety.
  - Docs: clarify sync entrypoints vs async-only APIs.

3) Circular-Import Hardening (Medium Impact/High Effort)
- Goal: finish decoupling DSL from execution with stable interfaces.
- Approach:
  - Create a target interfaces surface map (Step, Pipeline, Context, Agent protocols).
  - Move remaining TYPE_CHECKING/lazy imports out of DSL; depend on `domain.interfaces`.
  - Add architecture test: forbid core imports inside DSL modules; allow only interfaces.
  - Validate with `make all` + smoke CLI/runner creation.

4) Typed Context & Scratchpad Maturity (Medium Impact/Med Effort)
- Goal: deepen validation for step I/O keys and enforce typed mappings; scratchpad has been removed (legacy payloads rejected).
- Approach:
  - Extend step input/output key validation to branches/parallel/import routers.
  - Add mapping helpers to translate prior scratchpad usages to typed fields; emit guided
    errors when keys conflict.
  - Expand tests for branch-aware validation and strict enforcement flags.

5) Verification & Guardrails (Always-On)
- Run `make all` before closeout of each track.
- Lint/checks block reintroduction: no `serialize_jsonable`, no ad-hoc thread-based coroutine runners, enforce interfaces-only imports in DSL.

### Execution Order & Milestones (updated)
- Week 1: Serialization tests cleanup (zero shim) + guardrail checks; Async bridge verification.
- Week 2: Circular-import interface mapping draft; continue serialization test removal if needed.
- Week 3: Circular-import migrations (incremental, with architecture tests).
- Week 4: Typed-context validations/mappings; stabilize tests; final `make all`.

### Definition of Done (per track)
- Serialization: `rg serialize_jsonable flujo/ tests/` returns zero; function deleted; tests/docs updated to new pattern.
- Async bridge: no thread-spawned asyncio runs remain; portal utility shared; tests cover running-loop and shutdown paths.
- Circular imports: DSL imports are interface-only; architecture test passes; no lazy imports needed for core separation.
- Typed context: branch/parallel/import validation active; legacy scratchpad payloads rejected; new guidance documented.

### Executable Tickets (owners, scope, status)

1) Serialization Standardization — Zero Shim (Owner: @core-runtime)
- Scope: Replace all test/fixture/benchmark usages with `model_dump(mode="json")` or `_serialize_for_json`; delete serializer-spec tests; remove `serialize_jsonable` and exports.
- Status: Runtime done; **tests/benchmarks pending**; function still present.
- AC: `rg serialize_jsonable flujo/ tests/` yields zero; function removed; docs updated; guardrail tests enforce zero presence.

2) Interfaces/Import Cleanup (Owner: @core-arch)
- Scope: Consolidate DSL imports to `domain.interfaces`; remove TYPE_CHECKING shims and lazy imports from DSL modules; add architecture test banning core imports inside DSL.
- Status: In progress.
- AC: Architecture test passes; no DSL module imports core directly; `make all` green; no new lazy imports added.

3) Typed-Context Validation Depth (Owner: @core-validation)
- Scope: Extend input/output key contract validation to branch/parallel/import routers; add mappings/guided errors for legacy scratchpad keys; expand tests.
- Status: In progress (scratchpad removed; mapping helpers ongoing).
- AC: Branch/parallel/import validation enforced; legacy scratchpad payloads yield explicit errors; new tests cover positive/negative paths; docs updated.
