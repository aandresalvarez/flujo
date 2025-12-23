## Strategic Remediation Plan (Pending Items) — Zero-Shim Edition

Purpose: address the remaining gaps highlighted in the architectural debt review with a
sequenced, low-risk rollout. Optimizes for correctness first, then DX/maintainability.

### Current Status (per track)
- Serialization: ✅ **COMPLETE**. `serialize_jsonable` removed; `rg serialize_jsonable flujo/ tests/ docs/` yields zero; tests/benchmarks use `model_dump(mode="json")` / `_serialize_for_json`. Guardrail test to forbid reintroduction still missing.
- Async/Sync Bridge: ✅ **COMPLETE**. Shared `run_sync` (anyio.run) used across CLI/telemetry/state; tests cover running-loop + shutdown; no BlockingPortal usage.
- Circular Imports: ✅ **COMPLETE**. DSL uses `domain.interfaces`; guardrail test blocks module-level core imports.
- Typed Context: ⚠️ **PARTIAL**. Scratchpad rejected and branch/parallel/import validation active; codemod + fixers + lints shipped, but some docs/examples still reference scratchpad (e.g., `llm.md`, `FLUJO_TEAM_GUIDE.md`).

### Prioritized Tracks

1) Serialization Standardization (Highest Impact/Medium Effort, Zero Shim)
- Goal: remove `serialize_jsonable` entirely (no shim); standardize on Pydantic v2 `model_dump(mode="json")` and `_serialize_for_json`.
- Risks mitigated: inconsistent serialization, hidden circular refs, performance drift.
- Approach:
  - Inventory: `rg serialize_jsonable flujo/ tests/ docs/` (expect zero; now zero).
  - Migration slices:
    1) Runtime ✅ (done).
    2) Tests/benchmarks/fixtures ✅ (no serialize_jsonable tests remain).
  - Hardening: guardrail test to forbid any `serialize_jsonable` still pending.
  - Exit: function and exports removed ✅; docs/cookbook updated ✅.

2) Async/Sync Bridge Unification (High Impact/Low-Med Effort)
- Goal: single, battle-tested strategy (`run_sync` via anyio.run) for running async from sync.
- Approach:
  - Replace ad-hoc thread spawning with shared `run_sync`; enforce async-only entrypoints where appropriate.
  - Add unit tests: running under existing loop, shutdown paths, double-invoke safety. ✅
  - Docs: clarify sync entrypoints vs async-only APIs (explicit guidance still thin).

3) Circular-Import Hardening (Medium Impact/High Effort)
- Goal: finish decoupling DSL from execution with stable interfaces.
- Approach:
  - Create a target interfaces surface map (Step, Pipeline, Context, Agent protocols).
  - Move remaining TYPE_CHECKING/lazy imports out of DSL; depend on `domain.interfaces` (core imports already removed).
  - Add architecture test: forbid core imports inside DSL modules; allow only interfaces. ✅
  - Validate with `make all` + smoke CLI/runner creation.

4) Typed Context & Scratchpad Maturity (Medium Impact/Med Effort)
- Goal: deepen validation for step I/O keys and enforce typed mappings; scratchpad has been removed (legacy payloads rejected).
- Approach:
  - Extend step input/output key validation to branches/parallel/import routers. ✅
  - Add mapping helpers to translate prior scratchpad usages to typed fields; emit guided
    errors when keys conflict. ✅ (codemod + fixers + lints)
  - Expand tests for branch-aware validation and strict enforcement flags. ✅

5) Verification & Guardrails (Always-On)
- Run `make all` before closeout of each track.
- Lint/checks block reintroduction: interfaces-only imports in DSL ✅; no BlockingPortal usage outside shared bridge ✅.
- Gaps: explicit `serialize_jsonable` ban test missing; no guardrail for ad-hoc thread-based coroutine runners.

### Execution Order & Milestones (updated)
- Week 1: Serialization cleanup ✅; async bridge verification ✅.
- Week 2-3: Circular-import interface mapping ✅ with guardrail test.
- Week 4: Typed-context validations ✅; doc cleanup for scratchpad references still open.

### Definition of Done (per track)
- Serialization: `rg serialize_jsonable flujo/ tests/` returns zero; function deleted; tests/docs updated to new pattern.
- Async bridge: no thread-spawned asyncio runs remain; shared `run_sync` utility; tests cover running-loop and shutdown paths.
- Circular imports: DSL imports are interface-only; architecture test passes; no lazy imports needed for core separation.
- Typed context: branch/parallel/import validation active; legacy scratchpad payloads rejected; new guidance documented.

### Executable Tickets (owners, scope, status)

1) Serialization Standardization — Zero Shim (Owner: @core-runtime)
- Scope: Replace all test/fixture/benchmark usages with `model_dump(mode="json")` or `_serialize_for_json`; delete serializer-spec tests; remove `serialize_jsonable` and exports.
- Status: ✅ Done for code/tests/docs; guardrail test to ban `serialize_jsonable` reintroduction still pending.
- AC: `rg serialize_jsonable flujo/ tests/` yields zero; function removed; docs updated; guardrail tests enforce zero presence.

2) Interfaces/Import Cleanup (Owner: @core-arch)
- Scope: Consolidate DSL imports to `domain.interfaces`; remove TYPE_CHECKING shims and lazy imports from DSL modules; add architecture test banning core imports inside DSL.
- Status: ✅ Complete for core imports; TYPE_CHECKING remains for DSL-local types.
- AC: Architecture test passes; no DSL module imports core directly; `make all` green; no new lazy imports added.

3) Typed-Context Validation Depth (Owner: @core-validation)
- Scope: Extend input/output key contract validation to branch/parallel/import routers; add mappings/guided errors for legacy scratchpad keys; expand tests.
- Status: ⚠️ Mostly complete. Validation + mappings + tests done; doc cleanup for legacy scratchpad references still open.
- AC: Branch/parallel/import validation enforced; legacy scratchpad payloads yield explicit errors; new tests cover positive/negative paths; docs updated.
