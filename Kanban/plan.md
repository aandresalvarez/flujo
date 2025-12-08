## Strategic Remediation Plan (Pending Items)

Purpose: address the remaining gaps highlighted in the architectural debt review with a
sequenced, low-risk rollout. Optimizes for correctness first, then DX/maintainability.

### Prioritized Tracks

1) Serialization Standardization (Highest Impact/Medium Effort)
- Goal: remove `serialize_jsonable` from runtime; standardize on Pydantic v2
  `model_dump(mode="json")` + field serializers.
- Risks mitigated: inconsistent serialization, hidden circular refs, performance drift.
- Approach:
  - Inventory: `rg serialize_jsonable flujo/ tests/` to produce a callsite matrix (runtime
    vs tests; hot paths vs fixtures).
  - API plan: mark `serialize_jsonable` as deprecated (warnings), introduce
    `serialize_to_jsonable` adapter that delegates to `model_dump` for models and to
    `_serialize_for_json` for primitives; keep behavior stable during migration.
  - Migration slices:
    1) Core/state/backends and cache → replace with `_serialize_for_json` or
       `model_dump(mode="json")`.
    2) Application/core + processors/agents.
    3) Tests/benchmarks/fixtures (convert to new helpers or direct `model_dump`).
  - Hardening: add regression tests that forbid new `serialize_jsonable` imports in core,
    allow only in `tests/legacy_serialization/` during transition.
  - Exit: remove helper, delete legacy tests, keep cookbook updated.

2) Async/Sync Bridge Unification (High Impact/Low-Med Effort)
- Goal: single, battle-tested strategy (anyio BlockingPortal) for running async from sync.
- Risks mitigated: “Event loop is closed” flakiness, shutdown leaks.
- Approach:
  - Replace ad-hoc thread spawning (e.g., `flujo/telemetry/prometheus.py::run_coroutine`)
    with shared portal utility (same as sqlite bridge) or enforce async-only entrypoints
    where appropriate.
  - Add unit tests: running under existing loop, shutdown paths, double-invoke safety.
  - Docs: clarify sync entrypoints vs async-only APIs.

3) Circular-Import Hardening (Medium Impact/High Effort)
- Goal: finish decoupling DSL from execution with stable interfaces.
- Approach:
  - Create a target interfaces surface map (Step, Pipeline, Context, Agent protocols).
  - Slice migrations: move remaining TYPE_CHECKING/lazy imports out of DSL modules; depend
    on `domain.interfaces` from core; ensure DSL modules avoid core imports.
  - Add architecture test: forbid core imports inside DSL modules; allow only interfaces.
  - Validate with `make all` + smoke CLI/runner creation.

4) Typed Context & Scratchpad Maturity (Medium Impact/Med Effort)
- Goal: deepen validation for step I/O keys and enforce typed mappings while keeping
  scratchpad reserved for framework metadata.
- Approach:
  - Extend step input/output key validation to branches/parallel/import routers.
  - Add mapping helpers to translate prior scratchpad usages to typed fields; emit guided
    errors when keys conflict.
  - Expand tests for branch-aware validation and strict enforcement flags.

5) Verification & Guardrails (Always-On)
- Run `make all` before closeout of each track.
- Add lint/checks to block reintroduction: no new `serialize_jsonable`, no ad-hoc
  thread-based coroutine runners, enforce interfaces-only imports in DSL.

### Execution Order & Milestones
- Week 1: Track 1 (serialization) slice 1 + Track 2 prometheus fix; add guardrail tests.
- Week 2: Track 1 slices 2–3; begin Track 3 interface mapping draft.
- Week 3: Track 3 migrations (incremental, with architecture tests).
- Week 4: Track 4 validations/mappings; stabilize tests; final `make all`.

### Definition of Done (per track)
- Serialization: zero runtime references to `serialize_jsonable`; tests updated; docs
  reflect new pattern.
- Async bridge: no thread-spawned asyncio runs remain; portal utility shared; tests cover
  running-loop and shutdown paths.
- Circular imports: DSL imports are interface-only; architecture test passes; no lazy
  imports needed for core separation.
- Typed context: branch/parallel/import validation active; scratchpad reserved keys
  enforced; new guidance documented.

### Executable Tickets (owners & acceptance)

1) Serialization Standardization (Owner: @core-runtime)
- Scope: Replace runtime usages of `serialize_jsonable` with `model_dump(mode="json")`
  or `_serialize_for_json`; add deprecation warning shim; migrate tests/fixtures.
- AC: `rg serialize_jsonable flujo/` yields zero; core/tests pass; cookbook updated; lint
  blocks new imports outside legacy fixtures (if any remain temporarily).

2) Interfaces/Import Cleanup (Owner: @core-arch)
- Scope: Consolidate DSL imports to use `domain.interfaces`; remove TYPE_CHECKING shims
  and lazy imports from DSL modules; add architecture test banning core imports inside DSL.
- AC: Architecture test passes; no DSL module imports core directly; `make all` green; no
  new lazy imports added.

3) Typed-Context Validation Depth (Owner: @core-validation)
- Scope: Extend input/output key contract validation to branch/parallel/import routers;
  add mappings/guided errors for legacy scratchpad keys; expand tests.
- AC: Branch/parallel/import validation enforced; scratchpad conflicts yield explicit
  errors; new tests cover positive/negative paths; docs updated.

