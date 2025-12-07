## Flujo Type-Safety Execution Plan (Repo-Accurate)

Purpose: deliver compile-time confidence without breaking core architectural rules (policy-driven execution, control-flow exceptions, context idempotency, proactive quota, centralized config). Each phase lists scope, impact on Flujo, and rationale.

### Guardrails (non-negotiable)
- No type-based branching inside executor core; routing stays policy-driven.
- Control-flow exceptions (`PausedException`, `PipelineAbortSignal`, `InfiniteRedirectError`) are never swallowed or coerced into data failures.
- Context isolation/idempotency is preserved (`ContextManager.isolate()` for loops/parallel).
- Quota pattern remains proactive (Reserve → Execute → Reconcile); no reactive checks.
- Config remains centralized via `infra.config_manager`; no env/toml reads in domain logic.
- Serialization changes must honor Pydantic v2 and existing custom serialization rules.

### Phase 0 — Baseline Hardening (Non-breaking; preparatory)
- Actions:
  - Add TypeGuards for step outcomes and replace unchecked `cast(...)` in executor/policies.
  - Introduce typed fakes in `tests/test_types` (agent runner, usage meter, cache) and migrate tests off `MagicMock`; keep `_detect_mock_objects` only until fakes are universal, then remove.
  - Ensure DSL sets `__step_input_type__` / `__step_output_type__` from agent signatures; add a lint to surface `Any` defaults.
  - Add a CI lint to fail on new `Any` in core/DSL and new `cast(...)` in core.
- Impact on Flujo:
  - Safer executor hot path; fewer runtime surprises from mock leakage.
  - Type information propagates into validation without altering DSL signatures.
- Alignment & Need:
  - Honors control-flow exception and policy contracts (no new branching in core).
  - Reduces risk while keeping compatibility; builds foundation for stricter steps.

### Phase 1 — Context Evolution (Breaking: no legacy scratchpad semantics)
- Actions:
  - Provide typed context mixins (Pydantic BaseModel-bound) for common capabilities; forbid user data in `scratchpad` (reserved for framework metadata only).
  - Add `typed_context` factory and docs; require contexts to declare needed fields instead of stringly `scratchpad`.
  - Enforce `input_keys` / `output_keys` linting between steps and context models; violations fail validation.
  - Remove legacy fallback behaviors that permit untyped context access.
- Impact on Flujo:
  - Developers get guided contracts for context data; existing pipelines keep working.
  - Improved lint feedback reduces runtime `KeyError`/shape mismatches.
- Alignment & Need:
  - Respects context idempotency (ContextManager patterns unchanged).
  - Centralizes configuration/types; no policy-core changes.

### Phase 2 — DSL Type Continuity (Breaking: no implicit Any chaining)
- Actions:
  - Make `Pipeline.steps` typed as `Sequence[Step[PipeInT, PipeOutT]]`; preserve generics across `__rshift__`; no implicit `Any` acceptance.
  - Tighten `Pipeline.validate_graph`: incompatible or `Any`-flow chains fail validation; remove dict→object bridge except where explicitly annotated and justified.
  - Add TypeGuard-backed helpers for outcomes/results to eliminate remaining casts in policies/executor.
- Impact on Flujo:
  - IDE/mypy catches incompatible chains; pipelines become self-describing.
  - Legacy users have an escape hatch to avoid sudden breakage.
- Alignment & Need:
  - Maintains policy dispatch model; no isinstance branching in executor.
  - Drives down `Any` usage (goal: major reduction in DSL/core).

### Phase 3 — Enforcement & Cleanup (Breaking; remove legacy/unused code)
- Actions:
  - Require explicit type params on Steps in public DSL (`Step[str, JSONObject]`); default-`Any` is illegal (lint + runtime error).
  - Ban user data in `scratchpad`; require typed context fields or validated `input_keys`/`output_keys`.
  - Remove `_detect_mock_objects` after tests fully migrate to typed fakes.
  - Gate releases on `make typecheck` + tightened lints; add regression suites for typed chaining, context typing, and TypeGuards.
  - Remove all legacy/unused code paths that support untyped chaining, loose DSL imports, and permissive context access.
- Impact on Flujo:
  - Compile-time enforcement of data flow; fewer production type faults.
  - Cleaner context contract and smaller surface for hidden coupling.
- Alignment & Need:
  - Matches FLUJO_TEAM_GUIDE type-safety mandates.
  - Completes migration from “weak edges” to typed orchestration.

### Deliverables & Traceability
- ADR/RFC documenting the breaking shift to strict types (no loose mode), removal of legacy untyped paths, and assurance that policy-driven execution/control-flow exceptions/quota patterns are unchanged.
- Tickets per phase:
  - TypeGuards + executor/policy refactor
  - Typed fakes + test migration (and eventual removal of `_detect_mock_objects`)
  - Context mixins + `input_keys`/`output_keys` linting
  - Pipeline typing/linting (no loose-mode shim)
  - Scratchpad ban + docs + removal of unused code paths
- Toggles/config:
  - `strict_dsl` profile flag (CI default on; local opt-out removed post-release)
  - `context_strict_mode` flag (CI default on; local opt-out removed post-release)
- Tests: unit (TypeGuards, pipeline validation, context lints), integration (strict vs loose pipelines), regression for mocks→fakes.

### Success Metrics
- `cast(...)` in core: driven to 0 (replaced by TypeGuards).
- `Any` in DSL/core: substantial reduction; default-`Any` disallowed in strict mode.
- Mock usage in core paths: 0; tests use typed fakes.
- Context contract violations: caught at lint/typecheck time.
