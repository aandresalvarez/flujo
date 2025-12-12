## Flujo Type-Safety Execution Plan (Repo-Accurate, answers open questions)

Purpose: deliver compile-time confidence without breaking core architectural rules (policy-driven execution, control-flow exceptions, context idempotency, proactive quota, centralized config). This revision answers rollout, compatibility, adapter governance, perf budget, codemod/tooling, observability, and sequencing questions.

### Guardrails (non-negotiable)
- No type-based branching inside executor core; routing stays policy-driven.
- Control-flow exceptions (`PausedException`, `PipelineAbortSignal`, `InfiniteRedirectError`) are never swallowed or coerced into data failures.
- Context isolation/idempotency is preserved (`ContextManager.isolate()` for loops/parallel).
- Quota pattern remains proactive (Reserve → Execute → Reconcile); no reactive checks.
- Config remains centralized via `infra.config_manager`; no env/toml reads in domain logic.
- Serialization changes must honor Pydantic v2 and existing custom serialization rules.

### Strict Defaults & Adapter Constraints (agent pipelines)
- Strict-only stance: no loose mode, no implicit `Any` paths. `strict_dsl` and `context_strict_mode` are always on (CI and local); opt-outs are removed after a timeboxed migration window.
- Adapters as last resort with governance: `Unknown`/`TypedAny`, `DictContextAdapter`, `SchemaAdapter` exist only for explicit, narrow bridges (external schema boundary). Each use must be allowlisted with owner + rationale; CI lint fails unapproved adapters.
- Ergonomics: provide decorators/helpers to derive step IO types from agent signatures, a light `TypedContext` helper for concise context declarations, and typed fakes for tests to prevent mock leakage.
- Tests: regression coverage for strict paths plus the narrow adapter paths; ensure adapters cannot reintroduce silent `Any` flows. Validate policy-driven dispatch/control-flow exceptions remain unchanged.
- Docs: remove legacy loose-mode guidance; all examples use strict typing, typed contexts, and explicit adapters only where justified.

### Rollout & Migration (strict defaults + scratchpad ban)
1) Warn stage (timeboxed): emit deprecation warnings in CI/local for loose DSL, scratchpad writes, and untyped contexts; publish migration guide.
2) Soft-fail stage: CI fails new code paths; legacy paths log warnings. Provide codemods to rewrite `scratchpad[...]` to typed fields and `safe_merge_context_updates`.
3) Hard-fail default: strict on by default; temporary override flag allowed only for legacy suites with expiration date.
4) Cleanup: remove override; strict-only remains. Deprecation calendar published with dates for each stage.

### Adapter Governance & CI Enforcement
- Allowlist file with adapter identifiers, owner, rationale, expiry/review date.
- Lint/CI rule: new adapter instantiation must reference allowlist token; else fail. No generic adapter imports outside approved boundaries.
- Ownership: core maintainers approve additions; PRs adding adapters must include boundary tests and justification.
  
### Performance Budget & Benchmarks
- Define hot-path budget for executor + policy dispatch; perf canary in CI fast tier with fail-fast threshold. Full benchmarks run on schedule.
- Any perf regression requires root-cause fix; thresholds are not raised to mask regressions.

### Observability & Instrumentation
- Metrics: counts of `cast` in core, `Any` occurrences, adapter invocations, mock usage in tests, scratchpad writes.
- Dashboards: track migration progress and fail CI on regression to baseline.

### Phase 0 — Baseline Hardening (Non-breaking; preparatory)
- Actions:
  - Add TypeGuards for step outcomes and replace unchecked `cast(...)` in executor/policies.
  - Introduce typed fakes in `tests/test_types` (agent runner, usage meter, cache) and migrate tests off `MagicMock`; keep `_detect_mock_objects` only until fakes are universal, then remove.
  - Ensure DSL sets `__step_input_type__` / `__step_output_type__` from agent signatures; add a lint to surface `Any` defaults.
  - Add a CI lint to fail on new `Any` in core/DSL and new `cast(...)` in core; add adapter allowlist lint scaffold.
- Impact on Flujo:
  - Safer executor hot path; fewer runtime surprises from mock leakage.
  - Type information propagates into validation without altering DSL signatures.
- Alignment & Need:
  - Honors control-flow exception and policy contracts (no new branching in core).
  - Reduces risk while keeping compatibility; builds foundation for stricter steps.

### Phase 1 — Context Evolution (Breaking: no legacy scratchpad semantics)
- Actions:
  - Provide typed context mixins (Pydantic BaseModel-bound) for common capabilities; forbid user data in `scratchpad` (reserved for framework metadata only).
  - Add `typed_context` factory and docs; require contexts to declare needed fields instead of stringly `scratchpad`; ship codemod + lint autofix to rewrite scratchpad access.
  - Enforce `input_keys` / `output_keys` linting between steps and context models; violations fail validation.
  - Remove legacy fallback behaviors that permit untyped context access.
- Impact on Flujo:
  - Developers get guided contracts for context data; legacy untyped contexts must be updated (breaking).
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
  - Adapters exist only for justified boundaries, not for preserving legacy loose typing.
- Alignment & Need:
  - Maintains policy dispatch model; no isinstance branching in executor.
  - Drives down `Any` usage (goal: major reduction in DSL/core).

### Phase 3 — Enforcement & Cleanup (Breaking; remove legacy/unused code)
- Actions:
  - Require explicit type params on Steps in public DSL (`Step[str, JSONObject]`); default-`Any` is illegal (lint + runtime error).
  - Ban user data in `scratchpad`; require typed context fields or validated `input_keys`/`output_keys`.
  - Remove `_detect_mock_objects` after tests fully migrate to typed fakes.
  - Gate releases on `make typecheck` + tightened lints; add regression suites for typed chaining, context typing, TypeGuards, and adapter allowlist.
  - Remove all legacy/unused code paths that support untyped chaining, loose DSL imports, and permissive context access.
- Impact on Flujo:
  - Compile-time enforcement of data flow; fewer production type faults.
  - Cleaner context contract and smaller surface for hidden coupling.
- Alignment & Need:
  - Matches FLUJO_TEAM_GUIDE type-safety mandates.
  - Completes migration from “weak edges” to typed orchestration.

### Rollout Sequencing & Dependencies
- Phase 0 must land before Phase 1 (typed fakes + TypeGuards + lints in place).
- Phase 1 (context) and Phase 2 (DSL continuity) can proceed in parallel after Phase 0, but Phase 2 hard enforcement waits for Phase 1 warning/soft-fail completion.
- Phase 3 is blocked on: typed fakes adopted; adapter allowlist lint active; codemod + migration guide published; warning/soft-fail stages completed.
- Ownership: core maintainers drive adapter allowlist and perf canary; type-safety leads own lints/codemods; docs lead owns migration guides.

### Deliverables & Traceability
- ADR/RFC documenting the breaking shift to strict types (no loose mode), removal of legacy untyped paths, and assurance that policy-driven execution/control-flow exceptions/quota patterns are unchanged.
- Tickets per phase:
  - TypeGuards + executor/policy refactor
  - Typed fakes + test migration (and eventual removal of `_detect_mock_objects`)
  - Context mixins + `input_keys`/`output_keys` linting
  - Pipeline typing/linting (no loose-mode shim)
  - Scratchpad ban + docs + removal of unused code paths
- Toggles/config:
  - `strict_dsl` profile flag (always on; no opt-out after compatibility window)
  - `context_strict_mode` flag (always on; no opt-out after compatibility window)
- Tests: unit (TypeGuards, pipeline validation, context lints), integration (strict path plus justified adapter path), regression for mocks→fakes, adapter allowlist, and perf canary thresholds.
- Docs cleanup: remove or rewrite any legacy loose-mode documentation; ensure public docs show only strict typing, typed contexts, and explicit adapters where justified. Include migration guide + codemod usage.

### Success Metrics
- `cast(...)` in core: driven to 0 (replaced by TypeGuards).
- `Any` in DSL/core: substantial reduction; default-`Any` disallowed in strict mode.
- Mock usage in core paths: 0; tests use typed fakes.
- Context contract violations and adapter misuse: caught at lint/typecheck time; perf canary stays within budget.
- Observability shows declining scratchpad writes and untyped chains to near-zero before hard-fail.
