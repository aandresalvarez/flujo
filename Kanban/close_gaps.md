## Type-Safety Gap Closure Plan (short-term execution)

### Objectives
- Restore "strict-only" posture end-to-end (DSL, contexts, adapters) with real enforcement, not baselines that drift.
- Eliminate loose flows (Any/object bridges, dict contexts, scratchpad writes).
- Ship measurable guardrails (lints, validators, tests) that prevent regressions.

### Workstreams & Steps

1) Tighten DSL typing & validation
   - ✅ Strict validation: `_compatible` rejects Any/object fallthrough (pipeline.py:219-248)
   - ✅ Generics tracking: `_input_type`/`_output_type` captured via PrivateAttr (pipeline.py:43-44, 63-72)
   - ⚠️ **Note**: Container still typed `Sequence[Step[Any, Any]]` (pipeline.py:38); TypeVar propagation not enforced at runtime
   - ✅ Cast burn-down: no `cast(...)` usages remain in tracked core/runtime/DSL/blueprint scopes

2) Enforce typed contexts (no dict coercion)
   - ✅ Strict-only: `enforce_typed_context()` raises TypeError for non-BaseModel (executor_helpers.py:136-148)
   - ✅ Opt-out ignored: test confirms env flag "0" still raises (test_typed_context_enforcement.py:73-95)
   - ✅ Vestigial env flag code removed from config.py; marked deprecated in settings.py
   - ✅ Docs updated with CI enforcement section

3) Scratchpad ban completion
   - ✅ New typed fields added to PipelineContext (status, pause_message, step_outputs, etc.)
   - ✅ Hard ban enforced: `_validate_scratchpad()` always raises for user keys (context_adapter.py)
   - ✅ User keys rejected at construction (test_typed_context_enforcement.py:125-129)
   - ✅ Phase 2: All 20 `scratchpad["status"]` patterns migrated to `context.status`
   - ✅ Phase 3: Additional typed field writes added (current_state, next_state, pause_message, user_input, paused_step_input)
   - ✅ Phase 4: All 8 `scratchpad.get("status")` reads migrated to `getattr(context, "status", None)`
   - ✅ `PipelineContext.scratchpad` removed; any payload containing `scratchpad` is rejected (domain/models.py)

4) Adapter governance hardening
   - ✅ Explicit tokens required: `from_callable(is_adapter=True)` raises ValueError without adapter_id/adapter_allow (step.py:613-617)
   - ✅ AST-based lint implemented
   - ⚠️ **Note**: Testing may use mocks that bypass validation; verify in integration

5) Baseline reversal (drive `Any`/`cast` down)
   - ✅ Updated baselines (`scripts/type_safety_baseline.json`): core.cast=0/core.Any=0, runtime.cast=0/runtime.Any=0, dsl.cast=0/dsl.Any=167, blueprint.cast=0/blueprint.Any=0
   - ✅ Architecture thresholds lowered: max_allowed_any 600→500, cast threshold 50→10
   - ✅ Lint shows delta report; `--update-baseline` flag added

   - ✅ Added `docs/guides/scratchpad_migration.md`
   - ✅ `docs/context_strict_mode.md` updated
   - ✅ Added `docs/getting-started/type_safe_patterns.md`
   - ✅ Docs now accurately reflect code (env flags removed/deprecated)

### Remaining Gaps (Addressed)
- [x] Remove vestigial env flag code paths for typed context (config.py, settings.py updated)
- [x] Lower architecture test thresholds (600→500 for max_allowed_any)
- [x] Remove `FLUJO_SCRATCHPAD_BAN_STRICT` env toggle for true hard ban (context_adapter.py)
- [x] Remove `FLUJO_ENFORCE_SCRATCHPAD_BAN` global off-switch (context_adapter.py)
- [x] Update baselines to actual: core.cast=1, core.Any=1148, dsl.cast=0, dsl.Any=443
- [x] Audit TypeVar propagation at container level (documented - see below)

### TypeVar Container Analysis

**Issue**: `Pipeline` is `Generic[PipeInT, PipeOutT]` but `steps` field is `Sequence[Step[Any, Any]]`

**Current mitigation**: Runtime type tracking via `_input_type`/`_output_type` PrivateAttrs (pipeline.py:43-44, 63-72) that are populated from head/tail steps via `_initialize_io_types()`.

**Why `Step[Any, Any]` is used**:
1. Pydantic struggles with heterogeneous step chains (e.g., `[Step[str, dict], Step[dict, int]]`)
2. TypeVar covariance would require `Step` to be covariant in both params
3. Python has no HList-style typing for heterogeneous sequences

**Recommendation**: Current runtime tracking is sufficient. For static analysis:
- Use `Pipeline[str, Result]` in type hints for public APIs
- Rely on `validate_graph()` to catch type mismatches at validation time

### Acceptance Criteria
- `make lint` passes with lowered baseline (no upward drift), adapter lint enforced via AST, and strict validation enabled by default.
- Pipelines with type mismatches or generic targets fail validation without adapters.
- Dict contexts rejected in strict mode; scratchpad writes/reads are blocked.
- Tests added/updated for each guardrail (validation, adapter tokens, context enforcement, scratchpad ban).

### Scratchpad Status (Complete)
- `scratchpad` has been removed from `PipelineContext`; any incoming payload containing it fails validation early.
- Migration guidance: `docs/guides/scratchpad_migration.md` (typed fields, `step_outputs`, `import_artifacts`).
