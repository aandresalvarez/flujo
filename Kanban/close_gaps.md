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
   - 🔄 Cast burn-down: progressing; remaining in policies/cache/result handlers

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
   - ⚠️ **Remaining**: `scratchpad` field still exists on `PipelineContext` (domain/models.py:636-642)
   - 🔜 **Next**: Remove redundant dual-writes, migrate remaining readers, then remove field

4) Adapter governance hardening
   - ✅ Explicit tokens required: `from_callable(is_adapter=True)` raises ValueError without adapter_id/adapter_allow (step.py:613-617)
   - ✅ AST-based lint implemented
   - ⚠️ **Note**: Testing may use mocks that bypass validation; verify in integration

5) Baseline reversal (drive `Any`/`cast` down)
   - ✅ Updated baselines: core.cast=1, core.Any=1148, dsl.cast=0, dsl.Any=443
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

++++++
remaining work
Here’s a focused migration plan to fully remove `scratchpad` and finish type safety:

Scope
- Remove `scratchpad` entirely from runtime: no field on `PipelineContext`, no `_allow_scratchpad` backdoors, no legacy migration in `context_adapter`, no scratchpad-based tests/fixtures.
- Preserve backwards compatibility only at the edges by clear, early failures (or documented explicit adapters), not by silently accepting scratchpad.

Plan (ordered)

0) Baseline & guardrails
- Run `make test-fast` once to capture the current failure set after scratchpad removal.
- Keep `make lint`/`make typecheck` as final gates.

1) Core model/runtime hardening (already started)
- `PipelineContext`: no `scratchpad`, validator rejects any payload containing it.
- `context_adapter`: remove `_allow_scratchpad`; reject `scratchpad` updates with clear error; no legacy merge.
- `steps` property: use `step_outputs` only.

2) Test suite migration (bulk of the work)
Triage by area; for each, replace scratchpad usage with typed fields or import_artifacts/step_outputs.

- HITL & pause/resume:
  - Files: `tests/integration/test_hitl_integration.py`, `test_hitl_pipeline.py`, `test_stateful_hitl.py`, `test_hitl_loop_resume_fix.py`, `test_hitl_branch_schemas.py`, HITL migration suites.
  - Actions: use `status`, `pause_message`, `paused_step_input`, `hitl_data`, `hitl_history`, `step_outputs`.
  - Remove scratchpad assertions/fixtures.

- Loop/parallel/state-machine:
  - Files: `tests/integration/test_parallel_step.py`, `test_state_machine_transitions_integration.py`, loop recipe/agentic loop tests.
  - Actions: use `loop_iteration_index`, `loop_step_index`, `loop_last_output`, `step_outputs`/typed fields; update any sink_to scratchpad processors to step_outputs/import_artifacts.

- Import/input routing:
  - Files: `tests/unit/test_import_input_precedence.py`, `test_import_hitl_and_input_routing.py`, import pipeline tests.
  - Actions: map legacy scratchpad outputs to `import_artifacts` or `step_outputs`; adjust assertions.

- Context mixins / enforcement:
  - Files: `tests/domain/test_context_mixins.py`, `tests/application/core/test_typed_context_enforcement.py`, any lenient fixtures.
  - Actions: remove scratchpad fixtures; assert typed fields; ensure validators fail on scratchpad.

- Tracing/console/task replay:
  - Files: `tests/unit/infra/test_console_tracer_paused.py`, `tests/unit/tracing/test_trace_manager_post_run_state.py`, `tests/unit/test_replay_executor.py`, task client lifecycle tests.
  - Actions: set `pause_message`/`status` on context; snapshots use typed fields only.

- Misc policies/utilities:
  - Files: `tests/unit/test_granular_step_policy.py`, `tests/integration/test_parallel_step.py` sink_to, any custom context subclasses with scratchpad fields.
  - Actions: replace scratchpad fields with typed equivalents or `import_artifacts.extras`.

3) Runtime utilities & processors
- Search for `sink_to="scratchpad``, `context.scratchpad[...]`, `scratchpad.` templates.
- For built-ins/processors, write to `step_outputs` or `import_artifacts`; for loop/HITL metadata, use typed loop/HITL fields.

4) Docs and samples
- Update docs/examples to remove scratchpad references; point to typed fields and `step_outputs`/`import_artifacts`.

5) Validation sweep
- `rg scratchpad flujo/ tests/` should return only comments/docs.
- Ensure no `_allow_scratchpad` or legacy flags remain.

6) Full verification
- Run `make test-fast`.
- Run `make lint` and `make typecheck`.
- If time permits, `make all`.

Working approach
- Iterate in slices: migrate a cluster of tests/files, run a targeted pytest subset, then continue.
- Keep changes mechanical and typed-field aligned (status/pause_message/paused_step_input/hitl_data/step_outputs/import_artifacts/loop_*).
- Avoid reintroducing compatibility shims; fail fast on any scratchpad payload.

If you want, I’ll start with the highest-churn suites (HITL/pause/resume and loop/parallel) and proceed in the order above.