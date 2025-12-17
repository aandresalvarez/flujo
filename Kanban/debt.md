# Technical Debt (Verified)

This file tracks *actionable*, code-grounded debt that still exists in Flujo and keeps a paper trail
of what was resolved vs intentionally re-scoped.

Rules for updates:
- Every backlog item must include: impact, effort, evidence (how to verify), and exit criteria.
- “Resolved” items must include verifiable evidence (a quick `rg` command and/or a file pointer).
- If a fix is pending, link the PR and keep the item out of “Resolved” until it lands in `main`.

## Current backlog (ordered by ROI: impact ÷ effort)

### In progress (open PRs)

1) Gate test-mode state backend override in CLI (Impact: High, Effort: S)
   - Problem: `FLUJO_TEST_MODE=1` can cause `load_backend_from_config()` to ignore configured `state_uri`
     and use an isolated temp SQLite DB when `FLUJO_STATE_URI` is not set.
   - Evidence: see test-mode branches under `load_backend_from_config()` in `flujo/cli/config.py`.
   - Fix: only override configured persistence when a test-only opt-in is present
     (`FLUJO_TEST_STATE_DIR` or explicit ephemeral flags like `FLUJO_EPHEMERAL_STATE`).
   - Exit: with `FLUJO_TEST_MODE=1` and no explicit override, CLI honors `flujo.toml` `state_uri`; tests
     set `FLUJO_TEST_STATE_DIR` for hermetic runs.
   - Tracking: PR `#563` — https://github.com/aandresalvarez/flujo/pull/563

2) Remove runtime import in `@step` decorator path (Impact: Medium, Effort: S)
   - Problem: `flujo/domain/dsl/step_decorators.py` imports `Step`/`StepConfig` inside `step()` to avoid
     a circular import with `flujo/domain/dsl/step.py` re-exporting the decorators.
   - Evidence: `rg -n "Import here to avoid circular imports" flujo/domain/dsl/step_decorators.py`.
   - Fix: make module-scope imports safe (e.g., import decorators only after `Step` is defined, or split
     a small `step_core.py` module).
   - Exit: no imports of `.step` inside the decorator call path; `make lint` passes.
   - Tracking: PR `#564` — https://github.com/aandresalvarez/flujo/pull/564

3) Blueprint `processing` parsing is duplicated + untyped (Impact: Medium, Effort: S/M)
   - Problem: `_make_step_from_blueprint()` normalizes `processing` (dict → validated dict) and builders
     later validate/attach processing metadata again.
   - Evidence: `flujo/domain/blueprint/loader_steps.py` (processing normalization) and
     `flujo/domain/blueprint/loader_steps_misc.py` (`_attach_processing_meta`).
   - Fix: represent `processing` as `ProcessingConfigModel` on `BlueprintStepModel` and attach metadata
     from the typed model; remove duplicate parsing in the loader.
   - Exit: `BlueprintStepModel.processing` is typed; processing validation errors are not swallowed;
     YAML tests for `processing` still pass.
   - Tracking: PR `#565` — https://github.com/aandresalvarez/flujo/pull/565

4) Mermaid renderer redundant runtime imports (Impact: Low/Medium, Effort: S)
   - Fix: remove internal “runtime import to avoid circular dependency” blocks in `pipeline_mermaid.py`.
   - Evidence: `rg -n "Runtime import to avoid circular dependency" flujo/domain/dsl/pipeline_mermaid.py`
     returns empty.
   - Tracking: PR `#562` — https://github.com/aandresalvarez/flujo/pull/562

### Backlog (not started)

5) Blueprint loader legacy fallback (builder registry vs plugin class loading) (Impact: Medium, Effort: M)
   - Problem: `_make_step_from_blueprint()` falls back to `flujo.framework.registry.get_step_class`,
     mixing plugin loading with builder dispatch.
   - Evidence: `rg -n "get_step_class" flujo/domain/blueprint/loader_steps.py`.
   - Fix: either deprecate and gate behind an explicit flag, or move plugin resolution behind a dedicated
     extension hook so the loader has a single responsibility.
   - Exit: blueprint step construction uses one explicit path; legacy fallback is opt-in/deprecated.

6) ExecutorCore compatibility shims still exercised internally (Impact: Medium/High, Effort: L)
   - Problem: internal code paths still call compatibility shims (`execute_simple_step`,
     `execute_step_compat` / `_execute_simple_step`), increasing the chance of behavioral drift.
   - Evidence: `rg -n "execute_simple_step|execute_step_compat|_execute_simple_step" flujo/application/core`.
   - Fix: route all internal execution through `ExecutionFrame` + `ExecutorCore.execute()`; keep wrappers
     only for external compatibility, then remove in a major release.
   - Exit: no internal call sites use the shims; shims are deprecated and tested.

7) Telemetry/test helpers read env vars directly at import time (Impact: Low, Effort: M)
   - Problem: `flujo/infra/telemetry.py` reads `CI`/`FLUJO_TEST_MODE` at import time; `flujo/cli/test_setup.py`
     guards on `PYTEST_CURRENT_TEST`/`CI`.
   - Evidence: `rg -n "os\\.getenv\\(\\\"CI\\\"\\)|PYTEST_CURRENT_TEST" flujo/infra/telemetry.py flujo/cli/test_setup.py`.
   - Fix: evaluate whether this can be deferred until after settings are available without introducing
     import cycles; otherwise document as a bootstrap exception.
   - Exit: either moved to settings-based init, or explicitly documented as intentional.

8) Deprecated global agents module hook (Impact: Low, Effort: S)
   - Problem: `flujo/agents/recipes.py` uses module `__getattr__` to raise helpful errors for removed globals.
   - Evidence: `rg -n "def __getattr__" flujo/agents/recipes.py`.
   - Fix: keep until the next major release; then remove along with other deprecated surfaces.
   - Exit: removed in a major cleanup with release notes.

## Resolved / Re-scoped (keep this section honest)

1) `_force_setattr` workaround removed
   - Evidence: `rg -n "_force_setattr" flujo` returns empty.

2) Pydantic `arbitrary_types_allowed` scatter removed (single boundary definition)
   - Evidence: `rg -n "arbitrary_types_allowed" flujo` returns only `flujo/domain/base_model.py`.

3) Removed implicit pytest detection in config manager (explicit `FLUJO_TEST_MODE` only)
   - Evidence: `rg -n "PYTEST_CURRENT_TEST" flujo/infra/config_manager.py` returns empty.
   - Note: remaining `PYTEST_CURRENT_TEST` usage is limited to the test-only helper `flujo/cli/test_setup.py`.

4) Governance tool allowlist uses centralized settings (no direct env reads in registry)
   - Evidence: `rg -n "FLUJO_GOVERNANCE_TOOL_ALLOWLIST" flujo/infra/skill_registry.py` returns empty, and
     the setting is defined in `flujo/infra/settings.py`.

5) Fixed TOML override type mismatch for `governance_tool_allowlist` (`list[str]` → `str`)
   - Evidence: `flujo/infra/config_manager.py` normalizes list values before `setattr`, and
     `tests/unit/test_config_manager.py` includes a TOML list test case.

6) Versioned SQLite migrations (no ad-hoc runtime schema patching)
   - Evidence: `rg -n "PRAGMA user_version" flujo/state/backends/sqlite_migrations.py` shows versioned migration flow.

7) Legacy `step_executor` path removed from the core surface area
   - Evidence: `rg -n "\\bstep_executor\\b" flujo` returns empty.

8) Scratchpad writes removed from execution policies (typed fields only)
   - Evidence: `rg -n "scratchpad writes removed" flujo/application/core/policies` shows typed-only comments/paths.

9) CI architecture gates no longer re-run full suites inside the architecture job
   - Evidence: `rg -n "GITHUB_ACTIONS" tests/architecture` shows CI-only skips/overrides.
