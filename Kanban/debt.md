# Technical Debt (Verified)

This file tracks *actionable* debt that still exists in the current repo (grounded in code),
and keeps a paper trail of what was resolved vs intentionally re-scoped.

## Current backlog (ordered by ROI: impact ÷ effort)

1) Test-mode state backend override in CLI
   - Impact: reduce risk of accidental writes to temp DBs when `FLUJO_TEST_MODE=1` is set outside tests.
   - Evidence: `flujo/cli/config.py` `load_backend_from_config()` can ignore `flujo.toml` `state_uri` when
     `settings.test_mode` is enabled and `FLUJO_STATE_URI` is not set.
   - Fix: keep tests hermetic without surprising production by requiring an explicit test-only override
     (e.g., `FLUJO_TEST_STATE_DIR` / `FLUJO_EPHEMERAL_STATE`) before ignoring TOML `state_uri`.

2) DSL decorator import cycle (runtime import in decorator path)
   - Impact: clearer dependency graph; better static analysis/IDE support.
   - Evidence: `flujo/domain/dsl/step_decorators.py` imports `Step`/`StepConfig` inside `step()` to avoid a
     cycle with `flujo/domain/dsl/step.py` re-exporting the decorators.
   - Fix: break the cycle so decorators can import symbols normally (e.g., lazy export in `step.py` or a
     small `step_core.py` module for `Step`/`StepConfig`).

3) Blueprint loader mixes registry dispatch, legacy class loading, and config parsing
   - Impact: simpler loader; fewer edge cases; easier to extend.
   - Evidence: `flujo/domain/blueprint/loader_steps.py` `_make_step_from_blueprint()` parses `processing`
     and then falls back to `flujo.framework.registry` when no builder is registered.
   - Fix: move `processing` normalization into builders; define an explicit deprecation/flag path for the
     legacy fallback rather than silently mixing concerns.

4) Executor compatibility shims still exercised internally
   - Impact: reduce surface area for behavioral drift between execution paths.
   - Evidence: `flujo/application/core/executor_helpers.py` (`execute_simple_step`, `execute_step_compat`)
     and callers in loop orchestration.
   - Fix: route callers through `ExecutionFrame` + policy dispatch; keep shims as thin wrappers until a
     deprecation window closes.

5) Deprecated global agents module hook
   - Impact: low; mostly cosmetic.
   - Evidence: `flujo/agents/recipes.py` module `__getattr__`.
   - Fix: keep unless profiling proves it matters; remove in a major version cleanup.

## Resolved / Re-scoped (keep this section honest)

1) `# mypy: ignore-errors` in core policies
   - Resolved: removed module-level ignores; policies now typecheck under `mypy --strict`.
   - Evidence: `rg "^# mypy: ignore-errors" flujo/application/core/policies` returns empty.
   - Key files: `flujo/application/core/policies/parallel_policy.py`,
     `flujo/application/core/policies/loop_iteration_runner.py`,
     `flujo/application/core/policies/agent_policy_run.py`.

2) Blueprint loader special-cases
   - Resolved: StateMachine and YAML boolean-key coercion are handled via registry/builder paths.
   - Key files: `flujo/domain/blueprint/builder_registry.py`, `flujo/domain/blueprint/loader_models.py`,
     `flujo/domain/blueprint/loader_steps.py`.

3) SQLite runtime schema patching
   - Resolved: schema changes are versioned via `PRAGMA user_version` migrations.
   - Key files: `flujo/state/backends/sqlite_migrations.py`, `flujo/state/backends/sqlite_core.py`.

4) Scratchpad scaffolding
   - Resolved: scratchpad enforcement/messages are centralized; execution-time duplicate checks were reduced.
   - Key files: `flujo/utils/scratchpad.py`, `flujo/application/core/context_adapter.py`,
     `flujo/application/run_session.py`, `flujo/application/core/policies/parallel_policy.py`.

5) Legacy `step_executor` execution path
   - Resolved: legacy `step_executor` plumbing removed from core and tests.

6) Test/CI env hooks in CLI paths
   - Resolved: CLI behavior relies on explicit `settings.test_mode` (`FLUJO_TEST_MODE`) rather than implicit `CI`;
     `PYTEST_*` env vars are only consulted when `test_mode` is already enabled for test DB isolation.

7) Pydantic v2.11 deprecation warnings during fallback context merges
   - Resolved: `ContextManager.merge` fallback path avoids triggering Pydantic instance-access deprecations.
   - Key file: `flujo/application/core/context_manager.py`.

8) Pydantic `arbitrary_types_allowed` config scatter
   - Resolved: `arbitrary_types_allowed=True` is now defined only on the domain boundary base model.
   - Key files: `flujo/domain/base_model.py`, `flujo/domain/dsl/step.py`, `flujo/domain/resources.py`,
     `flujo/domain/models.py`.

9) CLI validation helpers bypassed strict typing
   - Resolved: removed module-level `# mypy: ignore-errors` and made helpers pass `mypy --strict`.
   - Key file: `flujo/cli/helpers_validation.py`.

10) Mermaid visualization runtime imports
   - Resolved: `pipeline_mermaid.py` no longer performs redundant runtime imports of core DSL step types.
   - Key file: `flujo/domain/dsl/pipeline_mermaid.py`.
