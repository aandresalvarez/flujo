# Technical Debt (Verified)

This file tracks *actionable* debt that still exists in the current repo (grounded in code),
and keeps a paper trail of what was resolved vs intentionally re-scoped.

## Current backlog (ordered by ROI: impact ÷ effort)

- None (verified)

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
