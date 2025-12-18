# Technical Debt (Verified)

This file tracks *actionable*, code-grounded debt that still exists in Flujo and keeps a paper trail
of what was resolved vs intentionally re-scoped.

Rules for updates:
- Every backlog item must include: impact, effort, evidence (how to verify), and exit criteria.
- “Resolved” items must include verifiable evidence (a quick `rg` command and/or a file pointer).
- If a fix is pending, link the PR and keep the item out of “Resolved” until it lands in `main`.

## Current backlog (ordered by ROI: impact ÷ effort)

- None (verified)

## Resolved / Re-scoped (keep this section honest)

1) Gate test-mode state backend override in CLI
   - Evidence: `rg -n "FLUJO_TEST_STATE_DIR|FLUJO_EPHEMERAL_STATE" flujo/cli/config.py` shows explicit opt-ins.
   - PR: #563

2) Remove runtime import in `@step` decorator path
   - Evidence: `rg -n "Import here to avoid circular imports" flujo/domain/dsl/step_decorators.py` returns empty.
   - PR: #564

3) Blueprint `processing` config typed + validated
   - Evidence: `rg -n "processing: Optional\\[ProcessingConfigModel\\]" flujo/domain/blueprint/loader_models.py`.
   - PR: #565

4) Mermaid renderer redundant runtime imports removed
   - Evidence: `rg -n "Runtime import to avoid circular dependency" flujo/domain/dsl/pipeline_mermaid.py`
     returns empty.
   - PR: #562

5) Blueprint loader legacy fallback removed
   - Evidence: `rg -n "get_step_class|framework\\.registry" flujo/domain/blueprint/loader_steps.py` returns empty.
   - PR: #566

6) ExecutorCore internal shim calls removed
   - Evidence: `rg -n "execute_simple_step|_execute_simple_step" flujo/application/core` shows only shim
     definitions/aliases.
   - PR: #567

7) Telemetry avoids import-time env reads
   - Evidence: `rg -n "os\\.getenv\\(\\\"CI\\\"\\)" flujo/infra/telemetry.py` is only used inside
     `init_telemetry()`.
   - PR: #568

8) `_force_setattr` workaround removed
   - Evidence: `rg -n "_force_setattr" flujo` returns empty.

9) Pydantic `arbitrary_types_allowed` scatter removed (single boundary definition)
   - Evidence: `rg -n "arbitrary_types_allowed" flujo` returns only `flujo/domain/base_model.py`.

10) Removed implicit pytest detection in config manager (explicit `FLUJO_TEST_MODE` only)
    - Evidence: `rg -n "PYTEST_CURRENT_TEST" flujo/infra/config_manager.py` returns empty.
    - Note: remaining `PYTEST_CURRENT_TEST` usage is limited to the test-only helper `flujo/cli/test_setup.py`.

11) Governance tool allowlist uses centralized settings (no direct env reads in registry)
   - Evidence: `rg -n "FLUJO_GOVERNANCE_TOOL_ALLOWLIST" flujo/infra/skill_registry.py` returns empty, and
     the setting is defined in `flujo/infra/settings.py`.

12) Fixed TOML override type mismatch for `governance_tool_allowlist` (`list[str]` → `str`)
   - Evidence: `flujo/infra/config_manager.py` normalizes list values before `setattr`, and
     `tests/unit/test_config_manager.py` includes a TOML list test case.

13) Versioned SQLite migrations (no ad-hoc runtime schema patching)
   - Evidence: `rg -n "PRAGMA user_version" flujo/state/backends/sqlite_migrations.py` shows versioned migration flow.

14) Legacy `step_executor` path removed from the core surface area
   - Evidence: `rg -n "\\bstep_executor\\b" flujo` returns empty.

15) Scratchpad writes removed from execution policies (typed fields only)
   - Evidence: `rg -n "scratchpad writes removed" flujo/application/core/policies` shows typed-only comments/paths.

16) CLI validation helpers bypassed strict typing
   - Resolved: removed module-level `# mypy: ignore-errors` and made helpers pass `mypy --strict`.
   - Key file: `flujo/cli/helpers_validation.py`.

17) CI architecture gates no longer re-run full suites inside the architecture job
    - Evidence: `rg -n "GITHUB_ACTIONS" tests/architecture` shows CI-only skips/overrides.

18) Deprecated global agents module hook (intentionally retained until next major release)
    - Rationale: We keep module `__getattr__` in `flujo/agents/recipes.py` to raise clear upgrade errors for
      removed globals. This is a deprecation UX shim and not a runtime hot path (only triggers on missing attrs).
    - Evidence: `rg -n "def __getattr__" flujo/agents/recipes.py` shows the intentional deprecation hook.
