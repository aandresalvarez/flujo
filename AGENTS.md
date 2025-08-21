This is the concise contributor guide for the Flujo repository. For deeper architectural context and rationale, see the full `FLUJO_TEAM_GUIDE.md`.

## Project Structure & Module Organization
- **Source**: `flujo/` — Core library (agents, application, cli, domain, processors, steps, utils, telemetry).
- **Tests**: `tests/` — `unit/`, `integration/`, `e2e/`, `benchmarks/`.
- **Docs & Samples**: `docs/`, `examples/`, `assets/`.
- **Dev Scripts**: `scripts/`.
- **Root**: `pyproject.toml`, `Makefile`, `flujo.toml`, `README.md`.

## Build, Test, and Development Commands
- **Setup**: `make install` — Creates `.venv` and syncs dependencies via `uv`.
- **CLI**: `uv run flujo --help` — Lists available commands.
- **Format/Lint**: `make format` / `make lint`.
- **Typecheck**: `make typecheck` — Runs `mypy --strict` on `flujo/`.
- **Tests**: `make test-fast` (quick subset), `make test` (full), `make testcov` (HTML coverage to `htmlcov/`).
- **All-in-One Check**: `make all` — Runs format, lint, typecheck, and tests. **This must pass before any PR is merged.**
- **Package**: `make package` — Builds distributions to `dist/`.

## Coding Style & Naming Conventions
- **Language**: Python 3.11+, 4-space indent, 100-col lines.
- **Typing**: Full type hints are mandatory. Must pass `mypy --strict`.
- **Naming**: Files/modules `snake_case.py`; functions/vars `snake_case`; classes `PascalCase`; constants `UPPER_SNAKE_CASE`.
- **Formatting**: `ruff format`; lint with `ruff check` (fix or justify).

## Testing Guidelines
- **Framework**: `pytest` with markers `fast`, `slow`, `serial`, `benchmark`, `e2e`.
- **Layout**: Place tests under `tests/<area>/test_*.py`.
- **Run**: Iterate with `make test-fast`; verify with `make test`.
- **Coverage**: Review via `make testcov` (HTML report).
- **Critical Principle**: Fix root causes. **Never** change test expectations or performance thresholds simply to make a failing build "green." Test failures are signals of real regressions.

## Commit & Pull Request Guidelines
- **Commits**: `Type(Scope): summary` (e.g., `Fix(serialization): handle BaseModel recursion`).
- **PRs**: Clear description, linked issues (e.g., `Fixes #123`), tests added/updated, rationale, logs/screenshots.
- **PR Gate**: `make all` **must pass with zero errors** before a PR can be merged. No exceptions.

---

## **Core Architectural Principles (Non-Negotiable)**

Adherence to these patterns is mandatory for all contributions.

### ✅ **1. Policy-Driven Execution**
- **The Rule**: All step execution logic belongs in a dedicated **policy** class in `flujo/application/core/step_policies.py`.
- **The Anti-Pattern**: **Never** add step-specific logic (`if isinstance(step, ...):`) to `ExecutorCore`. The core is a dispatcher only.

### ✅ **2. Control Flow Exception Safety**
- **The Rule**: Control flow exceptions (`PausedException`, `PipelineAbortSignal`, `InfiniteRedirectError`) orchestrate the workflow.
- **The Fatal Anti-Pattern**: **Never** catch a control flow exception and convert it into a data failure (e.g., a `StepResult(success=False)`). Always re-raise it to let the runner handle orchestration.

### ✅ **3. Context Idempotency**
- **The Rule**: Step execution must be idempotent. A failed attempt must not "poison" the context for subsequent retries.
- **The Pattern**: **Always** use `ContextManager.isolate()` to create a pristine context copy for each iteration of a complex step (e.g., `LoopStep`, `ParallelStep`). Only merge the context back upon successful completion of the iteration/branch.

### ✅ **4. Proactive Quota System**
- **The Rule**: Resource limits are enforced proactively via the **Reserve -> Execute -> Reconcile** pattern.
- **The Anti-Pattern**: **Never** introduce reactive, post-execution checks. The `breach_event` and legacy "governor" patterns are disallowed. Use `Quota.split()` for parallel branches.

### ✅ **5. Centralized Configuration**
- **The Rule**: Access all configuration via `flujo.infra.config_manager` and its helpers (`get_settings()`, etc.).
- **The Anti-Pattern**: **Never** read `flujo.toml` or environment variables directly in policies or domain logic.

### ✅ **6. Agent Creation**
- **The Rule**: Use the factories in `flujo.agents` to create agents.
  - `make_agent_async`: For a production-ready agent with retries, timeouts, and auto-repair.
  - `make_agent`: For low-level `pydantic-ai` agent creation (less common).
