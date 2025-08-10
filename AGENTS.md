# Repository Guidelines

## Architecture & Patterns
- Source of truth: see `FLUJO_TEAM_GUIDE.md` for deep guidance.
- Policy-driven design: add/modify behaviors via policies (not in `ExecutorCore`).
- Control-flow exceptions: never swallow; re-raise (`PausedException`, `PipelineAbortSignal`).
- Context: use `ContextManager` and `safe_merge_context_updates` (no direct field mutation).
- Configuration: access via `ConfigManager` and `get_settings()`; do not read `flujo.toml` directly.
- Agents: create via `flujo.agents.factory.make_agent` or wrapper `make_agent_async`.

## Project Structure & Module Organization
- `flujo/`: Core library (agents, application, cli, domain, processors, steps, utils, telemetry).
- `tests/`: Organized by type: `unit/`, `integration/`, `e2e/`, `benchmarks/`.
- `examples/`, `docs/`, `assets/`: Samples, documentation, and static assets.
- `scripts/`: Developer utilities.
- Root: `Makefile`, `pyproject.toml`, `README.md`, `flujo.toml`.

## Build, Test, and Development Commands
- `make install`: Create `.venv` and sync deps via `uv` (required).
- `make format` / `make lint`: Format with Ruff; run lint checks.
- `make typecheck`: Strict mypy on `flujo/`.
- `make test`: Full pytest run. Use `make test-fast` for parallel subset; `make testcov` for HTML coverage in `htmlcov/`.
- `make all`: Format, lint, typecheck, tests. `make package`: build dists to `dist/`.
- CLI: `uv run flujo --help` (after `make install`).

## Coding Style & Naming Conventions
- Python 3.11+, 4-space indent, line length 100 (Ruff).
- Full type hints; code must pass `mypy --strict`.
- Naming: files/modules `snake_case.py`; functions/vars `snake_case`; classes `PascalCase`; constants `UPPER_SNAKE_CASE`.
- Formatting: `ruff format`; Linting: `ruff check` (fix or justify).

## Testing Guidelines
- Framework: `pytest` with markers (`fast`, `slow`, `serial`, `benchmark`, `e2e`).
- Layout: put tests under `tests/<area>/test_*.py`.
- Run: `make test-fast` during iteration; `make test` before PR; review coverage via `make testcov`.
- Engineering principle: fix root causes—do not change test expectations or raise perf thresholds to “green” builds (see team guide).

## Commit & Pull Request Guidelines
- Commits: `Type(Scope): summary` (e.g., `Fix(serialization): handle BaseModel recursion`).
- PRs: clear description, linked issues (e.g., `Fixes #123`), tests added/updated, rationale, and relevant logs/screenshots. Ensure `make all` passes.
