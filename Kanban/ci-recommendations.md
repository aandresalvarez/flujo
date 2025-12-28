# CI Review and Recommendations

## Scope
This review covers GitHub Actions workflows and CI-related scripts in the Flujo repo.

## Principles and Assumptions
- CI should be deterministic and aligned with repo entrypoints (prefer Makefile targets).
- Release artifacts must be built from the tag commit, not from `main`.
- Docs builds include API docs via mkdocstrings, so docs CI must run on code changes too.
- Avoid redundant work, but never at the expense of correctness or coverage signal.

## Current Landscape
- Main CI runs on `main` and tags, with lint/typecheck, fast+slow tests, coverage, evals, packaging smoke, and optional perf/flake/ultra-slow jobs. `.github/workflows/ci.yml`
- PR checks run quality gates, sharded fast tests (3.13/3.14), architecture/security tests, coverage, and perf. `.github/workflows/pr-checks.yml`
- Docs build runs on PR + main pushes when docs or API inputs change; deploy-docs is path-filtered. `.github/workflows/docs.yml` `.github/workflows/deploy-docs.yml`
- Tag pushes trigger a consolidated release workflow that builds/tests once and publishes to PyPI and GitHub releases. `.github/workflows/release.yml`
- Manual E2E runs are separate and only on demand. `.github/workflows/e2e_tests.yml`

## Recommendations

### 1) Keep the tag release pipeline consolidated (single build from tag)
Ensure the release workflow checks out the tag commit and publishes both PyPI and GitHub release assets from a single build output. Preserve the PyPI environment gate and least-privilege permissions so releases match the tag exactly. `.github/workflows/release.yml`

### 2) Gate docs CI and deploy on actual docs inputs (not docs-only)
Docs builds include API docs (`mkdocstrings` reads `flujo/**`), so path filters must include code changes too. Add `paths` for `docs/**`, `mkdocs.yml`, `README.md` (if used), `flujo/**`, `scripts/check_docs_links.py`, `pyproject.toml`, and `uv.lock`. This avoids docs builds on unrelated changes while still catching API doc breakages. `.github/workflows/docs.yml` `.github/workflows/deploy-docs.yml` `mkdocs.yml`

### 3) Remove redundant unit-only job
`test-unit` duplicates coverage already exercised by `test-fast` across the same code paths. Remove it and keep Python 3.14 coverage in the `test-fast` matrix. If CI time is still high, prefer reducing coverage to one Python version rather than running a separate unit-only job. `.github/workflows/pr-checks.yml`

### 4) Run coverage once per PR (and name artifacts clearly if multi-version)
Coverage collection is expensive. Run coverage only on a single Python version (e.g., 3.13) and keep the other matrix entry non-coverage, or version the artifact names and update the download pattern accordingly. This reduces runtime without losing the fast-test signal. `.github/workflows/pr-checks.yml`

### 5) Shard fast tests safely using pytest-split
You already have `pytest-split`, but `make test-shard` runs the full suite. Add a `test-shard-fast` target (same markers as `test-fast`, `-p no:randomly`) and replace single `-n 2` runs with a shard matrix (e.g., 2 or 4 shards). This shortens wall-clock time without expanding the suite. `Makefile` `.github/workflows/ci.yml` `.github/workflows/pr-checks.yml`

### 6) Use narrower extras now; add `--frozen` only when `uv.lock` is tracked
Most jobs use `uv sync --all-extras`, which is slow and unnecessary. Use `uv sync --extra dev --extra templating --extra skills --extra aop-extras` for lint/typecheck/tests (mypy imports these optional modules) and `uv sync --extra docs` for docs, and keep docs CI on `uv` rather than `pip install`. Only add `--frozen` once `uv.lock` is checked into the repo. Packaging can keep `--extra dev` until a dedicated build extra exists. `.github/workflows/ci.yml` `.github/workflows/pr-checks.yml` `.github/workflows/docs.yml` `.github/workflows/deploy-docs.yml`

### 7) Avoid redundant dependency sync during typecheck
`make typecheck` runs `uv sync` again. Use `make typecheck-fast` in CI after deps are installed, or add a `CI=1` guard to skip the extra sync. This reduces CI time without weakening checks. `Makefile`

### 8) Align uv cache/version across workflows
Only `ci.yml` pins `UV_VERSION` and `UV_CACHE_DIR`. Standardize these in PR checks and docs workflows to maximize cache hits and avoid version drift. `.github/workflows/ci.yml` `.github/workflows/pr-checks.yml` `.github/workflows/docs.yml`

## Suggested Order of Implementation
1. Consolidate tag release workflow and ensure tag-based build.
2. Add docs `paths` filters that include code inputs.
3. Remove redundant `test-unit` job.
4. Run coverage once per PR (and name artifacts clearly if multi-version).
5. Replace fast test runs with sharded `test-shard-fast`.
6. Tighten `uv sync` usage and use `typecheck-fast`.
7. Standardize uv cache/version across workflows.
