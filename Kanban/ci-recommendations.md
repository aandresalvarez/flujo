# CI Review and Recommendations

## Scope
This review covers GitHub Actions workflows and CI-related scripts in the Flujo repo.

## Assumptions
Docs CI should run only when `docs/**` or `mkdocs.yml` change, per your response.

## Current Landscape
- Main CI runs on `main` and tags, with lint/typecheck, fast+slow tests, coverage, evals, packaging smoke, and optional perf/flake/ultra-slow jobs. `.github/workflows/ci.yml`
- PR checks run quality gates, fast tests (3.13/3.14), unit tests, architecture/security tests, coverage, and perf. `.github/workflows/pr-checks.yml`
- Docs build runs on PR + main pushes, and deploy-docs rebuilds on every main push. `.github/workflows/docs.yml` `.github/workflows/deploy-docs.yml`
- Tag pushes trigger both PyPI publish and GitHub release workflows, each re-running tests and builds. `.github/workflows/release.yml` `.github/workflows/github-release.yml`
- Manual E2E runs are separate and only on demand. `.github/workflows/e2e_tests.yml`

## Recommendations

### 1) Consolidate tag release pipeline
Both `release.yml` and `github-release.yml` re-run tests/builds on tag pushes, and `github-release.yml` builds from `main` instead of the tag. Consolidate into one workflow that runs on tag pushes, checks out the tag commit, and publishes both PyPI and GitHub release assets using a single build output. This removes duplicated work and ensures releases match the tag exactly. `.github/workflows/release.yml` `.github/workflows/github-release.yml`

### 2) Gate docs CI and deploy on docs changes only
Change `docs.yml` and `deploy-docs.yml` to use `paths` so they only run when `docs/**`, `mkdocs.yml`, and any doc-included files (like `README.md` if used by mkdocs) change. This removes a full docs build on unrelated code changes. `.github/workflows/docs.yml` `.github/workflows/deploy-docs.yml`

### 3) Remove redundant unit-only job or path-filter it
`test-unit` duplicates coverage already exercised by `test-fast`. Remove it, or make it conditional on changes in `tests/unit/**` or core runtime paths. This reduces redundant runtime while keeping coverage via `test-fast`. `.github/workflows/pr-checks.yml`

### 4) Fix coverage artifact naming and run coverage once
`test-fast` runs on 3.13 and 3.14 in PRs but uploads the same artifact name (`coverage-fast-pr`). Either run coverage only on 3.13 or make the artifact name versioned. This avoids upload collisions and cuts time. `.github/workflows/pr-checks.yml`

### 5) Shard fast tests using existing pytest-split support
You already have `pytest-split` and a `make test-shard` target. Replace single `-n 2` runs with a shard matrix (for example 2 or 4 shards), each using `make test-shard SHARD_INDEX=... SHARD_COUNT=...`. This shortens wall-clock time without dropping any tests. `Makefile` `.github/workflows/ci.yml` `.github/workflows/pr-checks.yml`

### 6) Use `uv sync --frozen` and narrower extras
Most jobs use `uv sync --all-extras`, which is slow and unnecessary. Use `--frozen` (lockfile only) and install only needed extras per job: `--extra dev` for lint/typecheck/test, `--extra docs` for docs, and no extras for packaging smoke unless required. This speeds up installs while remaining deterministic. `.github/workflows/ci.yml` `.github/workflows/pr-checks.yml` `.github/workflows/docs.yml` `.github/workflows/deploy-docs.yml`

### 7) Avoid redundant dependency sync during typecheck
`make typecheck` runs `uv sync --all-extras` again, which repeats work already done in the job. Add a `typecheck-ci` target (or a `CI=1` guard) that skips the second sync. This reduces CI time without weakening checks. `Makefile`

### 8) Align uv cache/version across workflows
Only `ci.yml` pins `UV_VERSION` and `UV_CACHE_DIR`. Standardize these in PR checks and docs workflows to maximize cache hits and avoid version drift. `.github/workflows/ci.yml` `.github/workflows/pr-checks.yml` `.github/workflows/docs.yml`

## Suggested Order of Implementation
1. Consolidate tag release workflow and ensure tag-based build.
2. Fix coverage artifact naming and reduce coverage to one Python version.
3. Add docs `paths` filters.
4. Replace fast test runs with sharded `make test-shard`.
5. Tighten `uv sync` usage and add `typecheck-ci`.
6. Standardize uv cache/version across workflows.
