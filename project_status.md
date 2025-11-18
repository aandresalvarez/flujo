# Project Status — November 17, 2025

This note is meant to replace the stale status docs scattered through the repo. It explains what Flujo is today and captures the current state of the code, tests, and known issues as of **Mon Nov 17 09:51:18 PST 2025** (updated after rerunning the attached bug demo evidence).

---

## What Flujo Is

- **Product**: Flujo is a conversational AI workflow server/CLI that lets you design, run, and inspect pipelines that combine LLM “agents,” HITL (human-in-the-loop) steps, and custom Python skills. The README describes the intended experience as “idea → production in a single conversation” via `flujo init`, `flujo create`, and `flujo run`, with additional tooling under `flujo lens` for replay/debug (`README.md`).
- **Core packages**: All runtime code lives under `flujo/` (agents, application/executor, CLI, processors, steps, utils, telemetry). Tooling and docs live in `scripts/`, `docs/`, `examples/`, and `assets/`.
- **Build/Test tooling**: Dependency and toolchain management is centralized in `pyproject.toml`/`uv.lock`; `make install`, `make test-fast`, and `make all` still define the expected workflow.

---

## Current Snapshot

| Area | Status |
| --- | --- |
| **Version** | `pyproject.toml` is pinned at `0.4.38` with notes referencing the HITL loop resume fix (`pyproject.toml:10`). |
| **Branch health** | Local `main` diverges from `origin/main` and contains a large `bug_reports/bug_demo` evidence pack, new SQLite shards, and edits to `flujo/application/core/step_policies.py` (`git status -sb`). |
| **Docs** | Numerous historical status docs (e.g., `FINAL_STATUS.md`, `TESTS_STATUS.md`) contradict each other. Treat them as archival only. |
| **Focus area** | All recent code churn is inside `DefaultLoopStepExecutor` and related HITL resume logic, indicating that nested-loop HITL resumes remain the top priority (`flujo/application/core/step_policies.py`). |
| **External reports** | The `bug_reports/bug_demo` folder documents a critical “nested loop still broken” scenario captured on Oct 4, 2025 with traces/debriefs meant for escalation to the Flujo team (`bug_reports/bug_demo/NESTED_LOOP_STILL_BROKEN.md`). |

---

## Verification Performed Today

All commands were run from `/Users/alvaro1/Documents/Coral/Code/flujo/flujo`.

| Command | Purpose | Result |
| --- | --- | --- |
| `uv run pytest tests/integration/test_hitl_loop_minimal.py` | Validates the minimal HITL loop regression test referenced by older docs. | ✅ Pass (0.25s). Confirms the minimal reproduction now works. |
| `uv run pytest tests/integration/test_hitl_loop_resume_simple.py` | Covers the richer HITL loop resume scenarios. | ✅ Pass (0.43s). All four parametrized cases succeed. |
| `uv run pytest tests/integration/test_caching_and_fallbacks.py::test_loop_step_fallback_continues` | Previously flagged as failing due to plugin retry vs fallback interplay. | ✅ Pass (0.14s). No repro of the earlier failure. |
| `uv run pytest tests/integration/test_hitl_loop_regression.py` | New regression test that loads the YAML from `bug_reports/bug_demo/test_hitl_loop_local.yaml` and exercises three pause/resume cycles. | ✅ Pass (0.18s). Confirms the loop now exits cleanly with `context.action == 'finish'`. |
| `uv run python - <<'PY' …` (manual runner driving `bug_demo` YAML) | Uses `Flujo` runner + `InMemoryBackend` to simulate CLI resumes with responses `yes, yes, ""`. | ✅ Confirms the stub pipeline finishes with context `count==3`, `action=="finish"`, and no nested loop spans. |

> **Implication**: The automated regression tests inside this repo are now green, so the “tests failing” statements in `TESTS_STATUS.md` and `FINAL_STATUS.md` are outdated.

---

## Outstanding Issues & Open Questions

1. **External repro (clarification pipeline)**  
   - The `bug_reports/bug_demo` evidence kit points to `/Users/alvaro1/Documents/Coral/Code/cohortgen/projects/clarification`, but that directory does not exist on this machine (only `projects/clar/`). We still need the real project path to capture a fresh production trace after the fixes. The Oct 4, 2025 trace remains the most recent external evidence.

2. **Stateful artifacts in the repo**  
   - SQLite DB shards (`flujo_ops.db*`, `tmp_dbg.db-*`, `tmp_perf.db-*`) and large JSON traces are tracked in the working tree. Confirm whether they should be committed or moved to `output/`/`.gitignore`.

3. **Docs vs reality drift**  
   - Multiple “FINAL” reports co-exist with conflicting claims. When making decisions, rely on current code/tests plus this document rather than historical summaries.

4. **Policy code complexity**  
   - `DefaultLoopStepExecutor` has grown substantially (thousands of lines) and still mixes resume detection, context merging, and telemetry. Further refactors may be needed to keep policy logic comprehensible and testable.

5. **.flujo project state**  
   - Running CLI repros still emits `Error: Database directory .../.flujo does not exist`. The evidence pack assumes a `flujo init` was run in `bug_demo/`. Decide whether to check in a minimal `.flujo/` folder or update instructions to run `flujo init` before reproductions.

---

## Suggested Next Steps

1. **Capture a fresh clarification trace** using the real project path referenced in the bug demo so we can confirm the fix in a production pipeline (the repo currently lacks that directory).
2. **Clean up the evidence pack**: either convert `bug_reports/bug_demo` into a formal issue/pr or archive it to keep `main` focused.
3. **Create/init the `.flujo/` directory under `bug_reports/bug_demo`** (or document the requirement) so CLI reproductions stop emitting the missing-DB warning.
4. **Run `make test-fast` (and eventually `make all`)** before additional merges to ensure no other regressions exist outside the targeted HITL coverage.
5. **Document the verified behavior** in `docs/testing.md` or a changelog entry so future contributors know the current baseline for HITL in loops (and point them to the new regression test).

---

_Maintainer note_: Update this file whenever new verification runs or bug investigations change the status so contributors have a single source of truth.
