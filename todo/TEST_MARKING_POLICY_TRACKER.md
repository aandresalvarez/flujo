# Tracking Issue: Enforce Test Marking Policy (Benchmarks & HITL)

Status: Open
Owner: Core Team
Priority: High (stability of fast suite)

## Summary
Ensure all benchmark and HITL/stateful tests are correctly marked so `make test-fast` remains reliable and fast without ad-hoc file filters.

## Required Policy
- Benchmarks: mark `@pytest.mark.benchmark` and `@pytest.mark.slow` (prefer module-level `pytestmark`).
- HITL/stateful resume (SQLite, interactive steps): mark `@pytest.mark.slow` and `@pytest.mark.serial`.
- Trace replay/persistence integration: mark `@pytest.mark.slow`.
- Fast subset relies on markers only: `not slow and not veryslow and not serial and not benchmark`.

## Tasks
- [ ] Audit `tests/benchmarks/` and ensure module-level `pytestmark = [pytest.mark.benchmark, pytest.mark.slow]`.
- [ ] Audit HITL modules (search: `HumanInTheLoopStep`, `hitl`, or `resume_async`) and enforce `slow+serial`.
- [ ] Audit trace replay/persistence tests (search: `replay_from_trace`, `SQLiteBackend`) and enforce `slow`.
- [ ] Add a pre-commit doc lint or CI check that warns on unmarked benchmark/HITL files (best-effort regex).
- [ ] Keep Makefile `test-fast` selection marker-based only (no file-name kexprs).

## References
- AGENTS.md: Testing Guidelines → Test Marking Policy
- FLUJO_TEAM_GUIDE.md: Testing Standards (Markers & Fast/Slow Split)
- docs/testing.md: Slow tests guidance and Marking Guidance (Required)

