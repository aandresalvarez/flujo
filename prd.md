# PRD: Ephemeral Runner Mode for Performance Tests

## Background

- Performance-focused integration tests such as `tests/integration/test_performance_with_context_updates.py` execute very small pipelines (often one step) in tight loops to enforce latency regressions.
- `tests/conftest.py:create_test_flujo` always injects `NoOpStateBackend`, which forces every `Flujo` run (even in micro benchmarks) through the full persistence pipeline.
- `RunSession` and `StateManager` serialize the entire `PipelineContext` via `.model_dump()` on every run, then `NoOpStateBackend` re-serializes with `safe_serialize`. The `PerformanceContext` used in the failing test contains ~30 KB of nested data, so serialization dominates runtime when repeated tens of times.
- In CI this overhead pushes the “high-frequency” regression test beyond its 10 s threshold, tripping the `make all` quality gate even though mainline logic is healthy.

## Problem Statement

We lack a way to run pipelines in an “ephemeral” mode that skips state persistence entirely. Tests that only care about runtime still pay the full serialization cost, which accumulates into multi-second latency on shared runners and causes false positives in performance gates.

## Goals

1. Provide a first-class way to disable persistence for pipelines that do not need resume/inspection.
2. Keep persistence enabled by default so existing behavior and durability guarantees remain unchanged.
3. Allow tests (and potentially lightweight CLI commands) to opt into the faster mode with a single flag, without re-implementing runner wiring.

## Non-Goals

- Changing how production runs persist state.
- Altering quality-gate thresholds; the test should pass because the pipeline is faster, not because the bar is lowered.

## Proposed Solution

1. **Runner Flag**
   - Add a `persist_state: bool = True` parameter to `Flujo.__init__` and propagate it into `RunSession`.
   - When `persist_state` is `False`, skip creating a `StateManager` altogether or reuse one configured with a `NullStateBackend`.

2. **StateManager Awareness**
   - Teach `StateManager` helper methods (`persist_workflow_state`, `_serializer` usage, etc.) to no-op immediately when persistence is disabled so we avoid hash/serialization work.

3. **Test Harness Update**
   - Extend `create_test_flujo` with `persist_state: bool = True`.
   - Performance-focused tests call `create_test_flujo(..., persist_state=False)`.

4. **Documentation**
   - Add a short section to `AGENTS.md` (or `FLUJO_TEAM_GUIDE.md`) describing the new flag and when to use it.

## Acceptance Criteria

- When `persist_state=False`, performance tests no longer invoke `StateManager.persist_workflow_state` or `NoOpStateBackend.save_state`.
- Default behavior (flag omitted) matches current persistence semantics; existing persistence tests continue to pass.
- `tests/integration/test_performance_with_context_updates.py::test_performance_with_context_updates_high_frequency` runs under 10 s inside `make all`.
- Unit/integration coverage demonstrating both code paths (persisting vs. ephemeral) is in place.

## Risks & Mitigations

- **Accidentally disabling persistence in tests that assert on saved state**: Keep the flag opt-in and document clearly.
- **Divergence between persisted vs. ephemeral modes**: Minimal risk because the change only short-circuits persistence hooks; step execution remains identical.

