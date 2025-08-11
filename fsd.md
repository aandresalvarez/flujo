 Refactor: Flujo Runner (runner.py)

Status: Implemented per FLUJO_TEAM_GUIDE

Summary
- Introduced ExecutionManager at `flujo/application/core/execution_manager.py` to own the step-by-step execution loop.
- `Flujo._execute_steps` now delegates to `ExecutionManager.execute_steps`.
- `Flujo.run_async` focuses on setup/teardown: pipeline resolution, context creation, state load/save, and hook dispatching.
- State and usage policies handled via dedicated managers (`StateManager`, `UsageGovernor`, `StepCoordinator`, `TypeValidator`).

Key Behaviors
- Control-flow exceptions (`PausedException`, `PipelineAbortSignal`) are never swallowed and are re-raised/propagated appropriately.
- Context updates use `ContextManager.merge` and safe injection; no direct field mutation.
- Configuration and telemetry follow composition-root practices; no direct `flujo.toml` reads.
- Final state persistence consolidated in `ExecutionManager.persist_final_state` with support for:
  - Completed runs (normal, HITL resumption, crash recovery)
  - Paused/failed runs with accurate index and last output

Benefits
- Single-responsibility runner, improved readability and testability
- Reusable execution loop, easier to validate in isolation
- Safer state persistence and resumability semantics

Files Touched
- `flujo/application/runner.py`: Delegates execution and finalization to `ExecutionManager`.
- `flujo/application/core/execution_manager.py`: Core execution and final state persistence.
- `flujo/application/core/state_manager.py`, `step_coordinator.py`, `type_validator.py`, `context_manager.py`: Coordination utilities.
