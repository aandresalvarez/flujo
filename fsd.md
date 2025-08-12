 I compiled the remaining gaps and turned them into concrete, high-signal tasks with exact files and expectations.

### Tasks to complete FSD-008 (status: outcome-first mostly complete; remaining policy early-returns and choke-point tests pending)

- Policy normalization to pure outcomes
  - Audit and replace remaining `return StepResult(...)` in `flujo/application/core/step_policies.py` with typed outcomes:
    - Use `to_outcome(sr)` when you already have a `StepResult`.
    - Or construct `Success(step_result=sr)` / `Failure(error=..., feedback=..., step_result=sr)` explicitly where appropriate.
    - Example area to fix:
```2655:2685:flujo/application/core/step_policies.py
return StepResult(
    name=loop_step.name,
    success=False,
    ...
)
```
  - Remove re-raises of `PausedException` inside policy codepaths that are only normal control-flow translations; return `Paused(...)` instead where the policy itself decides to pause. For nested calls that return `Paused`, prefer propagating the `Paused` outcome upward rather than re-raising, unless a legacy contract enforces raising at that boundary.

- Verify and finalize policy method signatures
  - Confirm all `execute(...)` signatures for policies and Protocols explicitly return `-> StepOutcome[StepResult]`. Fix any outliers in `step_policies.py`.

- Add the ExecutorCore choke-point test (FSD 3.4 acceptance)
  - Implemented: `tests/application/core/test_executor_core.py::test_execute_converts_unexpected_exception_to_failure_outcome`

- Strengthen ExecutionManager outcome handling tests
  - Implemented chunk pass-through and aborted termination in `tests/application/core/test_execution_manager.py`.

- Runner outcome-streaming tests (FSD 3.6 acceptance)
  - Implemented in `tests/application/test_runner.py` including paused path.

- Align fsd.md status with actual implementation
  - Update `fsd.md` header status (line 4) from “Outcome-first complete...” to a truthful status (e.g., “Outcome-first mostly complete; remaining policy early-returns and choke-point tests pending”).
  - Link to this task list in the doc for traceability.

- Documentation verification
  - Ensure `docs/advanced/typed_outcomes.md` documents:
    - Policy return contract (`StepOutcome[StepResult]`).
    - `run_outcomes_async` usage patterns and migration notes.
    - `to_outcome`/`unwrap` utilities in `flujo/domain/outcomes.py`.

- Type- and lint-gate the changes
  - Run `make install` and then `make all`; resolve any mypy/lint issues from signature changes and new tests.
  - Keep strict return types in all edited code to satisfy the “consistent type safety” rule.

- Legacy compatibility guardrails
  - Add a focused test that setting `FLUJO_WARN_LEGACY=1` emits a deprecation warning when:
    - Calling `ExecutorCore.execute(step, data, ...)` non-frame path.
    - Using the legacy runner path (`run_async` yielding `PipelineResult`) instead of `run_outcomes_async`.

- Cleanup dead artifacts
  - Audit and remove or integrate `apply_me.patch` and `tmp_step_policies.py` if they are legacy scaffolding. Ensure no duplicate implementations of `to_outcome`/`unwrap`.

- Policy-specific tests (additions)
  - `DefaultSimpleStepExecutor`: success and failure return `Success`/`Failure`.
  - `DefaultLoopStepExecutor`: error in `iteration_input_mapper` returns a typed `Failure` (no raw `StepResult`).
  - `DefaultParallelStepExecutor`: returns `StepOutcome`; verify context merge is not applied on failed branches.
  - `DefaultConditionalStepExecutor`: handles nested policy outcomes correctly and returns typed outcomes, including pass-through of `Paused` and `Failure`.

- ExecutorCore frame-path guarantees
  - Add tests asserting that when called with an `ExecutionFrame`, `ExecutorCore.execute(...)` always returns a `StepOutcome[...]` on all codepaths, including:
    - Cache hit path returns `Success(step_result=...)`.
    - HITL returns `Paused` (frame path).
    - Parallel/Loop/Conditional/Router policies yield typed outcomes.

- Paused handling end-to-end
  - Add an end-to-end test where a policy returns `Paused` (not raised), and verify:
    - `ExecutorCore.execute(frame)` returns `Paused`.
    - `ExecutionManager` converts to `PipelineAbortSignal` path as designed, with context `scratchpad` populated.

- CI tightening
  - Ensure CI runs with `FLUJO_WARN_LEGACY=1` and fails on warnings treated as errors in tests targeting legacy paths to catch regressions.

Status update
- Created a precise, ordered task list tied to concrete files and acceptance checks to finish FSD-008 fully, focusing on normalizing remaining policy returns, adding missing tests, updating docs/status, and tightening type/lint gates.

- Clarified the minimal code edits required: wrap any raw `StepResult` returns to `StepOutcome` and avoid PausedException raises where an outcome is expected.

- Highlighted the choke-point integration test and runner outcome tests as the biggest missing acceptance artifacts.

- Cleaned up documentation and CI guardrails to prevent regression.

- **Policy normalization**: Convert any `return StepResult(...)` to `to_outcome(...)` in `step_policies.py`, and prefer `Paused(...)` over raising in policy-level pause paths.
- **Add tests**: ExecutorCore choke-point; ExecutionManager for `Chunk`/`Aborted`; Runner `run_outcomes_async` happy/streaming/paused; frame-path guarantees; policy-specific unit tests.
- **Docs + status**: Fix `fsd.md` status line; expand `typed_outcomes.md` with contracts and runner API.
- **Type/lint**: Run `make all` and resolve issues.
- **Legacy guardrails**: Add tests for `FLUJO_WARN_LEGACY`.
- **Cleanup**: Remove/integrate `apply_me.patch` and `tmp_step_policies.py`.