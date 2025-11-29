# Fix Kanban — Gate Blockers

## To Do

### Monolith refactors (architecture gate)
  - due: 2025-12-12
  - tags: [architecture, monolith, refactor]
  - priority: medium
  - workload: Large
  - steps:
      - [x] Slim `flujo/application/core/agent_orchestrator.py` (1489 LOC) and `flujo/application/runner.py` (1497 LOC) (<1200 LOC each) aligned with Phase 4 plan.  
        Sub-steps (agent orchestrator):
          - [x] Extract the retry/fallback attempt loop into `agent_execution_runner.py` (<1200 LOC) and keep the orchestrator wrapper thin.  
          - [x] Extract plugin handling into a small helper (e.g., `agent_plugin_runner.py`) so runner stays lean.  
        Sub-steps (runner):
          - [x] Split runner orchestration into execution/telemetry helpers (e.g., `runner_execution.py`, `runner_telemetry.py`) to drop main file under the gate. *(execution/resume/replay moved to `runner_execution.py`; telemetry remains inline but file now 1041 LOC)*  
        Challenges:
          - High coupling between agent orchestrator and retry/fallback state (attempt counters, token/cost tracking, telemetry hooks) increases extraction risk.
          - Plugin dispatch relies on inline mutation of `processed_output` and primary token accounting; needs clear return types per `docs/advanced/typing_guide.md` (no `Any`).
          - Limited targeted tests for agent orchestrator: rely on `tests/application/test_runner.py` and integration flows, so we need incremental, small extractions with frequent runs.
        Run: `pytest tests/application/test_runner.py`
      - [ ] Plan/phase refactors for remaining monoliths: `flujo/cli/helpers.py`, `flujo/cli/dev_commands.py`, `flujo/domain/blueprint/loader.py`, `flujo/builtins.py`, `flujo/validation/linters.py`, `flujo/state/backends/sqlite.py`, `flujo/domain/dsl/pipeline.py`.  
        Run: `pytest tests/architecture/test_type_safety_compliance.py::TestArchitectureCompliance::test_no_monolith_files`
    ```md
    Goal: clear monolith gate (<1200 LOC) for all flagged files.
    ```

## Done

### Robustness fixes
  - due: 2025-12-05
  - tags: [robustness, concurrency, memory]
  - priority: high
  - workload: Medium
  - steps:
      - [x] Fix signal handling cancellation path so `tests/robustness/test_error_recovery.py::TestErrorRecovery::test_signal_handling_during_execution` cancels tasks correctly.  
        Run: `pytest tests/robustness/test_error_recovery.py::TestErrorRecovery::test_signal_handling_during_execution`
      - [x] Optimize high-concurrency handling to meet <150ms budget in `tests/robustness/test_performance_regression.py::TestScalabilityRegression::test_high_concurrency_handling`.  
        Run: `pytest tests/robustness/test_performance_regression.py::TestScalabilityRegression::test_high_concurrency_handling`
      - [x] Resolve cache thread-safety import path (`ModuleNotFoundError: flujo.infrastructure`) in `tests/robustness/test_concurrency_safety.py::TestConcurrencySafety::test_cache_thread_safety`.  
        Run: `pytest tests/robustness/test_concurrency_safety.py::TestConcurrencySafety::test_cache_thread_safety`
      - [x] Eliminate executor memory leak (819 new objects) in `tests/robustness/test_memory_leak_detection.py::TestMemoryLeakDetection::test_no_executor_memory_leak_on_repeated_execution`.  
        Run: `pytest tests/robustness/test_memory_leak_detection.py::TestMemoryLeakDetection::test_no_executor_memory_leak_on_repeated_execution`
    ```md
    Goal: robustness suite fully green under concurrency and leak detection.
    ```

### HITL/pipeline integration fixes
  - due: 2025-12-05
  - tags: [hitl, pipeline, integration]
  - priority: high
  - workload: Medium
  - steps:
      - [x] Fix HITL sink-to nested branch propagation (scratchpad should carry `user_input`) to pass `tests/integration/test_hitl_sink_to_nested.py::test_hitl_sink_to_in_conditional_branch`.  
        Run: `pytest tests/integration/test_hitl_sink_to_nested.py::test_hitl_sink_to_in_conditional_branch`
      - [x] Fix HITL loop resume context mutation (PipelineContext.steps setter error) for `tests/integration/test_hitl_loop_resume_fix.py::test_hitl_in_loop_no_nesting_on_resume`.  
        Run: `pytest tests/integration/test_hitl_loop_resume_fix.py::test_hitl_in_loop_no_nesting_on_resume`
    ```md
    Goal: HITL state/loop resume behaves correctly with context updates.
    ```

### Loop monolith split
  - tags: [architecture, monolith]
  - status: done
  - notes: `loop_policy.py` now 338 LOC, `loop_iteration_runner.py` 774 LOC (both under 1200 gate).
  - steps:
      - [x] Extract while-loop runner into `loop_iteration_runner.py`.
      - [x] Extract conversation/history handling into `loop_history.py`.
      - [ ] (optional) Extract HITL handling/resume state into a `loop_hitl_orchestrator.py` helper.
      - [ ] (optional) Extract iteration mappers/output mappers into a `loop_mapper.py` module.
    Run: `pytest tests/unit/test_loop_step_policy.py tests/application/core/test_executor_core_execute_loop.py`
