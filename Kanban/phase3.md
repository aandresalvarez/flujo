# Phase 3 Kanban — Policy Decoupling

## To Do

- None (remaining Phase 3 items are now in Done; run gates after merging).

## Done

### Adapt policies to StepPolicy

  - due: 2025-12-03
  - tags: [phase3, policies]
  - priority: high
  - workload: Medium
  - steps:
      - [x] Update remaining default policies (loop/parallel/conditional/dynamic-router/HITL/cache/import) to subclass `StepPolicy[...]` and expose `handles_type` (simple/agent already converted).  
        Run: `pytest tests/application/core/test_step_policies.py tests/application/core/test_executor_core.py`
      - [x] Register policies via `create_default_registry(core)` using StepPolicy instances for defaults; remove callable adapters once converted.  
        Run: `pytest tests/application/core/test_executor_core_conditional_step_dispatch.py tests/application/core/test_executor_core_loop_step_dispatch.py`
    ```md
    Goal: make policies registry-friendly and explicit about handled types.
    ```

### Typing & docs alignment

  - due: 2025-12-06
  - tags: [typing, docs]
  - priority: medium
  - workload: Small
  - steps:
      - [x] Update `FSD.md` Phase 3 section to reflect the new registry API (`create_default_registry(core)`, fallback support) and current StepPolicy status.  
        Run: `pytest tests/architecture/test_architecture_compliance.py`
      - [x] Update `AGENTS.md` to reference the registry API and StepPolicy conversions (simple/agent done; remaining policies pending).  
        Run: `pytest tests/architecture/test_architecture_compliance.py`
    ```md
    Goal: maintain strict typing and documentation parity.
    ```

### Regression gates

  - due: 2025-12-07
  - tags: [tests, regression]
  - priority: medium
  - workload: Small
  - steps:
      - [x] Run focused executor-core suite after wiring:  
        `pytest tests/application/core/test_executor_core.py tests/application/core/test_executor_core_conditional_step_dispatch.py tests/application/core/test_executor_core_loop_step_dispatch.py`
      - [x] Run fast suite to confirm no regressions:  
        `make test-fast` (PASS `output/controlled_test_run_20251128_111200.log`; latest PASS `output/controlled_test_run_20251128_111457.log`)
    ```md
    Goal: ensure Phase 3 changes ship without regressions.
    ```

### Typing & docs alignment (typing)

  - due: 2025-12-06
  - tags: [typing, docs]
  - priority: medium
  - workload: Small
  - steps:
      - [x] Keep `make typecheck` green after registry injection; fix Any leaks in new modules.  
        Run: `make typecheck`
    ```md
    Goal: maintain strict typing and documentation parity.
    ```

### DI-friendly custom policy tests

  - due: 2025-12-05
  - tags: [phase3, testing]
  - priority: medium
  - workload: Medium
  - steps:
      - [x] Add a unit test proving custom registry injection executes a mock policy.  
        Run: `pytest tests/application/core/test_policy_registry.py::test_custom_policy_injection`
      - [x] Add regression ensuring fallback policy is used for unknown steps.  
        Run: `pytest tests/application/core/test_policy_registry.py::test_fallback_policy_used`
    ```md
    Goal: validate DI surface and fallback behavior.
    ```

### Inject registry into ExecutorCore

  - due: 2025-12-04
  - tags: [phase3, executor-core]
  - priority: high
  - workload: Medium
  - steps:
      - [x] Allow `ExecutorCore(policy_registry=...)` and default to `create_default_registry()`.  
        Run: `pytest tests/application/core/test_executor_core.py tests/application/core/test_executor_core_chokepoint.py`
      - [x] Route dispatch through the registry (no inline branching); keep control-flow exceptions unaltered.  
        Run: `pytest tests/application/core/test_executor_core_conditional_step_dispatch.py tests/application/core/test_executor_core_loop_step_dispatch.py`
    ```md
    Goal: ExecutorCore becomes registry-driven without step-specific checks.
    ```

### Build policy registry skeleton

  - due: 2025-12-02
  - tags: [phase3, policies, core]
  - priority: high
  - workload: Medium
  - defaultExpanded: true
  - steps:
      - [x] Add `flujo/application/core/policy_registry.py` with `StepPolicy` protocol/base and `PolicyRegistry` (register/get/register_fallback).  
        Run: `make typecheck`
      - [x] Add `create_default_registry()` factory; export from package `__init__.py`.  
        Run: `pytest tests/application/core/test_policy_registry.py`
    ```md
    Goal: introduce a first-class registry to decouple policy resolution.
    ```

### Reference (Phase 2 baseline)

  - due: 2025-11-27
  - tags: [phase2, reference]
  - priority: low
    ```md
    Baseline: ExecutorCore slimmed to ~603 LOC; make test-fast passing (508/508).
    ```
