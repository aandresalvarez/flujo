# ADR-001: Policy-Driven Execution

- **Status:** Accepted
- **Date:** 2025-12-02
- **Context:** ExecutorCore used to contain step-type conditionals, making orchestration brittle and hard to extend.
- **Decision:** All step execution logic lives in dedicated `StepPolicy` classes registered via `create_default_registry(core)` in `flujo/application/core/step_policies.py`. The dispatcher never inspects step types directly.
- **Consequences:**
  - ✅ New step behaviors are added by implementing `StepPolicy` and registering it.
  - ✅ ExecutorCore remains a composition root/dispatcher only.
  - ❌ No `isinstance` branching on steps inside ExecutorCore or runners.

