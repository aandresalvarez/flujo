Flujo application/core module map

This package is the execution engine for Flujo. It is intentionally decomposed to
keep policy-driven execution and avoid monolithic executors. Step execution logic
remains in `step_policies.py`.

Top-level entry points (stable)
- executor_core.py: composition root and orchestrator for core execution.
- executor_protocols.py: protocol interfaces for pluggable components.
- step_policies.py: step execution logic. Do not move policies out of core.

Core subpackages
- agents/: agent execution runners, fallbacks, and orchestration helpers.
- context/: context adapters, isolation strategies, and update helpers.
- execution/: dispatch, execution management, and step/result handlers.
- orchestration/: pipeline, loop, HITL, and conditional orchestrators.
- policy/: policy registry, primitives, handlers, and governance policy.
- policies/: concrete step policy executors (routing via step_policies).
- runtime/: runtime builder and default components/managers.
- state/: state persistence and step history.
- support/: shared helpers, type utilities, and validation support.

Compatibility
- Root-level modules continue to re-export for existing imports; new code should
  prefer the subpackage locations for clarity.
