This proposal is aligned with `FLUJO_TEAM_GUIDE.md` and `AGENTS.md`. The goal is to
improve discoverability without violating policy-driven execution or dependency
boundaries.

## Non-negotiables (from the guides)
- Step execution logic lives in `flujo/application/core/step_policies.py`. Do not move
  it or add step-specific branching in `ExecutorCore`.
- Policies are registered via `create_default_registry(core)` and routed by the
  dispatcher.
- Control-flow exceptions must be re-raised; do not convert them to `StepResult`.
- Complex steps must isolate context via `ContextManager.isolate()` and merge only on
  success.
- Configuration access goes through `flujo.infra.config_manager` only.

## Current module map (source of truth)

- Executor composition root: `flujo/application/core/executor_core.py`
- Protocols: `flujo/application/core/executor_protocols.py`
- Default components: `flujo/application/core/runtime/default_components.py`
- Policies: `flujo/application/core/step_policies.py`
- Runner facade and orchestration: `flujo/application/runner*.py` and
  `flujo/application/runner_components/`

## Structural issues and aligned improvements

### 1) Root builtins crowding (safe with shims)
Observation: multiple `builtins_*.py` modules live at the top level.
Recommendation: introduce `flujo/builtins/` for organization, but keep legacy
`flujo/builtins_*.py` modules as compatibility shims that re-export and warn. Do
not remove old modules until a major version or a documented deprecation window.

### 2) infra vs infrastructure (keep compatibility)
Observation: `flujo/infrastructure/` is a compatibility shim (currently caching).
Recommendation: keep `flujo/infrastructure` as a shim that re-exports from
`flujo/infra` and emits deprecation warnings. Move implementation to `flujo/infra`
only if it does not change import paths for existing users/tests.

### 3) CacheStep placement (avoid domain -> infra dependency)
Observation: `CacheStep` lives in `flujo/steps/cache_step.py`, while other steps
are in `flujo/domain/dsl/`.
Constraint: `domain` must not depend on `infra`.
Recommendation: either keep `CacheStep` in `flujo/steps`, or, if you want it in
`domain`, first move the cache backend protocol into a domain-level interface and
keep infra implementations in `flujo/infra`. Re-export `CacheStep` in
`flujo/domain/dsl/__init__.py` for discoverability instead of relocating it.

### 4) application/core discoverability (re-map, do not relocate policies)
Observation: `flujo/application/core/` is large.
Recommendation: keep `step_policies.py` in place. Improve navigation via:
- a short module map in `flujo/application/core/README.md`
- consistent file naming (`execution_*`, `*_manager`, `*_handler`) within core
- explicit re-exports in `flujo/application/core/__init__.py`

### 5) Facade directories (public API only)
Observation: `flujo/models` and `flujo/type_definitions` re-export internal modules.
Recommendation: keep them only if they are intended as stable public entry points.
If internal, deprecate and remove with a migration path.

## Compatible target structure (example)

```text
flujo/
├── application/
│   ├── core/
│   │   ├── executor_core.py
│   │   ├── executor_protocols.py
│   │   ├── step_policies.py
│   │   ├── runtime/
│   │   │   └── default_components.py
│   │   ├── execution/
│   │   │   └── execution_manager.py
│   │   ├── orchestration/
│   │   │   └── step_coordinator.py
│   │   ├── context/
│   │   │   └── context_manager.py
│   │   ├── policy/
│   │   │   └── policy_registry.py
│   │   ├── state/
│   │   │   └── state_manager.py
│   │   ├── support/
│   │   │   └── type_validator.py
│   │   └── ...
│   ├── runner.py
│   ├── runner_execution.py
│   └── runner_components/
├── domain/
│   └── dsl/
├── infra/
├── steps/                # Optional; keep if CacheStep stays infra-backed
└── ...
```

## Immediate action items (low risk)

1) Add a short module map doc under `flujo/application/core/` to improve onboarding.
2) If reorganizing builtins, add compatibility shims and deprecation warnings.
3) Re-export `CacheStep` from `flujo/domain/dsl/__init__.py` for discoverability.
