# First‑Class Extensible Step Primitives

This document explains how Flujo supports first‑class, extensible Step primitives via a framework registry and policy‑driven execution. It also shows how to add your own high‑level orchestration steps and make them available in the Python DSL, YAML loader, and the AI Architect.

## Overview

Flujo’s executor is policy‑driven: concrete step types resolve to execution policies that implement their semantics. FSD‑025 introduces a public registry for adding new `Step` subclasses as first‑class primitives, with full YAML and Architect integration.

Key components:

- `flujo.framework.registry`
  - `register_step_type(step_class)`: Map a `kind` string → `Step` subclass
  - `register_policy(step_class, policy_instance)`: Map `Step` subclass → policy instance
  - `get_step_class(kind)`: Resolve a custom `kind` during YAML load
- YAML loader: delegates custom `kind` to the registry; instantiates the registered model via `model_validate()`
- ExecutorCore: initializes its `PolicyRegistry` with the framework policy registry and routes execution accordingly
- Architect skill: `flujo.builtins.get_framework_schema` exposes JSON Schemas for all registered step primitives

## Registry API

```python
from typing import Any
from flujo.framework import registry
from flujo.domain.dsl.step import Step

class MyStep(Step[Any, Any]):
    kind = "MyStep"

class MyPolicy:
    async def execute(self, core, frame):
        # Return a StepOutcome (Success/Failure/Paused)
        from flujo.domain.models import StepResult, Success
        return Success(step_result=StepResult(name="MyStep", output=None, success=True))

# Register at import time (e.g., in your package __init__.py)
registry.register_step_type(MyStep)
registry.register_policy(MyStep, MyPolicy())
```

Rules:
- `step_class` must be a subclass of `Step` and declare a class attr `kind: str`
- Duplicate `kind` registrations raise `ConfigurationError`
- Policies can be callables that accept an `ExecutionFrame` or objects exposing `execute(core, frame)`

## YAML Loader Integration

Custom steps become first‑class YAML primitives:

```yaml
version: "0.1"
steps:
  - kind: MyStep
    name: X
    # ... custom fields supported by your Pydantic model
```

During load, the loader checks built‑in kinds; otherwise it calls `registry.get_step_class(kind)` and instantiates the Pydantic model via `model_validate(your_yaml_dict)`.

## ExecutorCore Dispatch

`ExecutorCore` bootstraps its `PolicyRegistry` with the global policy registry from `flujo.framework.registry`. If a policy object exposes `execute(core, frame)`, the core binds it into a callable so the dispatch signature remains uniform.

## AI Architect Integration

Flujo’s Architect learns about registered primitives via the builtin skill:

- `flujo.builtins.get_framework_schema` returns `{"steps": {kind: json_schema}}` where `json_schema` is produced by your step model’s `model_json_schema()`.

You can import `flujo.builtins` (or run the CLI) to ensure skills are registered, then query the skill registry to fetch the schema at runtime.

## Best Practices

- Keep complex orchestration logic in policies; keep `Step` models declarative and strongly typed
- Use Pydantic validators to coerce nested YAML (e.g., converting lists of steps into Pipelines)
- Maintain context isolation and merging via `ContextManager` helpers from the core (see Team Guide)
- Add tests: registry behavior, YAML parsing for your `kind`, and execution (policy path)

