---
title: Skills & Plugins
---

# Built-in Skill Lifecycle (Flujo Standard)

This guide documents how built-in skills (plugins/skills) are registered and resolved in
Flujo. It applies to the `flujo/builtins/*.py` modules (legacy `builtins_*.py` shims
remain for compatibility) and any new skills added to the registry.

## Registration Pattern

Use the typed `SkillRegistration` dataclass from `flujo.infra.skill_models` to register
skills through `SkillRegistry`. This keeps metadata consistent (schemas, side effects,
auth requirements) and enforces typed JSON via `JSONObject`.

```python
from flujo.infra.skill_models import SkillRegistration
from flujo.infra.skill_registry import get_skill_registry

def register_skills() -> None:
    reg = get_skill_registry()
    reg.register(
        **SkillRegistration(
            id="flujo.builtins.example",
            factory=lambda **_params: example_agent,
            description="Short description",
            input_schema={
                "type": "object",
                "properties": {"foo": {"type": "string"}},
                "required": ["foo"],
            },
            output_schema={"type": "object"},
            side_effects=False,
        ).__dict__
    )
```

## Factory Expectations

- Skill factories should be callables accepting keyword params only; the registry passes
  `params` from YAML/CLI.
- Use `JSONObject` for all schemas/metadata.
- Factories must be safe to import and invoke in registry setup (no side effects on import).

## Graceful Degradation

- Optional dependencies (e.g., network clients) must fail closed: return a safe default
  (empty list/object) rather than raising when a dependency is missing.
- Avoid side effects unless explicitly flagged (`side_effects=True`).
- Keep `auth_required`/`auth_scope` accurate for downstream enforcement.

## Resolution

- Access via `get_skill_registry().get(skill_id, scope=..., version=...)`.
- Built-ins register under the `default` scope; tenant scopes remain isolated.

## Validation Checklist

- [ ] Registered via `SkillRegistration` with `JSONObject` schemas
- [ ] Factory callable accepts kwargs only and handles missing deps gracefully
- [ ] `side_effects` / `auth_*` fields set correctly
- [ ] Tests cover the registration (`pytest tests/unit/test_builtins.py`)
