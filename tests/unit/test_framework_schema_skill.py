from __future__ import annotations

import asyncio


def test_get_framework_schema_includes_state_machine() -> None:
    # Import builtins to ensure the skill is registered
    import flujo.builtins as _  # noqa: F401
    from flujo.infra.skill_registry import get_skill_registry

    reg = get_skill_registry()
    entry = reg.get("flujo.builtins.get_framework_schema")
    assert entry is not None
    factory = entry.get("factory")
    assert callable(factory)
    coro_fn = factory()
    assert callable(coro_fn)

    async def _run():
        result = await coro_fn()
        assert isinstance(result, dict)
        steps = result.get("steps") or {}
        assert "StateMachine" in steps

    asyncio.run(_run())
