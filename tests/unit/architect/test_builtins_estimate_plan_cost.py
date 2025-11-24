from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_estimate_plan_cost_sums_registry_estimates() -> None:
    from tests.conftest import get_registered_factory
    from flujo.infra.skill_registry import get_skill_registry

    reg = get_skill_registry()
    # Register two fake skills and inject est_cost metadata
    reg.register("test.skill_a", lambda **_: (lambda: None))
    reg.register("test.skill_b", lambda **_: (lambda: None))
    entry_a = reg.get("test.skill_a") or {}
    entry_b = reg.get("test.skill_b") or {}
    entry_a["est_cost"] = 0.01
    entry_b["est_cost"] = 0.05

    plan = [
        {"name": "A", "agent": {"id": "test.skill_a", "params": {}}},
        {"name": "B", "agent": {"id": "test.skill_b", "params": {}}},
        {"name": "Unknown", "agent": {"id": "not.registered", "params": {}}},
    ]

    factory = get_registered_factory("flujo.builtins.estimate_plan_cost")
    estimate = factory()
    out = await estimate(plan)
    assert isinstance(out, dict)
    assert abs(out.get("plan_estimated_cost_usd", 0.0) - 0.06) < 1e-6
