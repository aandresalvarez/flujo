from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_visualize_plan_generates_mermaid() -> None:
    from tests.conftest import get_registered_factory

    plan = [
        {"name": "Step One", "agent": {"id": "flujo.builtins.stringify", "params": {}}},
        {"name": "Step Two", "agent": {"id": "flujo.builtins.stringify", "params": {}}},
    ]
    factory = get_registered_factory("flujo.builtins.visualize_plan")
    visualize = factory()
    out = await visualize(plan)
    assert isinstance(out, dict)
    g = out.get("plan_mermaid_graph", "")
    assert g.startswith("graph TD")
    assert 'S1["Step One"]' in g
    assert 'S2["Step Two"]' in g


@pytest.mark.asyncio
async def test_visualize_plan_empty() -> None:
    from tests.conftest import get_registered_factory

    factory = get_registered_factory("flujo.builtins.visualize_plan")
    visualize = factory()
    out = await visualize([])
    assert isinstance(out, dict)
    g = out.get("plan_mermaid_graph", "")
    assert g.startswith("graph TD")
