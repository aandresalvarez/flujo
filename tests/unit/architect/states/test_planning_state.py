from __future__ import annotations

import pytest

from flujo.architect.context import ArchitectContext
from flujo.architect.states.planning import run_planner_agent


@pytest.mark.asyncio
async def test_planning_fallback_uses_available_http_get(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLUJO_ARCHITECT_AGENTIC_PLANNER", "0")

    ctx = ArchitectContext(
        user_goal="fetch https://example.com and save to file output.txt",
        available_skills=[
            {"id": "flujo.builtins.http_get"},
            {"id": "flujo.builtins.fs_write_file"},
        ],
    )

    out = await run_planner_agent(None, context=ctx)

    plan = out["execution_plan"]
    assert isinstance(plan, list)
    assert len(plan) >= 2
    first = plan[0]
    assert isinstance(first, dict)
    assert first.get("agent", {}).get("id") == "flujo.builtins.http_get"
    assert plan[1].get("agent", {}).get("id") == "flujo.builtins.fs_write_file"
