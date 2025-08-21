from __future__ import annotations

import pytest

from flujo.architect.builder import _run_planner_agent
from flujo.architect.context import ArchitectContext


@pytest.mark.asyncio
async def test_planner_agent_disabled_falls_back_to_heuristics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Disable agentic planner explicitly
    monkeypatch.setenv("FLUJO_ARCHITECT_AGENTIC_PLANNER", "0")

    ctx = ArchitectContext(user_goal="Fetch https://example.com", available_skills=[])

    out = await _run_planner_agent(None, context=ctx)

    assert isinstance(out, dict)
    assert "execution_plan" in out
    plan = out["execution_plan"]
    assert isinstance(plan, list) and len(plan) >= 1
    # Heuristic should pick http_get when goal has URL
    first = plan[0]
    assert isinstance(first, dict)
    agent = first.get("agent")
    assert isinstance(agent, dict)
    assert agent.get("id") == "flujo.builtins.http_get"
