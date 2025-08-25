from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_planner_and_toolmatcher_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unit-level check: planner/tool-matcher fall back to stringify/Echo Input when no skills available."""
    from flujo.architect import builder
    from flujo.architect.context import ArchitectContext
    from flujo.infra.skill_registry import get_skill_registry

    # Force state machine mode for consistency across environments
    monkeypatch.setenv("FLUJO_ARCHITECT_STATE_MACHINE", "1")

    # Mock registry to return empty dict to simulate unavailable skills
    reg = get_skill_registry()
    orig_get = reg.get
    reg.get = lambda _sid: {}
    try:
        ctx = ArchitectContext(initial_prompt="x", user_goal="Fetch and process web content")

        # Make plan and ensure default Echo Input step is present when no skills are available
        out = await builder._run_planner_agent(context=ctx)
        plan = out.get("execution_plan")
        assert isinstance(plan, list) and len(plan) >= 1
        assert any((s.get("name") == "Echo Input") for s in plan if isinstance(s, dict))

        # Prepare mapping items and run tool matcher once
        prep = await builder._prepare_for_map(context=ctx)
        assert isinstance(prep.get("prepared_steps_for_mapping"), list)

        # Directly exercise tool matcher heuristic
        sel = await builder._match_one_tool({"step_name": "Echo Input", "purpose": ""}, context=ctx)
        assert sel.get("chosen_agent_id") == "flujo.builtins.stringify"
    finally:
        reg.get = orig_get
