from __future__ import annotations

import pytest

from flujo.architect.builder import _match_one_tool
from flujo.architect.context import ArchitectContext
from flujo.infra.skill_registry import get_skill_registry


@pytest.mark.asyncio
async def test_tool_matcher_disabled_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    # Disable agentic tool matcher explicitly
    monkeypatch.setenv("FLUJO_ARCHITECT_AGENTIC_TOOLMATCHER", "0")

    ctx = ArchitectContext(available_skills=[])
    step_item = {"step_name": "Any", "purpose": "search the web for x"}

    out = await _match_one_tool(step_item, context=ctx)
    assert isinstance(out, dict)
    assert out.get("chosen_agent_id") == "flujo.builtins.stringify"


@pytest.mark.asyncio
async def test_tool_matcher_agent_failure_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    # Register a failing tool matcher to simulate LLM/agent errors
    reg = get_skill_registry()

    async def _failing_agent(_payload):
        raise RuntimeError("simulated agent failure")

    reg.register("flujo.architect.tool_matcher", lambda: _failing_agent, description="failing")

    ctx = ArchitectContext(available_skills=[])
    step_item = {"step_name": "Any", "purpose": "search the web for x"}

    out = await _match_one_tool(step_item, context=ctx)
    # Should not raise; should fallback to safe default
    assert out.get("chosen_agent_id") == "flujo.builtins.stringify"
