from __future__ import annotations

import pytest

from flujo.agents.wrapper import AsyncAgentWrapper


class KwOnlyAgent:
    def __init__(self) -> None:
        self.kwargs: dict | None = None

    async def run(self, payload, **kwargs):  # type: ignore[no-untyped-def]
        self.kwargs = dict(kwargs)
        return {"output": {"ok": True}}


@pytest.mark.fast
@pytest.mark.asyncio
async def test_wrapper_passes_response_format_when_agent_accepts_kwargs():
    agent = KwOnlyAgent()
    wrapper = AsyncAgentWrapper(agent)
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
    wrapper.enable_structured_output(json_schema=schema, name="wrapped")

    await wrapper.run_async("hello")
    assert isinstance(agent.kwargs, dict)
    rf = agent.kwargs.get("response_format")
    assert isinstance(rf, dict) and rf.get("type") == "json_schema"
    assert rf.get("json_schema", {}).get("name") == "wrapped"
