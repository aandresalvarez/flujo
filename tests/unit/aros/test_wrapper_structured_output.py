from __future__ import annotations

import pytest

from flujo.agents.wrapper import AsyncAgentWrapper


class FakeAgent:
    def __init__(self) -> None:
        self.last_kwargs: dict | None = None

    async def run(self, payload, **kwargs):  # type: ignore[no-untyped-def]
        self.last_kwargs = dict(kwargs)
        # mimic pydantic-ai AgentRunResult-like shape
        return {"output": {"ok": True}}


@pytest.mark.fast
@pytest.mark.asyncio
async def test_wrapper_attaches_response_format_when_enabled():
    agent = FakeAgent()
    wrapper = AsyncAgentWrapper(agent)
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
    wrapper.enable_structured_output(json_schema=schema, name="test")
    await wrapper.run_async("hello")
    assert agent.last_kwargs is not None
    rf = agent.last_kwargs.get("response_format")
    assert isinstance(rf, dict)
    assert rf.get("type") == "json_schema"
    assert isinstance(rf.get("json_schema"), dict)
    assert rf.get("json_schema", {}).get("name") == "test"


@pytest.mark.fast
@pytest.mark.asyncio
async def test_wrapper_does_not_override_existing_response_format():
    agent = FakeAgent()
    wrapper = AsyncAgentWrapper(agent)
    # User passes explicit response_format; wrapper should not override
    await wrapper.run_async("hello", response_format={"type": "json_object"})
    assert agent.last_kwargs is not None
    rf = agent.last_kwargs.get("response_format")
    assert rf == {"type": "json_object"}


@pytest.mark.fast
@pytest.mark.asyncio
async def test_wrapper_json_object_hint_when_no_schema():
    agent = FakeAgent()
    wrapper = AsyncAgentWrapper(agent)
    wrapper.enable_structured_output(json_schema=None, name=None)
    await wrapper.run_async("hello")
    assert agent.last_kwargs is not None
    rf = agent.last_kwargs.get("response_format")
    assert isinstance(rf, dict)
    assert rf.get("type") in {"json_object", "json_schema"}


class NoKwAgent:
    def __init__(self) -> None:
        self.called = False

    async def run(self, payload):  # type: ignore[no-untyped-def]
        # No **kwargs param: wrapper must not pass response_format
        self.called = True
        return {"output": {"ok": True}}


@pytest.mark.fast
@pytest.mark.asyncio
async def test_wrapper_does_not_pass_response_format_when_not_supported():
    from flujo.domain.agent_result import FlujoAgentResult

    agent = NoKwAgent()
    wrapper = AsyncAgentWrapper(agent)
    wrapper.enable_structured_output(json_schema={"type": "object"}, name="x")
    # Should not raise even though underlying run doesn't accept response_format
    res = await wrapper.run_async("hello")
    assert agent.called is True
    # Wrapper now returns FlujoAgentResult; check output inside
    assert isinstance(res, FlujoAgentResult)
    assert isinstance(res.output, dict)
