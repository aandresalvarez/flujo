import pytest

from flujo.testing.replay import ReplayAgent, ReplayError


@pytest.mark.asyncio
async def test_replay_agent_returns_canned_response() -> None:
    agent = ReplayAgent({"step_a:attempt_1": {"ok": 1}})
    out = await agent.run(step_name="step_a", attempt_number=1)
    assert out == {"ok": 1}


@pytest.mark.asyncio
async def test_replay_agent_missing_key_raises() -> None:
    agent = ReplayAgent({"step_a:attempt_1": {"ok": 1}})
    with pytest.raises(ReplayError):
        await agent.run(step_name="step_b", attempt_number=1)
