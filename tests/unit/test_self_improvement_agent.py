import pytest
from flujo.prompts import SELF_IMPROVE_SYS
from flujo.application.self_improvement import SelfImprovementAgent, ImprovementReport


class DummyAgent:
    async def run(self, prompt: str) -> ImprovementReport:
        return ImprovementReport(suggestions=[])


@pytest.mark.asyncio
async def test_self_improve_sys_contains_examples() -> None:
    assert "EXAMPLE 1" in SELF_IMPROVE_SYS
    assert "EXAMPLE 2" in SELF_IMPROVE_SYS


@pytest.mark.asyncio
async def test_self_improvement_agent_parses_json() -> None:
    agent = SelfImprovementAgent(DummyAgent())
    report = await agent.run("ctx")
    assert isinstance(report, ImprovementReport)
    assert report.suggestions == []


@pytest.mark.asyncio
async def test_self_improvement_agent_handles_bad_json() -> None:
    class BadAgent:
        async def run(self, prompt: str) -> str:
            return "not json"

    agent = SelfImprovementAgent(BadAgent())
    with pytest.raises(Exception):
        await agent.run("ctx")
