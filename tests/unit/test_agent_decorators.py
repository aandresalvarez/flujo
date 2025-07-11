import pytest
from pydantic import BaseModel

from flujo.agents import validated_agent, monitored_agent
from flujo.domain.agent_protocol import AsyncAgentProtocol
from flujo.monitor import global_monitor, FailureType
from flujo.exceptions import AgentIOValidationError


class InModel(BaseModel):
    value: int


class OutModel(BaseModel):
    doubled: int


@validated_agent(InModel, OutModel)
class SimpleAgent(AsyncAgentProtocol[InModel, OutModel]):
    async def run(self, data: InModel, **kwargs) -> OutModel:
        return OutModel(doubled=data.value * 2)


@monitored_agent("mon_agent")
class MonitoredAgent(AsyncAgentProtocol[str, str]):
    async def run(self, data: str, **kwargs) -> str:
        return data.upper()


@monitored_agent("combo")
@validated_agent(InModel, OutModel)
class ComboAgent(AsyncAgentProtocol[InModel, OutModel]):
    async def run(self, data: InModel, **kwargs) -> OutModel:
        return OutModel(doubled=data.value * 2)


@pytest.mark.asyncio
async def test_validated_agent_raises() -> None:
    agent = SimpleAgent()
    with pytest.raises(AgentIOValidationError):
        await agent.run({"value": "oops"})


@pytest.mark.asyncio
async def test_monitored_agent_records() -> None:
    global_monitor.calls.clear()
    agent = MonitoredAgent()
    result = await agent.run("hi")
    assert result == "HI"
    assert len(global_monitor.calls) == 1
    rec = global_monitor.calls[0]
    assert rec["agent_name"] == "mon_agent"
    assert rec["success"] is True
    assert rec["output_data"] == "HI"


@pytest.mark.asyncio
async def test_decorator_composition_success() -> None:
    global_monitor.calls.clear()
    agent = ComboAgent()
    result = await agent.run({"value": 3})
    assert result.doubled == 6
    assert len(global_monitor.calls) == 1
    rec = global_monitor.calls[0]
    assert rec["success"] is True
    assert rec["failure_type"] is None


@pytest.mark.asyncio
async def test_decorator_composition_validation_error() -> None:
    global_monitor.calls.clear()
    agent = ComboAgent()
    with pytest.raises(AgentIOValidationError):
        await agent.run({"value": "bad"})
    assert len(global_monitor.calls) == 1
    rec = global_monitor.calls[0]
    assert rec["success"] is False
    assert rec["failure_type"] == FailureType.VALIDATION_ERROR
