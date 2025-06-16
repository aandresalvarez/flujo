import pytest

from flujo.application.flujo_engine import Flujo
from flujo.domain import Step
from flujo.domain.models import Checklist, ChecklistItem
from flujo.testing.utils import StubAgent
from flujo.infra.agents import AsyncAgentWrapper

class TypeCheckingAgent:
    async def run(self, data):
        assert isinstance(data, dict)
        return "ok"

@pytest.mark.asyncio
async def test_pydantic_models_are_serialized_for_agents():
    first = Step("produce", StubAgent([Checklist(items=[ChecklistItem(description="a")])]))
    second = Step("consume", AsyncAgentWrapper(TypeCheckingAgent()))
    pipeline = first >> second
    runner = Flujo(pipeline)

    result = await runner.run_async(None)

    assert result.step_history[-1].output == "ok"
