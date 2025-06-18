import pytest
from pydantic import BaseModel

from flujo.application.flujo_engine import Flujo
from flujo.domain import Step
from flujo.domain.models import Checklist, ChecklistItem
from flujo.testing.utils import StubAgent, gather_result
from flujo.infra.agents import AsyncAgentWrapper


class TypeCheckingAgent:
    async def run(self, data):
        assert isinstance(data, dict)
        return "ok"


class KwargCheckingAgent:
    async def run(self, data, *, pipeline_context):
        assert isinstance(pipeline_context, dict)
        return pipeline_context.get("foo")


@pytest.mark.asyncio
async def test_pydantic_models_are_serialized_for_agents():
    first = Step("produce", StubAgent([Checklist(items=[ChecklistItem(description="a")])]))
    second = Step("consume", AsyncAgentWrapper(TypeCheckingAgent()))
    pipeline = first >> second
    runner = Flujo(pipeline)

    result = await gather_result(runner, None)

    assert result.step_history[-1].output == "ok"


class SimpleContext(BaseModel):
    foo: str


@pytest.mark.asyncio
async def test_pipeline_context_serialized_for_agent_kwargs():
    first = Step("produce", StubAgent(["x"]))
    second = Step("consume", AsyncAgentWrapper(KwargCheckingAgent()))
    pipeline = first >> second
    runner = Flujo(
        pipeline,
        context_model=SimpleContext,
        initial_context_data={"foo": "bar"},
    )

    result = await gather_result(runner, None)

    assert result.step_history[-1].output == "bar"
