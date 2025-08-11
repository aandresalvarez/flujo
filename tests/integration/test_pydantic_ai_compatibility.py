import pytest
from flujo.domain.models import BaseModel

from flujo.domain import Step
from flujo.domain.models import Checklist, ChecklistItem
from flujo.testing.utils import StubAgent, gather_result
from flujo.agents import AsyncAgentWrapper
from tests.conftest import create_test_flujo


class TypeCheckingAgent:
    async def run(self, data):
        assert isinstance(data, dict)
        return "ok"


class KwargCheckingAgent:
    async def run(self, data, *, context: "SimpleContext") -> str:
        if isinstance(context, dict):
            return context.get("foo", "")
        return context.foo


@pytest.mark.asyncio
async def test_pydantic_models_are_serialized_for_agents():
    first = Step.model_validate(
        {"name": "produce", "agent": StubAgent([Checklist(items=[ChecklistItem(description="a")])])}
    )
    second = Step.model_validate(
        {"name": "consume", "agent": AsyncAgentWrapper(TypeCheckingAgent())}
    )
    pipeline = first >> second
    runner = create_test_flujo(pipeline)

    result = await gather_result(runner, None)

    assert result.step_history[-1].output == "ok"


class SimpleContext(BaseModel):
    foo: str


@pytest.mark.asyncio
async def test_pipeline_context_serialized_for_agent_kwargs():
    first = Step.model_validate({"name": "produce", "agent": StubAgent(["x"])})
    second = Step.model_validate(
        {"name": "consume", "agent": AsyncAgentWrapper(KwargCheckingAgent())}
    )
    pipeline = first >> second
    runner = create_test_flujo(
        pipeline,
        context_model=SimpleContext,
        initial_context_data={"foo": "bar"},
    )

    result = await gather_result(runner, None)

    assert result.step_history[-1].output == "bar"
