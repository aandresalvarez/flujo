import pytest
from flujo import Step, Flujo, AgentProcessors
from flujo.testing.utils import StubAgent, gather_result
from flujo.processors import AddContextVariables, StripMarkdownFences
from pydantic import BaseModel


class Ctx(BaseModel):
    user_id: str
    session_id: str


@pytest.mark.asyncio
async def test_prompt_processor_adds_context() -> None:
    agent = StubAgent(["ok"])
    step = Step(
        "s",
        agent,
        processors=AgentProcessors(
            prompt_processors=[AddContextVariables(vars=["user_id", "session_id"])]
        ),
    )
    runner = Flujo(
        step, context_model=Ctx, initial_context_data={"user_id": "u1", "session_id": "abc"}
    )
    await gather_result(runner, "hi")
    assert agent.inputs[0].startswith("--- CONTEXT ---")


@pytest.mark.asyncio
async def test_output_processor_strips_markdown() -> None:
    agent = StubAgent(['```json\n{"a":1}\n```'])
    step = Step(
        "s",
        agent,
        processors=AgentProcessors(output_processors=[StripMarkdownFences(language="json")]),
    )
    runner = Flujo(step)
    result = await gather_result(runner, "in")
    assert result.step_history[0].output == '{"a":1}'
