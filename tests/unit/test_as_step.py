import pytest

from flujo.recipes.agentic_loop import AgenticLoop
from flujo.domain.commands import FinishCommand
from flujo.testing.utils import StubAgent
from flujo.domain.pipeline_dsl import Step
from flujo.application.runner import Flujo


@pytest.mark.asyncio
async def test_agentic_loop_as_step_basic() -> None:
    planner = StubAgent([FinishCommand(final_answer="done")])
    loop = AgenticLoop(planner, {})
    step = loop.as_step(name="loop")

    assert isinstance(step, Step)

    result = await step.arun("goal")
    assert result.final_pipeline_context.command_log[-1].execution_result == "done"


@pytest.mark.asyncio
async def test_flujo_as_step_basic() -> None:
    base_step = Step.model_validate({"name": "s", "agent": StubAgent(["ok"])})
    runner = Flujo(base_step)

    step = runner.as_step(name="runner")
    assert isinstance(step, Step)

    result = await step.arun("hi")
    assert result.step_history[-1].output == "ok"
