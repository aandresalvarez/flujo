import pytest

from flujo.recipes.factories import make_agentic_loop_pipeline
from flujo.domain.commands import FinishCommand
from flujo.testing.utils import StubAgent
from flujo.domain.dsl import Step
from flujo.application.runner import Flujo


@pytest.mark.asyncio
async def test_agentic_loop_as_step_basic() -> None:
    planner = StubAgent([FinishCommand(final_answer="done")])
    pipeline = make_agentic_loop_pipeline(planner_agent=planner, agent_registry={})
    runner = Flujo(pipeline)
    step = runner.as_step(name="loop")

    assert isinstance(step, Step)

    result = await step.arun("goal")
    # Check the execution result from the step history instead of command log
    assert result.step_history[-1].output.execution_result == "done"


@pytest.mark.asyncio
async def test_flujo_as_step_basic() -> None:
    base_step = Step.model_validate({"name": "s", "agent": StubAgent(["ok"])})
    runner = Flujo(base_step)

    step = runner.as_step(name="runner")
    assert isinstance(step, Step)

    result = await step.arun("hi")
    assert result.step_history[-1].output == "ok"
