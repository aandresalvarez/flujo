import pytest

from flujo import Flujo, Step
from flujo.recipes.agentic_loop import AgenticLoop
from flujo.testing.utils import StubAgent, gather_result
from flujo.domain.commands import FinishCommand, RunAgentCommand
from flujo.domain.models import PipelineContext, PipelineResult


@pytest.mark.asyncio
async def test_agentic_loop_as_composable_step() -> None:
    planner = StubAgent(
        [
            RunAgentCommand(agent_name="tool", input_data="hi"),
            FinishCommand(final_answer="done"),
        ]
    )
    tool = StubAgent(["tool-output"])
    loop = AgenticLoop(planner, {"tool": tool})

    pipeline = loop.as_step(name="loop")
    runner = Flujo(pipeline, context_model=PipelineContext)

    result = await gather_result(
        runner,
        "goal",
        initial_context_data={"initial_prompt": "goal"},
    )
    assert result.final_pipeline_context.command_log[-1].execution_result == "done"


@pytest.mark.asyncio
async def test_pipeline_of_pipelines_via_as_step() -> None:
    step1 = Step("a", StubAgent([1]))
    step2 = Step("b", StubAgent([2]))

    sub_runner1 = Flujo(step1, context_model=PipelineContext)
    sub_runner2 = Flujo(step2, context_model=PipelineContext)

    first = sub_runner1.as_step(name="first")

    async def extract_fn(pr: PipelineResult) -> int:
        return pr.step_history[-1].output

    extract = Step.from_mapper(
        extract_fn,
        name="extract",
    )
    master = first >> extract >> sub_runner2.as_step(name="second")
    runner = Flujo(master, context_model=PipelineContext)

    result = await gather_result(
        runner,
        0,
        initial_context_data={"initial_prompt": "goal"},
    )

    assert isinstance(result.step_history[0].output, PipelineResult)
    assert result.step_history[1].output == 1
    inner_result = result.step_history[2].output
    assert isinstance(inner_result, PipelineResult)
    assert inner_result.step_history[-1].output == 2
