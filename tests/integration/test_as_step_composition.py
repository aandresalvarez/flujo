import pytest

from flujo import Flujo, Step
from flujo.recipes.factories import make_agentic_loop_pipeline
from flujo.testing.utils import StubAgent, gather_result
from flujo.domain.commands import FinishCommand, RunAgentCommand
from flujo.domain.models import PipelineContext, PipelineResult
from flujo.domain.resources import AppResources
from flujo.exceptions import ContextInheritanceError


@pytest.mark.asyncio
async def test_agentic_loop_as_composable_step() -> None:
    planner = StubAgent(
        [
            RunAgentCommand(agent_name="tool", input_data="hi"),
            FinishCommand(final_answer="done"),
        ]
    )
    tool = StubAgent(["tool-output"])
    pipeline = make_agentic_loop_pipeline(planner_agent=planner, agent_registry={"tool": tool})

    # Create a Flujo runner first, then get the as_step
    flujo_runner = Flujo(pipeline, context_model=PipelineContext)
    pipeline_step = flujo_runner.as_step(name="loop")
    runner = Flujo(pipeline_step, context_model=PipelineContext)

    result = await gather_result(
        runner,
        "goal",
        initial_context_data={"initial_prompt": "goal"},
    )
    assert result.final_pipeline_context.command_log[-1].execution_result == "done"


@pytest.mark.asyncio
async def test_pipeline_of_pipelines_via_as_step() -> None:
    step1 = Step.model_validate({"name": "a", "agent": StubAgent([1])})
    step2 = Step.model_validate({"name": "b", "agent": StubAgent([2])})

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


@pytest.mark.asyncio
async def test_as_step_context_propagation() -> None:
    class Incrementer:
        async def run(self, data: int, *, context: PipelineContext | None = None) -> dict:
            assert context is not None
            current = context.scratchpad.get("counter", 0)
            return {"scratchpad": {"counter": current + data}}

    inner_runner = Flujo(
        Step.model_validate({"name": "inc", "agent": Incrementer(), "updates_context": True}),
        context_model=PipelineContext,
    )

    pipeline = inner_runner.as_step(name="inner")
    runner = Flujo(pipeline, context_model=PipelineContext)

    result = await gather_result(
        runner,
        2,
        initial_context_data={"initial_prompt": "goal", "scratchpad": {"counter": 1}},
    )

    assert result.final_pipeline_context.scratchpad["counter"] == 3


@pytest.mark.asyncio
async def test_as_step_resource_propagation() -> None:
    class Res(AppResources):
        counter: int = 0

    class UseRes:
        async def run(self, data: int, *, resources: Res) -> int:
            resources.counter += data
            return resources.counter

    inner_runner = Flujo(
        Step.model_validate({"name": "res", "agent": UseRes()}),
        context_model=PipelineContext,
    )

    pipeline = inner_runner.as_step(name="inner")
    res = Res()
    runner = Flujo(pipeline, context_model=PipelineContext, resources=res)

    await gather_result(
        runner,
        5,
        initial_context_data={"initial_prompt": "goal"},
    )

    assert res.counter == 5


@pytest.mark.asyncio
async def test_as_step_initial_prompt_sync() -> None:
    planner = StubAgent(
        [
            RunAgentCommand(agent_name="tool", input_data="hi"),
            FinishCommand(final_answer="done"),
        ]
    )
    tool = StubAgent(["tool-output"])
    pipeline = make_agentic_loop_pipeline(planner_agent=planner, agent_registry={"tool": tool})

    # Create a Flujo runner first, then get the as_step
    flujo_runner = Flujo(pipeline, context_model=PipelineContext)
    pipeline_step = flujo_runner.as_step(name="inner")
    runner = Flujo(pipeline_step, context_model=PipelineContext)

    result = await gather_result(
        runner,
        "goal",
        initial_context_data={"initial_prompt": "wrong"},
    )

    assert result.final_pipeline_context.initial_prompt == "wrong"


@pytest.mark.asyncio
async def test_as_step_inherit_context_false() -> None:
    class Incrementer:
        async def run(self, data: int, *, context: PipelineContext | None = None) -> dict:
            assert context is not None
            current = context.scratchpad.get("counter", 0)
            return {"scratchpad": {"counter": current + data}}

    inner_runner = Flujo(
        Step.model_validate({"name": "inc", "agent": Incrementer(), "updates_context": True}),
        context_model=PipelineContext,
    )

    pipeline = inner_runner.as_step(name="inner", inherit_context=False)
    runner = Flujo(pipeline, context_model=PipelineContext)

    result = await gather_result(
        runner,
        2,
        initial_context_data={"initial_prompt": "goal", "scratchpad": {"counter": 1}},
    )

    assert result.final_pipeline_context.scratchpad["counter"] == 1


class ChildCtx(PipelineContext):
    extra: int


@pytest.mark.asyncio
async def test_as_step_context_inheritance_error() -> None:
    step = Step.model_validate({"name": "s", "agent": StubAgent(["ok"])})

    inner_runner = Flujo(step, context_model=ChildCtx)
    pipeline = inner_runner.as_step(name="inner")
    runner = Flujo(pipeline, context_model=PipelineContext)

    with pytest.raises(ContextInheritanceError) as exc:
        await gather_result(
            runner,
            "goal",
            initial_context_data={"initial_prompt": "goal"},
        )
    assert exc.value.missing_fields == ["extra"]
