import pytest

from flujo import Step
from flujo.recipes.factories import make_agentic_loop_pipeline
from flujo.testing.utils import StubAgent, gather_result
from flujo.domain.commands import FinishCommand, RunAgentCommand
from flujo.domain.models import PipelineContext, PipelineResult, ImportArtifacts
from flujo.domain.resources import AppResources
from tests.conftest import create_test_flujo


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
    flujo_runner = create_test_flujo(pipeline, context_model=PipelineContext)
    pipeline_step = flujo_runner.as_step(name="loop")
    runner = create_test_flujo(pipeline_step, context_model=PipelineContext)

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

    sub_runner1 = create_test_flujo(step1, context_model=PipelineContext)
    sub_runner2 = create_test_flujo(step2, context_model=PipelineContext)

    first = sub_runner1.as_step(name="first")

    async def extract_fn(pr: PipelineResult) -> int:
        return pr.step_history[-1].output

    extract = Step.from_mapper(
        extract_fn,
        name="extract",
    )
    master = first >> extract >> sub_runner2.as_step(name="second")
    runner = create_test_flujo(master, context_model=PipelineContext)

    result = await gather_result(
        runner,
        0,
        initial_context_data={"initial_prompt": "goal"},
    )

    first_out = result.step_history[0].output
    if isinstance(first_out, PipelineResult):
        assert first_out.step_history[-1].output == 1
    assert result.step_history[1].output == 1
    inner_result = result.step_history[2].output
    if isinstance(inner_result, PipelineResult):
        assert inner_result.step_history[-1].output == 2


@pytest.mark.asyncio
async def test_as_step_context_propagation() -> None:
    class Incrementer:
        async def run(self, data: int, *, context: PipelineContext | None = None) -> dict:
            assert context is not None
            extras = getattr(context.import_artifacts, "extras", {}) or {}
            current = extras.get("counter", 0)
            new_extras = dict(extras)
            new_extras["counter"] = current + data
            return {"import_artifacts": {"extras": new_extras}}

    inner_runner = create_test_flujo(
        Step.model_validate({"name": "inc", "agent": Incrementer(), "updates_context": True}),
        context_model=PipelineContext,
    )

    pipeline = inner_runner.as_step(name="inner")
    runner = create_test_flujo(pipeline, context_model=PipelineContext)

    result = await gather_result(
        runner,
        2,
        initial_context_data={
            "initial_prompt": "goal",
            "import_artifacts": ImportArtifacts(extras={"counter": 1}),
        },
    )

    ia = result.final_pipeline_context.import_artifacts
    extras = ia.extras if hasattr(ia, "extras") else ia.get("extras", {})  # type: ignore[arg-type]
    assert extras.get("counter") == 3


@pytest.mark.asyncio
async def test_as_step_resource_propagation() -> None:
    class Res(AppResources):
        counter: int = 0

    class UseRes:
        async def run(self, data: int, *, resources: Res) -> int:
            resources.counter += data
            return resources.counter

    inner_runner = create_test_flujo(
        Step.model_validate({"name": "res", "agent": UseRes()}),
        context_model=PipelineContext,
    )

    pipeline = inner_runner.as_step(name="inner")
    res = Res()
    runner = create_test_flujo(pipeline, context_model=PipelineContext, resources=res)

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
    flujo_runner = create_test_flujo(pipeline, context_model=PipelineContext)
    pipeline_step = flujo_runner.as_step(name="inner")
    runner = create_test_flujo(pipeline_step, context_model=PipelineContext)

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
            extras = getattr(context.import_artifacts, "extras", {}) or {}
            current = extras.get("counter", 0)
            new_extras = dict(extras)
            new_extras["counter"] = current + data
            return {"import_artifacts": {"extras": new_extras}}

    inner_runner = create_test_flujo(
        Step.model_validate({"name": "inc", "agent": Incrementer(), "updates_context": True}),
        context_model=PipelineContext,
    )

    pipeline = inner_runner.as_step(name="inner", inherit_context=False)
    runner = create_test_flujo(pipeline, context_model=PipelineContext)

    result = await gather_result(
        runner,
        2,
        initial_context_data={
            "initial_prompt": "goal",
            "import_artifacts": {"extras": {"counter": 1}},
        },
    )

    assert result.final_pipeline_context.import_artifacts.extras.get("counter") == 1


class ChildCtx(PipelineContext):
    extra: int


@pytest.mark.asyncio
async def test_as_step_context_inheritance_error() -> None:
    step = Step.model_validate({"name": "s", "agent": StubAgent(["ok"])})

    inner_runner = create_test_flujo(step, context_model=ChildCtx)
    pipeline = inner_runner.as_step(name="inner")
    runner = create_test_flujo(pipeline, context_model=PipelineContext)

    # ✅ ENHANCED ERROR HANDLING: System gracefully handles context inheritance failures
    # Previous behavior: Context inheritance errors raised as exceptions
    # Enhanced behavior: Context errors converted to step failures with detailed feedback
    # This provides better error recovery and prevents pipeline crashes
    result = await gather_result(
        runner,
        "goal",
        initial_context_data={"initial_prompt": "goal"},
    )

    # Enhanced: Context inheritance failure handled gracefully as step failure
    assert len(result.step_history) > 0
    step_result = result.step_history[-1]
    assert step_result.success is False
    assert "Failed to inherit context" in (step_result.feedback or "")
    assert "extra" in (step_result.feedback or "")  # Missing field mentioned in feedback


@pytest.mark.asyncio
async def test_direct_context_inheritance_error():
    from flujo.domain.models import PipelineContext
    from flujo.domain.dsl.step import Step
    from flujo.application.runner import Flujo
    from flujo.testing.utils import StubAgent

    class ChildCtx(PipelineContext):
        extra: int

    # Create the problematic setup
    step = Step.model_validate({"name": "s", "agent": StubAgent(["ok"])})
    inner_runner = Flujo(step, context_model=ChildCtx)
    pipeline_step = inner_runner.as_step(name="inner")
    runner = Flujo(pipeline_step, context_model=PipelineContext)

    # Enhanced: Context inheritance error returns graceful failure
    results = []
    async for result in runner.run_async("goal", initial_context_data={"initial_prompt": "goal"}):
        results.append(result)

    # Enhanced: Verify graceful failure instead of exception
    assert len(results) > 0
    pipeline_result = results[-1]
    assert pipeline_result.step_history[0].success is False
    assert (
        "contextinheritanceerror" in pipeline_result.step_history[0].feedback.lower()
        or "context inheritance" in pipeline_result.step_history[0].feedback.lower()
    )
