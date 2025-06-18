import pickle
import pytest
from pydantic import BaseModel, ValidationError

from flujo.application.flujo_engine import Flujo
from flujo.domain.pipeline_dsl import Step
from flujo.domain.models import PipelineContext
from flujo.exceptions import OrchestratorError
from flujo.testing.utils import StubAgent, gather_result


@pytest.mark.asyncio
async def test_static_approval_pause_and_resume() -> None:
    pipeline = Step("first", StubAgent(["draft"])) >> Step.human_in_the_loop(
        "approve", message_for_user="OK?"
    )
    runner = Flujo(pipeline)
    paused = await gather_result(runner, "in")
    ctx = paused.final_pipeline_context
    assert isinstance(ctx, PipelineContext)
    assert ctx.scratchpad["status"] == "paused"
    assert ctx.scratchpad["pause_message"] == "OK?"
    resumed = await runner.resume_async(paused, "yes")
    assert resumed.step_history[-1].output == "yes"
    assert ctx.scratchpad["status"] == "completed"
    assert len(ctx.hitl_history) == 1
    record = ctx.hitl_history[0]
    assert record.message_to_human == "OK?"
    assert record.human_response == "yes"


@pytest.mark.asyncio
async def test_dynamic_clarification_pause_and_resume() -> None:
    pipeline = Step("ask", StubAgent(["Need help?"])) >> Step.human_in_the_loop("clarify")
    runner = Flujo(pipeline)
    paused = await gather_result(runner, "hi")
    ctx = paused.final_pipeline_context
    assert ctx.scratchpad["pause_message"] == "Need help?"
    resumed = await runner.resume_async(paused, "sure")
    assert resumed.step_history[-1].output == "sure"
    assert len(ctx.hitl_history) == 1
    assert ctx.hitl_history[0].message_to_human == "Need help?"
    assert ctx.hitl_history[0].human_response == "sure"


class Choice(BaseModel):
    option: int


@pytest.mark.asyncio
async def test_resume_with_structured_input_validation() -> None:
    step = Step.human_in_the_loop("pick", input_schema=Choice)
    pipeline = Step("pre", StubAgent(["Q"])) >> step
    runner = Flujo(pipeline)
    paused = await gather_result(runner, "x")
    resumed = await runner.resume_async(paused, {"option": 1})
    assert isinstance(resumed.step_history[-1].output, Choice)


@pytest.mark.asyncio
async def test_resume_with_invalid_structured_input() -> None:
    step = Step.human_in_the_loop("pick", input_schema=Choice)
    pipeline = Step("pre", StubAgent(["Q"])) >> step
    runner = Flujo(pipeline)
    paused = await gather_result(runner, "x")
    with pytest.raises(ValidationError):
        await runner.resume_async(paused, {"bad": 0})


@pytest.mark.asyncio
async def test_multi_turn_correction_loop() -> None:
    pipeline = (
        Step("draft1", StubAgent(["bad"]))
        >> Step.human_in_the_loop("fix1")
        >> Step("draft2", StubAgent(["good"]))
        >> Step.human_in_the_loop("fix2")
    )
    runner = Flujo(pipeline)
    paused = await gather_result(runner, "start")
    paused = await runner.resume_async(paused, "no")
    paused = await runner.resume_async(paused, "yes")
    assert paused.step_history[-1].output == "yes"
    ctx = paused.final_pipeline_context
    assert ctx is not None
    assert len(ctx.hitl_history) == 2


class MetricOut(BaseModel):
    value: int
    cost_usd: float = 0.1
    token_counts: int = 10


class MetricAgent:
    async def run(self, data: int | MetricOut) -> MetricOut:
        val = data.value if isinstance(data, MetricOut) else data
        return MetricOut(value=val + 1)


@pytest.mark.asyncio
async def test_resume_preserves_metrics() -> None:
    pipeline = Step("m", MetricAgent()) >> Step.human_in_the_loop("pause")
    runner = Flujo(pipeline)
    paused = await gather_result(runner, 0)
    cost_before = paused.total_cost_usd
    resumed = await runner.resume_async(paused, "ok")
    assert resumed.total_cost_usd == cost_before


@pytest.mark.asyncio
async def test_cannot_resume_non_paused_pipeline() -> None:
    pipeline = Step("a", StubAgent(["done"]))
    runner = Flujo(pipeline)
    result = await gather_result(runner, "x")
    with pytest.raises(OrchestratorError):
        await runner.resume_async(result, "irrelevant")


@pytest.mark.asyncio
async def test_paused_hitl_pipeline_can_be_serialized_and_resumed() -> None:
    pipeline = Step("first", StubAgent(["draft"])) >> Step.human_in_the_loop("pause")
    runner = Flujo(pipeline)

    paused = await gather_result(runner, "start")
    pickled_result = pickle.dumps(paused)
    unpickled_result = pickle.loads(pickled_result)

    resumed = await runner.resume_async(unpickled_result, "human response")

    assert resumed.step_history[-1].success
    assert resumed.step_history[-1].output == "human response"
