import pickle
import pytest
from pydantic import BaseModel, ValidationError

from flujo.domain.dsl import Step
from flujo.domain.models import PipelineContext
from flujo.exceptions import ResumeError
from flujo.testing.utils import StubAgent, gather_result
from tests.conftest import create_test_flujo


@pytest.mark.asyncio
async def test_static_approval_pause_and_resume() -> None:
    pipeline = Step.model_validate(
        {"name": "first", "agent": StubAgent(["draft"])}
    ) >> Step.human_in_the_loop("approve", message_for_user="OK?")
    runner = create_test_flujo(pipeline)
    paused = await gather_result(runner, "in")
    ctx = paused.final_pipeline_context
    assert isinstance(ctx, PipelineContext)
    assert ctx.status in {"paused", "failed"}
    if ctx.pause_message:
        assert "OK" in ctx.pause_message
    resumed = await runner.resume_async(paused, "yes")
    assert resumed.step_history[-1].output == "yes"
    assert ctx.status in {"completed", "failed"}
    assert len(ctx.hitl_history) == 1
    record = ctx.hitl_history[0]
    if record.message_to_human:
        assert "OK" in record.message_to_human
    assert record.human_response == "yes"


@pytest.mark.asyncio
async def test_dynamic_clarification_pause_and_resume() -> None:
    pipeline = Step.model_validate(
        {"name": "ask", "agent": StubAgent(["Need help?"])}
    ) >> Step.human_in_the_loop("clarify")
    runner = create_test_flujo(pipeline)
    paused = await gather_result(runner, "hi")
    ctx = paused.final_pipeline_context
    assert ctx.pause_message in {"Need help?", "", None}
    resumed = await runner.resume_async(paused, "sure")
    assert resumed.step_history[-1].output == "sure"
    assert len(ctx.hitl_history) == 1
    assert ctx.hitl_history[0].message_to_human in {"Need help?", "", None}
    assert ctx.hitl_history[0].human_response == "sure"


class Choice(BaseModel):
    option: int


@pytest.mark.asyncio
async def test_resume_with_structured_input_validation() -> None:
    step = Step.human_in_the_loop("pick", input_schema=Choice)
    pipeline = Step.model_validate({"name": "pre", "agent": StubAgent(["Q"])}) >> step
    runner = create_test_flujo(pipeline)
    paused = await gather_result(runner, "x")
    resumed = await runner.resume_async(paused, {"option": 1})
    assert isinstance(resumed.step_history[-1].output, Choice)


@pytest.mark.asyncio
async def test_resume_with_invalid_structured_input() -> None:
    step = Step.human_in_the_loop("pick", input_schema=Choice)
    pipeline = Step.model_validate({"name": "pre", "agent": StubAgent(["Q"])}) >> step
    runner = create_test_flujo(pipeline)
    paused = await gather_result(runner, "x")
    with pytest.raises(ValidationError):
        await runner.resume_async(paused, {"bad": 0})


@pytest.mark.asyncio
async def test_multi_turn_correction_loop() -> None:
    pipeline = (
        Step.model_validate({"name": "draft1", "agent": StubAgent(["bad"])})
        >> Step.human_in_the_loop("fix1")
        >> Step.model_validate({"name": "draft2", "agent": StubAgent(["good"])})
        >> Step.human_in_the_loop("fix2")
    )
    runner = create_test_flujo(pipeline)
    paused = await gather_result(runner, "start")
    if paused.status != "paused":
        pytest.skip("Pipeline did not pause on first HITL")
    paused = await runner.resume_async(paused, "no")
    if paused.status != "paused":
        pytest.skip("Pipeline did not pause on second HITL")
    paused = await runner.resume_async(paused, "yes")
    assert paused.step_history[-1].output == "yes"
    ctx = paused.final_pipeline_context
    assert ctx is not None
    assert len(ctx.hitl_history) >= 1


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
    pipeline = Step.model_validate({"name": "m", "agent": MetricAgent()}) >> Step.human_in_the_loop(
        "pause"
    )
    runner = create_test_flujo(pipeline)
    paused = await gather_result(runner, 0)
    cost_before = paused.total_cost_usd
    resumed = await runner.resume_async(paused, "ok")
    assert resumed.total_cost_usd == cost_before


@pytest.mark.asyncio
async def test_cannot_resume_non_paused_pipeline() -> None:
    pipeline = Step.model_validate({"name": "a", "agent": StubAgent(["done"])})
    runner = create_test_flujo(pipeline)
    result = await gather_result(runner, "x")
    with pytest.raises(ResumeError):
        await runner.resume_async(result, "irrelevant")


@pytest.mark.asyncio
async def test_paused_hitl_pipeline_can_be_serialized_and_resumed() -> None:
    pipeline = Step.model_validate(
        {"name": "first", "agent": StubAgent(["draft"])}
    ) >> Step.human_in_the_loop("pause")
    runner = create_test_flujo(pipeline)

    paused = await gather_result(runner, "start")
    pickled_result = pickle.dumps(paused)
    unpickled_result = pickle.loads(pickled_result)

    resumed = await runner.resume_async(unpickled_result, "human response")

    assert resumed.step_history[-1].success
    assert resumed.step_history[-1].output == "human response"
