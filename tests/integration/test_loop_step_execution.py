import pytest
from pydantic import BaseModel

from pydantic_ai_orchestrator.application.pipeline_runner import PipelineRunner
from pydantic_ai_orchestrator.domain import Step, Pipeline, LoopStep


class IncrementAgent:
    async def run(self, data: int, **kwargs) -> int:
        return data + 1


@pytest.mark.asyncio
async def test_basic_loop_until_condition_met() -> None:
    body = Pipeline.from_step(Step("inc", IncrementAgent()))
    loop = Step.loop_until(
        name="loop",
        loop_body_pipeline=body,
        exit_condition_callable=lambda out, ctx: out >= 3,
        max_loops=5,
    )
    runner = PipelineRunner(loop)
    result = await runner.run_async(0)
    step_result = result.step_history[-1]
    assert step_result.success is True
    assert step_result.attempts == 3
    assert step_result.output == 3


@pytest.mark.asyncio
async def test_loop_max_loops_reached() -> None:
    body = Pipeline.from_step(Step("inc", IncrementAgent()))
    loop = Step.loop_until(
        name="loop",
        loop_body_pipeline=body,
        exit_condition_callable=lambda out, ctx: out > 10,
        max_loops=2,
    )
    runner = PipelineRunner(loop)
    result = await runner.run_async(0)
    step_result = result.step_history[-1]
    assert step_result.success is False
    assert step_result.attempts == 2


class Ctx(BaseModel):
    counter: int = 0


@pytest.mark.asyncio
async def test_loop_with_context_modification() -> None:
    class IncRecordAgent:
        async def run(self, x: int, *, pipeline_context: Ctx | None = None) -> int:
            if pipeline_context:
                pipeline_context.counter += 1
            return x + 1

    body = Pipeline.from_step(Step("inc", IncRecordAgent()))
    loop = Step.loop_until(
        name="loop_ctx",
        loop_body_pipeline=body,
        exit_condition_callable=lambda out, ctx: ctx and ctx.counter >= 2,
        max_loops=5,
    )
    runner = PipelineRunner(loop, context_model=Ctx)
    result = await runner.run_async(0)
    step_result = result.step_history[-1]
    assert step_result.success is True
    assert result.final_pipeline_context.counter == 2
