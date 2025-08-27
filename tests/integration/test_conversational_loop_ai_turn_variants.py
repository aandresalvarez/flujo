import pytest

from flujo.domain.dsl.step import Step
from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.dsl.loop import LoopStep
from flujo.application.runner import Flujo
from flujo.domain.models import PipelineContext, ConversationRole


@pytest.mark.asyncio
@pytest.mark.slow
@pytest.mark.serial
async def test_conversational_loop_ai_turn_source_all_agents():
    async def a1(msg: str, *, context: PipelineContext | None = None) -> str:
        return "A1 says hi"

    async def a2(msg: str, *, context: PipelineContext | None = None) -> str:
        return "A2 confirms"

    step1 = Step.from_callable(a1, name="a1")
    step2 = Step.from_callable(a2, name="a2")

    body = Pipeline.from_step(step1) >> step2

    def _exit(last_output: str, _ctx: PipelineContext | None) -> bool:
        return True  # single iteration

    loop = LoopStep(
        name="multi_agents",
        loop_body_pipeline=body,
        exit_condition_callable=_exit,
        max_retries=1,
    )
    loop.meta["conversation"] = True
    loop.meta["ai_turn_source"] = "all_agents"
    pipeline = Pipeline.from_step(loop)
    runner = Flujo(pipeline)

    result = None
    async for r in runner.run_async("Start"):
        result = r
    assert result is not None
    success_attr = getattr(result, "success", None)
    if success_attr is None:
        assert getattr(result, "step_history", None), "No steps executed"
        assert result.step_history[-1].success is True
    else:
        assert success_attr is True
    ctx = result.final_pipeline_context  # type: ignore[union-attr]
    assert isinstance(ctx, PipelineContext)
    asst_texts = [
        t.content.lower() for t in ctx.conversation_history if t.role == ConversationRole.assistant
    ]
    assert any("a1" in s or "says hi" in s for s in asst_texts)
    assert any("a2" in s or "confirms" in s for s in asst_texts)


@pytest.mark.asyncio
@pytest.mark.slow
@pytest.mark.serial
async def test_conversational_loop_ai_turn_source_named_steps():
    async def a1(msg: str, *, context: PipelineContext | None = None) -> str:
        return "A1 only"

    async def a2(msg: str, *, context: PipelineContext | None = None) -> str:
        return "A2 should not appear"

    step1 = Step.from_callable(a1, name="a1")
    step2 = Step.from_callable(a2, name="a2")
    body = Pipeline.from_step(step1) >> step2

    def _exit(last_output: str, _ctx: PipelineContext | None) -> bool:
        return True

    loop = LoopStep(
        name="named_src",
        loop_body_pipeline=body,
        exit_condition_callable=_exit,
        max_retries=1,
    )
    loop.meta["conversation"] = True
    loop.meta["ai_turn_source"] = "named_steps"
    loop.meta["named_steps"] = ["a1"]

    pipeline = Pipeline.from_step(loop)
    runner = Flujo(pipeline)

    result = None
    async for r in runner.run_async("Start"):
        result = r
    assert result is not None
    success_attr = getattr(result, "success", None)
    if success_attr is None:
        assert getattr(result, "step_history", None), "No steps executed"
        assert result.step_history[-1].success is True
    else:
        assert success_attr is True
    ctx = result.final_pipeline_context  # type: ignore[union-attr]
    assert isinstance(ctx, PipelineContext)
    asst_texts = [
        t.content.lower() for t in ctx.conversation_history if t.role == ConversationRole.assistant
    ]
    assert any("a1" in s or "only" in s for s in asst_texts)
    assert not any("a2" in s or "should not appear" in s for s in asst_texts)
