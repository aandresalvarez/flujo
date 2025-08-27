import pytest

from flujo.domain.dsl.step import Step
from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.dsl.loop import LoopStep
from flujo.application.runner import Flujo
from flujo.domain.models import PipelineContext, ConversationRole


@pytest.mark.asyncio
@pytest.mark.slow
@pytest.mark.serial
async def test_nested_conversation_inner_scoped():
    # Inner loop agents
    async def inner_clarify(_: str, *, context: PipelineContext | None = None) -> str:
        return "Inner question"

    async def inner_finalize(_: str, *, context: PipelineContext | None = None) -> str:
        return "finish"

    s_ic = Step.from_callable(inner_clarify, name="inner_clarify")
    s_if = Step.from_callable(inner_finalize, name="inner_finalize")
    inner_body = Pipeline.from_step(s_ic) >> s_if

    def inner_exit(last_output: str, _ctx: PipelineContext | None) -> bool:
        return str(last_output).strip().lower() == "finish"

    inner_loop = LoopStep(
        name="inner_loop",
        loop_body_pipeline=inner_body,
        exit_condition_callable=inner_exit,
        max_retries=2,
    )
    inner_loop.meta["conversation"] = True

    # Outer loop body: just run inner loop once
    async def outer_stub(_: str, *, context: PipelineContext | None = None) -> str:
        return "ok"

    outer_stub_step = Step.from_callable(outer_stub, name="outer_stub")
    outer_body = Pipeline.from_step(inner_loop) >> outer_stub_step

    def outer_exit(last_output: str, _ctx: PipelineContext | None) -> bool:
        return True

    outer_loop = LoopStep(
        name="outer_loop",
        loop_body_pipeline=outer_body,
        exit_condition_callable=outer_exit,
        max_retries=1,
    )
    # Note: only inner loop has conversation: true enabled

    pipeline = Pipeline.from_step(outer_loop)
    runner = Flujo(pipeline)

    # Run
    result = None
    async for r in runner.run_async("Nested Start"):
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
    # Verify assistant turn from inner loop made it into history
    assert any(
        (t.role == ConversationRole.assistant and "inner question" in t.content.lower())
        for t in ctx.conversation_history
    )
