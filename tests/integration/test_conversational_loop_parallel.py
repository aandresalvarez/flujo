import pytest

from flujo.domain.dsl.step import Step
from flujo.domain.dsl.pipeline import Pipeline
from flujo.application.runner import Flujo
from flujo.domain.models import PipelineContext, ConversationRole


@pytest.mark.asyncio
@pytest.mark.slow
@pytest.mark.serial
async def test_conversational_loop_parallel_all_agents():
    async def a(_: str, *, context: PipelineContext | None = None) -> str:
        return "branch A"

    async def b(_: str, *, context: PipelineContext | None = None) -> str:
        return "branch B"

    sa = Step.from_callable(a, name="sa")
    sb = Step.from_callable(b, name="sb")

    # Build a parallel via Step.parallel: two branches "pa" and "pb"
    par = Step.parallel(name="par", branches={"pa": sa, "pb": sb})

    # Loop body: parallel only; exit immediately after first iteration
    body = Pipeline.from_step(par)

    def _exit(_out: str, _ctx: PipelineContext | None) -> bool:
        return True

    from flujo.domain.dsl.loop import LoopStep

    loop = LoopStep(
        name="par_loop",
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
    asst = [
        t.content.lower() for t in ctx.conversation_history if t.role == ConversationRole.assistant
    ]
    # Expect both branch outputs present as assistant turns
    assert any("branch a" in s for s in asst)
    assert any("branch b" in s for s in asst)
