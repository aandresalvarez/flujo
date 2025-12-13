import pytest

from flujo.domain.dsl.step import Step
from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.dsl.loop import LoopStep
from flujo.application.runner import Flujo
from flujo.domain.models import PipelineContext, ConversationRole


@pytest.mark.asyncio
@pytest.mark.slow
@pytest.mark.serial
async def test_conversational_loop_hitl_pause_and_resume(tmp_path):
    # Agents
    async def clarify_agent(msg: str, *, context: PipelineContext | None = None) -> str:
        return "Please provide the deadline"

    # Loop body: clarify -> HITL -> finalize
    clarify = Step.from_callable(clarify_agent, name="clarify")
    hitl = Step.human_in_the_loop(name="hitl", message_for_user="Provide deadline")

    async def finalize_agent(_: str, *, context: PipelineContext | None = None) -> str:
        return "finish"

    finalize = Step.from_callable(finalize_agent, name="finalize")
    body = Pipeline.from_step(clarify) >> hitl >> finalize

    def _exit(last_output: str, _ctx: PipelineContext | None) -> bool:
        return str(last_output).strip().lower() == "finish"

    loop = LoopStep(
        name="clarification_loop",
        loop_body_pipeline=body,
        exit_condition_callable=_exit,
        max_retries=3,
    )
    loop.meta["conversation"] = True
    loop.meta["history_management"] = {"strategy": "truncate_turns", "max_turns": 20}

    pipeline = Pipeline.from_step(loop)
    runner = Flujo(pipeline)

    # First run should pause at HITL
    paused = None
    async for res in runner.run_async("Initial Goal", run_id="conv-run-hitl"):
        paused = res
        break
    assert paused is not None
    ctx = paused.final_pipeline_context  # type: ignore[union-attr]
    assert isinstance(ctx, PipelineContext)
    assert ctx.status == "paused"

    # Resume with human input
    resumed = await runner.resume_async(paused, human_input="Tomorrow")
    assert resumed.success is True
    rctx = resumed.final_pipeline_context
    assert isinstance(rctx, PipelineContext)

    # Verify history contains: initial user, assistant clarify, user from HITL, assistant finalize
    roles = [t.role for t in rctx.conversation_history]
    # Must contain at least one user and one assistant after resume
    assert ConversationRole.user in roles and ConversationRole.assistant in roles
    assert any(
        "deadline" in t.content.lower()
        for t in rctx.conversation_history
        if t.role == ConversationRole.assistant
    )
    assert any(
        "tomorrow" in t.content.lower()
        for t in rctx.conversation_history
        if t.role == ConversationRole.user
    )
