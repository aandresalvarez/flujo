import pytest
from pathlib import Path

from flujo.domain.dsl.step import Step
from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.dsl.loop import LoopStep
from flujo.application.runner import Flujo
from flujo.domain.models import PipelineContext, ConversationRole, PipelineResult
from flujo.application.core.state_manager import StateManager
from flujo.state.backends.sqlite import SQLiteBackend


@pytest.mark.asyncio
@pytest.mark.slow
@pytest.mark.serial
async def test_sqlite_pause_resume_persists_conversation(tmp_path: Path) -> None:
    db = SQLiteBackend(tmp_path / "ops.db")

    async def clarify(_: str, *, context: PipelineContext | None = None) -> str:
        return "Ask: deadline?"

    clarify_step = Step.from_callable(clarify, name="clarify")
    hitl = Step.human_in_the_loop(name="hitl", message_for_user="Provide deadline")

    async def finalize(_: str, *, context: PipelineContext | None = None) -> str:
        return "finish"

    finalize_step = Step.from_callable(finalize, name="finalize")

    body = Pipeline.from_step(clarify_step) >> hitl >> finalize_step

    def _exit(last_output: str, _ctx: PipelineContext | None) -> bool:
        return str(last_output).strip().lower() == "finish"

    loop = LoopStep(
        name="conv",
        loop_body_pipeline=body,
        exit_condition_callable=_exit,
        max_retries=3,
    )
    loop.meta["conversation"] = True
    loop.meta["ai_turn_source"] = "all_agents"
    loop.meta["user_turn_sources"] = ["hitl"]
    pipeline = Pipeline.from_step(loop)

    run_id = "conv-sqlite-1"
    runner = Flujo(pipeline, state_backend=db)

    # First run: pause
    paused = None
    async for r in runner.run_async("Goal", run_id=run_id):
        paused = r
        break
    assert paused is not None

    # Simulate restart: reconstruct paused result from backend
    sm = StateManager[PipelineContext](db)
    ctx, last_output, idx, created_at, pname, pver, step_history = await sm.load_workflow_state(
        run_id, PipelineContext
    )
    assert ctx is not None
    paused_reconstructed: PipelineResult[PipelineContext] = PipelineResult(
        step_history=step_history,
        total_cost_usd=0.0,
        total_tokens=0,
        final_pipeline_context=ctx,
    )

    # New runner instance resumes using reconstructed result
    runner2 = Flujo(pipeline, state_backend=db)
    resumed = await runner2.resume_async(paused_reconstructed, human_input="Tomorrow")
    # Accept PipelineResult return; verify final step success
    if hasattr(resumed, "success"):
        assert resumed.success is True  # type: ignore[attr-defined]
    else:
        assert resumed.step_history and resumed.step_history[-1].success is True
    rctx = resumed.final_pipeline_context
    assert isinstance(rctx, PipelineContext)
    # Verify conversation history includes seeded initial, assistant clarify, user resume
    roles = [t.role for t in rctx.conversation_history]
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
