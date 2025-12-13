import pytest

from flujo.domain.dsl.step import Step
from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.dsl.loop import LoopStep
from flujo.domain.dsl.step import BranchKey
from flujo.application.runner import Flujo
from flujo.domain.models import PipelineContext, ConversationRole


@pytest.mark.asyncio
async def test_conversational_loop_captures_nested_hitl_user_turns():
    """Ensure conversation:true captures HITL answers nested in a ConditionalStep.

    Repro structure:
      loop (conversation:true)
        - clarify (simple agent) -> returns 'ask' until it sees >= 2 turns
        - conditional on clarify output:
            true  -> HITL (ask_user_for_clarification)
            false -> finalize (returns 'done')
      exit when last output == 'done'
    """

    # Agent that asks once, then finishes after seeing user's clarification captured in history
    async def clarify_agent(_: str, *, context: PipelineContext | None = None) -> str:
        if context is not None and len(context.conversation_history) >= 2:
            return "finish"
        return "ask"

    clarify = Step.from_callable(clarify_agent, name="check_if_more_info_needed")

    # Conditional routing: 'ask' -> HITL branch, else -> finalize branch
    def _cond(prev_out: str, _ctx: PipelineContext | None) -> BranchKey:
        return "true" if str(prev_out).strip().lower() == "ask" else "false"

    hitl = Step.human_in_the_loop(
        name="ask_user_for_clarification", message_for_user="Provide more details"
    )
    # Optional: mark as updates_context to mirror YAML convention; not required by policy
    try:
        hitl.updates_context = True  # type: ignore[attr-defined]
    except Exception:
        pass

    async def finalize_agent(_: str, *, context: PipelineContext | None = None) -> str:
        return "done"

    finalize = Step.from_callable(finalize_agent, name="finalize")

    from flujo.domain.dsl import Pipeline as _P

    branches = {
        "true": _P.from_step(hitl),
        "false": _P.from_step(finalize),
    }
    conditional = Step.branch_on(
        name="ask_question_if_needed",
        condition_callable=_cond,
        branches=branches,
    )

    body = Pipeline.from_step(clarify) >> conditional

    def _exit(last_output: str, _ctx: PipelineContext | None) -> bool:
        return str(last_output).strip().lower() == "done"

    loop = LoopStep(
        name="clarification_loop",
        loop_body_pipeline=body,
        exit_condition_callable=_exit,
        max_retries=3,
    )
    loop.meta["conversation"] = True
    loop.meta["history_management"] = {"strategy": "truncate_turns", "max_turns": 20}

    pipeline = Pipeline.from_step(loop)
    runner: Flujo[object, object, PipelineContext] = Flujo(pipeline)

    # First run: pauses at nested HITL inside conditional branch
    paused = None
    async for res in runner.run_async("Initial Goal", run_id="conv-nested-hitl"):
        paused = res
        break
    assert paused is not None
    ctx = paused.final_pipeline_context  # type: ignore[union-attr]
    assert isinstance(ctx, PipelineContext)
    assert ctx.status == "paused"

    # Resume with user's clarification
    resumed = await runner.resume_async(paused, human_input="I want details today")
    rctx = resumed.final_pipeline_context
    assert isinstance(rctx, PipelineContext)

    # Verify conversation history includes the nested HITL user turn and an assistant turn
    roles = [t.role for t in rctx.conversation_history]
    assert ConversationRole.user in roles and ConversationRole.assistant in roles

    # Must include our human input from the nested HITL as a user turn
    assert any(
        "details today" in t.content.lower()
        for t in rctx.conversation_history
        if t.role == ConversationRole.user
    )

    # Loop should exit by condition (not by max_loops)
    loop_result = next(sr for sr in resumed.step_history if sr.name == "clarification_loop")
    assert loop_result.metadata_.get("exit_reason") in {"condition", "max_retries", None}
