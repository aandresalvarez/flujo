import pytest

from flujo.domain.dsl.step import Step
from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.dsl.loop import LoopStep
from flujo.domain.dsl.step import BranchKey
from flujo.application.runner import Flujo
from flujo.domain.models import PipelineContext, ConversationRole
from flujo.domain.dsl.import_step import ImportStep


@pytest.mark.asyncio
async def test_import_step_preserves_hitl_state_on_pause_resume() -> None:
    """Regression: Ensure ImportStep merges child context on pause so resume progresses.

    Child structure:
      loop (conversation:true)
        - clarify (simple agent) -> returns 'ask' until it sees >= 2 turns
        - conditional on clarify output:
            true  -> HITL (ask_user_for_clarification)
            false -> finalize (returns 'done')
      exit when last output == 'done'

    The parent wraps this child pipeline in ImportStep with inherit_context=true.
    On first run: pause is raised by nested HITL; ImportStep must merge child state
    (conversation/hitl) back to parent context before propagating pause. On resume,
    the child should see the user's answer and finish instead of looping forever.
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

    child_body = Pipeline.from_step(clarify) >> conditional

    def _exit(last_output: str, _ctx: PipelineContext | None) -> bool:
        return str(last_output).strip().lower() == "done"

    loop = LoopStep(
        name="clarification_loop",
        loop_body_pipeline=child_body,
        exit_condition_callable=_exit,
        max_retries=3,
    )
    loop.meta["conversation"] = True
    loop.meta["history_management"] = {"strategy": "truncate_turns", "max_turns": 20}

    child_pipeline = Pipeline.from_step(loop)

    # Wrap as ImportStep in a parent pipeline
    import_step = ImportStep(
        name="run_child",
        pipeline=child_pipeline,
        inherit_context=True,
        input_to="initial_prompt",
        inherit_conversation=True,
        propagate_hitl=True,
        updates_context=True,
    )

    parent = Pipeline.from_step(import_step)
    runner: Flujo[object, object, PipelineContext] = Flujo(parent)

    # First run pauses at nested HITL inside import
    paused = None
    async for res in runner.run_async("Find patients with flu", run_id="import-nested-hitl"):
        paused = res
        break
    assert paused is not None
    ctx = paused.final_pipeline_context  # type: ignore[union-attr]
    assert isinstance(ctx, PipelineContext)
    assert ctx.status == "paused"

    # Resume with user's clarification; should advance and finish
    resumed = await runner.resume_async(paused, human_input="count over all time")
    rctx = resumed.final_pipeline_context
    assert isinstance(rctx, PipelineContext)

    # Verify conversation history includes the nested HITL user turn
    roles = [t.role for t in rctx.conversation_history]
    assert ConversationRole.user in roles and ConversationRole.assistant in roles
    assert any(
        "count" in t.content.lower()
        for t in rctx.conversation_history
        if t.role == ConversationRole.user
    )

    # Loop should exit by condition (not max_loops)
    # Inspect the ImportStep result's child step history for the loop outcome
    import_results = [sr for sr in resumed.step_history if sr.name == "run_child"]
    assert import_results, "expected ImportStep result in history"
    child_sr_list = import_results[-1].step_history or []
    assert child_sr_list, "expected child step history on ImportStep result"
    child_loop = next(
        (sr for sr in child_sr_list if sr.name == "clarification_loop"), child_sr_list[-1]
    )
    assert child_loop.success is True
    assert child_loop.metadata_.get("exit_reason") == "condition"
