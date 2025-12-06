import pytest

from flujo.application.core.executor_core import ExecutorCore
from flujo.application.core.step_policies import DefaultLoopStepExecutor
from flujo.application.core.executor_helpers import make_execution_frame
from flujo.domain.dsl.step import Step
from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.dsl.loop import LoopStep
from flujo.domain.models import PipelineContext, Success


@pytest.mark.asyncio
async def test_conversation_history_skips_finish_artifact():
    """Assistant turn with action='finish' must not be logged into conversation_history."""

    class _FinishAgent:
        async def run(self, payload, context=None, resources=None, **kwargs):
            # Emulate clarification agent finish output
            return {"action": "finish", "question": None}

    body = Pipeline.from_step(Step(name="check_if_more_info_needed", agent=_FinishAgent()))

    loop = LoopStep(
        name="clarification_loop",
        loop_body_pipeline=body,
        exit_condition_callable=lambda _o, _c: True,
        max_loops=1,
    )
    # Enable conversational mode and ensure assistant turns would be captured from this step
    loop.meta["conversation"] = True
    loop.meta["ai_turn_source"] = "named_steps"
    loop.meta["user_turn_sources"] = ["hitl"]
    loop.meta["named_steps"] = ["check_if_more_info_needed"]
    loop.meta["history_management"] = {"strategy": "truncate_tokens", "max_tokens": 999}

    core = ExecutorCore()
    ctx = PipelineContext()
    # Seed an initial goal input to appear as first user turn in history
    data = "find patients"

    frame = make_execution_frame(
        core,
        loop,
        data=data,
        context=ctx,
        resources=None,
        limits=None,
        context_setter=None,
        stream=False,
        on_chunk=None,
        fallback_depth=0,
        result=None,
        quota=None,
    )
    outcome = await DefaultLoopStepExecutor().execute(core, frame)

    assert isinstance(outcome, Success)
    final_ctx = outcome.step_result.branch_context
    assert final_ctx is not None
    hist = getattr(final_ctx, "conversation_history", [])
    # Expect only initial user turn; no assistant 'finish' artifact
    assert hist and hist[0].role.value == "user"
    # There should be no assistant message equal to the finish artifact
    assert all(
        (t.role.value != "assistant") or ("action='finish'" not in t.content.lower()) for t in hist
    )
