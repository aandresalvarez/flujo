import asyncio
import pytest

from flujo.domain.dsl.step import Step
from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.dsl.loop import LoopStep
from flujo.application.runner import Flujo
from flujo.domain.models import PipelineContext, ConversationRole


@pytest.mark.slow
@pytest.mark.serial
def test_conversational_loop_injection_and_history_evolution():
    calls = {"clarify": 0}

    async def clarify_agent(msg: str, *, context: PipelineContext | None = None) -> str:
        # First call asks a question; second signals finish
        calls["clarify"] += 1
        if calls["clarify"] == 1:
            return "What is the deadline?"
        return "finish"

    async def planner_agent(_: str, *, context: PipelineContext | None = None) -> str:
        # Echo a summary that should include prior turns when injected
        # In a real chat model, it would use the injected history implicitly.
        # Here we just return a fixed string to mark completion.
        return "Final Plan"

    clarify = Step.from_callable(clarify_agent, name="clarify", updates_context=False)
    planner = Step.from_callable(planner_agent, name="plan", updates_context=False)

    body = Pipeline.from_step(clarify) >> planner

    def exit_when_finished(last_output: str, ctx: PipelineContext | None) -> bool:
        # Loop should stop after second clarify returns 'finish'
        try:
            return str(last_output).strip().lower() == "finish"
        except Exception:
            return False

    loop = LoopStep(
        name="clarification_loop",
        loop_body_pipeline=body,
        exit_condition_callable=exit_when_finished,
        max_retries=3,
    )
    # Enable conversation features and set simple history management
    loop.meta["conversation"] = True
    loop.meta["history_management"] = {"strategy": "truncate_turns", "max_turns": 10}

    pipeline = Pipeline.from_step(loop)

    runner = Flujo(pipeline)
    # Run with initial input; loop should:
    # - seed initial user turn
    # - append assistant turn from clarify("What is the deadline?")
    # - then planner runs; exit after clarify returns "finish"
    result = asyncio.get_event_loop().run_until_complete(runner.run_async("Initial Goal"))

    assert result.success is True
    ctx = result.final_pipeline_context
    assert isinstance(ctx, PipelineContext)
    # Verify history has at least the seeded user turn and one assistant turn
    assert len(ctx.conversation_history) >= 2
    roles = [t.role for t in ctx.conversation_history]
    assert roles[0] == ConversationRole.user
    # Find an assistant turn containing the clarify output
    assert any(
        (t.role == ConversationRole.assistant and "deadline" in t.content.lower())
        for t in ctx.conversation_history
    )
