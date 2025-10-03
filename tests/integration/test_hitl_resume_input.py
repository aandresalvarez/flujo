"""Integration tests for resume_input template variable in HITL pipelines."""

from __future__ import annotations

import pytest

from flujo.application.runner import Flujo
from flujo.domain.models import PipelineContext
from flujo.domain.dsl import HumanInTheLoopStep, Pipeline, Step
from flujo.testing.utils import gather_result


@pytest.mark.slow
@pytest.mark.serial
async def test_resume_input_available_after_hitl():
    """Test that resume_input is available in templates after HITL step."""
    # Create pipeline with HITL step followed by a step that uses resume_input
    hitl_step = HumanInTheLoopStep(name="ask_user", message_for_user="What is your name?")

    async def echo(data: str) -> str:
        return data

    echo_step = Step.from_callable(echo, name="echo_response")
    echo_step.meta["templated_input"] = "User said: {{ resume_input }}"

    pipeline = Pipeline.from_step(hitl_step) >> echo_step
    runner = Flujo(pipeline, context_model=PipelineContext)

    # Run until pause
    paused = None
    async for result in runner.run_async(""):
        paused = result
        break

    assert paused is not None

    # Resume with human input
    resumed = await runner.resume_async(paused, "Alice")

    assert resumed.success
    assert resumed.output == "User said: Alice"


@pytest.mark.slow
@pytest.mark.serial
async def test_resume_input_in_conditional_expression():
    """Test that resume_input works in conditional expressions."""
    # Create pipeline with HITL step followed by conditional based on resume_input
    hitl_step = HumanInTheLoopStep(name="ask_continue", message_for_user="Continue? (yes/no)")

    async def approve() -> str:
        return "approved"

    async def deny() -> str:
        return "denied"

    from flujo.domain.dsl import ConditionalStep

    approve_step = Step.from_callable(approve, name="approve")
    deny_step = Step.from_callable(deny, name="deny")

    # Create conditional with expression using resume_input
    from flujo.utils.expressions import compile_expression_to_callable

    condition_fn = compile_expression_to_callable("resume_input.lower() == 'yes'")

    conditional = ConditionalStep(
        name="check_response",
        condition=condition_fn,
        branches={
            True: Pipeline.from_step(approve_step),
            False: Pipeline.from_step(deny_step),
        },
    )

    pipeline = Pipeline.from_step(hitl_step) >> conditional
    runner = Flujo(pipeline, context_model=PipelineContext)

    # Test with "yes" response
    paused = None
    async for result in runner.run_async(""):
        paused = result
        break

    assert paused is not None

    resumed = await runner.resume_async(paused, "yes")

    assert resumed.success
    assert resumed.output == "approved"


@pytest.mark.slow
@pytest.mark.serial
async def test_resume_input_none_without_hitl():
    """Test that resume_input is None before first HITL."""

    # Create a simple step that tries to use resume_input when no HITL has happened
    async def check_resume(data: str) -> str:
        return data

    step = Step.from_callable(check_resume, name="check")
    # Try to use resume_input in template - should be undefined/None
    step.meta["templated_input"] = "resume_input value: {{ resume_input | default('not_set') }}"

    pipeline = Pipeline.from_step(step)
    runner = Flujo(pipeline, context_model=PipelineContext)

    final_result = await gather_result(runner, "test")
    assert final_result.success
    # resume_input should show as not_set before HITL
    assert "not_set" in final_result.output


@pytest.mark.slow
@pytest.mark.serial
async def test_resume_input_updates_after_each_hitl():
    """Test that resume_input updates with each HITL response."""
    # Create pipeline with multiple HITL steps
    hitl1 = HumanInTheLoopStep(name="ask_first", message_for_user="First question?")
    hitl2 = HumanInTheLoopStep(name="ask_second", message_for_user="Second question?")

    async def combine(data: str) -> str:
        return data

    combine_step = Step.from_callable(combine, name="combine")
    combine_step.meta["templated_input"] = "Last answer: {{ resume_input }}"

    pipeline = Pipeline.from_step(hitl1) >> hitl2 >> combine_step
    runner = Flujo(pipeline, context_model=PipelineContext)

    # Run until first pause
    paused1 = None
    async for result in runner.run_async(""):
        paused1 = result
        break

    assert paused1 is not None

    # Resume with first answer
    paused2 = await runner.resume_async(paused1, "answer1")

    # Resume with second answer
    final = await runner.resume_async(paused2, "answer2")

    assert final.success
    # Should show the SECOND answer (most recent HITL)
    assert "answer2" in final.output
