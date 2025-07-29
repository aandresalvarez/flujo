"""
Test Loop Steps with Context Updates

This test verifies that loop steps work correctly with @step(updates_context=True)
decorated functions in the loop body. This combination was identified as potentially
problematic due to context state management across iterations.
"""

from __future__ import annotations

import pytest
from typing import Any

from flujo import Step, Pipeline, step
from flujo.domain.models import PipelineContext
from flujo.testing.utils import gather_result
from tests.conftest import create_test_flujo, NoOpStateBackend


class LoopContext(PipelineContext):
    """Context for testing loop steps with context updates."""

    initial_prompt: str = "test"
    iteration_count: int = 0
    accumulated_value: int = 0
    loop_exit_reason: str = ""
    final_state: dict = {}


@step(updates_context=True)
async def increment_iteration(data: Any, *, context: LoopContext) -> dict:
    """Increment iteration count and return context update."""
    context.iteration_count += 1
    context.accumulated_value += data if isinstance(data, (int, float)) else 1

    return {
        "iteration_count": context.iteration_count,
        "accumulated_value": context.accumulated_value,
        "current_data": data,
    }


@step(updates_context=True)
async def conditional_exit(data: Any, *, context: LoopContext) -> dict:
    """Check if we should exit the loop based on context state."""
    should_exit = context.accumulated_value >= 10 or context.iteration_count >= 5

    if should_exit:
        context.loop_exit_reason = (
            f"Exited at iteration {context.iteration_count} with value {context.accumulated_value}"
        )

    return {
        "should_exit": should_exit,
        "exit_reason": context.loop_exit_reason,
    }


@step
async def finalize_state(data: Any, *, context: LoopContext) -> dict:
    """Finalize the loop state."""
    # Only update fields that exist in the context
    context.final_state = {
        "total_iterations": context.iteration_count,
        "final_accumulated_value": context.accumulated_value,
        "exit_reason": context.loop_exit_reason,
        "last_data": data,
    }

    # Return the final state as the output, but don't try to update context fields
    return {
        "final_state": context.final_state,
        "message": "Loop completed successfully",
    }


@pytest.mark.asyncio
async def test_loop_with_context_updates_basic():
    """Test basic loop execution with context updates."""

    # Create a simple loop that increments context state
    loop_body = Pipeline.from_step(increment_iteration)

    loop_step = Step.loop_until(
        name="test_loop",
        loop_body_pipeline=loop_body,
        exit_condition_callable=lambda data, context: context.iteration_count >= 3,
        max_loops=5,
    )

    runner = create_test_flujo(loop_step, context_model=LoopContext)
    result = await gather_result(runner, 5)

    # Verify loop executed correctly with context updates
    # FIXED: Context updates are now properly applied between iterations
    assert result.step_history[-1].success is True  # Loop exits successfully when condition met
    assert result.step_history[-1].attempts == 3  # Should exit after 3 iterations
    # Context updates are now properly applied
    assert result.final_pipeline_context.iteration_count >= 3
    assert result.final_pipeline_context.accumulated_value >= 3


@pytest.mark.asyncio
async def test_loop_with_context_updates_complex():
    """Test complex loop execution with multiple context-updating steps."""

    # Create a loop body with multiple context-updating steps
    # Use the >> operator to chain steps into a pipeline
    loop_body = increment_iteration >> conditional_exit

    loop_step = Step.loop_until(
        name="complex_loop",
        loop_body_pipeline=loop_body,
        exit_condition_callable=lambda data, context: context.accumulated_value >= 10,
        max_loops=10,
    )

    # Add finalization step
    pipeline = loop_step >> finalize_state

    runner = create_test_flujo(pipeline, context_model=LoopContext)
    result = await gather_result(runner, 3)

    # Verify complex loop execution with context updates
    # FIXED: Context updates are now properly applied between iterations
    assert result.step_history[-1].success is True  # Loop exits successfully when condition met
    assert result.step_history[-1].attempts >= 1  # Should have executed at least one iteration
    # Context updates are now properly applied
    assert result.final_pipeline_context.iteration_count >= 1
    assert result.final_pipeline_context.accumulated_value >= 1
    # Should have final state since loop completed successfully
    assert result.final_pipeline_context.final_state != {}


@pytest.mark.asyncio
async def test_loop_with_context_updates_mapper_conflicts():
    """Test loop execution with mappers that might conflict with context updates."""

    def iteration_mapper(output: Any, context: LoopContext, iteration: int) -> int:
        """Mapper that might conflict with context updates."""
        # Handle both dict and int outputs from steps
        if isinstance(output, dict):
            return output.get("current_data", 1) + iteration
        elif isinstance(output, (int, float)):
            return output + iteration
        else:
            # Default fallback
            return 1 + iteration

    def exit_condition(data: Any, context: LoopContext) -> bool:
        """Exit condition that depends on context state."""
        return context.iteration_count >= 3 or context.accumulated_value >= 15

    loop_body = Pipeline.from_step(increment_iteration)

    loop_step = Step.loop_until(
        name="mapper_loop",
        loop_body_pipeline=loop_body,
        exit_condition_callable=exit_condition,
        iteration_input_mapper=iteration_mapper,
        max_loops=5,
    )

    runner = create_test_flujo(loop_step, context_model=LoopContext)
    result = await gather_result(runner, 2)

    # Verify mapper doesn't conflict with context updates
    # FIXED: Context updates are now properly applied between iterations
    assert result.step_history[-1].success is True  # Loop exits successfully when condition met
    assert result.step_history[-1].attempts >= 1  # Should have executed at least one iteration
    # Context updates are now properly applied
    assert result.final_pipeline_context.iteration_count >= 1
    assert result.final_pipeline_context.accumulated_value >= 1


@pytest.mark.asyncio
async def test_loop_with_context_updates_max_loops():
    """Test loop execution that hits max_loops with context updates."""

    loop_body = Pipeline.from_step(increment_iteration)

    loop_step = Step.loop_until(
        name="max_loops_loop",
        loop_body_pipeline=loop_body,
        exit_condition_callable=lambda data, context: False,  # Never exit naturally
        max_loops=3,
    )

    runner = create_test_flujo(loop_step, context_model=LoopContext)
    result = await gather_result(runner, 1)

    # Verify loop fails due to max_loops when exit condition is never met
    assert result.step_history[-1].success is False  # Loop fails due to max_loops
    assert "max_loops" in result.step_history[-1].feedback.lower()
    # Context updates should still be applied even when hitting max_loops
    assert result.final_pipeline_context.iteration_count >= 1
    assert result.final_pipeline_context.accumulated_value >= 1


@pytest.mark.asyncio
async def test_loop_with_context_updates_error_handling():
    """Test loop execution with context updates when steps fail."""

    @step(updates_context=True)
    async def failing_step(data: Any, *, context: LoopContext) -> dict:
        """Step that fails after updating context."""
        context.iteration_count += 1
        if context.iteration_count >= 2:
            raise RuntimeError("Intentional failure")
        return {"status": "success"}

    loop_body = Pipeline.from_step(failing_step)

    loop_step = Step.loop_until(
        name="error_loop",
        loop_body_pipeline=loop_body,
        exit_condition_callable=lambda data, context: context.iteration_count >= 3,
        max_loops=5,
    )

    runner = create_test_flujo(loop_step, context_model=LoopContext)
    result = await gather_result(runner, 1)

    # Verify error handling with context updates
    # FIXED: Context updates are now properly applied between iterations
    # Loop should exit when condition is met, even if some iterations fail
    assert result.step_history[-1].success is False  # Should fail due to step errors
    assert "loop exited by condition" in result.step_history[-1].feedback.lower()
    # Context updates should still be applied
    assert result.final_pipeline_context.iteration_count >= 1


@pytest.mark.asyncio
async def test_loop_with_context_updates_state_isolation():
    """Test that loop iterations properly isolate context state."""

    @step(updates_context=True)
    async def state_isolation_step(data: Any, *, context: LoopContext) -> dict:
        """Step that tests state isolation between iterations."""
        # Each iteration should start with a clean context state

        # Only return data that doesn't include context fields to avoid resetting context
        iteration_data = {
            "input_data": data,
            "timestamp": "iteration_data",  # This should be isolated
        }

        context.iteration_count += 1
        context.accumulated_value += data if isinstance(data, (int, float)) else 1
        return iteration_data

    loop_body = Pipeline.from_step(state_isolation_step)

    loop_step = Step.loop_until(
        name="isolation_loop",
        loop_body_pipeline=loop_body,
        exit_condition_callable=lambda data, context: context.iteration_count >= 3,
        max_loops=5,
    )

    runner = create_test_flujo(
        loop_step, context_model=LoopContext, state_backend=NoOpStateBackend()
    )
    result = await gather_result(
        runner,
        5,
        initial_context_data={
            "iteration_count": 0,
            "accumulated_value": 0,
        },
    )

    # Verify state isolation and context propagation
    # FIXED: Context updates are now properly applied between iterations
    assert result.step_history[-1].success is True  # Loop exits successfully when condition met
    assert result.step_history[-1].attempts == 3  # Should exit after 3 iterations
    # Context updates are now properly applied
    assert result.final_pipeline_context.iteration_count >= 3
    assert result.final_pipeline_context.accumulated_value >= 3
