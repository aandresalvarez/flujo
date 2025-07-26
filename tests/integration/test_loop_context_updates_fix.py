"""
Integration tests for the LoopStep context update bug fix.

This test suite verifies that @step(updates_context=True) works correctly
in loop iterations, ensuring context updates are properly applied between iterations.
"""

import pytest
from typing import Any

from flujo import Step, Pipeline, step, Flujo
from flujo.domain.models import PipelineContext


class LoopTestContext(PipelineContext):
    """Test context for loop context update verification."""

    iteration_count: int = 0
    is_clear: bool = False
    current_value: str = ""
    accumulated_data: dict[str, Any] = {}


@step(updates_context=True)
async def context_updating_step(data: str, *, context: LoopTestContext) -> dict[str, Any]:
    """
    A step that updates the context and returns updates.

    This simulates the behavior described in the bug report where
    context updates from @step(updates_context=True) were not being
    applied between loop iterations.
    """
    # Update context state
    context.iteration_count += 1
    context.current_value = data

    # Simulate the bug scenario: agent determines if definition is clear
    if "clear" in data.lower() or context.iteration_count >= 3:
        context.is_clear = True
        return {"is_clear": True, "current_value": data, "iteration_count": context.iteration_count}
    else:
        # Simulate clarification needed
        context.is_clear = False
        context.accumulated_data[f"iteration_{context.iteration_count}"] = data
        return {
            "is_clear": False,
            "current_value": f"{data} (clarified)",
            "iteration_count": context.iteration_count,
            "accumulated_data": context.accumulated_data,
        }


def exit_condition(output: dict[str, Any], context: LoopTestContext) -> bool:
    """Exit condition that checks the context state."""
    print(f"  [Loop Check] Context 'is_clear' flag is: {context.is_clear}")
    print(f"  [Loop Check] Context 'iteration_count' is: {context.iteration_count}")
    return context.is_clear


def input_mapper(initial_input: str, context: LoopTestContext) -> str:
    """Map initial input to loop body input."""
    return context.current_value if context.current_value else initial_input


def iteration_mapper(last_output: dict[str, Any], context: LoopTestContext, iteration: int) -> str:
    """Map output from previous iteration to next iteration input."""
    if isinstance(last_output, dict) and "current_value" in last_output:
        return last_output["current_value"]
    return context.current_value


@pytest.mark.asyncio
async def test_loop_context_updates_basic():
    """Test basic loop execution with context updates."""

    # Create loop body with context-updating step
    loop_body = Pipeline.from_step(context_updating_step)

    loop_step = Step.loop_until(
        name="test_context_updates",
        loop_body_pipeline=loop_body,
        exit_condition_callable=exit_condition,
        max_loops=5,
        initial_input_to_loop_body_mapper=input_mapper,
        iteration_input_mapper=iteration_mapper,
    )

    # Create runner with context model
    runner = Flujo(
        loop_step, pipeline_name="test_loop_context_updates", context_model=LoopTestContext
    )

    # Test with initial data that should become clear
    initial_context_data = {
        "initial_prompt": "test",
        "current_value": "clear definition",
        "is_clear": False,
        "iteration_count": 0,
    }

    result = None
    async for item in runner.run_async(
        "clear definition", initial_context_data=initial_context_data
    ):
        result = item

    assert result is not None
    assert result.final_pipeline_context is not None

    # Verify context updates were applied correctly
    final_context = result.final_pipeline_context
    assert final_context.is_clear is True
    assert final_context.iteration_count >= 1
    assert final_context.current_value == "clear definition"

    # Verify loop exited successfully
    assert result.step_history[-1].success is True


@pytest.mark.asyncio
async def test_loop_context_updates_multiple_iterations():
    """Test loop execution with multiple iterations requiring context updates."""

    # Create loop body with context-updating step
    loop_body = Pipeline.from_step(context_updating_step)

    loop_step = Step.loop_until(
        name="test_multiple_iterations",
        loop_body_pipeline=loop_body,
        exit_condition_callable=exit_condition,
        max_loops=5,
        initial_input_to_loop_body_mapper=input_mapper,
        iteration_input_mapper=iteration_mapper,
    )

    # Create runner with context model
    runner = Flujo(
        loop_step, pipeline_name="test_multiple_iterations", context_model=LoopTestContext
    )

    # Test with initial data that requires multiple clarifications
    initial_context_data = {
        "initial_prompt": "test",
        "current_value": "ambiguous definition",
        "is_clear": False,
        "iteration_count": 0,
    }

    result = None
    async for item in runner.run_async(
        "ambiguous definition", initial_context_data=initial_context_data
    ):
        result = item

    assert result is not None
    assert result.final_pipeline_context is not None

    # Verify context updates were applied correctly across iterations
    final_context = result.final_pipeline_context
    assert final_context.is_clear is True  # Should become clear after 3 iterations
    assert final_context.iteration_count >= 3
    assert len(final_context.accumulated_data) >= 2  # Should have accumulated data from iterations

    # Verify loop exited successfully
    assert result.step_history[-1].success is True


@pytest.mark.asyncio
async def test_loop_context_updates_max_loops():
    """Test loop execution that hits max_loops without becoming clear."""

    # Create loop body with context-updating step
    loop_body = Pipeline.from_step(context_updating_step)

    loop_step = Step.loop_until(
        name="test_max_loops",
        loop_body_pipeline=loop_body,
        exit_condition_callable=exit_condition,
        max_loops=2,  # Set low max_loops to test failure case
        initial_input_to_loop_body_mapper=input_mapper,
        iteration_input_mapper=iteration_mapper,
    )

    # Create runner with context model
    runner = Flujo(loop_step, pipeline_name="test_max_loops", context_model=LoopTestContext)

    # Test with initial data that won't become clear in 2 iterations
    initial_context_data = {
        "initial_prompt": "test",
        "current_value": "very ambiguous definition",
        "is_clear": False,
        "iteration_count": 0,
    }

    result = None
    async for item in runner.run_async(
        "very ambiguous definition", initial_context_data=initial_context_data
    ):
        result = item

    assert result is not None
    assert result.final_pipeline_context is not None

    # Verify context updates were still applied correctly
    final_context = result.final_pipeline_context
    assert final_context.iteration_count == 2  # Should have done 2 iterations
    assert final_context.is_clear is False  # Should not be clear yet
    assert len(final_context.accumulated_data) == 2  # Should have data from both iterations

    # Verify loop failed due to max_loops
    assert result.step_history[-1].success is False
    assert "max_loops" in result.step_history[-1].feedback.lower()


@pytest.mark.asyncio
async def test_loop_context_updates_complex_state():
    """Test loop execution with complex state management."""

    @step(updates_context=True)
    async def complex_state_step(data: str, *, context: LoopTestContext) -> dict[str, Any]:
        """Step that manages complex state across iterations."""
        context.iteration_count += 1

        # Simulate complex state management
        if context.iteration_count == 1:
            context.current_value = f"{data} - first clarification"
            context.accumulated_data["stage"] = "initial"
            return {
                "current_value": context.current_value,
                "accumulated_data": context.accumulated_data,
                "is_clear": False,
            }
        elif context.iteration_count == 2:
            context.current_value = f"{data} - second clarification"
            context.accumulated_data["stage"] = "refined"
            context.accumulated_data["clarifications"] = ["first", "second"]
            return {
                "current_value": context.current_value,
                "accumulated_data": context.accumulated_data,
                "is_clear": False,
            }
        else:
            # Third iteration should make it clear
            context.current_value = f"{data} - final version"
            context.accumulated_data["stage"] = "final"
            context.accumulated_data["clarifications"].append("final")
            context.is_clear = True
            return {
                "current_value": context.current_value,
                "accumulated_data": context.accumulated_data,
                "is_clear": True,
            }

    # Create loop body with complex state step
    loop_body = Pipeline.from_step(complex_state_step)

    loop_step = Step.loop_until(
        name="test_complex_state",
        loop_body_pipeline=loop_body,
        exit_condition_callable=exit_condition,
        max_loops=5,
        initial_input_to_loop_body_mapper=input_mapper,
        iteration_input_mapper=iteration_mapper,
    )

    # Create runner with context model
    runner = Flujo(loop_step, pipeline_name="test_complex_state", context_model=LoopTestContext)

    # Test with initial data
    initial_context_data = {
        "initial_prompt": "test",
        "current_value": "complex definition",
        "is_clear": False,
        "iteration_count": 0,
        "accumulated_data": {},
    }

    result = None
    async for item in runner.run_async(
        "complex definition", initial_context_data=initial_context_data
    ):
        result = item

    assert result is not None
    assert result.final_pipeline_context is not None

    # Verify complex state was managed correctly
    final_context = result.final_pipeline_context
    assert final_context.is_clear is True
    assert final_context.iteration_count == 3
    assert final_context.accumulated_data["stage"] == "final"
    assert len(final_context.accumulated_data["clarifications"]) == 3
    assert "final version" in final_context.current_value

    # Verify loop exited successfully
    assert result.step_history[-1].success is True


@pytest.mark.asyncio
async def test_loop_context_updates_error_handling():
    """Test loop execution with error handling for context updates."""

    @step(updates_context=True)
    async def error_prone_step(data: str, *, context: LoopTestContext) -> dict[str, Any]:
        """Step that might cause errors during context updates."""
        context.iteration_count += 1

        # Simulate an error condition
        if context.iteration_count == 2:
            # This should not break the loop, just log an error
            context.current_value = f"{data} - error occurred"
            return {"current_value": context.current_value, "is_clear": False}

        # Normal operation
        context.current_value = f"{data} - iteration {context.iteration_count}"
        if context.iteration_count >= 3:
            context.is_clear = True
            return {"current_value": context.current_value, "is_clear": True}
        else:
            return {"current_value": context.current_value, "is_clear": False}

    # Create loop body with error-prone step
    loop_body = Pipeline.from_step(error_prone_step)

    loop_step = Step.loop_until(
        name="test_error_handling",
        loop_body_pipeline=loop_body,
        exit_condition_callable=exit_condition,
        max_loops=5,
        initial_input_to_loop_body_mapper=input_mapper,
        iteration_input_mapper=iteration_mapper,
    )

    # Create runner with context model
    runner = Flujo(loop_step, pipeline_name="test_error_handling", context_model=LoopTestContext)

    # Test with initial data
    initial_context_data = {
        "initial_prompt": "test",
        "current_value": "test definition",
        "is_clear": False,
        "iteration_count": 0,
    }

    result = None
    async for item in runner.run_async(
        "test definition", initial_context_data=initial_context_data
    ):
        result = item

    assert result is not None
    assert result.final_pipeline_context is not None

    # Verify loop completed despite potential errors
    final_context = result.final_pipeline_context
    assert final_context.is_clear is True
    assert final_context.iteration_count >= 3

    # Verify loop exited successfully
    assert result.step_history[-1].success is True
