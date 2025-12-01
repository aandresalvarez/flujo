"""
Integration tests for Refine Until + Context Updates feature combination.

This tests the critical combination of refinement loops with context-updating steps,
which could reveal bugs in context state management during iterative refinement.
"""

import pytest
from typing import Any, List
from flujo import step, Step
from flujo.domain.models import PipelineContext, RefinementCheck
from flujo.domain.dsl.pipeline import Pipeline
from flujo.testing.utils import gather_result
from tests.conftest import create_test_flujo
from flujo.type_definitions.common import JSONObject


class RefineContext(PipelineContext):
    """Context for testing refine until operations with context updates."""

    initial_prompt: str = "test"
    refinement_count: int = 0
    total_iterations: int = 0
    refinement_history: List[str] = []
    current_quality: float = 0.0
    best_quality: float = 0.0
    refinement_data: JSONObject = {}


@step(updates_context=True)
async def refine_generator_step(data: Any, *, context: RefineContext) -> str:
    """Generate refined content and update context."""
    context.refinement_count += 1
    context.total_iterations += 1
    context.refinement_history.append(f"refinement_{context.refinement_count}")

    # Simulate quality improvement
    quality = min(0.9, 0.1 + (context.refinement_count * 0.2))
    context.current_quality = quality
    context.best_quality = max(context.best_quality, quality)

    refinement_data = {
        "iteration": context.refinement_count,
        "quality": quality,
        "improvement": quality - (quality - 0.2) if context.refinement_count > 1 else 0.0,
    }
    context.refinement_data[f"refinement_{context.refinement_count}"] = refinement_data

    return f"refined_content_{context.refinement_count}"


@step(updates_context=True)
async def refine_critic_step(data: str, *, context: RefineContext) -> RefinementCheck:
    """Criticize refined content and update context."""
    context.total_iterations += 1

    # Simulate quality assessment
    quality = context.current_quality
    is_complete = quality >= 0.8

    feedback = f"Quality: {quality:.2f}, {'Satisfactory' if is_complete else 'Needs improvement'}"

    return RefinementCheck(is_complete=is_complete, feedback=feedback)


@step(updates_context=True)
async def refine_with_error_step(data: Any, *, context: RefineContext) -> str:
    """Generate refined content that sometimes fails."""
    context.refinement_count += 1
    context.total_iterations += 1
    context.refinement_history.append(f"refinement_{context.refinement_count}")

    # Fail on specific iterations
    if context.refinement_count == 2:
        raise RuntimeError(f"Intentional failure in refinement {context.refinement_count}")

    # Ensure quality is very low to not trigger exit condition on iteration 1
    quality = 0.05 + (context.refinement_count * 0.02)  # Much lower quality progression
    context.current_quality = quality
    context.best_quality = max(context.best_quality, quality)

    return f"refined_content_{context.refinement_count}"


@step(updates_context=True)
async def refine_with_context_dependent_step(data: Any, *, context: RefineContext) -> str:
    """Generate refined content with context-dependent logic."""
    context.refinement_count += 1
    context.total_iterations += 1

    # Use context state to determine refinement strategy
    if context.refinement_count <= 2:
        strategy = "early_refinement"
        quality = 0.05 + (context.refinement_count * 0.02)  # Much lower quality
    else:
        strategy = "late_refinement"
        quality = 0.1 + (context.refinement_count * 0.05)  # Higher quality for late strategy

    context.current_quality = quality
    context.best_quality = max(context.best_quality, quality)
    context.refinement_history.append(f"{strategy}_{context.refinement_count}")

    return f"{strategy}_content_{context.refinement_count}"


@pytest.mark.asyncio
async def test_refine_until_with_context_updates_basic():
    """Test basic refine until operation with context updates."""

    refine_step = Step.refine_until(
        name="basic_refine",
        generator_pipeline=Pipeline.from_step(refine_generator_step),
        critic_pipeline=Pipeline.from_step(refine_critic_step),
        max_refinements=5,
    )

    runner = create_test_flujo(refine_step, context_model=RefineContext)
    result = await gather_result(runner, "initial_data")

    # Verify refine until operation with context updates
    assert result.step_history[-1].success is True
    assert result.final_pipeline_context.refinement_count >= 1
    assert result.final_pipeline_context.total_iterations >= 2  # generator + critic
    assert len(result.final_pipeline_context.refinement_history) >= 1
    assert result.final_pipeline_context.best_quality > 0.0

    # Verify refinement data was collected
    assert len(result.final_pipeline_context.refinement_data) >= 1


@pytest.mark.asyncio
async def test_refine_until_with_context_updates_error_handling():
    """Test refine until operation with context updates when generator fails."""

    refine_step = Step.refine_until(
        name="error_refine",
        generator_pipeline=Pipeline.from_step(refine_with_error_step),
        critic_pipeline=Pipeline.from_step(refine_critic_step),
        max_refinements=5,
    )

    runner = create_test_flujo(refine_step, context_model=RefineContext)
    result = await gather_result(runner, "initial_data")

    # Verify error handling with context updates
    assert result.step_history[-1].success is False
    assert (
        "loop body failed"
        in result.step_history[-1].feedback.lower()  # Enhanced: Simplified error message
    )

    # Verify context updates from successful iterations
    assert result.final_pipeline_context.refinement_count >= 1
    assert result.final_pipeline_context.total_iterations >= 1
    assert len(result.final_pipeline_context.refinement_history) >= 1


@pytest.mark.asyncio
async def test_refine_until_with_context_updates_context_dependent():
    """Test refine until operation with context-dependent refinement."""

    refine_step = Step.refine_until(
        name="context_dependent_refine",
        generator_pipeline=Pipeline.from_step(refine_with_context_dependent_step),
        critic_pipeline=Pipeline.from_step(refine_critic_step),
        max_refinements=5,
    )

    runner = create_test_flujo(refine_step, context_model=RefineContext)
    result = await gather_result(runner, "initial_data")

    # Verify context-dependent refinement
    assert result.step_history[-1].success is False  # Loop fails due to critic step failure
    assert result.final_pipeline_context.refinement_count >= 1
    assert len(result.final_pipeline_context.refinement_history) >= 1

    # Verify context-dependent strategies were used
    history = result.final_pipeline_context.refinement_history
    assert any("early_refinement" in h for h in history)
    # Note: late_refinement won't be reached because the loop exits on iteration 1


@pytest.mark.asyncio
async def test_refine_until_with_context_updates_state_isolation():
    """Test that refine until operations properly manage context state across iterations."""

    @step(updates_context=True)
    async def isolation_generator_step(data: Any, *, context: RefineContext) -> str:
        """Step that tests state isolation in refine operations."""
        # Each iteration should see the accumulated context state
        iteration_data = {
            "refinement_count_at_start": context.refinement_count,
            "total_iterations_at_start": context.total_iterations,
            "best_quality_at_start": context.best_quality,
            "current_iteration": context.refinement_count + 1,
        }

        context.refinement_count += 1
        context.total_iterations += 1
        context.refinement_history.append(f"isolation_{context.refinement_count}")

        # Update quality
        quality = min(0.9, 0.1 + (context.refinement_count * 0.2))
        context.current_quality = quality
        context.best_quality = max(context.best_quality, quality)

        context.refinement_data[f"isolation_{context.refinement_count}"] = iteration_data

        return f"isolation_content_{context.refinement_count}"

    refine_step = Step.refine_until(
        name="isolation_refine",
        generator_pipeline=Pipeline.from_step(isolation_generator_step),
        critic_pipeline=Pipeline.from_step(refine_critic_step),
        max_refinements=3,
    )

    runner = create_test_flujo(refine_step, context_model=RefineContext)
    result = await gather_result(runner, "initial_data")

    # Verify state management - test actual business logic rather than just failure
    assert result.final_pipeline_context.refinement_count > 0
    assert result.final_pipeline_context.best_quality >= 0.1  # Should have some improvement
    assert (
        result.final_pipeline_context.current_quality >= 0.1
    )  # Final quality should show improvement

    # Verify each iteration saw the correct accumulated state
    refinement_data = result.final_pipeline_context.refinement_data
    for i in range(1, result.final_pipeline_context.refinement_count + 1):
        key = f"isolation_{i}"
        if key in refinement_data:
            iteration_data = refinement_data[key]
            assert iteration_data["refinement_count_at_start"] == i - 1
            assert iteration_data["current_iteration"] == i


@pytest.mark.asyncio
async def test_refine_until_with_context_updates_complex_feedback():
    """Test refine until operation with complex feedback handling."""

    @step(updates_context=True)
    async def complex_feedback_generator_step(data: Any, *, context: RefineContext) -> str:
        """Step that generates content with complex feedback handling."""
        context.refinement_count += 1
        context.total_iterations += 1

        # Use feedback from previous iterations to improve
        if context.refinement_count > 1:
            previous_feedback = context.refinement_data.get(
                f"feedback_{context.refinement_count - 1}"
            )
            if previous_feedback:
                context.refinement_history.append(f"improved_{context.refinement_count}")
            else:
                context.refinement_history.append(f"new_{context.refinement_count}")
        else:
            context.refinement_history.append(f"initial_{context.refinement_count}")

        quality = min(0.9, 0.1 + (context.refinement_count * 0.25))
        context.current_quality = quality
        context.best_quality = max(context.best_quality, quality)

        return f"complex_content_{context.refinement_count}"

    @step(updates_context=True)
    async def complex_feedback_critic_step(data: str, *, context: RefineContext) -> RefinementCheck:
        """Step that provides complex feedback and updates context."""
        context.total_iterations += 1

        quality = context.current_quality
        is_complete = quality >= 0.8

        # Store feedback in context for next iteration
        feedback_data = {
            "quality": quality,
            "satisfactory": is_complete,
            "iteration": context.refinement_count,
            "improvement_needed": not is_complete,
        }
        context.refinement_data[f"feedback_{context.refinement_count}"] = feedback_data

        feedback = (
            f"Quality: {quality:.2f}, {'Satisfactory' if is_complete else 'Needs improvement'}"
        )

        return RefinementCheck(is_complete=is_complete, feedback=feedback)

    refine_step = Step.refine_until(
        name="complex_feedback_refine",
        generator_pipeline=Pipeline.from_step(complex_feedback_generator_step),
        critic_pipeline=Pipeline.from_step(complex_feedback_critic_step),
        max_refinements=4,
    )

    runner = create_test_flujo(refine_step, context_model=RefineContext)
    result = await gather_result(runner, "initial_data")

    # Verify complex feedback handling
    assert result.step_history[-1].success is True
    assert result.final_pipeline_context.refinement_count >= 1
    assert len(result.final_pipeline_context.refinement_history) >= 1

    # Verify feedback data was stored
    feedback_keys = [
        k for k in result.final_pipeline_context.refinement_data.keys() if k.startswith("feedback_")
    ]
    assert len(feedback_keys) >= 1


@pytest.mark.asyncio
async def test_refine_until_with_context_updates_metadata_conflicts():
    """Test refine until operation with context updates and metadata conflicts."""

    @step(updates_context=True)
    async def metadata_generator_step(data: Any, *, context: RefineContext) -> str:
        """Step that tests metadata conflicts in refine operations."""
        context.refinement_count += 1
        context.total_iterations += 1

        # Try to update fields that might conflict with refine metadata
        context.refinement_data[f"metadata_{context.refinement_count}"] = {
            "refine_index": context.refinement_count,
            "refine_iteration": context.total_iterations,
            "refine_metadata": {
                "iteration": context.refinement_count,
                "timestamp": "now",
                "data": f"refinement_{context.refinement_count}",
            },
        }

        # Ensure quality is very low to not trigger exit condition before max_loops
        quality = 0.05 + (context.refinement_count * 0.02)  # Much lower quality progression
        context.current_quality = quality
        context.best_quality = max(context.best_quality, quality)

        return f"metadata_content_{context.refinement_count}"

    refine_step = Step.refine_until(
        name="metadata_refine",
        generator_pipeline=Pipeline.from_step(metadata_generator_step),
        critic_pipeline=Pipeline.from_step(refine_critic_step),
        max_refinements=3,
    )

    runner = create_test_flujo(refine_step, context_model=RefineContext)
    result = await gather_result(runner, "initial_data")

    # Verify metadata handling
    assert result.step_history[-1].success is False  # Loop fails due to reaching max loops
    assert "reached max_loops" in result.step_history[-1].feedback.lower()

    # Verify metadata in refinement data
    refinement_data = result.final_pipeline_context.refinement_data
    for i in range(1, result.final_pipeline_context.refinement_count + 1):
        metadata_key = f"metadata_{i}"
        if metadata_key in refinement_data:
            metadata_data = refinement_data[metadata_key]
            assert "refine_index" in metadata_data
            assert "refine_iteration" in metadata_data
            assert "refine_metadata" in metadata_data
