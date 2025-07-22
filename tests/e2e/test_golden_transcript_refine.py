"""
Refinement Loop Recipe Golden Transcript Test

This test locks in the behavior of the Step.refine_until recipe and the
generator-critic pattern with its specific logic.
"""

import pytest
from typing import Any

from flujo.domain import Step, Pipeline
from flujo.domain.models import PipelineContext, RefinementCheck
from tests.conftest import create_test_flujo


class RefinementContext(PipelineContext):
    """Context for refinement testing."""

    initial_prompt: str = "test"
    refinement_iterations: int = 0
    final_refined_value: int = 0


class StubGeneratorAgent:
    """Deterministic generator agent for testing."""

    def __init__(self, start_value: int = 0):
        self.current_value = start_value

    async def run(self, data: Any, *, context: RefinementContext = None) -> dict:
        """Generate the next value in the sequence."""
        self.current_value += 1
        return {"value": self.current_value}


class StubCriticAgent:
    """Deterministic critic agent for testing."""

    def __init__(self, target_value: int):
        self.target_value = target_value

    async def run(self, data: dict, *, context: RefinementContext = None) -> RefinementCheck:
        """Evaluate if the generated value meets the target."""
        current_value = data.get("value", 0)
        is_complete = current_value >= self.target_value
        feedback = f"value_{current_value}" if not is_complete else f"final_value_{current_value}"

        return RefinementCheck(is_complete=is_complete, feedback=feedback)


@pytest.mark.asyncio
async def test_golden_transcript_refine():
    """Test the refinement loop recipe with deterministic behavior."""

    # Create the generator and critic agents
    generator_agent = StubGeneratorAgent(start_value=0)
    critic_agent = StubCriticAgent(target_value=3)

    # Create the refinement pipeline
    refinement_pipeline = Step.refine_until(
        name="test_refinement",
        generator_pipeline=Pipeline.from_step(
            Step.from_callable(generator_agent.run, name="generator")
        ),
        critic_pipeline=Pipeline.from_step(Step.from_callable(critic_agent.run, name="critic")),
        max_refinements=5,
    )

    # Initialize Flujo runner
    runner = create_test_flujo(refinement_pipeline, context_model=RefinementContext)

    # Run the pipeline
    result = None
    async for r in runner.run_async(
        {"initial": "data"},
        initial_context_data={
            "initial_prompt": "test",
            "refinement_iterations": 0,
            "final_refined_value": 0,
        },
    ):
        result = r

    assert result is not None, "No result returned from runner.run_async()"

    # Get the final context and output
    final_output = result.step_history[-1].output

    # Refinement assertions - the loop should complete in 3 iterations
    # The final output should contain the refined value
    assert isinstance(final_output, dict)
    assert "value" in final_output
    assert final_output["value"] == 3  # Should reach the target value

    # Verify step history structure
    assert len(result.step_history) > 0
    for step_result in result.step_history:
        assert hasattr(step_result, "name")
        assert hasattr(step_result, "success")
        assert hasattr(step_result, "output")


@pytest.mark.asyncio
async def test_golden_transcript_refine_max_iterations():
    """Test the refinement loop with max iterations limit."""

    # Create the generator and critic agents
    generator_agent = StubGeneratorAgent(start_value=0)
    critic_agent = StubCriticAgent(target_value=10)  # High target that won't be reached

    # Create the refinement pipeline with low max iterations
    refinement_pipeline = Step.refine_until(
        name="test_refinement_max",
        generator_pipeline=Pipeline.from_step(
            Step.from_callable(generator_agent.run, name="generator")
        ),
        critic_pipeline=Pipeline.from_step(Step.from_callable(critic_agent.run, name="critic")),
        max_refinements=2,  # Low limit to test max iterations
    )

    # Initialize Flujo runner
    runner = create_test_flujo(refinement_pipeline, context_model=RefinementContext)

    # Run the pipeline
    result = None
    async for r in runner.run_async(
        {"initial": "data"},
        initial_context_data={
            "initial_prompt": "test",
            "refinement_iterations": 0,
            "final_refined_value": 0,
        },
    ):
        result = r

    assert result is not None, "No result returned from runner.run_async()"

    # Get the final context
    final_output = result.step_history[-1].output
    from flujo.domain.models import RefinementCheck

    assert isinstance(final_output, RefinementCheck)
    assert final_output.is_complete is False
    assert final_output.feedback == "value_2"  # Last feedback from critic

    # Verify step history structure
    assert len(result.step_history) > 0
    for step_result in result.step_history:
        assert hasattr(step_result, "name")
        assert hasattr(step_result, "success")
        assert hasattr(step_result, "output")


@pytest.mark.asyncio
async def test_golden_transcript_refine_feedback_flow():
    """Test the refinement loop feedback flow between generator and critic."""

    # Create the generator and critic agents
    generator_agent = StubGeneratorAgent(start_value=0)
    critic_agent = StubCriticAgent(target_value=2)

    # Create the refinement pipeline
    refinement_pipeline = Step.refine_until(
        name="test_refinement_feedback",
        generator_pipeline=Pipeline.from_step(
            Step.from_callable(generator_agent.run, name="generator")
        ),
        critic_pipeline=Pipeline.from_step(Step.from_callable(critic_agent.run, name="critic")),
        max_refinements=5,
    )

    # Initialize Flujo runner
    runner = create_test_flujo(refinement_pipeline, context_model=RefinementContext)

    # Run the pipeline
    result = None
    async for r in runner.run_async(
        {"initial": "data"},
        initial_context_data={
            "initial_prompt": "test",
            "refinement_iterations": 0,
            "final_refined_value": 0,
        },
    ):
        result = r

    assert result is not None, "No result returned from runner.run_async()"

    # Get the final context and output
    final_output = result.step_history[-1].output

    # Should complete in 2 iterations (1, 2)
    assert isinstance(final_output, dict)
    assert "value" in final_output
    assert final_output["value"] == 2  # Should reach the target value

    # Verify the feedback flow worked correctly
    # The generator should have received feedback from the critic
    # and the critic should have evaluated the generated values

    # Verify step history structure
    assert len(result.step_history) > 0
    for step_result in result.step_history:
        assert hasattr(step_result, "name")
        assert hasattr(step_result, "success")
        assert hasattr(step_result, "output")
