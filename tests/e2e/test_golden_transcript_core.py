"""
Core Orchestration Golden Transcript Test

This test locks in the behavior of the fundamental, low-level control flow primitives
and their interactions with context, resources, and resilience features.
"""

import pytest

from flujo.application.runner import Flujo
from examples.golden_pipeline import create_golden_pipeline, GoldenContext


@pytest.mark.asyncio
async def test_golden_transcript_core():
    """Test the core orchestration primitives with deterministic behavior."""

    # Create the golden pipeline
    pipeline = create_golden_pipeline()

    # Test data for both branches
    test_data = {"branch": "A", "items": ["ITEM1", "ITEM2", "ITEM3"]}

    # Initialize Flujo runner
    runner = Flujo(pipeline, context_model=GoldenContext)

    # Run the pipeline
    result = None
    async for r in runner.run_async(
        test_data,
        initial_context_data={
            "initial_prompt": "Test prompt for core orchestration",
            "initial_data": "Test data for core orchestration",
        },
    ):
        result = r

    assert result is not None, "No result returned from runner.run_async()"

    # Get the final output
    final_output = result.step_history[-1].output

    # Core orchestration assertions
    assert isinstance(final_output, dict)
    assert final_output["conditional_path"] == "A"

    # Map over assertions
    assert len(final_output["map_over_results"]) == 3
    assert all(isinstance(item, str) for item in final_output["map_over_results"])
    assert all(item.isupper() for item in final_output["map_over_results"])

    # Parallel branch assertions
    assert len(final_output["parallel_branch_results"]) >= 1
    assert final_output["parallel_failures"] >= 0

    # Metric tracking assertions
    assert final_output["total_cost_usd"] > 0
    assert final_output["total_tokens"] > 0

    # Context state assertions
    assert final_output["initial_prompt"] == "Test prompt for core orchestration"
    assert final_output["initial_data"] == "Test data for core orchestration"

    # Verify step history structure
    assert len(result.step_history) > 0
    for step_result in result.step_history:
        assert hasattr(step_result, "name")
        assert hasattr(step_result, "success")
        assert hasattr(step_result, "output")


@pytest.mark.asyncio
async def test_golden_transcript_core_branch_b():
    """Test the core orchestration primitives with branch B (loop and refinement)."""

    # Create the golden pipeline
    pipeline = create_golden_pipeline()

    # Test data for branch B
    test_data = {"branch": "B", "items": ["ITEM1", "ITEM2", "ITEM3"]}

    # Initialize Flujo runner
    runner = Flujo(pipeline, context_model=GoldenContext)

    # Run the pipeline
    result = None
    async for r in runner.run_async(
        test_data,
        initial_context_data={
            "initial_prompt": "Test prompt for branch B",
            "initial_data": "Test data for branch B",
        },
    ):
        result = r

    assert result is not None, "No result returned from runner.run_async()"

    # Get the final output
    final_output = result.step_history[-1].output

    # Core orchestration assertions for branch B
    assert isinstance(final_output, dict)
    assert final_output["conditional_path"] == "B"

    # Loop step assertions
    assert final_output["loop_iterations"] >= 2
    assert final_output["loop_final_value"] >= 2
    assert final_output["fallback_triggered"] is True
    assert final_output["retry_attempts"] >= 1

    # Refinement step assertions
    assert final_output["refine_iterations"] >= 1
    assert final_output["refine_final_value"] >= 1

    # Metric tracking assertions
    assert final_output["total_cost_usd"] > 0
    assert final_output["total_tokens"] > 0

    # Context state assertions
    assert final_output["initial_prompt"] == "Test prompt for branch B"
    assert final_output["initial_data"] == "Test data for branch B"

    # Verify step history structure
    assert len(result.step_history) > 0
    for step_result in result.step_history:
        assert hasattr(step_result, "name")
        assert hasattr(step_result, "success")
        assert hasattr(step_result, "output")
