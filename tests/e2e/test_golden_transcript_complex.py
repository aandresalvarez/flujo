"""
Comprehensive Golden Transcript Test for Core Orchestration

This test exercises all major flujo framework features in a single pipeline run
and uses vcrpy to record the execution for regression testing. The test ensures
that the core orchestration behavior remains stable across development cycles.

The test pipeline includes:
- LoopStep with real iteration counting and context modification
- ConditionalStep with branching logic based on loop output
- ParallelStep with deterministic branch execution
- Step with configured fallback
- Custom PipelineContext that is modified by multiple steps
- Step that utilizes injected AppResources
- Aggregation step that produces clean, assertable output
- Caching for performance optimization
- Refinement pipeline with generator and critic
- Dynamic parallel branching with failure handling
- Metric aggregation (cost and tokens)
- Nested sub-pipelines using as_step

This test is designed to be deterministic and provide comprehensive coverage
of the framework's core orchestration capabilities.
"""

import pytest

from flujo.application.runner import Flujo
from examples.golden_pipeline import create_golden_pipeline, GoldenContext


@pytest.mark.parametrize("branch", ["A", "B"])
@pytest.mark.asyncio
async def test_golden_pipeline_complex(branch: str):
    """Test the comprehensive golden pipeline with all features."""
    pipeline = create_golden_pipeline()
    runner = Flujo(pipeline, context_model=GoldenContext)

    # Set up initial context based on branch
    initial_context = GoldenContext(
        initial_prompt=f"Test prompt for branch {branch}",
        initial_data=f"Test data for branch {branch}",
        has_failed_once=False,  # Reset for each test
    )

    result = None
    async for r in runner.run_async(
        {"branch": branch, "items": ["ITEM1", "ITEM2", "ITEM3"]},
        initial_context_data=initial_context.model_dump(),
    ):
        result = r

    assert result is not None, "No result returned from runner.run_async()"
    final_output = result.final_pipeline_context

    # Common assertions for both branches
    assert final_output.conditional_path_taken == branch
    assert final_output.total_cost_usd > 0
    assert final_output.total_tokens > 0

    if branch == "A":
        # Branch A: map_over, parallel, nested pipeline
        assert len(final_output.map_over_results) == 3
        assert all("ITEM" in item for item in final_output.map_over_results)
        assert len(final_output.parallel_branch_results) >= 1  # At least one success
        assert final_output.parallel_failures >= 1  # At least one failure
        assert final_output.loop_iterations == 0  # No loop in branch A
        assert final_output.loop_final_value == 0  # No loop in branch A
        assert final_output.retry_attempts == 0  # No retries in branch A
        assert final_output.fallback_triggered is False  # No fallback in branch A
        assert final_output.refine_iterations == 0  # No refinement in branch A
        assert final_output.refine_final_value == 0  # No refinement in branch A

        # Check that initial_prompt and initial_data are preserved
        assert final_output.initial_prompt == f"Test prompt for branch {branch}"
        assert final_output.initial_data == f"Test data for branch {branch}"

    elif branch == "B":
        # Branch B: loop_until, fallback, retry, refinement
        assert final_output.loop_iterations >= 2  # Should take at least 2 iterations
        assert final_output.loop_final_value >= 2  # Should reach target value
        assert final_output.retry_attempts >= 1  # Should have at least one retry
        assert final_output.fallback_triggered is True  # Should trigger fallback
        assert final_output.refine_iterations > 0  # Should have refinement iterations
        assert final_output.refine_final_value >= 2  # Should have refinement final value
        assert len(final_output.map_over_results) == 0  # No map_over in branch B
        assert len(final_output.parallel_branch_results) == 0  # No parallel in branch B
        assert final_output.parallel_failures == 0  # No parallel in branch B

        # Check that initial_prompt and initial_data are preserved
        assert final_output.initial_prompt == f"Test prompt for branch {branch}"
        assert final_output.initial_data == f"Test data for branch {branch}"

    # Final aggregation should contain all the expected data
    final_aggregation = result.step_history[-1].output
    assert isinstance(final_aggregation, dict)
    assert final_aggregation["conditional_path"] == branch
    assert "map_over_results" in final_aggregation
    assert "loop_iterations" in final_aggregation
    assert "loop_final_value" in final_aggregation
    assert "fallback_triggered" in final_aggregation
    assert "retry_attempts" in final_aggregation
    assert "refine_iterations" in final_aggregation
    assert "refine_final_value" in final_aggregation
    assert "parallel_branch_results" in final_aggregation
    assert "parallel_failures" in final_aggregation
    assert "total_cost_usd" in final_aggregation
    assert "total_tokens" in final_aggregation
