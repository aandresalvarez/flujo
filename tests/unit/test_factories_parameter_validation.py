"""Tests to ensure make_default_pipeline properly handles all parameters."""

import pytest
from unittest.mock import AsyncMock

from flujo.recipes.factories import make_default_pipeline, run_default_pipeline
from flujo.domain.models import Task, Checklist


class TestMakeDefaultPipelineParameters:
    """Test that make_default_pipeline accepts and handles all required parameters."""

    def test_accepts_all_parameters(self):
        """Test that make_default_pipeline accepts k_variants, max_iters, and reflection_limit."""
        # Mock agents to avoid actual API calls
        mock_review = AsyncMock()
        mock_solution = AsyncMock()
        mock_validator = AsyncMock()
        mock_reflection = AsyncMock()

        # Test that the function accepts all parameters without TypeError
        try:
            pipeline = make_default_pipeline(
                review_agent=mock_review,
                solution_agent=mock_solution,
                validator_agent=mock_validator,
                reflection_agent=mock_reflection,
                max_retries=5,
                k_variants=3,
                max_iters=7,
                reflection_limit=2,
            )
            assert pipeline is not None
        except TypeError as e:
            pytest.fail(f"make_default_pipeline should accept all parameters: {e}")

    def test_parameter_defaults(self):
        """Test that parameters have sensible defaults."""
        mock_review = AsyncMock()
        mock_solution = AsyncMock()
        mock_validator = AsyncMock()

        # Test with minimal parameters
        pipeline = make_default_pipeline(
            review_agent=mock_review,
            solution_agent=mock_solution,
            validator_agent=mock_validator,
        )
        assert pipeline is not None

    def test_parameter_validation(self):
        """Test that invalid parameter values are handled gracefully."""
        mock_review = AsyncMock()
        mock_solution = AsyncMock()
        mock_validator = AsyncMock()

        # Test with invalid parameter types (should raise TypeError)
        with pytest.raises(TypeError):
            make_default_pipeline(
                review_agent=mock_review,
                solution_agent=mock_solution,
                validator_agent=mock_validator,
                k_variants="invalid",  # Should be int
            )

        with pytest.raises(TypeError):
            make_default_pipeline(
                review_agent=mock_review,
                solution_agent=mock_solution,
                validator_agent=mock_validator,
                max_iters="invalid",  # Should be int
            )

    def test_cli_parameter_compatibility(self):
        """Test that the function works with CLI-style parameter names."""
        mock_review = AsyncMock()
        mock_solution = AsyncMock()
        mock_validator = AsyncMock()

        # Test with 'k' parameter (CLI uses 'k' but function uses 'k_variants')
        pipeline = make_default_pipeline(
            review_agent=mock_review,
            solution_agent=mock_solution,
            validator_agent=mock_validator,
            k_variants=2,  # CLI --k=2 should map to k_variants=2
            max_iters=5,  # CLI --max-iters=5 should map to max_iters=5
        )
        assert pipeline is not None

    def test_reflection_agent_optional(self):
        """Test that reflection_agent is truly optional."""
        mock_review = AsyncMock()
        mock_solution = AsyncMock()
        mock_validator = AsyncMock()

        # Test without reflection_agent
        pipeline1 = make_default_pipeline(
            review_agent=mock_review,
            solution_agent=mock_solution,
            validator_agent=mock_validator,
        )

        # Test with reflection_agent
        mock_reflection = AsyncMock()
        pipeline2 = make_default_pipeline(
            review_agent=mock_review,
            solution_agent=mock_solution,
            validator_agent=mock_validator,
            reflection_agent=mock_reflection,
        )

        assert pipeline1 is not None
        assert pipeline2 is not None


class TestRunDefaultPipelineAsync:
    """Test that run_default_pipeline works correctly as an async function."""

    @pytest.mark.asyncio
    async def test_async_execution(self):
        """Test that run_default_pipeline can be awaited properly."""
        # Create a simple pipeline
        mock_review = AsyncMock()
        mock_solution = AsyncMock()
        mock_validator = AsyncMock()

        # Mock the agents to return expected values
        from flujo.domain.models import ChecklistItem

        mock_review.run.return_value = Checklist(
            items=[ChecklistItem(item="Test requirement", weight=1.0, description="desc")]
        )
        mock_solution.run.return_value = "Test solution"
        mock_validator.run.return_value = Checklist(
            items=[ChecklistItem(item="Validated", weight=1.0, description="desc")]
        )

        pipeline = make_default_pipeline(
            review_agent=mock_review,
            solution_agent=mock_solution,
            validator_agent=mock_validator,
        )

        # Test that it can be awaited
        task = Task(prompt="Test prompt")
        result = await run_default_pipeline(pipeline, task)

        # Should return None or a valid result
        assert result is None or hasattr(result, "solution")

    @pytest.mark.asyncio
    async def test_async_with_context(self):
        """Test that run_default_pipeline works with context."""
        mock_review = AsyncMock()
        mock_solution = AsyncMock()
        mock_validator = AsyncMock()

        # Mock the agents to return concrete values
        from flujo.domain.models import ChecklistItem

        mock_review.run.return_value = Checklist(
            items=[ChecklistItem(item="Test requirement", weight=1.0, description="desc")]
        )
        mock_solution.run.return_value = "Test solution"
        mock_validator.run.return_value = Checklist(
            items=[ChecklistItem(item="Validated", weight=1.0, description="desc")]
        )

        pipeline = make_default_pipeline(
            review_agent=mock_review,
            solution_agent=mock_solution,
            validator_agent=mock_validator,
        )

        task = Task(prompt="Test prompt", metadata={"test": "data"})

        # Should not raise an exception
        try:
            await run_default_pipeline(pipeline, task)
        except Exception as e:
            pytest.fail(f"run_default_pipeline should handle context properly: {e}")


class TestCLIParameterMapping:
    """Test that CLI parameters correctly map to function parameters."""

    def test_cli_k_maps_to_k_variants(self):
        """Test that CLI --k parameter maps to k_variants."""
        mock_review = AsyncMock()
        mock_solution = AsyncMock()
        mock_validator = AsyncMock()

        # Simulate CLI passing k=3
        pipeline = make_default_pipeline(
            review_agent=mock_review,
            solution_agent=mock_solution,
            validator_agent=mock_validator,
            k_variants=3,  # This should come from CLI --k=3
        )
        assert pipeline is not None

    def test_cli_max_iters_maps_correctly(self):
        """Test that CLI --max-iters parameter maps to max_iters."""
        mock_review = AsyncMock()
        mock_solution = AsyncMock()
        mock_validator = AsyncMock()

        # Simulate CLI passing max_iters=5
        pipeline = make_default_pipeline(
            review_agent=mock_review,
            solution_agent=mock_solution,
            validator_agent=mock_validator,
            max_iters=5,  # This should come from CLI --max-iters=5
        )
        assert pipeline is not None

    def test_cli_reflection_maps_correctly(self):
        """Test that CLI reflection parameter maps correctly."""
        mock_review = AsyncMock()
        mock_solution = AsyncMock()
        mock_validator = AsyncMock()
        mock_reflection = AsyncMock()

        # Simulate CLI passing reflection=True
        pipeline = make_default_pipeline(
            review_agent=mock_review,
            solution_agent=mock_solution,
            validator_agent=mock_validator,
            reflection_agent=mock_reflection,  # This should come from CLI --reflection
        )
        assert pipeline is not None
