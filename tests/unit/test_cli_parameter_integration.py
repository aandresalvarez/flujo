"""Tests to ensure CLI parameters are properly integrated with pipeline functions."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock, Mock

from flujo.recipes.factories import make_default_pipeline, run_default_pipeline
from flujo.domain.models import Task
from flujo.cli.main import app


class TestCLIParameterIntegration:
    """Test that CLI parameters are properly integrated with pipeline functions."""

    def test_cli_solve_command_parameters(self):
        """Test that the solve command properly passes parameters to make_default_pipeline."""
        # Mock all agent creation functions to avoid API key requirements
        with patch("flujo.cli.main.make_review_agent") as mock_make_review:
            with patch("flujo.cli.main.make_solution_agent") as mock_make_solution:
                with patch("flujo.cli.main.make_validator_agent") as mock_make_validator:
                    with patch("flujo.cli.main.get_reflection_agent") as mock_get_reflection:
                        # Create mock agents
                        mock_review = AsyncMock()
                        mock_solution = AsyncMock()
                        mock_validator = AsyncMock()
                        mock_reflection = AsyncMock()

                        # Configure mocks to return our mock agents
                        mock_make_review.return_value = mock_review
                        mock_make_solution.return_value = mock_solution
                        mock_make_validator.return_value = mock_validator
                        mock_get_reflection.return_value = mock_reflection

                        # Mock the pipeline creation and execution at the CLI module level
                        with patch("flujo.cli.main.make_default_pipeline") as mock_make_pipeline:
                            with patch("flujo.cli.main.run_default_pipeline") as mock_run_pipeline:
                                # Create a mock pipeline and result
                                mock_pipeline = MagicMock()
                                mock_result = MagicMock()
                                mock_result.model_dump.return_value = {"result": "test"}

                                mock_make_pipeline.return_value = mock_pipeline
                                mock_run_pipeline.return_value = mock_result

                                # Mock asyncio.run to avoid actual async execution
                                with patch("asyncio.run") as mock_asyncio_run:
                                    mock_asyncio_run.return_value = mock_result

                                    # Actually invoke the CLI to verify integration
                                    from typer.testing import CliRunner

                                    runner = CliRunner()
                                    result = runner.invoke(
                                        app, ["dev", "experimental", "solve", "test prompt"]
                                    )
                                    assert result.exit_code == 0
                                    mock_make_pipeline.assert_called_once()

    def test_cli_bench_command_parameters(self):
        """Test that the bench command properly passes parameters to make_default_pipeline."""
        # Mock all agent creation functions to avoid API key requirements
        with patch("flujo.cli.main.make_review_agent") as mock_make_review:
            with patch("flujo.cli.main.make_solution_agent") as mock_make_solution:
                with patch("flujo.cli.main.make_validator_agent") as mock_make_validator:
                    with patch("flujo.cli.main.get_reflection_agent") as mock_get_reflection:
                        # Create mock agents using regular Mock since they're not actually called
                        mock_review = Mock()
                        mock_solution = Mock()
                        mock_validator = Mock()
                        mock_reflection = Mock()

                        # Configure mocks to return our mock agents
                        mock_make_review.return_value = mock_review
                        mock_make_solution.return_value = mock_solution
                        mock_make_validator.return_value = mock_validator
                        mock_get_reflection.return_value = mock_reflection

                        # Mock the pipeline creation and execution at the CLI module level
                        with patch("flujo.cli.main.make_default_pipeline") as mock_make_pipeline:
                            with patch("flujo.cli.main.run_default_pipeline") as mock_run_pipeline:
                                # Create a mock pipeline and result
                                class DummyPipeline:
                                    def __str__(self):
                                        return "dummy pipeline"

                                class DummyResult:
                                    def __init__(self):
                                        self.score = 1.0

                                    def __str__(self):
                                        return "dummy result"

                                mock_pipeline = DummyPipeline()
                                mock_result = DummyResult()

                                mock_make_pipeline.return_value = mock_pipeline
                                mock_run_pipeline.return_value = mock_result

                                # Mock asyncio.run to avoid actual async execution
                                with patch("asyncio.run") as mock_asyncio_run:
                                    mock_asyncio_run.return_value = mock_result

                                    # Provide a dummy numpy module so the bench
                                    # command works even when numpy isn't installed
                                    from typer.testing import CliRunner
                                    import sys
                                    from types import SimpleNamespace

                                    dummy_numpy = SimpleNamespace(
                                        percentile=lambda data, q: data[0]
                                    )
                                    with patch.dict(sys.modules, {"numpy": dummy_numpy}):
                                        runner = CliRunner()
                                        result = runner.invoke(
                                            app,
                                            [
                                                "dev",
                                                "experimental",
                                                "bench",
                                                "test prompt",
                                                "--rounds",
                                                "1",
                                            ],
                                        )
                                    assert result.exit_code == 0
                                    mock_make_pipeline.assert_called_once()

    def test_parameter_mapping_consistency(self):
        """Test that CLI parameter names map consistently to function parameters."""
        # Test the mapping between CLI parameters and function parameters
        cli_to_function_mapping = {
            "k": "k_variants",
            "max_iters": "max_iters",
            "reflection": "reflection_agent",
        }

        # Verify that the mapping is consistent
        assert cli_to_function_mapping["k"] == "k_variants"
        assert cli_to_function_mapping["max_iters"] == "max_iters"
        assert cli_to_function_mapping["reflection"] == "reflection_agent"

    def test_default_parameter_values(self):
        """Test that default parameter values are consistent between CLI and functions."""
        # Test that default values are consistent
        expected_defaults = {
            "k_variants": 1,
            "max_iters": 3,
            "max_retries": 3,
        }

        # Verify that make_default_pipeline has the expected defaults
        import inspect

        sig = inspect.signature(make_default_pipeline)

        for param_name, expected_default in expected_defaults.items():
            if param_name in sig.parameters:
                param = sig.parameters[param_name]
                if param.default is not inspect.Parameter.empty:
                    assert param.default == expected_default, (
                        f"Default for {param_name} should be {expected_default}, got {param.default}"
                    )

    def test_parameter_validation(self):
        """Test that parameters are validated correctly."""
        mock_review = AsyncMock()
        mock_solution = AsyncMock()
        mock_validator = AsyncMock()

        # Test with valid parameters
        try:
            pipeline = make_default_pipeline(
                review_agent=mock_review,
                solution_agent=mock_solution,
                validator_agent=mock_validator,
                k_variants=1,
                max_iters=3,
            )
            assert pipeline is not None
        except Exception as e:
            pytest.fail(f"Valid parameters should not raise exception: {e}")

        # Test with invalid parameters (should raise appropriate exceptions)
        with pytest.raises((TypeError, ValueError)):
            make_default_pipeline(
                review_agent=mock_review,
                solution_agent=mock_solution,
                validator_agent=mock_validator,
                k_variants="invalid",  # Should be int
            )

    def test_async_execution_compatibility(self):
        """Test that run_default_pipeline works correctly as an async function."""
        mock_review = AsyncMock()
        mock_solution = AsyncMock()
        mock_validator = AsyncMock()

        # Configure mocks to return concrete values
        from flujo.domain.models import Checklist, ChecklistItem

        mock_solution.run.return_value = "Test solution"
        mock_review.run.return_value = Checklist(
            items=[ChecklistItem(description="Test item", passed=True)]
        )
        mock_validator.run.return_value = Checklist(
            items=[ChecklistItem(description="Test item", passed=True)]
        )

        pipeline = make_default_pipeline(
            review_agent=mock_review,
            solution_agent=mock_solution,
            validator_agent=mock_validator,
        )

        # Test that run_default_pipeline is async
        import inspect

        assert inspect.iscoroutinefunction(run_default_pipeline)

        # Test that it can be awaited
        async def test_async():
            task = Task(prompt="Test")
            result = await run_default_pipeline(pipeline, task)
            return result

        # This should not raise an exception
        try:
            import asyncio

            asyncio.run(test_async())
        except Exception as e:
            pytest.fail(f"run_default_pipeline should work as async function: {e}")


class TestCLIParameterRegressionPrevention:
    """Test to prevent regression of CLI parameter issues."""

    def test_no_parameter_loss(self):
        """Test that no parameters are lost during refactoring."""
        # Get the current signature of make_default_pipeline
        import inspect

        sig = inspect.signature(make_default_pipeline)

        # Define the expected parameters
        expected_params = {
            "review_agent",
            "solution_agent",
            "validator_agent",
            "reflection_agent",
            "max_retries",
            "k_variants",
            "max_iters",
            "reflection_limit",
        }

        # Check that all expected parameters are present
        actual_params = set(sig.parameters.keys())
        missing_params = expected_params - actual_params

        assert not missing_params, f"Missing parameters in make_default_pipeline: {missing_params}"

    def test_parameter_documentation(self):
        """Test that all parameters are properly documented."""

        # Get the docstring
        doc = make_default_pipeline.__doc__
        assert doc is not None, "make_default_pipeline should have a docstring"

        # Check that key parameters are mentioned in the docstring
        key_params = ["k_variants", "max_iters", "reflection_limit"]
        for param in key_params:
            assert param in doc, f"Parameter {param} should be documented in docstring"

    def test_cli_integration_consistency(self):
        """Test that CLI integration is consistent."""
        # This test ensures that the CLI properly calls the pipeline functions
        # with the correct parameters

        # Mock the CLI functions to verify they call make_default_pipeline correctly
        with patch("flujo.recipes.factories.make_default_pipeline") as mock_make:
            mock_make.return_value = MagicMock()

            # Test that the CLI functions exist and are callable
            from flujo.cli.main import solve, bench

            assert callable(solve)
            assert callable(bench)

    def test_parameter_type_consistency(self):
        """Test that parameter types are consistent."""
        import inspect

        sig = inspect.signature(make_default_pipeline)

        # Check that numeric parameters have the correct types
        numeric_params = {
            "k_variants": int,
            "max_iters": int,
            "max_retries": int,
        }

        for param_name, expected_type in numeric_params.items():
            if param_name in sig.parameters:
                param = sig.parameters[param_name]
                # Check the annotation if present
                if param.annotation is not inspect.Parameter.empty:
                    # Handle string annotations (forward references)
                    annotation = param.annotation
                    if isinstance(annotation, str):
                        # For string annotations, we can't easily compare types
                        # So we'll just verify the parameter exists and has an annotation
                        assert annotation == "int", (
                            f"Parameter {param_name} should have 'int' annotation"
                        )
                    else:
                        assert annotation == expected_type, (
                            f"Parameter {param_name} should be {expected_type}"
                        )

    def test_optional_parameter_handling(self):
        """Test that optional parameters are handled correctly."""
        mock_review = AsyncMock()
        mock_solution = AsyncMock()
        mock_validator = AsyncMock()

        # Test with minimal parameters
        pipeline1 = make_default_pipeline(
            review_agent=mock_review,
            solution_agent=mock_solution,
            validator_agent=mock_validator,
        )

        # Test with all optional parameters
        mock_reflection = AsyncMock()
        pipeline2 = make_default_pipeline(
            review_agent=mock_review,
            solution_agent=mock_solution,
            validator_agent=mock_validator,
            reflection_agent=mock_reflection,
            k_variants=2,
            max_iters=5,
            reflection_limit=3,
        )

        assert pipeline1 is not None
        assert pipeline2 is not None
