"""Comprehensive tests for ExecutorCore ConditionalStep handling."""

import pytest
from unittest.mock import Mock, AsyncMock
from flujo.domain.dsl.conditional import ConditionalStep
from flujo.domain.dsl import Pipeline, Step
from flujo.domain.models import StepResult, UsageLimits
from flujo.application.core.ultra_executor import ExecutorCore


class TestExecutorCoreConditionalStep:
    """Test suite for ExecutorCore ConditionalStep handling."""

    @pytest.fixture
    def executor_core(self):
        """Create ExecutorCore instance for testing."""
        return ExecutorCore()

    @pytest.fixture
    def mock_conditional_step(self):
        """Create a mock ConditionalStep for testing."""
        conditional_step = Mock(spec=ConditionalStep)
        conditional_step.name = "test_conditional"
        conditional_step.condition_callable = Mock(return_value="branch_a")
        conditional_step.branches = {"branch_a": Mock(spec=Pipeline)}
        conditional_step.branches["branch_a"].steps = []
        conditional_step.default_branch_pipeline = None
        conditional_step.branch_input_mapper = None
        conditional_step.branch_output_mapper = None
        return conditional_step

    async def test_handle_conditional_step_method_exists(self, executor_core):
        """Test that _handle_conditional_step method exists."""
        assert hasattr(executor_core, "_handle_conditional_step")
        assert callable(executor_core._handle_conditional_step)

    async def test_handle_conditional_step_signature(self, executor_core, mock_conditional_step):
        """Test that _handle_conditional_step has correct signature."""
        method = executor_core._handle_conditional_step
        import inspect

        sig = inspect.signature(method)

        # Check required parameters
        expected_params = {
            "conditional_step",
            "data",
            "context",
            "resources",
            "limits",
            "context_setter",
        }
        actual_params = set(sig.parameters.keys())
        assert expected_params.issubset(actual_params)

    async def test_handle_conditional_step_basic_execution(
        self, executor_core, mock_conditional_step, monkeypatch
    ):
        """Test basic ConditionalStep execution through ExecutorCore."""
        # Mock the legacy handler to return success
        monkeypatch.setattr(
            "flujo.application.core.step_logic._handle_conditional_step",
            AsyncMock(return_value=StepResult(name="test_conditional", success=True)),
        )

        result = await executor_core._handle_conditional_step(
            mock_conditional_step,
            data="test_data",
            context=None,
            resources=None,
            limits=None,
            context_setter=None,
        )

        assert isinstance(result, StepResult)
        assert result.success is True

    async def test_handle_conditional_step_error_handling(
        self, executor_core, mock_conditional_step, monkeypatch
    ):
        """Test ConditionalStep error handling."""
        # Mock the legacy handler to raise an exception
        monkeypatch.setattr(
            "flujo.application.core.step_logic._handle_conditional_step",
            AsyncMock(side_effect=Exception("Test error")),
        )

        with pytest.raises(Exception, match="Test error"):
            await executor_core._handle_conditional_step(
                mock_conditional_step,
                data="test_data",
                context=None,
                resources=None,
                limits=None,
                context_setter=None,
            )

    async def test_handle_conditional_step_recursive_execution(
        self, executor_core, mock_conditional_step, monkeypatch
    ):
        """Test that ConditionalStep uses recursive step execution."""
        # Mock the legacy handler to capture the step_executor
        captured_step_executor = None

        async def mock_legacy_handler(*args, **kwargs):
            nonlocal captured_step_executor
            captured_step_executor = kwargs.get("step_executor")
            return StepResult(name="test_conditional", success=True)

        monkeypatch.setattr(
            "flujo.application.core.step_logic._handle_conditional_step", mock_legacy_handler
        )

        await executor_core._handle_conditional_step(
            mock_conditional_step,
            data="test_data",
            context=None,
            resources=None,
            limits=None,
            context_setter=None,
        )

        # Verify that a step_executor was passed
        assert captured_step_executor is not None
        assert callable(captured_step_executor)

    async def test_handle_conditional_step_parameter_passing(
        self, executor_core, mock_conditional_step, monkeypatch
    ):
        """Test that all parameters are correctly passed to legacy handler."""
        captured_args = None
        captured_kwargs = None

        async def mock_legacy_handler(*args, **kwargs):
            nonlocal captured_args, captured_kwargs
            captured_args = args
            captured_kwargs = kwargs
            return StepResult(name="test_conditional", success=True)

        monkeypatch.setattr(
            "flujo.application.core.step_logic._handle_conditional_step", mock_legacy_handler
        )

        test_data = "test_data"
        test_context = Mock()
        test_resources = Mock()
        test_limits = UsageLimits(total_cost_usd_limit=10.0)
        test_context_setter = Mock()

        await executor_core._handle_conditional_step(
            mock_conditional_step,
            data=test_data,
            context=test_context,
            resources=test_resources,
            limits=test_limits,
            context_setter=test_context_setter,
        )

        # Verify parameters were passed correctly
        assert captured_args[0] is mock_conditional_step
        assert captured_args[1] is test_data
        assert captured_args[2] is test_context
        assert captured_args[3] is test_resources

        assert captured_kwargs["context_model_defined"] is True
        assert captured_kwargs["usage_limits"] is test_limits
        assert captured_kwargs["context_setter"] is test_context_setter
        assert "step_executor" in captured_kwargs

    async def test_handle_conditional_step_step_executor_functionality(
        self, executor_core, mock_conditional_step, monkeypatch
    ):
        """Test that the step_executor function works correctly."""
        captured_step_executor = None
        test_step = Mock(spec=Step)
        test_step.name = "test_step"

        async def mock_legacy_handler(*args, **kwargs):
            nonlocal captured_step_executor
            captured_step_executor = kwargs.get("step_executor")

            # Test the step_executor by calling it
            if captured_step_executor:
                result = await captured_step_executor(test_step, "test_input", None, None)
                return result
            return StepResult(name="test_conditional", success=True)

        monkeypatch.setattr(
            "flujo.application.core.step_logic._handle_conditional_step", mock_legacy_handler
        )

        # Mock the execute method to return a known result
        original_execute = executor_core.execute
        executor_core.execute = AsyncMock(return_value=StepResult(name="test_step", success=True))

        try:
            await executor_core._handle_conditional_step(
                mock_conditional_step,
                data="test_data",
                context=None,
                resources=None,
                limits=None,
                context_setter=None,
            )

            # Verify that execute was called by the step_executor
            executor_core.execute.assert_called_once()

        finally:
            executor_core.execute = original_execute

    async def test_handle_conditional_step_with_limits_and_context_setter(
        self, executor_core, mock_conditional_step, monkeypatch
    ):
        """Test ConditionalStep with usage limits and context setter."""
        captured_kwargs = None

        async def mock_legacy_handler(*args, **kwargs):
            nonlocal captured_kwargs
            captured_kwargs = kwargs
            return StepResult(name="test_conditional", success=True)

        monkeypatch.setattr(
            "flujo.application.core.step_logic._handle_conditional_step", mock_legacy_handler
        )

        test_limits = UsageLimits(total_cost_usd_limit=5.0, total_tokens_limit=1000)
        test_context_setter = Mock()

        await executor_core._handle_conditional_step(
            mock_conditional_step,
            data="test_data",
            context=None,
            resources=None,
            limits=test_limits,
            context_setter=test_context_setter,
        )

        # Verify that limits and context_setter are passed through
        assert captured_kwargs["usage_limits"] is test_limits
        assert captured_kwargs["context_setter"] is test_context_setter

    async def test_handle_conditional_step_null_parameters(
        self, executor_core, mock_conditional_step, monkeypatch
    ):
        """Test ConditionalStep with null parameters."""
        captured_kwargs = None

        async def mock_legacy_handler(*args, **kwargs):
            nonlocal captured_kwargs
            captured_kwargs = kwargs
            return StepResult(name="test_conditional", success=True)

        monkeypatch.setattr(
            "flujo.application.core.step_logic._handle_conditional_step", mock_legacy_handler
        )

        await executor_core._handle_conditional_step(
            mock_conditional_step,
            data="test_data",
            context=None,
            resources=None,
            limits=None,
            context_setter=None,
        )

        # Verify that null parameters are handled correctly
        assert captured_kwargs["usage_limits"] is None
        assert captured_kwargs["context_setter"] is None
        assert captured_kwargs["context_model_defined"] is True

    async def test_handle_conditional_step_integration_with_execute_complex_step(
        self, executor_core, mock_conditional_step
    ):
        """Test that ConditionalStep is properly integrated in _execute_complex_step."""
        # Mock the _handle_conditional_step method to verify it's called
        original_handle_conditional = executor_core._handle_conditional_step
        executor_core._handle_conditional_step = AsyncMock(
            return_value=StepResult(name="test_conditional", success=True)
        )

        try:
            # Call _execute_complex_step with a ConditionalStep
            result = await executor_core._execute_complex_step(
                mock_conditional_step,
                data="test_data",
                context=None,
                resources=None,
                limits=None,
                stream=False,
                on_chunk=None,
                breach_event=None,
                context_setter=None,
            )

            # Verify that _handle_conditional_step was called
            executor_core._handle_conditional_step.assert_called_once()

            # Verify the result
            assert isinstance(result, StepResult)
            assert result.success is True

        finally:
            executor_core._handle_conditional_step = original_handle_conditional
