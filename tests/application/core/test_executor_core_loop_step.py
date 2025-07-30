"""Comprehensive tests for ExecutorCore LoopStep handling."""

import pytest
from unittest.mock import Mock, AsyncMock
from flujo.domain.dsl.loop import LoopStep
from flujo.domain.dsl import Pipeline
from flujo.domain.models import StepResult, UsageLimits
from flujo.application.core.ultra_executor import ExecutorCore


class TestExecutorCoreLoopStep:
    """Test suite for ExecutorCore LoopStep handling."""

    @pytest.fixture
    def executor_core(self):
        """Create ExecutorCore instance for testing."""
        return ExecutorCore()

    @pytest.fixture
    def mock_loop_step(self):
        """Create a mock LoopStep for testing."""
        loop_step = Mock(spec=LoopStep)
        loop_step.name = "test_loop"
        loop_step.max_loops = 3
        loop_step.loop_body_pipeline = Mock(spec=Pipeline)
        loop_step.loop_body_pipeline.steps = []
        loop_step.exit_condition_callable = Mock(return_value=False)
        loop_step.initial_input_to_loop_body_mapper = None
        loop_step.iteration_input_mapper = None
        loop_step.loop_output_mapper = None
        return loop_step

    async def test_handle_loop_step_method_exists(self, executor_core):
        """Test that _handle_loop_step method exists."""
        assert hasattr(executor_core, "_handle_loop_step")
        assert callable(executor_core._handle_loop_step)

    async def test_handle_loop_step_signature(self, executor_core, mock_loop_step):
        """Test that _handle_loop_step has correct signature."""
        method = executor_core._handle_loop_step
        import inspect

        sig = inspect.signature(method)

        # Check required parameters
        expected_params = {"loop_step", "data", "context", "resources", "limits", "context_setter"}
        actual_params = set(sig.parameters.keys())
        assert expected_params.issubset(actual_params)

    async def test_handle_loop_step_basic_execution(self, executor_core, mock_loop_step):
        """Test basic LoopStep execution through ExecutorCore."""
        # Mock the legacy handler to return success
        with pytest.MonkeyPatch().context() as m:
            m.setattr(
                "flujo.application.core.step_logic._handle_loop_step",
                AsyncMock(return_value=StepResult(name="test_loop", success=True)),
            )

            result = await executor_core._handle_loop_step(
                mock_loop_step,
                data="test_data",
                context=None,
                resources=None,
                limits=None,
                context_setter=None,
            )

            assert isinstance(result, StepResult)
            assert result.success is True

    async def test_handle_loop_step_error_handling(self, executor_core, mock_loop_step):
        """Test LoopStep error handling."""
        # Mock the legacy handler to raise an exception
        with pytest.MonkeyPatch().context() as m:
            m.setattr(
                "flujo.application.core.step_logic._handle_loop_step",
                AsyncMock(side_effect=Exception("Test error")),
            )

            with pytest.raises(Exception, match="Test error"):
                await executor_core._handle_loop_step(
                    mock_loop_step,
                    data="test_data",
                    context=None,
                    resources=None,
                    limits=None,
                    context_setter=None,
                )

    async def test_handle_loop_step_recursive_execution(self, executor_core, mock_loop_step):
        """Test that LoopStep uses recursive step execution."""
        # Mock the legacy handler to capture the step_executor
        captured_step_executor = None

        async def mock_legacy_handler(*args, **kwargs):
            nonlocal captured_step_executor
            captured_step_executor = kwargs.get("step_executor")
            return StepResult(name="test_loop", success=True)

        with pytest.MonkeyPatch().context() as m:
            m.setattr("flujo.application.core.step_logic._handle_loop_step", mock_legacy_handler)

            await executor_core._handle_loop_step(
                mock_loop_step,
                data="test_data",
                context=None,
                resources=None,
                limits=None,
                context_setter=None,
            )

            # Verify that a step_executor was passed
            assert captured_step_executor is not None
            assert callable(captured_step_executor)

    async def test_handle_loop_step_parameter_passing(self, executor_core, mock_loop_step):
        """Test that all parameters are correctly passed to legacy handler."""
        captured_args = None
        captured_kwargs = None

        async def mock_legacy_handler(*args, **kwargs):
            nonlocal captured_args, captured_kwargs
            captured_args = args
            captured_kwargs = kwargs
            return StepResult(name="test_loop", success=True)

        with pytest.MonkeyPatch().context() as m:
            m.setattr("flujo.application.core.step_logic._handle_loop_step", mock_legacy_handler)

            test_data = "test_data"
            test_context = Mock()
            test_resources = Mock()
            test_limits = UsageLimits(total_cost_usd_limit=10.0)
            test_context_setter = Mock()

            await executor_core._handle_loop_step(
                mock_loop_step,
                data=test_data,
                context=test_context,
                resources=test_resources,
                limits=test_limits,
                context_setter=test_context_setter,
            )

            # Verify parameters were passed correctly
            assert captured_args[0] is mock_loop_step
            assert captured_args[1] is test_data
            assert captured_args[2] is test_context
            assert captured_args[3] is test_resources

            assert captured_kwargs["context_model_defined"] is True
            assert captured_kwargs["usage_limits"] is test_limits
            assert captured_kwargs["context_setter"] is test_context_setter
            assert "step_executor" in captured_kwargs

    async def test_handle_loop_step_with_none_parameters(self, executor_core, mock_loop_step):
        """Test LoopStep handling with None parameters."""
        with pytest.MonkeyPatch().context() as m:
            m.setattr(
                "flujo.application.core.step_logic._handle_loop_step",
                AsyncMock(return_value=StepResult(name="test_loop", success=True)),
            )

            result = await executor_core._handle_loop_step(
                mock_loop_step,
                data=None,
                context=None,
                resources=None,
                limits=None,
                context_setter=None,
            )

            assert isinstance(result, StepResult)

    async def test_handle_loop_step_with_complex_limits(self, executor_core, mock_loop_step):
        """Test LoopStep with complex usage limits."""
        complex_limits = UsageLimits(
            total_cost_usd_limit=100.0,
            total_tokens_limit=10000,
            cost_per_minute_usd_limit=10.0,
            tokens_per_minute_limit=1000,
        )

        with pytest.MonkeyPatch().context() as m:
            m.setattr(
                "flujo.application.core.step_logic._handle_loop_step",
                AsyncMock(return_value=StepResult(name="test_loop", success=True)),
            )

            result = await executor_core._handle_loop_step(
                mock_loop_step,
                data="test_data",
                context=None,
                resources=None,
                limits=complex_limits,
                context_setter=None,
            )

            assert isinstance(result, StepResult)

    async def test_handle_loop_step_step_executor_functionality(
        self, executor_core, mock_loop_step
    ):
        """Test that the step_executor function works correctly."""
        captured_step_executor = None

        async def mock_legacy_handler(*args, **kwargs):
            nonlocal captured_step_executor
            captured_step_executor = kwargs.get("step_executor")
            return StepResult(name="test_loop", success=True)

        with pytest.MonkeyPatch().context() as m:
            m.setattr("flujo.application.core.step_logic._handle_loop_step", mock_legacy_handler)

            await executor_core._handle_loop_step(
                mock_loop_step,
                data="test_data",
                context=None,
                resources=None,
                limits=None,
                context_setter=None,
            )

            # Test that the step_executor can be called
            test_step = Mock()
            test_data = "test_data"
            test_context = Mock()
            test_resources = Mock()

            # Mock the execute method to return a success result
            executor_core.execute = AsyncMock(return_value=StepResult(name="test", success=True))

            result = await captured_step_executor(
                test_step, test_data, test_context, test_resources
            )

            assert isinstance(result, StepResult)
            assert result.success is True

            # Verify execute was called with correct parameters
            executor_core.execute.assert_called_once_with(
                test_step,
                test_data,
                context=test_context,
                resources=test_resources,
                limits=None,
                context_setter=None,
            )

    async def test_handle_loop_step_step_executor_with_extra_kwargs(
        self, executor_core, mock_loop_step
    ):
        """Test that step_executor handles extra kwargs correctly."""
        captured_step_executor = None

        async def mock_legacy_handler(*args, **kwargs):
            nonlocal captured_step_executor
            captured_step_executor = kwargs.get("step_executor")
            return StepResult(name="test_loop", success=True)

        with pytest.MonkeyPatch().context() as m:
            m.setattr("flujo.application.core.step_logic._handle_loop_step", mock_legacy_handler)

            test_limits = UsageLimits(total_cost_usd_limit=10.0)
            test_context_setter = Mock()

            await executor_core._handle_loop_step(
                mock_loop_step,
                data="test_data",
                context=None,
                resources=None,
                limits=test_limits,
                context_setter=test_context_setter,
            )

            # Mock the execute method
            executor_core.execute = AsyncMock(return_value=StepResult(name="test", success=True))

            # Test step_executor with extra kwargs
            test_step = Mock()
            result = await captured_step_executor(
                test_step,
                "data",
                None,
                None,
                usage_limits=test_limits,
                context_setter=test_context_setter,
            )

            assert isinstance(result, StepResult)

            # Verify execute was called with the extra kwargs
            executor_core.execute.assert_called_once_with(
                test_step,
                "data",
                context=None,
                resources=None,
                limits=test_limits,
                context_setter=test_context_setter,
            )
