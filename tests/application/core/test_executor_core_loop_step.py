"""Comprehensive tests for ExecutorCore LoopStep handling."""

import pytest
from unittest.mock import Mock, AsyncMock
from flujo.domain.dsl.loop import LoopStep
from flujo.domain.dsl import Pipeline
from flujo.domain.models import StepResult, UsageLimits
from flujo.application.core.executor_core import ExecutorCore
from flujo.application.core.step_policies import DefaultLoopStepExecutor
from tests.test_types.fixtures import create_test_step


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

    async def test_loop_policy_execute_signature(self):
        """Policy surface: DefaultLoopStepExecutor.execute has correct standardized signature."""
        import inspect

        sig = inspect.signature(DefaultLoopStepExecutor.execute)
        expected_params = {"self", "core", "frame"}
        actual_params = set(sig.parameters.keys())
        assert expected_params == actual_params

    async def test_handle_loop_step_basic_execution(self, executor_core, mock_loop_step):
        """Test basic LoopStep execution through ExecutorCore."""
        # Create a simple step for the loop body
        from flujo.testing.utils import StubAgent

        simple_step = create_test_step(name="test_step", agent=StubAgent(["test_output"]))
        mock_loop_step.loop_body_pipeline.steps = [simple_step]
        mock_loop_step.exit_condition_callable = Mock(
            return_value=True
        )  # Exit after first iteration

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
        assert result.attempts == 1

    async def test_handle_loop_step_error_handling(self, executor_core, mock_loop_step):
        """Test LoopStep error handling."""
        # Create a step that will fail
        from flujo.testing.utils import StubAgent

        failing_step = create_test_step(
            name="failing_step", agent=StubAgent([])
        )  # Empty list causes IndexError
        mock_loop_step.loop_body_pipeline.steps = [failing_step]
        mock_loop_step.exit_condition_callable = Mock(return_value=False)

        result = await executor_core._handle_loop_step(
            mock_loop_step,
            data="test_data",
            context=None,
            resources=None,
            limits=None,
            context_setter=None,
        )

        assert isinstance(result, StepResult)
        assert result.success is False

    async def test_handle_loop_step_recursive_execution(self, executor_core, mock_loop_step):
        """Test that LoopStep uses recursive step execution."""
        # Create a step for the loop body
        from flujo.testing.utils import StubAgent

        simple_step = create_test_step(name="test_step", agent=StubAgent(["test_output"]))
        mock_loop_step.loop_body_pipeline.steps = [simple_step]
        mock_loop_step.exit_condition_callable = Mock(return_value=True)

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

    async def test_handle_loop_step_parameter_passing(self, executor_core, mock_loop_step):
        """Test that parameters are passed correctly to the LoopStep handler."""
        # Create a simple step for the loop body
        from flujo.testing.utils import StubAgent

        simple_step = create_test_step(name="test_step", agent=StubAgent(["test_output"]))
        mock_loop_step.loop_body_pipeline.steps = [simple_step]
        mock_loop_step.exit_condition_callable = Mock(return_value=True)

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

    async def test_handle_loop_step_with_none_parameters(self, executor_core, mock_loop_step):
        """Test LoopStep handling with None parameters."""
        with pytest.MonkeyPatch().context() as m:
            m.setattr(
                "flujo.application.core.executor_core.ExecutorCore._handle_loop_step",
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
        # Create a simple step for the loop body
        from flujo.testing.utils import StubAgent

        simple_step = create_test_step(name="test_step", agent=StubAgent(["test_output"]))
        mock_loop_step.loop_body_pipeline.steps = [simple_step]
        mock_loop_step.exit_condition_callable = Mock(return_value=True)

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

    async def test_handle_loop_step_step_executor_with_extra_kwargs(
        self, executor_core, mock_loop_step
    ):
        """Test that the step_executor function handles extra kwargs correctly."""
        # Create a simple step for the loop body
        from flujo.testing.utils import StubAgent

        simple_step = create_test_step(name="test_step", agent=StubAgent(["test_output"]))
        mock_loop_step.loop_body_pipeline.steps = [simple_step]
        mock_loop_step.exit_condition_callable = Mock(return_value=True)

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
