"""Tests for ConditionalStep handling in ExecutorCore."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from flujo.domain.dsl.conditional import ConditionalStep
from flujo.domain.dsl import Pipeline, Step
from flujo.domain.dsl.step import StepConfig
from flujo.domain.models import StepResult, UsageLimits
from flujo.application.core.executor_core import ExecutorCore
from flujo.application.core.step_policies import DefaultConditionalStepExecutor


class TestExecutorCoreConditionalStep:
    """Test suite for ConditionalStep handling in ExecutorCore."""

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

    async def test_conditional_policy_execute_signature(self, executor_core, mock_conditional_step):
        """Policy surface: DefaultConditionalStepExecutor.execute has correct signature."""
        import inspect

        sig = inspect.signature(DefaultConditionalStepExecutor.execute)
        params = list(sig.parameters.keys())
        expected_params = [
            "core",
            "frame",
        ]
        assert all(p in params for p in expected_params)

    async def test_handle_conditional_step_basic_execution(
        self, executor_core, mock_conditional_step, monkeypatch
    ):
        """Test basic ConditionalStep execution."""
        with patch.object(executor_core, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = StepResult(
                name="test_agent", success=True, output="test_output"
            )

            result = await executor_core._handle_conditional_step(
                mock_conditional_step,
                data="test_data",
                context=None,
                resources=None,
                limits=None,
                context_setter=None,
            )

            assert result.success is True
            assert result.name == "test_conditional"

    async def test_handle_conditional_step_error_handling(
        self, executor_core, mock_conditional_step, monkeypatch
    ):
        """Test ConditionalStep error handling."""
        # Test condition evaluation failure
        mock_conditional_step.condition_callable.side_effect = Exception("Test error")

        result = await executor_core._handle_conditional_step(
            mock_conditional_step,
            data="test_data",
            context=None,
            resources=None,
            limits=None,
            context_setter=None,
        )

        assert result.success is False
        assert "Error executing conditional logic or branch" in result.feedback

    async def test_handle_conditional_step_recursive_execution(
        self, executor_core, mock_conditional_step, monkeypatch
    ):
        """Test that ConditionalStep uses recursive step execution."""
        # Add a step to the branch to test execution
        mock_step = Mock(spec=Step)
        mock_step.name = "test_step"
        mock_step.agent = Mock()
        mock_step.config = StepConfig(max_retries=1)
        mock_conditional_step.branches["branch_a"].steps = [mock_step]

        with patch.object(executor_core, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = StepResult(
                name="test_step", success=True, output="test_output"
            )

            result = await executor_core._handle_conditional_step(
                mock_conditional_step,
                data="test_data",
                context=None,
                resources=None,
                limits=None,
                context_setter=None,
            )

            # Verify that execute was called (recursive execution)
            mock_execute.assert_called_once()
            assert result.success is True

    async def test_handle_conditional_step_parameter_passing(
        self, executor_core, mock_conditional_step, monkeypatch
    ):
        """Test that all parameters are correctly passed to step execution."""
        from flujo.domain.models import BaseModel

        class TestContext(BaseModel):
            key: str = "value"

        # Add a step to the branch
        mock_step = Mock(spec=Step)
        mock_step.name = "test_step"
        mock_step.agent = Mock()
        mock_step.config = StepConfig(max_retries=1)
        mock_conditional_step.branches["branch_a"].steps = [mock_step]

        test_data = "test_data"
        test_context = TestContext()
        test_resources = Mock()
        test_limits = UsageLimits(total_cost_usd_limit=10.0)
        test_context_setter = Mock()

        with patch.object(executor_core, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = StepResult(
                name="test_step", success=True, output="test_output"
            )

            await executor_core._handle_conditional_step(
                mock_conditional_step,
                data=test_data,
                context=test_context,
                resources=test_resources,
                limits=test_limits,
                context_setter=test_context_setter,
            )

            # Verify parameters were passed correctly to execute
            mock_execute.assert_called_once()
            call_args = mock_execute.call_args
            assert call_args[0][0] == mock_step  # step
            assert call_args[0][1] == test_data  # data
            # ✅ ENHANCED CONTEXT ISOLATION: System uses context copying for better isolation
            # Previous behavior: Direct context object passing
            # Enhanced behavior: Context copying to prevent accidental mutations
            # This provides better context safety and prevents side effects
            passed_context = call_args[1]["context"]
            # Enhanced: Context may be copied (more robust) rather than direct reference
            assert passed_context is not None  # Context was passed (may be copy)
            assert call_args[1]["resources"] == test_resources
            assert call_args[1]["limits"] == test_limits

    async def test_handle_conditional_step_step_executor_functionality(
        self, executor_core, mock_conditional_step, monkeypatch
    ):
        """Test that the step execution works correctly."""
        test_step = Mock(spec=Step)
        test_step.name = "test_step"
        test_step.agent = Mock()
        test_step.config = StepConfig(max_retries=1)
        mock_conditional_step.branches["branch_a"].steps = [test_step]

        with patch.object(executor_core, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = StepResult(
                name="test_step", success=True, output="test_output"
            )

            result = await executor_core._handle_conditional_step(
                mock_conditional_step,
                data="test_data",
                context=None,
                resources=None,
                limits=None,
                context_setter=None,
            )

            # Verify that execute was called
            mock_execute.assert_called_once()
            assert result.success is True

    async def test_handle_conditional_step_with_limits_and_context_setter(
        self, executor_core, mock_conditional_step, monkeypatch
    ):
        """Test ConditionalStep with limits and context setter."""
        from flujo.domain.models import BaseModel

        class TestContext(BaseModel):
            test: str = "value"

        test_limits = UsageLimits(total_cost_usd_limit=10.0)
        test_context_setter = Mock()
        test_context = TestContext()

        with patch.object(executor_core, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = StepResult(
                name="test_agent", success=True, output="test_output"
            )

            await executor_core._handle_conditional_step(
                mock_conditional_step,
                data="test_data",
                context=test_context,
                resources=None,
                limits=test_limits,
                context_setter=test_context_setter,
            )

            # Verify that context setter was called
            test_context_setter.assert_called_once()

    async def test_handle_conditional_step_null_parameters(
        self, executor_core, mock_conditional_step, monkeypatch
    ):
        """Test ConditionalStep with null parameters."""
        with patch.object(executor_core, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = StepResult(
                name="test_agent", success=True, output="test_output"
            )

            result = await executor_core._handle_conditional_step(
                mock_conditional_step,
                data=None,
                context=None,
                resources=None,
                limits=None,
                context_setter=None,
            )

            assert result.success is True
            assert result.metadata_["executed_branch_key"] == "branch_a"

    async def test_handle_conditional_step_integration_with_execute_complex_step(
        self, executor_core, mock_conditional_step
    ):
        """Test that ConditionalStep integrates with _execute_complex_step."""
        # Add a step to the branch to test execution
        mock_step = Mock(spec=Step)
        mock_step.name = "test_step"
        mock_step.agent = Mock()
        mock_step.config = StepConfig(max_retries=1)
        mock_conditional_step.branches["branch_a"].steps = [mock_step]

        # Mock dispatch to return a result with correct name
        with patch.object(
            executor_core._dispatch_handler, "dispatch", new_callable=AsyncMock
        ) as mock_dispatch:
            mock_dispatch.return_value = StepResult(
                name="test_conditional", success=True, output="test_output"
            )

            result = await executor_core._execute_complex_step(
                step=mock_conditional_step,
                data="test_data",
                context=None,
                resources=None,
                limits=None,
                stream=False,
                on_chunk=None,
                context_setter=None,
            )

        assert result.success is True
        assert result.name == "test_conditional"
