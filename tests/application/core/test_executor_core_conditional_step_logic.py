"""Tests for ConditionalStep core logic migration in ExecutorCore."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from flujo.domain.dsl.conditional import ConditionalStep
from flujo.domain.dsl import Pipeline, Step
from flujo.domain.models import StepResult, UsageLimits
from flujo.application.core.executor_core import ExecutorCore

# Note: Tests that need telemetry capture should use the isolated_telemetry fixture
# which provides per-test isolation without requiring serial execution.


class TestExecutorCoreConditionalStepLogic:
    """Test suite for ConditionalStep core logic in ExecutorCore."""

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

    @pytest.fixture
    def mock_branch_step(self):
        """Create a mock branch step for testing."""
        step = Mock(spec=Step)
        step.name = "test_branch_step"
        step.agent = Mock()
        step.config = Mock()
        step.config.max_retries = 1
        return step

    async def test_condition_evaluation_success(self, executor_core, mock_conditional_step):
        """Test successful condition evaluation."""
        mock_conditional_step.condition_callable.return_value = "branch_a"

        with patch.object(executor_core, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = StepResult(
                name="test_branch_step", success=True, output="test_output"
            )

            result = await executor_core._handle_conditional_step(
                mock_conditional_step,
                data="test_data",
                context=None,
                resources=None,
                limits=None,
                context_setter=None,
            )

            # Verify condition was evaluated
            mock_conditional_step.condition_callable.assert_called_once_with("test_data", None)
            assert result.success is True

    async def test_condition_evaluation_failure(self, executor_core, mock_conditional_step):
        """Test condition evaluation failure."""
        mock_conditional_step.condition_callable.side_effect = Exception("Condition failed")

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

    async def test_branch_not_found_no_default(self, executor_core, mock_conditional_step):
        """Test when branch is not found and no default branch exists."""
        mock_conditional_step.condition_callable.return_value = "nonexistent_branch"
        mock_conditional_step.branches = {}
        mock_conditional_step.default_branch_pipeline = None

        result = await executor_core._handle_conditional_step(
            mock_conditional_step,
            data="test_data",
            context=None,
            resources=None,
            limits=None,
            context_setter=None,
        )

        assert result.success is False
        assert "No branch found for key 'nonexistent_branch'" in result.feedback

    async def test_branch_not_found_with_default(self, executor_core, mock_conditional_step):
        """Test when branch is not found but default branch exists."""
        mock_conditional_step.condition_callable.return_value = "nonexistent_branch"
        mock_conditional_step.branches = {}
        mock_conditional_step.default_branch_pipeline = Mock(spec=Pipeline)
        mock_conditional_step.default_branch_pipeline.steps = []

        with patch.object(executor_core, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = StepResult(
                name="test_branch_step", success=True, output="test_output"
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

    async def test_branch_execution_success(
        self, executor_core, mock_conditional_step, mock_branch_step
    ):
        """Test successful branch execution."""
        mock_conditional_step.branches["branch_a"].steps = [mock_branch_step]

        with patch.object(executor_core, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = StepResult(
                name="test_branch_step", success=True, output="test_output"
            )

            result = await executor_core._handle_conditional_step(
                mock_conditional_step,
                data="test_data",
                context=None,
                resources=None,
                limits=None,
                context_setter=None,
            )

            # Verify branch step was executed
            mock_execute.assert_called_once()
            assert result.success is True
            assert result.output == "test_output"

    async def test_branch_execution_failure(
        self, executor_core, mock_conditional_step, mock_branch_step
    ):
        """Test branch execution failure."""
        mock_conditional_step.branches["branch_a"].steps = [mock_branch_step]

        with patch.object(executor_core, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = StepResult(
                name="test_branch_step", success=False, feedback="Step failed"
            )

            result = await executor_core._handle_conditional_step(
                mock_conditional_step,
                data="test_data",
                context=None,
                resources=None,
                limits=None,
                context_setter=None,
            )

            assert result.success is False
            assert "Failure in branch 'branch_a'" in result.feedback

    async def test_branch_input_mapping(self, executor_core, mock_conditional_step):
        """Test branch input mapping functionality."""
        mock_conditional_step.branch_input_mapper = Mock(return_value="mapped_input")
        mock_conditional_step.branches["branch_a"].steps = []

        with patch.object(executor_core, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = StepResult(
                name="test_branch_step", success=True, output="test_output"
            )

            await executor_core._handle_conditional_step(
                mock_conditional_step,
                data="test_data",
                context=None,
                resources=None,
                limits=None,
                context_setter=None,
            )

            # Verify input mapper was called
            mock_conditional_step.branch_input_mapper.assert_called_once_with("test_data", None)

    async def test_branch_output_mapping(self, executor_core, mock_conditional_step):
        """Test branch output mapping functionality."""
        mock_conditional_step.branch_output_mapper = Mock(return_value="mapped_output")
        mock_conditional_step.branches["branch_a"].steps = []

        with patch.object(executor_core, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = StepResult(
                name="test_branch_step", success=True, output="test_output"
            )

            result = await executor_core._handle_conditional_step(
                mock_conditional_step,
                data="test_data",
                context=None,
                resources=None,
                limits=None,
                context_setter=None,
            )

            # Verify output mapper was called with the correct parameters
            # When there are no steps, the branch_output is the original input data
            mock_conditional_step.branch_output_mapper.assert_called_once_with(
                "test_data", "branch_a", None
            )
            assert result.output == "mapped_output"

    async def test_branch_output_mapper_exception(self, executor_core, mock_conditional_step):
        """Test branch output mapper exception handling."""
        mock_conditional_step.branch_output_mapper = Mock(side_effect=Exception("Mapper failed"))
        mock_conditional_step.branches["branch_a"].steps = []

        with patch.object(executor_core, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = StepResult(
                name="test_branch_step", success=True, output="test_output"
            )

            result = await executor_core._handle_conditional_step(
                mock_conditional_step,
                data="test_data",
                context=None,
                resources=None,
                limits=None,
                context_setter=None,
            )

            assert result.success is False
            assert "Branch output mapper raised an exception" in result.feedback

    async def test_metrics_accumulation(
        self, executor_core, mock_conditional_step, mock_branch_step
    ):
        """Test that metrics are properly accumulated from branch execution."""
        mock_conditional_step.branches["branch_a"].steps = [mock_branch_step]

        with patch.object(executor_core, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = StepResult(
                name="test_branch_step",
                success=True,
                output="test_output",
                latency_s=1.5,
                cost_usd=0.01,
                token_counts=100,
            )

            result = await executor_core._handle_conditional_step(
                mock_conditional_step,
                data="test_data",
                context=None,
                resources=None,
                limits=None,
                context_setter=None,
            )

            assert result.latency_s == 1.5
            assert result.cost_usd == 0.01
            assert result.token_counts == 100

    async def test_metadata_executed_branch_key(self, executor_core, mock_conditional_step):
        """Test that executed branch key is stored in metadata."""
        mock_conditional_step.branches["branch_a"].steps = []

        with patch.object(executor_core, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = StepResult(
                name="test_branch_step", success=True, output="test_output"
            )

            result = await executor_core._handle_conditional_step(
                mock_conditional_step,
                data="test_data",
                context=None,
                resources=None,
                limits=None,
                context_setter=None,
            )

            assert result.metadata_ is not None
            assert result.metadata_["executed_branch_key"] == "branch_a"

    async def test_context_setter_called_on_success(self, executor_core, mock_conditional_step):
        """Test that context setter is called on successful branch execution."""
        from flujo.domain.models import BaseModel

        class TestContext(BaseModel):
            pass

        mock_conditional_step.branches["branch_a"].steps = []
        mock_context_setter = Mock()

        with patch.object(executor_core, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = StepResult(
                name="test_branch_step", success=True, output="test_output"
            )

            await executor_core._handle_conditional_step(
                mock_conditional_step,
                data="test_data",
                context=TestContext(),
                resources=None,
                limits=None,
                context_setter=mock_context_setter,
            )

            # Verify context setter was called
            mock_context_setter.assert_called_once()

    async def test_multiple_branch_steps_execution(self, executor_core, mock_conditional_step):
        """Test execution of multiple steps in a branch."""
        step1 = Mock(spec=Step)
        step1.name = "step1"
        step1.agent = Mock()
        step1.config = Mock()
        step1.config.max_retries = 1

        step2 = Mock(spec=Step)
        step2.name = "step2"
        step2.agent = Mock()
        step2.config = Mock()
        step2.config.max_retries = 1

        mock_conditional_step.branches["branch_a"].steps = [step1, step2]

        with patch.object(executor_core, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.side_effect = [
                StepResult(name="step1", success=True, output="step1_output"),
                StepResult(name="step2", success=True, output="step2_output"),
            ]

            result = await executor_core._handle_conditional_step(
                mock_conditional_step,
                data="test_data",
                context=None,
                resources=None,
                limits=None,
                context_setter=None,
            )

            # Verify both steps were executed
            assert mock_execute.call_count == 2
            assert result.success is True
            assert result.output == "step2_output"

    async def test_telemetry_logging(
        self, executor_core, mock_conditional_step, isolated_telemetry
    ):
        """Test that telemetry logging is properly implemented."""
        mock_conditional_step.branches["branch_a"].steps = []

        with patch.object(executor_core, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = StepResult(
                name="test_branch_step", success=True, output="test_output"
            )

            await executor_core._handle_conditional_step(
                mock_conditional_step,
                data="test_data",
                context=None,
                resources=None,
                limits=None,
                context_setter=None,
            )

            # Verify info logging was called
            assert any(
                "Condition evaluated to branch key 'branch_a'" in msg
                for msg in isolated_telemetry.infos
            )

    async def test_error_handling_with_context(self, executor_core, mock_conditional_step):
        """Test error handling when context is provided."""
        from flujo.domain.models import BaseModel

        class TestContext(BaseModel):
            test: str = "value"

        mock_conditional_step.condition_callable.side_effect = Exception("Test error")
        context = TestContext()

        result = await executor_core._handle_conditional_step(
            mock_conditional_step,
            data="test_data",
            context=context,
            resources=None,
            limits=None,
            context_setter=None,
        )

        assert result.success is False
        assert "Error executing conditional logic or branch" in result.feedback

    async def test_branch_execution_with_resources(
        self, executor_core, mock_conditional_step, mock_branch_step
    ):
        """Test branch execution with resources parameter."""
        mock_conditional_step.branches["branch_a"].steps = [mock_branch_step]
        resources = {"api_key": "test_key"}

        with patch.object(executor_core, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = StepResult(
                name="test_branch_step", success=True, output="test_output"
            )

            await executor_core._handle_conditional_step(
                mock_conditional_step,
                data="test_data",
                context=None,
                resources=resources,
                limits=None,
                context_setter=None,
            )

            # Verify resources were passed to execute
            mock_execute.assert_called_once()
            call_args = mock_execute.call_args
            assert call_args[1]["resources"] == resources

    async def test_branch_execution_with_limits(
        self, executor_core, mock_conditional_step, mock_branch_step
    ):
        """Test branch execution with usage limits."""
        mock_conditional_step.branches["branch_a"].steps = [mock_branch_step]
        limits = UsageLimits(total_cost_usd_limit=1.0)

        with patch.object(executor_core, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = StepResult(
                name="test_branch_step", success=True, output="test_output"
            )

            await executor_core._handle_conditional_step(
                mock_conditional_step,
                data="test_data",
                context=None,
                resources=None,
                limits=limits,
                context_setter=None,
            )

            # Verify limits were passed to execute
            mock_execute.assert_called_once()
            call_args = mock_execute.call_args
            assert call_args[1]["limits"] == limits
