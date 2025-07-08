"""Unit tests for the new execution management components."""

import pytest
from unittest.mock import AsyncMock, Mock
from datetime import datetime

from flujo.application.core import (
    ExecutionManager,
    StateManager,
    UsageGovernor,
    StepCoordinator,
    TypeValidator,
)
from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.dsl.step import Step
from flujo.domain.models import (
    PipelineResult,
    StepResult,
    UsageLimits,
    PipelineContext,
)

from flujo.exceptions import UsageLimitExceededError, TypeMismatchError


class TestStateManager:
    """Test the StateManager component."""

    @pytest.fixture
    def state_manager(self):
        return StateManager()

    @pytest.fixture
    def mock_state_backend(self):
        backend = Mock()
        backend.load_state = AsyncMock()
        backend.save_state = AsyncMock()
        return backend

    @pytest.mark.asyncio
    async def test_load_workflow_state_no_backend(self, state_manager):
        """Test loading state when no backend is configured."""
        (
            context,
            output,
            idx,
            created,
            pipeline_name,
            pipeline_version,
        ) = await state_manager.load_workflow_state("test-id")
        assert context is None
        assert output is None
        assert idx == 0
        assert created is None
        assert pipeline_name is None
        assert pipeline_version is None

    @pytest.mark.asyncio
    async def test_load_workflow_state_no_run_id(self, mock_state_backend):
        """Test loading state when no run_id is provided."""
        state_manager = StateManager(mock_state_backend)
        (
            context,
            output,
            idx,
            created,
            pipeline_name,
            pipeline_version,
        ) = await state_manager.load_workflow_state("")
        assert context is None
        assert output is None
        assert idx == 0
        assert created is None
        assert pipeline_name is None
        assert pipeline_version is None

    @pytest.mark.asyncio
    async def test_load_workflow_state_not_found(self, mock_state_backend):
        """Test loading state when state doesn't exist."""
        mock_state_backend.load_state.return_value = None
        state_manager = StateManager(mock_state_backend)

        (
            context,
            output,
            idx,
            created,
            pipeline_name,
            pipeline_version,
        ) = await state_manager.load_workflow_state("test-id")
        assert context is None
        assert output is None
        assert idx == 0
        assert created is None
        assert pipeline_name is None
        assert pipeline_version is None

    @pytest.mark.asyncio
    async def test_persist_workflow_state_no_backend(self, state_manager):
        """Test persisting state when no backend is configured."""
        await state_manager.persist_workflow_state(
            run_id="test-id",
            context=PipelineContext(initial_prompt="test"),
            current_step_index=1,
            last_step_output="output",
            status="running",
        )
        # Should not raise any exceptions

    @pytest.mark.asyncio
    async def test_persist_workflow_state_no_run_id(self, mock_state_backend):
        """Test persisting state when no run_id is provided."""
        state_manager = StateManager(mock_state_backend)
        await state_manager.persist_workflow_state(
            run_id=None,
            context=PipelineContext(initial_prompt="test"),
            current_step_index=1,
            last_step_output="output",
            status="running",
        )
        mock_state_backend.save_state.assert_not_called()

    def test_get_run_id_from_context(self, state_manager):
        """Test extracting run_id from context."""
        context = PipelineContext(initial_prompt="test")
        context.run_id = "test-run-id"

        run_id = state_manager.get_run_id_from_context(context)
        assert run_id == "test-run-id"

    def test_get_run_id_from_context_none(self, state_manager):
        """Test extracting run_id when context is None."""
        run_id = state_manager.get_run_id_from_context(None)
        assert run_id is None


class TestUsageGovernor:
    """Test the UsageGovernor component."""

    @pytest.fixture
    def usage_governor(self):
        return UsageGovernor()

    @pytest.fixture
    def usage_limits(self):
        return UsageLimits(
            total_cost_usd_limit=10.0,
            total_tokens_limit=1000,
        )

    @pytest.fixture
    def pipeline_result(self):
        result = PipelineResult()
        result.step_history = [
            StepResult(name="step1", output="test", success=True, cost_usd=5.0, token_counts=500),
            StepResult(name="step2", output="test", success=True, cost_usd=3.0, token_counts=300),
        ]
        result.total_cost_usd = 8.0
        return result

    def test_check_usage_limits_no_limits(self, usage_governor, pipeline_result):
        """Test usage limit checking when no limits are configured."""
        # Should not raise any exceptions
        usage_governor.check_usage_limits(pipeline_result, None)

    def test_check_usage_limits_cost_exceeded(self, usage_limits, pipeline_result):
        """Test usage limit checking when cost limit is exceeded."""
        usage_governor = UsageGovernor(usage_limits)
        pipeline_result.total_cost_usd = 15.0

        with pytest.raises(UsageLimitExceededError, match="Cost limit of \\$10.0 exceeded"):
            usage_governor.check_usage_limits(pipeline_result, None)

    def test_check_usage_limits_tokens_exceeded(self, usage_limits, pipeline_result):
        """Test usage limit checking when token limit is exceeded."""
        usage_governor = UsageGovernor(usage_limits)
        pipeline_result.step_history[0].token_counts = 600
        pipeline_result.step_history[1].token_counts = 500

        with pytest.raises(UsageLimitExceededError, match="Token limit of 1000 exceeded"):
            usage_governor.check_usage_limits(pipeline_result, None)

    def test_update_telemetry_span(self, usage_limits, pipeline_result):
        """Test updating telemetry span with usage metrics."""
        usage_governor = UsageGovernor(usage_limits)
        mock_span = Mock()

        usage_governor.update_telemetry_span(mock_span, pipeline_result)

        mock_span.set_attribute.assert_any_call("flujo.total_cost_usd", 8.0)
        mock_span.set_attribute.assert_any_call("flujo.step_count", 2)
        mock_span.set_attribute.assert_any_call("flujo.total_tokens", 800)


class TestTypeValidator:
    """Test the TypeValidator component."""

    @pytest.fixture
    def type_validator(self):
        return TypeValidator()

    @pytest.fixture
    def mock_step(self):
        step = Mock(spec=Step)
        step.name = "test_step"
        step.__step_input_type__ = str
        step.__step_output_type__ = str
        return step

    def test_validate_step_output_no_next_step(self, type_validator, mock_step):
        """Test type validation when there's no next step."""
        # Should not raise any exceptions
        type_validator.validate_step_output(mock_step, "output", None)

    def test_validate_step_output_compatible_types(self, type_validator, mock_step):
        """Test type validation with compatible types."""
        next_step = Mock(spec=Step)
        next_step.name = "next_step"
        next_step.__step_input_type__ = str

        # Should not raise any exceptions
        type_validator.validate_step_output(mock_step, "string output", next_step)

    def test_validate_step_output_incompatible_types(self, type_validator, mock_step):
        """Test type validation with incompatible types."""
        next_step = Mock(spec=Step)
        next_step.name = "next_step"
        next_step.__step_input_type__ = int

        with pytest.raises(TypeMismatchError, match="Type mismatch"):
            type_validator.validate_step_output(mock_step, "string output", next_step)

    def test_validate_step_output_none_value(self, type_validator, mock_step):
        """Test type validation with None values."""
        next_step = Mock(spec=Step)
        next_step.name = "next_step"
        next_step.__step_input_type__ = str

        # Should raise TypeMismatchError when None is passed to str type
        with pytest.raises(TypeMismatchError, match="Type mismatch"):
            type_validator.validate_step_output(mock_step, None, next_step)

    def test_get_step_input_type(self, type_validator, mock_step):
        """Test getting step input type."""
        mock_step.__step_input_type__ = str
        input_type = type_validator.get_step_input_type(mock_step)
        assert input_type is str

    def test_get_step_output_type(self, type_validator, mock_step):
        """Test getting step output type."""
        mock_step.__step_output_type__ = int
        output_type = type_validator.get_step_output_type(mock_step)
        assert output_type is int


class TestStepCoordinator:
    """Test the StepCoordinator component."""

    @pytest.fixture
    def step_coordinator(self):
        return StepCoordinator()

    @pytest.fixture
    def mock_step(self):
        step = Mock(spec=Step)
        step.name = "test_step"
        step.__step_input_type__ = str
        step.__step_output_type__ = str
        return step

    @pytest.fixture
    def mock_step_executor(self):
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_execute_step_success(self, step_coordinator, mock_step, mock_step_executor):
        """Test successful step execution."""
        step_result = StepResult(name="test_step", output="success", success=True)

        # Create a proper async generator mock
        async def mock_executor(*args, **kwargs):
            yield step_result

        # Replace the mock with the actual function
        step_coordinator._run_step = mock_executor

        results = []
        async for item in step_coordinator.execute_step(
            mock_step, "input", None, step_executor=mock_executor
        ):
            results.append(item)

        assert len(results) == 1
        assert results[0] == step_result

    @pytest.mark.asyncio
    async def test_execute_step_failure(self, step_coordinator, mock_step, mock_step_executor):
        """Test failed step execution."""
        step_result = StepResult(name="test_step", output=None, success=False, feedback="error")

        # Create a proper async generator mock
        async def mock_executor(*args, **kwargs):
            yield step_result

        results = []
        async for item in step_coordinator.execute_step(
            mock_step, "input", None, step_executor=mock_executor
        ):
            results.append(item)

        assert len(results) == 1
        assert results[0] == step_result

    def test_update_pipeline_result(self, step_coordinator):
        """Test updating pipeline result with step result."""
        result = PipelineResult()
        step_result = StepResult(name="test", output="test", success=True, cost_usd=1.0)

        step_coordinator.update_pipeline_result(result, step_result)

        assert len(result.step_history) == 1
        assert result.step_history[0] == step_result
        assert result.total_cost_usd == 1.0


class TestExecutionManager:
    """Test the ExecutionManager component."""

    @pytest.fixture
    def mock_pipeline(self):
        pipeline = Mock(spec=Pipeline)
        step1 = Mock(spec=Step)
        step1.name = "step1"
        step1.__step_input_type__ = str
        step1.__step_output_type__ = str
        step2 = Mock(spec=Step)
        step2.name = "step2"
        step2.__step_input_type__ = str
        step2.__step_output_type__ = str
        pipeline.steps = [step1, step2]
        return pipeline

    @pytest.fixture
    def execution_manager(self, mock_pipeline):
        return ExecutionManager(mock_pipeline)

    @pytest.fixture
    def mock_step_executor(self):
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_execute_steps_basic(self, execution_manager, mock_step_executor):
        """Test basic step execution."""
        step1_result = StepResult(name="step1", output="output1", success=True)
        step2_result = StepResult(name="step2", output="output2", success=True)

        # Create a proper async generator mock that handles both steps
        async def mock_executor(step, data, context, resources, stream=False):
            if step.name == "step1":
                yield step1_result
            elif step.name == "step2":
                yield step2_result

        result = PipelineResult()
        results = []
        async for item in execution_manager.execute_steps(
            start_idx=0,
            data="input",
            context=None,
            result=result,
            step_executor=mock_executor,
        ):
            results.append(item)

        assert len(results) == 0  # No streaming output
        assert len(result.step_history) == 2  # Both steps in the pipeline were executed

    def test_set_final_context(self, execution_manager):
        """Test setting final context."""
        result = PipelineResult()
        context = PipelineContext(initial_prompt="test")

        execution_manager.set_final_context(result, context)

        assert result.final_pipeline_context == context

    @pytest.mark.asyncio
    async def test_persist_final_state(self, execution_manager):
        """Test persisting final state."""
        result = PipelineResult()
        result.step_history = [StepResult(name="step1", output="final", success=True)]
        context = PipelineContext(initial_prompt="test")

        # Mock the state manager
        execution_manager.state_manager = Mock()
        execution_manager.state_manager.persist_workflow_state = AsyncMock()
        execution_manager.state_manager.get_run_id_from_context.return_value = "test-run"

        await execution_manager.persist_final_state(
            run_id="test-run",
            context=context,
            result=result,
            start_idx=0,
            state_created_at=datetime.now(),
            final_status="completed",
        )

        execution_manager.state_manager.persist_workflow_state.assert_called_once()
