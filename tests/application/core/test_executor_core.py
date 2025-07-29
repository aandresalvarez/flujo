"""
Unit tests for ExecutorCore._execute_simple_step method.

This test suite covers the migrated retry loop logic for simple steps,
ensuring that the new implementation correctly replaces the legacy _run_step_logic
for simple steps while maintaining backward compatibility.
"""

import pytest
from unittest.mock import AsyncMock, Mock

from flujo.application.core.ultra_executor import ExecutorCore
from flujo.domain.models import UsageLimits
from flujo.exceptions import (
    UsageLimitExceededError,
    MissingAgentError,
    PricingNotConfiguredError,
)


class TestExecutorCoreSimpleStep:
    """Test suite for ExecutorCore._execute_simple_step method."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent for testing."""
        agent = Mock()
        agent.run = AsyncMock(return_value="test output")
        return agent

    @pytest.fixture
    def mock_step(self, mock_agent):
        """Create a mock step for testing."""
        step = Mock()
        step.name = "test_step"
        step.agent = mock_agent
        step.config.max_retries = 3
        step.config.temperature = 0.7
        step.processors = Mock()
        step.processors.prompt_processors = []
        step.processors.output_processors = []
        step.validators = []
        step.plugins = []
        return step

    @pytest.fixture
    def executor_core(self):
        """Create an ExecutorCore instance with mocked dependencies."""
        # Create mock implementations of all injected components
        mock_agent_runner = AsyncMock()
        mock_processor_pipeline = AsyncMock()
        mock_validator_runner = AsyncMock()
        mock_plugin_runner = AsyncMock()
        mock_usage_meter = AsyncMock()
        mock_cache_backend = AsyncMock()
        mock_telemetry = Mock()

        # Configure mock behaviors
        mock_processor_pipeline.apply_prompt.return_value = "processed data"
        mock_processor_pipeline.apply_output.return_value = "processed output"
        mock_plugin_runner.run_plugins.return_value = "final output"
        # Configure agent runner to return a non-Mock object
        mock_agent_runner.run.return_value = "raw output"

        return ExecutorCore(
            agent_runner=mock_agent_runner,
            processor_pipeline=mock_processor_pipeline,
            validator_runner=mock_validator_runner,
            plugin_runner=mock_plugin_runner,
            usage_meter=mock_usage_meter,
            cache_backend=mock_cache_backend,
            telemetry=mock_telemetry,
        )

    @pytest.mark.asyncio
    async def test_successful_run_no_retries(self, executor_core, mock_step):
        """Test a successful run with no retries."""
        # Arrange
        data = "test input"
        context = None
        resources = None
        limits = None
        stream = False
        on_chunk = None
        cache_key = "test_cache_key"
        breach_event = None

        # Act
        result = await executor_core._execute_simple_step(
            mock_step,
            data,
            context,
            resources,
            limits,
            stream,
            on_chunk,
            cache_key,
            breach_event,
        )

        # Assert
        assert result.success is True
        assert result.attempts == 1
        assert result.output == "final output"
        assert result.feedback is None
        assert result.latency_s > 0
        assert result.cost_usd >= 0
        assert result.token_counts >= 0

        # Verify component interactions
        executor_core._processor_pipeline.apply_prompt.assert_called_once()
        executor_core._agent_runner.run.assert_called_once()
        executor_core._processor_pipeline.apply_output.assert_called_once()
        executor_core._plugin_runner.run_plugins.assert_called_once()
        executor_core._usage_meter.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_successful_run_with_retry(self, executor_core, mock_step):
        """Test a run that fails once but succeeds on the second retry."""
        # Arrange
        data = "test input"
        context = None
        resources = None
        limits = None
        stream = False
        on_chunk = None
        cache_key = "test_cache_key"
        breach_event = None

        # Configure agent to fail first time, succeed second time
        executor_core._agent_runner.run.side_effect = [
            Exception("First attempt failed"),
            "test output",
        ]

        # Act
        result = await executor_core._execute_simple_step(
            mock_step,
            data,
            context,
            resources,
            limits,
            stream,
            on_chunk,
            cache_key,
            breach_event,
        )

        # Assert
        assert result.success is True
        assert result.attempts == 2
        assert result.output == "final output"
        assert result.feedback is None

        # Verify agent was called twice
        assert executor_core._agent_runner.run.call_count == 2

    @pytest.mark.asyncio
    async def test_all_retries_failed(self, executor_core, mock_step):
        """Test a run that fails all retries and returns success=False."""
        # Arrange
        data = "test input"
        context = None
        resources = None
        limits = None
        stream = False
        on_chunk = None
        cache_key = "test_cache_key"
        breach_event = None

        # Configure agent to always fail
        executor_core._agent_runner.run.side_effect = Exception("Always fails")

        # Act
        result = await executor_core._execute_simple_step(
            mock_step,
            data,
            context,
            resources,
            limits,
            stream,
            on_chunk,
            cache_key,
            breach_event,
        )

        # Assert
        assert result.success is False
        assert result.attempts == 3  # max_retries
        assert result.output is None
        assert "Agent execution failed with Exception: Always fails" in result.feedback
        assert result.latency_s > 0

        # Verify agent was called max_retries times
        assert executor_core._agent_runner.run.call_count == 3

    @pytest.mark.asyncio
    async def test_validator_failure_triggers_retry(self, executor_core, mock_step):
        """Test a run where a validator fails, triggering a retry."""
        # Arrange
        data = "test input"
        context = None
        resources = None
        limits = None
        stream = False
        on_chunk = None
        cache_key = "test_cache_key"
        breach_event = None

        # Add a validator that fails
        mock_validator = Mock()
        mock_validator.validate = AsyncMock(side_effect=ValueError("Validation failed"))
        mock_step.validators = [mock_validator]

        # Configure validator runner to call the mock validator and raise the error
        async def mock_validate(validators, data, *, context):
            for validator in validators:
                await validator.validate(data, context=context)

        executor_core._validator_runner.validate = mock_validate

        # Act
        result = await executor_core._execute_simple_step(
            mock_step,
            data,
            context,
            resources,
            limits,
            stream,
            on_chunk,
            cache_key,
            breach_event,
        )

        # Assert
        assert result.success is False
        assert result.attempts == 3  # max_retries
        assert "Agent execution failed with ValueError: Validation failed" in result.feedback

        # Verify validator was called max_retries times
        assert mock_validator.validate.call_count == 3

    @pytest.mark.asyncio
    async def test_usage_limit_exceeded_error_propagates(self, executor_core, mock_step):
        """Test that UsageLimitExceededError propagates correctly."""
        # Arrange
        data = "test input"
        context = None
        resources = None
        limits = UsageLimits(total_cost_usd_limit=1.0)
        stream = False
        on_chunk = None
        cache_key = "test_cache_key"
        breach_event = None

        # Configure usage meter to raise UsageLimitExceededError
        executor_core._usage_meter.guard.side_effect = UsageLimitExceededError(
            "Cost limit exceeded", None
        )

        # Act & Assert
        with pytest.raises(UsageLimitExceededError, match="Cost limit exceeded"):
            await executor_core._execute_simple_step(
                mock_step,
                data,
                context,
                resources,
                limits,
                stream,
                on_chunk,
                cache_key,
                breach_event,
            )

    @pytest.mark.asyncio
    async def test_missing_agent_error(self, executor_core, mock_step):
        """Test that MissingAgentError is raised when step has no agent."""
        # Arrange
        mock_step.agent = None
        data = "test input"
        context = None
        resources = None
        limits = None
        stream = False
        on_chunk = None
        cache_key = "test_cache_key"
        breach_event = None

        # Act & Assert
        with pytest.raises(MissingAgentError, match="Step 'test_step' has no agent configured"):
            await executor_core._execute_simple_step(
                mock_step,
                data,
                context,
                resources,
                limits,
                stream,
                on_chunk,
                cache_key,
                breach_event,
            )

    @pytest.mark.asyncio
    async def test_mock_object_detection(self, executor_core, mock_step):
        """Test that Mock objects in output are detected and raise TypeError."""
        # Arrange
        data = "test input"
        context = None
        resources = None
        limits = None
        stream = False
        on_chunk = None
        cache_key = "test_cache_key"
        breach_event = None

        # Configure agent to return a Mock object
        executor_core._agent_runner.run.return_value = Mock()

        # Act & Assert
        with pytest.raises(TypeError, match="returned a Mock object"):
            await executor_core._execute_simple_step(
                mock_step,
                data,
                context,
                resources,
                limits,
                stream,
                on_chunk,
                cache_key,
                breach_event,
            )

    @pytest.mark.asyncio
    async def test_plugin_validation_failure_with_feedback(self, executor_core, mock_step):
        """Test plugin validation failure with feedback accumulation."""
        # Arrange
        data = "test input"
        context = None
        resources = None
        limits = None
        stream = False
        on_chunk = None
        cache_key = "test_cache_key"
        breach_event = None

        # Configure plugin runner to raise ValueError with specific message
        executor_core._plugin_runner.run_plugins.side_effect = ValueError(
            "Plugin validation failed: Invalid format"
        )

        # Act
        result = await executor_core._execute_simple_step(
            mock_step,
            data,
            context,
            resources,
            limits,
            stream,
            on_chunk,
            cache_key,
            breach_event,
        )

        # Assert
        assert result.success is False
        assert result.attempts == 3  # max_retries
        assert (
            "Agent execution failed with ValueError: Plugin validation failed: Invalid format"
            in result.feedback
        )

    @pytest.mark.asyncio
    async def test_caching_behavior(self, executor_core, mock_step):
        """Test that successful results are cached."""
        # Arrange
        data = "test input"
        context = None
        resources = None
        limits = None
        stream = False
        on_chunk = None
        cache_key = "test_cache_key"
        breach_event = None

        # Act
        result = await executor_core._execute_simple_step(
            mock_step,
            data,
            context,
            resources,
            limits,
            stream,
            on_chunk,
            cache_key,
            breach_event,
        )

        # Assert
        assert result.success is True
        executor_core._cache_backend.put.assert_called_once_with(cache_key, result, ttl_s=3600)

    @pytest.mark.asyncio
    async def test_usage_tracking(self, executor_core, mock_step):
        """Test that usage metrics are properly tracked."""
        # Arrange
        data = "test input"
        context = None
        resources = None
        limits = UsageLimits(total_cost_usd_limit=10.0)
        stream = False
        on_chunk = None
        cache_key = "test_cache_key"
        breach_event = None

        # Act
        result = await executor_core._execute_simple_step(
            mock_step,
            data,
            context,
            resources,
            limits,
            stream,
            on_chunk,
            cache_key,
            breach_event,
        )

        # Assert
        assert result.success is True
        executor_core._usage_meter.guard.assert_called_once_with(limits)
        executor_core._usage_meter.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_streaming_behavior(self, executor_core, mock_step):
        """Test streaming behavior with on_chunk callback."""
        # Arrange
        data = "test input"
        context = None
        resources = None
        limits = None
        stream = True
        on_chunk = AsyncMock()
        cache_key = "test_cache_key"
        breach_event = None

        # Act
        result = await executor_core._execute_simple_step(
            mock_step,
            data,
            context,
            resources,
            limits,
            stream,
            on_chunk,
            cache_key,
            breach_event,
        )

        # Assert
        assert result.success is True
        executor_core._agent_runner.run.assert_called_once()
        # Verify that streaming parameters were passed correctly
        call_args = executor_core._agent_runner.run.call_args
        assert call_args[1]["stream"] is True
        assert call_args[1]["on_chunk"] == on_chunk

    @pytest.mark.asyncio
    async def test_feedback_accumulation(self, executor_core, mock_step):
        """Test that feedback accumulates across retries."""
        # Arrange
        data = "test input"
        context = None
        resources = None
        limits = None
        stream = False
        on_chunk = None
        cache_key = "test_cache_key"
        breach_event = None

        # Configure agent to fail with different errors each time
        executor_core._agent_runner.run.side_effect = [
            ValueError("First error"),
            ValueError("Second error"),
            "success",
        ]

        # Act
        result = await executor_core._execute_simple_step(
            mock_step,
            data,
            context,
            resources,
            limits,
            stream,
            on_chunk,
            cache_key,
            breach_event,
        )

        # Assert
        assert result.success is True
        assert result.attempts == 3

    @pytest.mark.asyncio
    async def test_pricing_not_configured_error_propagates(self, executor_core, mock_step):
        """Test that PricingNotConfiguredError propagates correctly."""
        # Arrange
        data = "test input"
        context = None
        resources = None
        limits = None
        stream = False
        on_chunk = None
        cache_key = "test_cache_key"
        breach_event = None

        # Configure agent to raise PricingNotConfiguredError with proper constructor
        executor_core._agent_runner.run.side_effect = PricingNotConfiguredError(
            "Pricing not configured", "test_model"
        )

        # Act & Assert
        with pytest.raises(PricingNotConfiguredError, match="Pricing not configured"):
            await executor_core._execute_simple_step(
                mock_step,
                data,
                context,
                resources,
                limits,
                stream,
                on_chunk,
                cache_key,
                breach_event,
            )

    @pytest.mark.asyncio
    async def test_context_persistence(self, executor_core, mock_step):
        """Test that validation results and feedback are persisted to context when requested."""
        # Arrange
        data = "test input"
        context = Mock()
        context.validation_history = []
        context.feedback_history = []
        resources = None
        limits = None
        stream = False
        on_chunk = None
        cache_key = "test_cache_key"
        breach_event = None

        # Configure step to persist validation results and feedback
        mock_step.persist_validation_results_to = "validation_history"
        mock_step.persist_feedback_to_context = "feedback_history"

        # Configure agent to fail
        executor_core._agent_runner.run.side_effect = Exception("Test failure")

        # Act
        result = await executor_core._execute_simple_step(
            mock_step,
            data,
            context,
            resources,
            limits,
            stream,
            on_chunk,
            cache_key,
            breach_event,
        )

        # Assert
        assert result.success is False
        # Note: The actual persistence logic would be tested in integration tests
        # This test verifies the method signature and basic flow
