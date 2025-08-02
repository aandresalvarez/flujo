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
        step.fallback_step = None  # Explicitly set to None to avoid triggering fallback logic
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

        executor = ExecutorCore(
            agent_runner=mock_agent_runner,
            processor_pipeline=mock_processor_pipeline,
            validator_runner=mock_validator_runner,
            plugin_runner=mock_plugin_runner,
            usage_meter=mock_usage_meter,
            cache_backend=mock_cache_backend,
            telemetry=mock_telemetry,
        )

        return executor

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
        assert result.output == "processed output"
        assert result.feedback is None
        assert result.latency_s > 0
        assert result.cost_usd >= 0
        assert result.token_counts >= 0

        # Verify component interactions
        executor_core._processor_pipeline.apply_prompt.assert_called_once()
        executor_core._agent_runner.run.assert_called_once()
        executor_core._processor_pipeline.apply_output.assert_called_once()
        # Plugin runner should NOT be called when step.plugins is empty
        executor_core._plugin_runner.run_plugins.assert_not_called()
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
        assert result.output == "processed output"
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

        # Set up step with actual plugins to test plugin validation failure
        mock_step.plugins = [Mock(), Mock()]  # Add actual plugins

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
    async def test_plugin_runner_not_called_when_plugins_empty(self, executor_core, mock_step):
        """Test that plugin runner is not called when step.plugins is empty."""
        # Arrange
        data = "test input"
        context = None
        resources = None
        limits = None
        stream = False
        on_chunk = None
        cache_key = "test_cache_key"
        breach_event = None

        # Ensure step has empty plugins list
        mock_step.plugins = []

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
        assert result.output == "processed output"
        # Plugin runner should NOT be called when plugins is empty
        executor_core._plugin_runner.run_plugins.assert_not_called()

    @pytest.mark.asyncio
    async def test_plugin_runner_not_called_when_plugins_none(self, executor_core, mock_step):
        """Test that plugin runner is not called when step.plugins is None."""
        # Arrange
        data = "test input"
        context = None
        resources = None
        limits = None
        stream = False
        on_chunk = None
        cache_key = "test_cache_key"
        breach_event = None

        # Ensure step has None plugins
        mock_step.plugins = None

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
        assert result.output == "processed output"
        # Plugin runner should NOT be called when plugins is None
        executor_core._plugin_runner.run_plugins.assert_not_called()

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
        # Check that guard was called with limits and step_history
        executor_core._usage_meter.guard.assert_called_once()
        call_args = executor_core._usage_meter.guard.call_args
        assert call_args[0][0] == limits  # First argument should be limits
        assert 'step_history' in call_args[1]  # Should have step_history as keyword argument
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


class TestExecutorCoreComplexStepClassification:
    """Test suite for ExecutorCore._is_complex_step method (FSD 6.1)."""

    @pytest.fixture
    def executor_core(self):
        """Create an ExecutorCore instance with mocked dependencies."""
        mock_agent_runner = AsyncMock()
        mock_processor_pipeline = AsyncMock()
        mock_validator_runner = AsyncMock()
        mock_plugin_runner = AsyncMock()
        mock_usage_meter = AsyncMock()
        mock_cache_backend = AsyncMock()
        mock_telemetry = Mock()

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
    async def test_fallback_steps_are_simple(self, executor_core):
        """Test that steps with fallbacks are classified as simple."""
        # Arrange
        step = Mock()
        step.name = "test_step"
        step.fallback_step = Mock()
        step.fallback_step.name = "fallback_step"

        # Configure step to not have plugins or meta (which would make it complex)
        step.plugins = None  # Explicitly set to None
        step.meta = None  # Explicitly set to None

        # Act
        is_complex = executor_core._is_complex_step(step)

        # Assert
        assert not is_complex, "Steps with fallbacks should be classified as simple"

    @pytest.mark.asyncio
    async def test_complex_steps_remain_complex(self, executor_core):
        """Test that truly complex steps are still classified as complex."""
        from flujo.domain.dsl.loop import LoopStep
        from flujo.domain.dsl.parallel import ParallelStep

        # Test LoopStep
        loop_step = Mock(spec=LoopStep)
        loop_step.name = "loop_step"
        assert executor_core._is_complex_step(loop_step)

        # Test ParallelStep
        parallel_step = Mock(spec=ParallelStep)
        parallel_step.name = "parallel_step"
        assert executor_core._is_complex_step(parallel_step)

    @pytest.mark.asyncio
    async def test_validation_steps_remain_complex(self, executor_core):
        """Test that validation steps are still classified as complex."""
        step = Mock()
        step.name = "validation_step"
        step.meta = {"is_validation_step": True}

        assert executor_core._is_complex_step(step)

    @pytest.mark.asyncio
    async def test_plugin_steps_remain_complex(self, executor_core):
        """Test that steps with plugins are still classified as complex."""
        step = Mock()
        step.name = "plugin_step"
        step.plugins = [Mock(), Mock()]

        assert executor_core._is_complex_step(step)

    @pytest.mark.asyncio
    async def test_simple_steps_without_fallbacks(self, executor_core):
        """Test that simple steps without fallbacks are classified as simple."""
        step = Mock()
        step.name = "simple_step"
        # No fallback_step attribute
        step.plugins = None  # Explicitly set to None
        step.meta = None  # Explicitly set to None

        is_complex = executor_core._is_complex_step(step)
        assert not is_complex

    @pytest.mark.asyncio
    async def test_steps_with_none_fallback(self, executor_core):
        """Test that steps with None fallback are classified as simple."""
        step = Mock()
        step.name = "step_with_none_fallback"
        step.fallback_step = None
        step.plugins = None  # Explicitly set to None
        step.meta = None  # Explicitly set to None

        is_complex = executor_core._is_complex_step(step)
        assert not is_complex

    @pytest.mark.asyncio
    async def test_cache_steps_remain_complex(self, executor_core):
        """Test that CacheStep is still classified as complex."""
        from flujo.steps.cache_step import CacheStep

        cache_step = Mock(spec=CacheStep)
        cache_step.name = "cache_step"

        assert executor_core._is_complex_step(cache_step)

    @pytest.mark.asyncio
    async def test_conditional_steps_remain_complex(self, executor_core):
        """Test that ConditionalStep is still classified as complex."""
        from flujo.domain.dsl.conditional import ConditionalStep

        conditional_step = Mock(spec=ConditionalStep)
        conditional_step.name = "conditional_step"

        assert executor_core._is_complex_step(conditional_step)

    @pytest.mark.asyncio
    async def test_hitl_steps_remain_complex(self, executor_core):
        """Test that HumanInTheLoopStep is still classified as complex."""
        from flujo.domain.dsl.step import HumanInTheLoopStep

        hitl_step = Mock(spec=HumanInTheLoopStep)
        hitl_step.name = "hitl_step"

        assert executor_core._is_complex_step(hitl_step)

    @pytest.mark.asyncio
    async def test_dynamic_router_steps_remain_complex(self, executor_core):
        """Test that DynamicParallelRouterStep is still classified as complex."""
        from flujo.domain.dsl.dynamic_router import DynamicParallelRouterStep

        router_step = Mock(spec=DynamicParallelRouterStep)
        router_step.name = "router_step"

        assert executor_core._is_complex_step(router_step)

    @pytest.mark.asyncio
    async def test_steps_with_fallbacks_and_plugins(self, executor_core):
        """Test that steps with both fallbacks and plugins are classified as complex (due to plugins)."""
        step = Mock()
        step.name = "step_with_fallback_and_plugins"
        step.fallback_step = Mock()
        step.fallback_step.name = "fallback_step"
        step.plugins = [Mock()]

        # Should be complex due to plugins, not fallback
        assert executor_core._is_complex_step(step)

    @pytest.mark.asyncio
    async def test_steps_with_fallbacks_and_validation(self, executor_core):
        """Test that steps with both fallbacks and validation are classified as complex (due to validation)."""
        step = Mock()
        step.name = "step_with_fallback_and_validation"
        step.fallback_step = Mock()
        step.fallback_step.name = "fallback_step"
        step.meta = {"is_validation_step": True}

        # Should be complex due to validation, not fallback
        assert executor_core._is_complex_step(step)


class TestExecutorCoreFallbackLogic:
    """Test suite for fallback logic in ExecutorCore._execute_simple_step method."""

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
    async def test_successful_fallback_execution(self, executor_core):
        """Test that successful fallbacks work correctly in ExecutorCore."""
        # Arrange
        primary_step = Mock()
        primary_step.name = "primary_step"
        primary_step.agent = Mock()
        primary_step.agent.run = AsyncMock(side_effect=Exception("Primary failed"))
        primary_step.config.max_retries = 1
        primary_step.config.temperature = 0.7
        primary_step.processors = Mock()
        primary_step.processors.prompt_processors = []
        primary_step.processors.output_processors = []
        primary_step.validators = []
        primary_step.plugins = []

        fallback_step = Mock()
        fallback_step.name = "fallback_step"
        fallback_step.agent = Mock()
        fallback_step.agent.run = AsyncMock(return_value="fallback success")
        fallback_step.config.max_retries = 1
        fallback_step.config.temperature = 0.7
        fallback_step.processors = Mock()
        fallback_step.processors.prompt_processors = []
        fallback_step.processors.output_processors = []
        fallback_step.validators = []
        fallback_step.plugins = []

        primary_step.fallback_step = fallback_step

        # Configure executor to handle fallback
        executor_core._agent_runner.run.side_effect = [
            Exception("Primary failed"),  # Primary fails
            "fallback success",  # Fallback succeeds
        ]

        # Configure the mocked execute method for fallback
        from flujo.domain.models import StepResult
        from unittest.mock import patch

        with patch.object(executor_core, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = StepResult(
                name="fallback_step",
                output="fallback success",
                success=True,
                attempts=1,
                latency_s=0.1,
                cost_usd=0.2,
                token_counts=23,  # 15 + 8
                feedback=None,
            )

            # Act
            result = await executor_core._execute_simple_step(
                primary_step,
                "test data",
                None,  # context
                None,  # resources
                None,  # limits
                False,  # stream
                None,  # on_chunk
                "cache_key",
                None,  # breach_event
            )

            # Assert
            assert result.success is True
            assert result.output == "fallback success"
            assert result.feedback is None
            assert result.metadata_["fallback_triggered"] is True
            assert "original_error" in result.metadata_

    @pytest.mark.asyncio
    async def test_failed_fallback_execution(self, executor_core):
        """Test that failed fallbacks work correctly in ExecutorCore."""
        # Arrange
        primary_step = Mock()
        primary_step.name = "primary_step"
        primary_step.agent = Mock()
        primary_step.agent.run = AsyncMock(side_effect=Exception("Primary failed"))
        primary_step.config.max_retries = 1
        primary_step.config.temperature = 0.7
        primary_step.processors = Mock()
        primary_step.processors.prompt_processors = []
        primary_step.processors.output_processors = []
        primary_step.validators = []
        primary_step.plugins = []

        fallback_step = Mock()
        fallback_step.name = "fallback_step"
        fallback_step.agent = Mock()
        fallback_step.agent.run = AsyncMock(side_effect=Exception("Fallback failed"))
        fallback_step.config.max_retries = 1
        fallback_step.config.temperature = 0.7
        fallback_step.processors = Mock()
        fallback_step.processors.prompt_processors = []
        fallback_step.processors.output_processors = []
        fallback_step.validators = []
        fallback_step.plugins = []

        primary_step.fallback_step = fallback_step

        # Configure executor to handle both failures
        executor_core._agent_runner.run.side_effect = [
            Exception("Primary failed"),  # Primary fails
            Exception("Fallback failed"),  # Fallback fails
        ]

        # Configure the mocked execute method for failed fallback
        from flujo.domain.models import StepResult
        from unittest.mock import patch

        with patch.object(executor_core, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = StepResult(
                name="fallback_step",
                output=None,
                success=False,
                attempts=1,
                latency_s=0.1,
                cost_usd=0.2,
                token_counts=23,
                feedback="Fallback error: Fallback failed",
            )

            # Act
            result = await executor_core._execute_simple_step(
                primary_step,
                "test data",
                None,  # context
                None,  # resources
                None,  # limits
                False,  # stream
                None,  # on_chunk
                "cache_key",
                None,  # breach_event
            )

            # Assert
            assert result.success is False
            assert "Original error" in result.feedback
            assert "Fallback error" in result.feedback

    @pytest.mark.asyncio
    async def test_fallback_metric_accounting(self, executor_core):
        """Test that fallback metrics are correctly accounted for."""
        # Arrange
        primary_step = Mock()
        primary_step.name = "primary_step"
        primary_step.agent = Mock()
        primary_step.agent.run = AsyncMock(side_effect=Exception("Primary failed"))
        primary_step.config.max_retries = 1
        primary_step.config.temperature = 0.7
        primary_step.processors = Mock()
        primary_step.processors.prompt_processors = []
        primary_step.processors.output_processors = []
        primary_step.validators = [Mock()]  # Add a validator to trigger validation
        primary_step.plugins = []

        fallback_step = Mock()
        fallback_step.name = "fallback_step"
        fallback_step.agent = Mock()
        fallback_step.agent.run = AsyncMock(return_value="fallback success")
        fallback_step.config.max_retries = 1
        fallback_step.config.temperature = 0.7
        fallback_step.processors = Mock()
        fallback_step.processors.prompt_processors = []
        fallback_step.processors.output_processors = []
        fallback_step.validators = []
        fallback_step.plugins = []

        primary_step.fallback_step = fallback_step

        # Configure executor to make primary step succeed in agent run but fail in validation
        executor_core._agent_runner.run.side_effect = [
            "primary success",  # Primary succeeds in agent run
            "fallback success",  # Fallback succeeds
        ]

        # Configure validator to fail for primary step
        executor_core._validator_runner.validate.side_effect = [
            ValueError("Validation failed"),  # Primary fails validation
            None,  # Fallback passes validation
        ]

        # Configure processor pipeline to return the agent output
        executor_core._processor_pipeline.apply_output.return_value = "primary success"

        # Configure the mocked execute method for fallback with specific metrics
        from flujo.domain.models import StepResult
        from unittest.mock import patch

        with patch.object(executor_core, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = StepResult(
                name="fallback_step",
                output="fallback success",
                success=True,
                attempts=1,
                latency_s=0.1,
                cost_usd=0.2,  # Fallback cost
                token_counts=23,  # 15 + 8
                feedback=None,
            )

            # Mock usage extraction to return specific values for primary step
            with patch(
                "flujo.application.core.ultra_executor.extract_usage_metrics"
            ) as mock_extract:
                mock_extract.side_effect = [
                    (10, 5, 0.1),  # Primary: 10 prompt, 5 completion, $0.1
                    (15, 8, 0.2),  # Fallback: 15 prompt, 8 completion, $0.2
                ]

                # Act
                result = await executor_core._execute_simple_step(
                    primary_step,
                    "test data",
                    None,  # context
                    None,  # resources
                    None,  # limits
                    False,  # stream
                    None,  # on_chunk
                    "cache_key",
                    None,  # breach_event
                )

                # Assert
                assert result.success is True
                assert result.cost_usd == 0.2  # Should be fallback cost only
                print(f"Expected token counts: 38, Actual: {result.token_counts}")
                print("Primary token counts from mock: 15, Fallback token counts: 23")
                assert result.token_counts == 38  # Should be sum: (10+5) + (15+8) = 38

    @pytest.mark.asyncio
    async def test_fallback_latency_accumulation(self, executor_core):
        """Test that fallback latency is correctly accumulated."""
        # Arrange
        primary_step = Mock()
        primary_step.name = "primary_step"
        primary_step.agent = Mock()
        primary_step.agent.run = AsyncMock(side_effect=Exception("Primary failed"))
        primary_step.config.max_retries = 1
        primary_step.config.temperature = 0.7
        primary_step.processors = Mock()
        primary_step.processors.prompt_processors = []
        primary_step.processors.output_processors = []
        primary_step.validators = []
        primary_step.plugins = []

        fallback_step = Mock()
        fallback_step.name = "fallback_step"
        fallback_step.agent = Mock()
        fallback_step.agent.run = AsyncMock(return_value="fallback success")
        fallback_step.config.max_retries = 1
        fallback_step.config.temperature = 0.7
        fallback_step.processors = Mock()
        fallback_step.processors.prompt_processors = []
        fallback_step.processors.output_processors = []
        fallback_step.validators = []
        fallback_step.plugins = []

        primary_step.fallback_step = fallback_step

        # Configure executor
        executor_core._agent_runner.run.side_effect = [
            Exception("Primary failed"),  # Primary fails
            "fallback success",  # Fallback succeeds
        ]

        # Configure the mocked execute method for fallback
        from flujo.domain.models import StepResult
        from unittest.mock import patch

        with patch.object(executor_core, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = StepResult(
                name="fallback_step",
                output="fallback success",
                success=True,
                attempts=1,
                latency_s=0.1,
                cost_usd=0.2,
                token_counts=23,
                feedback=None,
            )

            # Act
            result = await executor_core._execute_simple_step(
                primary_step,
                "test data",
                None,  # context
                None,  # resources
                None,  # limits
                False,  # stream
                None,  # on_chunk
                "cache_key",
                None,  # breach_event
            )

            # Assert
            assert result.success is True
            assert result.latency_s > 0  # Should have accumulated latency
            # Note: Exact timing depends on execution speed, so we just check it's positive

    @pytest.mark.asyncio
    async def test_fallback_with_none_feedback(self, executor_core):
        """Test fallback handling when primary step has no feedback."""
        # Arrange
        primary_step = Mock()
        primary_step.name = "primary_step"
        primary_step.agent = Mock()
        primary_step.agent.run = AsyncMock(side_effect=Exception("Primary failed"))
        primary_step.config.max_retries = 1
        primary_step.config.temperature = 0.7
        primary_step.processors = Mock()
        primary_step.processors.prompt_processors = []
        primary_step.processors.output_processors = []
        primary_step.validators = []
        primary_step.plugins = []

        fallback_step = Mock()
        fallback_step.name = "fallback_step"
        fallback_step.agent = Mock()
        fallback_step.agent.run = AsyncMock(return_value="fallback success")
        fallback_step.config.max_retries = 1
        fallback_step.config.temperature = 0.7
        fallback_step.processors = Mock()
        fallback_step.processors.prompt_processors = []
        fallback_step.processors.output_processors = []
        fallback_step.validators = []
        fallback_step.plugins = []

        primary_step.fallback_step = fallback_step

        # Configure executor
        executor_core._agent_runner.run.side_effect = [
            Exception("Primary failed"),  # Primary fails
            "fallback success",  # Fallback succeeds
        ]

        # Configure the mocked execute method for fallback
        from flujo.domain.models import StepResult
        from unittest.mock import patch

        with patch.object(executor_core, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = StepResult(
                name="fallback_step",
                output="fallback success",
                success=True,
                attempts=1,
                latency_s=0.1,
                cost_usd=0.2,
                token_counts=23,
                feedback=None,
            )

            # Act
            result = await executor_core._execute_simple_step(
                primary_step,
                "test data",
                None,  # context
                None,  # resources
                None,  # limits
                False,  # stream
                None,  # on_chunk
                "cache_key",
                None,  # breach_event
            )

            # Assert
            assert result.success is True
            assert result.feedback is None  # Should be cleared on successful fallback

    @pytest.mark.asyncio
    async def test_fallback_execution_exception_handling(self, executor_core):
        """Test that exceptions during fallback execution are handled gracefully."""
        # Arrange
        primary_step = Mock()
        primary_step.name = "primary_step"
        primary_step.agent = Mock()
        primary_step.agent.run = AsyncMock(side_effect=Exception("Primary failed"))
        primary_step.config.max_retries = 1
        primary_step.config.temperature = 0.7
        primary_step.processors = Mock()
        primary_step.processors.prompt_processors = []
        primary_step.processors.output_processors = []
        primary_step.validators = []
        primary_step.plugins = []

        fallback_step = Mock()
        fallback_step.name = "fallback_step"
        fallback_step.agent = Mock()
        fallback_step.agent.run = AsyncMock(return_value="fallback success")
        fallback_step.config.max_retries = 1
        fallback_step.config.temperature = 0.7
        fallback_step.processors = Mock()
        fallback_step.processors.prompt_processors = []
        fallback_step.processors.output_processors = []
        fallback_step.validators = []
        fallback_step.plugins = []

        primary_step.fallback_step = fallback_step

        # Configure executor to raise exception during fallback execution
        executor_core._agent_runner.run.side_effect = [
            Exception("Primary failed"),  # Primary fails
            Exception("Fallback execution failed"),  # Fallback execution fails
        ]

        # Configure the mocked execute method to raise an exception
        from unittest.mock import patch

        with patch.object(executor_core, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.side_effect = Exception("Fallback execution failed")

            # Act
            result = await executor_core._execute_simple_step(
                primary_step,
                "test data",
                None,  # context
                None,  # resources
                None,  # limits
                False,  # stream
                None,  # on_chunk
                "cache_key",
                None,  # breach_event
            )

            # Assert
            assert result.success is False
            assert "Fallback execution failed" in result.feedback
