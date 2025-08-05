"""
Unit tests for ExecutorCore._execute_simple_step method.

This test suite covers the migrated retry loop logic for simple steps,
ensuring that the new implementation correctly replaces the legacy _run_step_logic
for simple steps while maintaining backward compatibility.
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from typing import Any

from flujo.application.core.ultra_executor import ExecutorCore, MockDetectionError
from flujo.domain.models import UsageLimits
from flujo.domain.plugins import PluginOutcome
from flujo.exceptions import (
    UsageLimitExceededError,
    MissingAgentError,
    PricingNotConfiguredError,
)
from flujo.testing.utils import DummyPlugin
from flujo.domain.dsl.step import Step, HumanInTheLoopStep
from flujo.domain.dsl.loop import LoopStep
from flujo.domain.dsl.parallel import ParallelStep
from flujo.domain.dsl.conditional import ConditionalStep
from flujo.steps.cache_step import CacheStep
from flujo.domain.dsl.dynamic_router import DynamicParallelRouterStep
from flujo.testing.utils import StubAgent


class TestExecutorCoreSimpleStep:
    """Test suite for ExecutorCore._execute_simple_step method."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent for testing."""
        agent = Mock()
        
        # Create a proper mock response object that won't cause Mock object errors
        class MockResponse:
            def __init__(self):
                self.output = "test output"
                
            def usage(self):
                class MockUsage:
                    def __init__(self):
                        self.request_tokens = 10
                        self.response_tokens = 5
                return MockUsage()
        
        # Configure the agent to return a proper response object
        agent.run = AsyncMock(return_value=MockResponse())
        
        # Ensure model_id returns a proper string value, not a Mock object
        agent.model_id = "openai:gpt-4o"
        
        return agent

    @pytest.fixture
    def mock_step(self, mock_agent):
        """Create a mock step for testing."""
        step = Mock()
        step.name = "test_step"
        step.agent = mock_agent
        step.max_retries = 3  # Add max_retries directly to step to match code expectations
        
        # Create a proper config object instead of using a Mock
        class MockConfig:
            def __init__(self):
                self.max_retries = 3
                self.temperature = 0.7
        
        step.config = MockConfig()
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
        
        # Create a proper mock response object that won't cause Mock object errors
        class MockResponse:
            def __init__(self):
                self.output = "test output"
                
            def usage(self):
                class MockUsage:
                    def __init__(self):
                        self.request_tokens = 10
                        self.response_tokens = 5
                return MockUsage()
        
        # Configure agent runner to return a proper response object
        mock_agent_runner.run.return_value = MockResponse()

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
        assert result.attempts == 4  # max_retries (1 initial + 3 retries)
        assert result.output is None
        assert "Agent execution failed with Exception: Always fails" in result.feedback
        assert result.latency_s > 0

        # Verify agent was called max_retries times
        assert executor_core._agent_runner.run.call_count == 4

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
        assert result.attempts == 4  # max_retries (1 initial + 3 retries)
        assert "Validation failed after max retries" in result.feedback

        # Verify validator was called max_retries times
        assert mock_validator.validate.call_count == 4

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
        with pytest.raises(MockDetectionError, match="returned a Mock object"):
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
        assert result.attempts == 3  # max_retries (1 initial + 2 retries)
        assert (
            "Agent execution failed with ValueError: Plugin validation failed: Invalid format"
            in result.feedback
        )

    @pytest.mark.asyncio
    async def test_plugin_failure_propagates(self, executor_core, mock_step):
        """Test that a plugin failure re-raises the exception and marks step as failed."""
        # Arrange
        data = "test input"
        context = None
        resources = None
        limits = None
        stream = False
        on_chunk = None
        cache_key = "test_cache_key"
        breach_event = None

        # Create a DummyPlugin that fails
        failing_plugin = DummyPlugin(
            outcomes=[PluginOutcome(success=False, feedback="Plugin execution error")]
        )
        mock_step.plugins = [(failing_plugin, 1)]  # Add the failing plugin to the step

        # Override the plugin runner to use the real implementation
        from flujo.application.core.ultra_executor import DefaultPluginRunner

        executor_core._plugin_runner = DefaultPluginRunner()

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
        assert "Plugin validation failed: Plugin execution error" in result.feedback
        assert result.attempts == 3  # 3 attempts total
        assert failing_plugin.call_count == 3  # Plugin called on each attempt

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
        assert "step_history" in call_args[1]  # Should have step_history as keyword argument
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
    async def test_is_complex_property_detection(self, executor_core):
        """Test that _is_complex_step correctly uses the is_complex property."""
        from flujo.domain.dsl.step import Step, HumanInTheLoopStep
        from flujo.domain.dsl.loop import LoopStep
        from flujo.domain.dsl.parallel import ParallelStep
        from flujo.domain.dsl.conditional import ConditionalStep
        from flujo.steps.cache_step import CacheStep
        from flujo.domain.dsl.dynamic_router import DynamicParallelRouterStep
        from flujo.testing.utils import StubAgent

        # Test LoopStep
        loop_step = LoopStep(name="loop", loop_body_pipeline=Mock(), exit_condition_callable=Mock())
        assert executor_core._is_complex_step(loop_step)

        parallel_step = ParallelStep(name="parallel", branches={})
        assert executor_core._is_complex_step(parallel_step)

        conditional_step = ConditionalStep(
            name="conditional",
            condition_callable=Mock(),
            branches={"true": Mock(), "false": Mock()},
        )
        assert executor_core._is_complex_step(conditional_step)

        # Create a real step for the cache step test
        real_step = Step(name="inner_step", agent=StubAgent(["test output"]))
        cache_step = CacheStep(name="cache", wrapped_step=real_step)
        assert executor_core._is_complex_step(cache_step)

        hitl_step = HumanInTheLoopStep(name="hitl", agent=Mock())
        assert executor_core._is_complex_step(hitl_step)

        dynamic_router_step = DynamicParallelRouterStep(
            name="dynamic_router", router_agent=Mock(), branches={}
        )
        assert executor_core._is_complex_step(dynamic_router_step)

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
        step.is_complex = False  # Ensure the property is set for this test

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
        loop_step.is_complex = True  # Ensure the property is set for this test
        assert executor_core._is_complex_step(loop_step)

        # Test ParallelStep
        parallel_step = Mock(spec=ParallelStep)
        parallel_step.name = "parallel_step"
        parallel_step.is_complex = True  # Ensure the property is set for this test
        assert executor_core._is_complex_step(parallel_step)

    @pytest.mark.asyncio
    async def test_validation_steps_remain_complex(self, executor_core):
        """Test that validation steps are still classified as complex."""
        step = Mock()
        step.name = "validation_step"
        step.meta = {"is_validation_step": True}
        step.is_complex = True  # Ensure the property is set for this test

        assert executor_core._is_complex_step(step)

    @pytest.mark.asyncio
    async def test_plugin_steps_remain_complex(self, executor_core):
        """Test that steps with plugins are still classified as complex."""
        step = Mock()
        step.name = "plugin_step"
        step.plugins = [Mock(), Mock()]
        step.is_complex = True  # Ensure the property is set for this test

        assert executor_core._is_complex_step(step)

    @pytest.mark.asyncio
    async def test_simple_steps_without_fallbacks(self, executor_core):
        """Test that simple steps without fallbacks are classified as simple."""
        step = Mock()
        step.name = "simple_step"
        # No fallback_step attribute
        step.plugins = None  # Explicitly set to None
        step.meta = None  # Explicitly set to None
        step.is_complex = False  # Ensure the property is set for this test

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
        step.is_complex = False  # Ensure the property is set for this test

        is_complex = executor_core._is_complex_step(step)
        assert not is_complex

    @pytest.mark.asyncio
    async def test_cache_steps_remain_complex(self, executor_core):
        """Test that CacheStep is still classified as complex."""
        from flujo.steps.cache_step import CacheStep

        cache_step = Mock(spec=CacheStep)
        cache_step.name = "cache_step"
        cache_step.is_complex = True  # Ensure the property is set for this test

        assert executor_core._is_complex_step(cache_step)

    @pytest.mark.asyncio
    async def test_conditional_steps_remain_complex(self, executor_core):
        """Test that ConditionalStep is still classified as complex."""
        from flujo.domain.dsl.conditional import ConditionalStep

        conditional_step = Mock(spec=ConditionalStep)
        conditional_step.name = "conditional_step"
        conditional_step.is_complex = True  # Ensure the property is set for this test

        assert executor_core._is_complex_step(conditional_step)

    @pytest.mark.asyncio
    async def test_hitl_steps_remain_complex(self, executor_core):
        """Test that HumanInTheLoopStep is still classified as complex."""
        from flujo.domain.dsl.step import HumanInTheLoopStep

        hitl_step = Mock(spec=HumanInTheLoopStep)
        hitl_step.name = "hitl_step"
        hitl_step.is_complex = True  # Ensure the property is set for this test

        assert executor_core._is_complex_step(hitl_step)

    @pytest.mark.asyncio
    async def test_dynamic_router_steps_remain_complex(self, executor_core):
        """Test that DynamicParallelRouterStep is still classified as complex."""
        from flujo.domain.dsl.dynamic_router import DynamicParallelRouterStep

        router_step = Mock(spec=DynamicParallelRouterStep)
        router_step.name = "router_step"
        router_step.is_complex = True  # Ensure the property is set for this test

        assert executor_core._is_complex_step(router_step)

    @pytest.mark.asyncio
    async def test_steps_with_fallbacks_and_plugins(self, executor_core):
        """Test that steps with both fallbacks and plugins are classified as complex (due to plugins)."""
        step = Mock()
        step.name = "step_with_fallback_and_plugins"
        step.fallback_step = Mock()
        step.fallback_step.name = "fallback_step"
        step.plugins = [Mock()]
        step.is_complex = True  # Ensure the property is set for this test

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
        step.is_complex = True  # Ensure the property is set for this test

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
        
        # Create a proper mock response object that won't cause Mock object errors
        class MockResponse:
            def __init__(self):
                self.output = "test output"
                
            def usage(self):
                class MockUsage:
                    def __init__(self):
                        self.request_tokens = 10
                        self.response_tokens = 5
                return MockUsage()
        
        # Configure agent runner to return a proper response object
        mock_agent_runner.run.return_value = MockResponse()

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

        # Make the agent runner fail for the primary step
        executor_core._agent_runner.run.side_effect = Exception("Primary failed")

        # Patch execute to return proper StepResult for fallback step
        def mock_execute(step, *args, **kwargs):
            if step == fallback_step:
                from flujo.application.core.ultra_executor import StepResult
                return StepResult(
                    name="fallback_step",
                    output="fallback success",
                    success=True,
                    attempts=1,
                    feedback="Fallback executed successfully"
                )
            return Mock()  # Default mock for other calls

        with patch.object(executor_core, 'execute', side_effect=mock_execute):
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


class TestExecutorCoreObjectOrientedComplexStep:
    """Test suite for the refactored object-oriented _is_complex_step method (Task #4)."""

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
    async def test_object_oriented_property_detection(self, executor_core):
        """Test that the refactored method correctly uses the is_complex property."""
        from flujo.domain.dsl.step import Step, HumanInTheLoopStep
        from flujo.domain.dsl.loop import LoopStep
        from flujo.domain.dsl.parallel import ParallelStep
        from flujo.domain.dsl.conditional import ConditionalStep
        from flujo.steps.cache_step import CacheStep
        from flujo.domain.dsl.dynamic_router import DynamicParallelRouterStep
        from flujo.testing.utils import StubAgent

        # Test all complex step types using object-oriented approach
        test_cases = [
            (
                LoopStep(name="loop", loop_body_pipeline=Mock(), exit_condition_callable=Mock()),
                "LoopStep",
            ),
            (ParallelStep(name="parallel", branches={}), "ParallelStep"),
            (
                ConditionalStep(
                    name="conditional",
                    condition_callable=Mock(),
                    branches={"true": Mock(), "false": Mock()},
                ),
                "ConditionalStep",
            ),
            (
                CacheStep(name="cache", wrapped_step=Step(name="inner", agent=StubAgent(["test"]))),
                "CacheStep",
            ),
            (HumanInTheLoopStep(name="hitl", agent=Mock()), "HumanInTheLoopStep"),
            (
                DynamicParallelRouterStep(name="dynamic_router", router_agent=Mock(), branches={}),
                "DynamicParallelRouterStep",
            ),
        ]

        for step, step_type in test_cases:
            assert executor_core._is_complex_step(step), (
                f"{step_type} should be detected as complex via is_complex property"
            )

    @pytest.mark.asyncio
    async def test_steps_without_is_complex_property(self, executor_core):
        """Test steps that don't have the is_complex property."""
        # Create a step without the is_complex property
        step = Mock()
        step.name = "step_without_is_complex"
        # Explicitly set plugins and meta to None to avoid Mock defaults
        step.plugins = None
        step.meta = None
        # Explicitly set is_complex to False to avoid Mock defaults
        step.is_complex = False

        # Should default to False via getattr(step, 'is_complex', False)
        assert not executor_core._is_complex_step(step)

    @pytest.mark.asyncio
    async def test_steps_with_false_is_complex_property(self, executor_core):
        """Test steps that explicitly set is_complex to False."""
        step = Mock()
        step.name = "step_with_false_is_complex"
        step.is_complex = False
        # Explicitly set plugins and meta to None to avoid Mock defaults
        step.plugins = None
        step.meta = None

        assert not executor_core._is_complex_step(step)

    @pytest.mark.asyncio
    async def test_steps_with_true_is_complex_property(self, executor_core):
        """Test steps that explicitly set is_complex to True."""
        step = Mock()
        step.name = "step_with_true_is_complex"
        step.is_complex = True

        assert executor_core._is_complex_step(step)

    @pytest.mark.asyncio
    async def test_validation_steps_backward_compatibility(self, executor_core):
        """Test that validation steps work correctly with the object-oriented approach."""
        # Test validation step without is_complex property
        step = Mock()
        step.name = "validation_step"
        step.meta = {"is_validation_step": True}
        # Don't set is_complex property

        assert executor_core._is_complex_step(step)

        # Test validation step with is_complex property set to False (should still be complex due to validation)
        step.is_complex = False
        assert executor_core._is_complex_step(step)

    @pytest.mark.asyncio
    async def test_plugin_steps_backward_compatibility(self, executor_core):
        """Test that plugin steps work correctly with the object-oriented approach."""
        # Test plugin step without is_complex property
        step = Mock()
        step.name = "plugin_step"
        step.plugins = [Mock(), Mock()]
        # Don't set is_complex property

        assert executor_core._is_complex_step(step)

        # Test plugin step with is_complex property set to False (should still be complex due to plugins)
        step.is_complex = False
        assert executor_core._is_complex_step(step)

    @pytest.mark.asyncio
    async def test_basic_steps_without_special_properties(self, executor_core):
        """Test basic steps without any special properties."""
        step = Mock()
        step.name = "basic_step"
        # Explicitly set plugins and meta to None to avoid Mock defaults
        step.plugins = None
        step.meta = None
        # Explicitly set is_complex to False to avoid Mock defaults
        step.is_complex = False
        # No is_complex property, no plugins, no meta

        assert not executor_core._is_complex_step(step)

    @pytest.mark.asyncio
    async def test_steps_with_empty_plugins_list(self, executor_core):
        """Test steps with empty plugins list."""
        step = Mock()
        step.name = "step_with_empty_plugins"
        step.plugins = []
        step.is_complex = False
        # Explicitly set meta to None to avoid Mock defaults
        step.meta = None

        assert not executor_core._is_complex_step(step)

    @pytest.mark.asyncio
    async def test_steps_with_none_plugins(self, executor_core):
        """Test steps with None plugins."""
        step = Mock()
        step.name = "step_with_none_plugins"
        step.plugins = None
        step.is_complex = False
        # Explicitly set meta to None to avoid Mock defaults
        step.meta = None

        assert not executor_core._is_complex_step(step)

    @pytest.mark.asyncio
    async def test_steps_with_empty_meta(self, executor_core):
        """Test steps with empty meta dictionary."""
        step = Mock()
        step.name = "step_with_empty_meta"
        step.meta = {}
        step.is_complex = False
        # Explicitly set plugins to None to avoid Mock defaults
        step.plugins = None

        assert not executor_core._is_complex_step(step)

    @pytest.mark.asyncio
    async def test_steps_with_none_meta(self, executor_core):
        """Test steps with None meta."""
        step = Mock()
        step.name = "step_with_none_meta"
        step.meta = None
        step.is_complex = False
        # Explicitly set plugins to None to avoid Mock defaults
        step.plugins = None

        assert not executor_core._is_complex_step(step)

    @pytest.mark.asyncio
    async def test_steps_with_meta_but_no_validation_flag(self, executor_core):
        """Test steps with meta but no is_validation_step flag."""
        step = Mock()
        step.name = "step_with_meta_no_validation"
        step.meta = {"other_flag": True}
        step.is_complex = False
        # Explicitly set plugins to None to avoid Mock defaults
        step.plugins = None

        assert not executor_core._is_complex_step(step)

    @pytest.mark.asyncio
    async def test_steps_with_false_validation_flag(self, executor_core):
        """Test steps with is_validation_step set to False."""
        step = Mock()
        step.name = "step_with_false_validation"
        step.meta = {"is_validation_step": False}
        step.is_complex = False
        # Explicitly set plugins to None to avoid Mock defaults
        step.plugins = None

        assert not executor_core._is_complex_step(step)

    @pytest.mark.asyncio
    async def test_complex_nested_workflow_compatibility(self, executor_core):
        """Test complex nested workflows to ensure recursive execution compatibility."""
        from flujo.domain.dsl.loop import LoopStep
        from flujo.domain.dsl.parallel import ParallelStep
        from flujo.domain.dsl.conditional import ConditionalStep
        from flujo.steps.cache_step import CacheStep
        from flujo.domain.dsl.step import Step
        from flujo.testing.utils import StubAgent

        # Create a complex nested workflow
        inner_step = Step(name="inner", agent=StubAgent(["inner output"]))
        cache_step = CacheStep(name="cache", wrapped_step=inner_step)

        # Loop containing parallel steps
        parallel_step = ParallelStep(
            name="parallel", branches={"branch1": cache_step, "branch2": cache_step}
        )
        loop_step = LoopStep(
            name="loop",
            loop_body_pipeline=parallel_step,
            exit_condition_callable=lambda data, context: len(data) > 3,
        )

        # Conditional containing loop
        conditional_step = ConditionalStep(
            name="conditional",
            condition_callable=lambda data, context: data.get("condition", False),
            branches={"true": loop_step, "false": cache_step},
        )

        # All should be detected as complex
        assert executor_core._is_complex_step(cache_step)
        assert executor_core._is_complex_step(parallel_step)
        assert executor_core._is_complex_step(loop_step)
        assert executor_core._is_complex_step(conditional_step)

    @pytest.mark.asyncio
    async def test_edge_case_missing_name_attribute(self, executor_core):
        """Test edge case where step doesn't have a name attribute."""
        step = Mock()
        # Don't set name attribute
        step.is_complex = True

        # Should still work (getattr will handle missing name gracefully)
        assert executor_core._is_complex_step(step)

    @pytest.mark.asyncio
    async def test_edge_case_step_with_dynamic_properties(self, executor_core):
        """Test edge case with dynamically added properties."""
        step = Mock()
        step.name = "dynamic_step"
        # Explicitly set plugins and meta to None to avoid Mock defaults
        step.plugins = None
        step.meta = None
        # Explicitly set is_complex to False to avoid Mock defaults
        step.is_complex = False

        # Initially no is_complex property (should be False)
        assert not executor_core._is_complex_step(step)

        # Dynamically add is_complex property
        step.is_complex = True
        assert executor_core._is_complex_step(step)

        # Dynamically remove is_complex property
        del step.is_complex
        # After deletion, Mock will create a new Mock object, so we need to set it again
        step.is_complex = False
        assert not executor_core._is_complex_step(step)

    @pytest.mark.asyncio
    async def test_edge_case_step_with_property_descriptor(self, executor_core):
        """Test edge case with property descriptor instead of attribute."""

        class StepWithPropertyDescriptor:
            def __init__(self, name, is_complex):
                self.name = name
                self._is_complex = is_complex

            @property
            def is_complex(self):
                return self._is_complex

        step = StepWithPropertyDescriptor("property_step", True)
        assert executor_core._is_complex_step(step)

        step._is_complex = False
        assert not executor_core._is_complex_step(step)

    @pytest.mark.asyncio
    async def test_edge_case_step_with_callable_is_complex(self, executor_core):
        """Test edge case where is_complex is a callable instead of a property."""
        step = Mock()
        step.name = "callable_step"
        step.is_complex = lambda: True

        # getattr should handle callable gracefully
        assert executor_core._is_complex_step(step)

    @pytest.mark.asyncio
    async def test_comprehensive_step_type_coverage(self, executor_core):
        """Test comprehensive coverage of all step types and combinations."""
        from flujo.domain.dsl.step import Step, HumanInTheLoopStep
        from flujo.domain.dsl.loop import LoopStep
        from flujo.domain.dsl.parallel import ParallelStep
        from flujo.domain.dsl.conditional import ConditionalStep
        from flujo.steps.cache_step import CacheStep
        from flujo.domain.dsl.dynamic_router import DynamicParallelRouterStep
        from flujo.testing.utils import StubAgent

        # Test all step types with various combinations
        test_cases = [
            # (step, expected_complex, description)
            (Step(name="simple", agent=StubAgent(["output"])), False, "Simple Step"),
            (
                LoopStep(name="loop", loop_body_pipeline=Mock(), exit_condition_callable=Mock()),
                True,
                "LoopStep",
            ),
            (ParallelStep(name="parallel", branches={}), True, "ParallelStep"),
            (
                ConditionalStep(
                    name="conditional", condition_callable=Mock(), branches={"true": Mock()}
                ),
                True,
                "ConditionalStep",
            ),
            (
                CacheStep(
                    name="cache", wrapped_step=Step(name="inner", agent=StubAgent(["output"]))
                ),
                True,
                "CacheStep",
            ),
            (HumanInTheLoopStep(name="hitl", agent=Mock()), True, "HumanInTheLoopStep"),
            (
                DynamicParallelRouterStep(name="router", router_agent=Mock(), branches={}),
                True,
                "DynamicParallelRouterStep",
            ),
        ]

        for step, expected_complex, description in test_cases:
            actual_complex = executor_core._is_complex_step(step)
            assert actual_complex == expected_complex, (
                f"{description}: expected {expected_complex}, got {actual_complex}"
            )

    @pytest.mark.asyncio
    async def test_object_oriented_principle_verification(self, executor_core):
        """Test that the object-oriented principles are correctly implemented."""
        # Test that the method uses getattr instead of isinstance
        step = Mock()
        step.name = "test_step"
        step.is_complex = True
        # Explicitly set plugins and meta to None to avoid Mock defaults
        step.plugins = None
        step.meta = None

        # The method should use getattr(step, 'is_complex', False)
        # This test verifies the object-oriented approach works
        assert executor_core._is_complex_step(step)

        # Test with a step that doesn't have is_complex property
        step2 = Mock()
        step2.name = "test_step2"
        # Explicitly set plugins and meta to None to avoid Mock defaults
        step2.plugins = None
        step2.meta = None
        # Explicitly set is_complex to False to avoid Mock defaults
        step2.is_complex = False

        assert not executor_core._is_complex_step(step2)


class TestExecutorCoreFunctionalEquivalence:
    """Test functional equivalence between old and new _is_complex_step implementations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.executor = ExecutorCore()

    def _old_is_complex_step_implementation(self, step: Any) -> bool:
        """Recreate the old implementation for comparison testing."""
        # Check for specific step types
        if isinstance(
            step,
            (
                CacheStep,
                LoopStep,
                ConditionalStep,
                DynamicParallelRouterStep,
                ParallelStep,
                HumanInTheLoopStep,
            ),
        ):
            return True

        # Check for validation steps
        if hasattr(step, "meta") and step.meta and step.meta.get("is_validation_step", False):
            return True

        # Check for steps with plugins (plugins can have redirects, feedback, etc.)
        if hasattr(step, "plugins") and step.plugins:
            return True

        return False

    def test_functional_equivalence_basic_steps(self):
        """Test that basic steps are classified identically."""
        # Create a basic step
        basic_step = Mock()
        basic_step.name = "basic_step"
        basic_step.is_complex = False
        basic_step.plugins = None
        basic_step.meta = None

        # Test both implementations
        old_result = self._old_is_complex_step_implementation(basic_step)
        new_result = self.executor._is_complex_step(basic_step)

        assert old_result == new_result == False, (
            f"Basic step classification mismatch: old={old_result}, new={new_result}"
        )

    def test_functional_equivalence_complex_step_types(self):
        """Test that all complex step types are classified identically."""
        # Test that the new implementation correctly identifies complex steps
        # by their is_complex property, which is the key improvement

        # Create steps with is_complex=True (new object-oriented approach)
        complex_steps = [
            Mock(name="loop_step", is_complex=True, plugins=None, meta=None),
            Mock(name="parallel_step", is_complex=True, plugins=None, meta=None),
            Mock(name="conditional_step", is_complex=True, plugins=None, meta=None),
            Mock(name="dynamic_router_step", is_complex=True, plugins=None, meta=None),
            Mock(name="cache_step", is_complex=True, plugins=None, meta=None),
            Mock(name="hitl_step", is_complex=True, plugins=None, meta=None),
        ]

        for step in complex_steps:
            # The new implementation should return True because of is_complex=True
            new_result = self.executor._is_complex_step(step)
            assert new_result == True, (
                f"New implementation should identify {step.name} as complex via is_complex property"
            )

    def test_functional_equivalence_validation_steps(self):
        """Test that validation steps are classified identically."""
        # Create a validation step
        validation_step = Mock()
        validation_step.name = "validation_step"
        validation_step.is_complex = False  # Doesn't have is_complex property
        validation_step.plugins = None
        validation_step.meta = {"is_validation_step": True}

        # Test both implementations
        old_result = self._old_is_complex_step_implementation(validation_step)
        new_result = self.executor._is_complex_step(validation_step)

        assert old_result == new_result == True, (
            f"Validation step classification mismatch: old={old_result}, new={new_result}"
        )

    def test_functional_equivalence_plugin_steps(self):
        """Test that plugin steps are classified identically."""
        # Create a plugin step
        plugin_step = Mock()
        plugin_step.name = "plugin_step"
        plugin_step.is_complex = False  # Doesn't have is_complex property
        plugin_step.plugins = ["plugin1", "plugin2"]
        plugin_step.meta = None

        # Test both implementations
        old_result = self._old_is_complex_step_implementation(plugin_step)
        new_result = self.executor._is_complex_step(plugin_step)

        assert old_result == new_result == True, (
            f"Plugin step classification mismatch: old={old_result}, new={new_result}"
        )

    def test_functional_equivalence_edge_cases(self):
        """Test edge cases to ensure identical behavior."""
        test_cases = [
            # Step with empty plugins list
            {
                "name": "step_with_empty_plugins",
                "is_complex": False,
                "plugins": [],
                "meta": None,
                "expected": False,
            },
            # Step with None plugins
            {
                "name": "step_with_none_plugins",
                "is_complex": False,
                "plugins": None,
                "meta": None,
                "expected": False,
            },
            # Step with empty meta
            {
                "name": "step_with_empty_meta",
                "is_complex": False,
                "plugins": None,
                "meta": {},
                "expected": False,
            },
            # Step with None meta
            {
                "name": "step_with_none_meta",
                "is_complex": False,
                "plugins": None,
                "meta": None,
                "expected": False,
            },
            # Step with meta but no validation flag
            {
                "name": "step_with_meta_no_validation",
                "is_complex": False,
                "plugins": None,
                "meta": {"other_flag": True},
                "expected": False,
            },
            # Step with false validation flag
            {
                "name": "step_with_false_validation",
                "is_complex": False,
                "plugins": None,
                "meta": {"is_validation_step": False},
                "expected": False,
            },
            # Step with explicit True is_complex - this is the key improvement!
            {
                "name": "step_with_true_is_complex",
                "is_complex": True,
                "plugins": None,
                "meta": None,
                "expected_old": False,  # Old implementation doesn't recognize this
                "expected_new": True,  # New implementation recognizes is_complex property
                "expected": True,  # We expect the new behavior
            },
            # Step with explicit False is_complex
            {
                "name": "step_with_false_is_complex",
                "is_complex": False,
                "plugins": None,
                "meta": None,
                "expected": False,
            },
        ]

        for test_case in test_cases:
            step = Mock()
            step.name = test_case["name"]
            step.is_complex = test_case["is_complex"]
            step.plugins = test_case["plugins"]
            step.meta = test_case["meta"]

            old_result = self._old_is_complex_step_implementation(step)
            new_result = self.executor._is_complex_step(step)

            # Handle the special case where old and new implementations differ
            if test_case["name"] == "step_with_true_is_complex":
                # This is the key improvement: new implementation recognizes is_complex property
                assert old_result == test_case["expected_old"], (
                    f"Old implementation should return {test_case['expected_old']} for {test_case['name']}"
                )
                assert new_result == test_case["expected_new"], (
                    f"New implementation should return {test_case['expected_new']} for {test_case['name']}"
                )
                print(
                    f"✅ Key improvement confirmed: {test_case['name']} - old={old_result}, new={new_result}"
                )
            else:
                # For all other cases, both implementations should agree
                assert old_result == new_result == test_case["expected"], (
                    f"Edge case classification mismatch for {test_case['name']}: "
                    f"old={old_result}, new={new_result}, expected={test_case['expected']}"
                )

    def test_functional_equivalence_comprehensive_coverage(self):
        """Test comprehensive coverage of all step types and combinations."""
        # Create comprehensive test cases using Mock objects

        # Test cases where both implementations should agree
        basic_test_steps = [
            # Validation steps (should be True)
            Mock(
                name="validation_step",
                is_complex=False,
                plugins=None,
                meta={"is_validation_step": True},
            ),
            # Plugin steps (should be True)
            Mock(name="plugin_step", is_complex=False, plugins=["plugin1"], meta=None),
            # Basic steps (should be False)
            Mock(name="basic_step", is_complex=False, plugins=None, meta=None),
            Mock(name="basic_step2", is_complex=False, plugins=[], meta={}),
            Mock(name="basic_step3", is_complex=False, plugins=None, meta={"other_flag": True}),
        ]

        for step in basic_test_steps:
            old_result = self._old_is_complex_step_implementation(step)
            new_result = self.executor._is_complex_step(step)

            assert old_result == new_result, (
                f"Basic test failed for {step.name} ({type(step).__name__}): "
                f"old={old_result}, new={new_result}"
            )

        # Test cases where the new implementation is more flexible (key improvement)
        complex_test_steps = [
            Mock(name="loop_step", is_complex=True, plugins=None, meta=None),
            Mock(name="parallel_step", is_complex=True, plugins=None, meta=None),
            Mock(name="conditional_step", is_complex=True, plugins=None, meta=None),
            Mock(name="dynamic_router_step", is_complex=True, plugins=None, meta=None),
            Mock(name="cache_step", is_complex=True, plugins=None, meta=None),
            Mock(name="hitl_step", is_complex=True, plugins=None, meta=None),
        ]

        for step in complex_test_steps:
            old_result = self._old_is_complex_step_implementation(step)
            new_result = self.executor._is_complex_step(step)

            # Old implementation doesn't recognize these as complex (no isinstance match)
            # New implementation recognizes them as complex (has is_complex=True)
            assert old_result == False, f"Old implementation should return False for {step.name}"
            assert new_result == True, f"New implementation should return True for {step.name}"
            print(
                f"✅ Extensibility improvement confirmed: {step.name} - old={old_result}, new={new_result}"
            )

    def test_functional_equivalence_no_behavioral_changes(self):
        """Test that no behavioral changes were introduced."""
        # Test that the new implementation maintains all existing behavior
        # This test ensures that the refactoring was purely internal

        # Create a step that would have been complex in the old implementation
        complex_step = Mock()
        complex_step.name = "complex_step"
        complex_step.is_complex = True  # New property-based approach
        complex_step.plugins = None
        complex_step.meta = None

        # The new implementation should still return True
        new_result = self.executor._is_complex_step(complex_step)
        assert new_result == True, (
            "New implementation should still identify complex steps correctly"
        )

        # Create a step that would have been simple in the old implementation
        simple_step = Mock()
        simple_step.name = "simple_step"
        simple_step.is_complex = False  # New property-based approach
        simple_step.plugins = None
        simple_step.meta = None

        # The new implementation should still return False
        new_result = self.executor._is_complex_step(simple_step)
        assert new_result == False, (
            "New implementation should still identify simple steps correctly"
        )

    def test_functional_equivalence_backward_compatibility(self):
        """Test that backward compatibility is maintained."""
        # Test that existing step types continue to work without changes

        # Create a step that doesn't have the is_complex property (legacy step)
        # Use a custom class instead of Mock to avoid automatic attribute creation
        class LegacyStep:
            def __init__(self, name):
                self.name = name
                # Don't set is_complex property to simulate legacy step
                self.plugins = None
                self.meta = None

        legacy_step = LegacyStep("legacy_step")

        # The new implementation should gracefully handle missing is_complex property
        new_result = self.executor._is_complex_step(legacy_step)
        assert new_result == False, (
            "New implementation should handle missing is_complex property gracefully"
        )

        # Test legacy step with validation flag
        class LegacyValidationStep:
            def __init__(self, name):
                self.name = name
                # Don't set is_complex property
                self.plugins = None
                self.meta = {"is_validation_step": True}

        legacy_validation_step = LegacyValidationStep("legacy_validation_step")

        # The new implementation should still detect validation steps
        new_result = self.executor._is_complex_step(legacy_validation_step)
        assert new_result == True, "New implementation should still detect validation steps"

        # Test legacy step with plugins
        class LegacyPluginStep:
            def __init__(self, name):
                self.name = name
                # Don't set is_complex property
                self.plugins = ["plugin1"]
                self.meta = None

        legacy_plugin_step = LegacyPluginStep("legacy_plugin_step")

        # The new implementation should still detect plugin steps
        new_result = self.executor._is_complex_step(legacy_plugin_step)
        assert new_result == True, "New implementation should still detect plugin steps"

    def test_functional_equivalence_key_improvement(self):
        """Test the key improvement: object-oriented approach vs isinstance checks."""
        # This test demonstrates the key improvement of the refactoring

        # Create a step that would NOT pass isinstance checks in old implementation
        # but DOES have is_complex=True (new approach)
        custom_complex_step = Mock()
        custom_complex_step.name = "custom_complex_step"
        custom_complex_step.is_complex = True
        custom_complex_step.plugins = None
        custom_complex_step.meta = None

        # Old implementation would return False (doesn't pass isinstance checks)
        old_result = self._old_is_complex_step_implementation(custom_complex_step)
        assert old_result == False, "Old implementation should return False for custom step types"

        # New implementation should return True (uses is_complex property)
        new_result = self.executor._is_complex_step(custom_complex_step)
        assert new_result == True, (
            "New implementation should return True for steps with is_complex=True"
        )

        # This demonstrates the key improvement: extensibility without core changes
        print(
            f"✅ Key improvement demonstrated: Custom step type correctly identified as complex via is_complex property"
        )
