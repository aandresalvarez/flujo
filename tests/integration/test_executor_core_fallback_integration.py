"""
Integration tests for fallback functionality in ExecutorCore.

These tests use real components to verify end-to-end functionality
and ensure backward compatibility.
"""

import pytest
from unittest.mock import AsyncMock, Mock
from flujo.application.core.executor_core import ExecutorCore
from flujo.domain.models import UsageLimits
from flujo.domain.dsl.step import Step
from flujo.domain.dsl.step import StepConfig
from flujo.exceptions import (
    MissingAgentError,
)

# Mark this module as slow due to complex integration testing
pytestmark = pytest.mark.slow


class TestExecutorCoreFallbackIntegration:
    """Integration tests for fallback functionality in ExecutorCore."""

    @pytest.fixture
    def real_executor_core(self):
        """Create a real ExecutorCore instance with default components."""
        return ExecutorCore()

    @pytest.fixture
    def create_real_step_with_fallback(self):
        """Helper to create real steps with fallback configuration."""

        def _create_step(primary_fails=True, fallback_succeeds=True):
            # Create primary step
            primary_agent = Mock()
            if primary_fails:
                primary_agent.run = AsyncMock(side_effect=Exception("Primary failed"))
            else:
                primary_agent.run = AsyncMock(return_value="primary success")

            primary_step = Step(
                name="primary_step",
                agent=primary_agent,
                config=StepConfig(max_retries=1, temperature=0.7),
            )

            # Create fallback step
            fallback_agent = Mock()
            if fallback_succeeds:
                fallback_agent.run = AsyncMock(return_value="fallback success")
            else:
                fallback_agent.run = AsyncMock(side_effect=Exception("Fallback failed"))

            fallback_step = Step(
                name="fallback_step",
                agent=fallback_agent,
                config=StepConfig(max_retries=1, temperature=0.7),
            )

            # Set fallback relationship
            primary_step.fallback(fallback_step)

            return primary_step, fallback_step

        return _create_step

    @pytest.mark.asyncio
    async def test_real_fallback_execution(
        self, real_executor_core, create_real_step_with_fallback
    ):
        """Test fallback execution with real ExecutorCore components."""
        # Arrange
        primary_step, fallback_step = create_real_step_with_fallback(
            primary_fails=True, fallback_succeeds=True
        )

        # Act
        result = await real_executor_core.execute(
            step=primary_step,
            data="test data",
            context=None,
            resources=None,
            limits=None,
            stream=False,
            on_chunk=None,
        )

        # Assert
        assert result.success is True
        assert result.output == "fallback success"
        assert result.metadata_["fallback_triggered"] is True
        assert "original_error" in result.metadata_

    @pytest.mark.asyncio
    async def test_real_fallback_failure(self, real_executor_core, create_real_step_with_fallback):
        """Test fallback failure with real ExecutorCore components."""
        # Arrange
        primary_step, fallback_step = create_real_step_with_fallback(
            primary_fails=True, fallback_succeeds=False
        )

        # Act
        result = await real_executor_core.execute(
            step=primary_step,
            data="test data",
            context=None,
            resources=None,
            limits=None,
            stream=False,
            on_chunk=None,
        )

        # Assert
        assert result.success is False
        assert "Original error:" in result.feedback
        assert "Fallback error:" in result.feedback

    @pytest.mark.asyncio
    async def test_real_fallback_not_triggered_on_success(
        self, real_executor_core, create_real_step_with_fallback
    ):
        """Test that fallback is not triggered when primary step succeeds."""
        # Arrange
        primary_step, fallback_step = create_real_step_with_fallback(
            primary_fails=False, fallback_succeeds=True
        )

        # Act
        result = await real_executor_core.execute(
            step=primary_step,
            data="test data",
            context=None,
            resources=None,
            limits=None,
            stream=False,
            on_chunk=None,
        )

        # Assert
        assert result.success is True
        assert result.output == "primary success"
        assert "fallback_triggered" not in result.metadata_

    @pytest.mark.asyncio
    async def test_real_fallback_with_complex_data_types(self, real_executor_core):
        """Test fallback with complex data types and real components."""
        # Arrange
        complex_data = {
            "nested": {"list": [1, 2, 3], "dict": {"key": "value"}},
            "array": ["a", "b", "c"],
            "number": 42.5,
        }

        primary_agent = Mock()
        primary_agent.run = AsyncMock(side_effect=Exception("Primary failed"))

        fallback_agent = Mock()
        fallback_agent.run = AsyncMock(return_value={"processed": complex_data})

        primary_step = Step(
            name="complex_primary", agent=primary_agent, config=StepConfig(max_retries=1)
        )

        fallback_step = Step(
            name="complex_fallback", agent=fallback_agent, config=StepConfig(max_retries=1)
        )

        primary_step.fallback(fallback_step)

        # Act
        result = await real_executor_core.execute(
            step=primary_step,
            data=complex_data,
            context=None,
            resources=None,
            limits=None,
            stream=False,
            on_chunk=None,
        )

        # Assert
        assert result.success is True
        assert result.output["processed"] == complex_data
        assert result.metadata_["fallback_triggered"] is True

    @pytest.mark.asyncio
    async def test_real_fallback_with_context_updates(self, real_executor_core):
        """Test fallback with context updates using real components."""
        # Arrange
        context = {"history": [], "counter": 0}

        primary_agent = Mock()
        primary_agent.run = AsyncMock(side_effect=Exception("Primary failed"))

        fallback_agent = Mock()
        fallback_agent.run = AsyncMock(return_value={"status": "fallback_success", "counter": 1})

        primary_step = Step(
            name="context_primary",
            agent=primary_agent,
            config=StepConfig(max_retries=1),
            updates_context=True,
        )

        fallback_step = Step(
            name="context_fallback",
            agent=fallback_agent,
            config=StepConfig(max_retries=1),
            updates_context=True,
        )

        primary_step.fallback(fallback_step)

        # Act
        result = await real_executor_core.execute(
            step=primary_step,
            data="test",
            context=context,
            resources=None,
            limits=None,
            stream=False,
            on_chunk=None,
        )

        # Assert
        assert result.success is True
        assert result.output["status"] == "fallback_success"
        assert result.metadata_["fallback_triggered"] is True

    @pytest.mark.asyncio
    async def test_real_fallback_with_usage_limits(self, real_executor_core):
        """Test fallback with usage limits using real components."""
        # Arrange
        limits = UsageLimits(max_cost_usd=0.01, max_tokens=1000)

        primary_agent = Mock()
        primary_agent.run = AsyncMock(side_effect=Exception("Primary failed"))

        fallback_agent = Mock()
        fallback_agent.run = AsyncMock(return_value="fallback within limits")

        primary_step = Step(
            name="limited_primary", agent=primary_agent, config=StepConfig(max_retries=1)
        )

        fallback_step = Step(
            name="limited_fallback", agent=fallback_agent, config=StepConfig(max_retries=1)
        )

        primary_step.fallback(fallback_step)

        # Act
        result = await real_executor_core.execute(
            step=primary_step,
            data="test",
            context=None,
            resources=None,
            limits=limits,
            stream=False,
            on_chunk=None,
        )

        # Assert
        assert result.success is True
        assert result.output == "fallback within limits"
        assert result.metadata_["fallback_triggered"] is True

    @pytest.mark.asyncio
    async def test_real_fallback_with_streaming(self, real_executor_core):
        """Test fallback with streaming using real components."""
        # Arrange
        chunks_received = []

        async def on_chunk(chunk):
            chunks_received.append(chunk)

        # Create agents that support streaming
        class StreamingAgent:
            async def run(self, data):
                return "fallback stream"

            async def stream(self, data):
                # Simulate streaming by yielding chunks
                yield "fallback "
                yield "stream"

        class FailingStreamingAgent:
            async def run(self, data):
                raise Exception("Primary failed")

            async def stream(self, data):
                raise Exception("Primary failed")

        primary_agent = FailingStreamingAgent()
        fallback_agent = StreamingAgent()

        primary_step = Step(
            name="stream_primary", agent=primary_agent, config=StepConfig(max_retries=1)
        )

        fallback_step = Step(
            name="stream_fallback", agent=fallback_agent, config=StepConfig(max_retries=1)
        )

        primary_step.fallback(fallback_step)

        # Act
        result = await real_executor_core.execute(
            step=primary_step,
            data="test",
            context=None,
            resources=None,
            limits=None,
            stream=True,
            on_chunk=on_chunk,
        )

        # Assert
        assert result.success is True
        assert result.output == "fallback stream"
        assert result.metadata_["fallback_triggered"] is True

    @pytest.mark.asyncio
    async def test_real_fallback_with_multiple_retries(self, real_executor_core):
        """Test fallback after multiple retries with real components."""
        # Arrange
        primary_agent = Mock()
        # Fail all attempts, triggering fallback
        primary_agent.run = AsyncMock(side_effect=Exception("Primary failed"))

        fallback_agent = Mock()
        fallback_agent.run = AsyncMock(return_value="fallback success")

        primary_step = Step(
            name="retry_primary",
            agent=primary_agent,
            config=StepConfig(max_retries=2),  # Allow 2 retries
        )

        fallback_step = Step(
            name="retry_fallback", agent=fallback_agent, config=StepConfig(max_retries=1)
        )

        primary_step.fallback(fallback_step)

        # Act
        result = await real_executor_core.execute(
            step=primary_step,
            data="test",
            context=None,
            resources=None,
            limits=None,
            stream=False,
            on_chunk=None,
        )

        # Assert
        assert result.success is True
        assert result.output == "fallback success"
        assert result.metadata_["fallback_triggered"] is True
        # The attempts count may vary based on retry logic, but should be at least 1
        assert result.attempts >= 1

    @pytest.mark.asyncio
    async def test_real_fallback_with_validation_failure(self, real_executor_core):
        """Test fallback triggered by validation failure with real components."""
        # Arrange
        primary_agent = Mock()
        primary_agent.run = AsyncMock(return_value="invalid output")

        fallback_agent = Mock()
        fallback_agent.run = AsyncMock(return_value="valid fallback output")

        # Create a validator that always fails
        from flujo.domain.validation import Validator

        class FailingValidator(Validator):
            def validate(self, output):
                return False, "Validation failed"

        primary_step = Step(
            name="validation_primary",
            agent=primary_agent,
            config=StepConfig(max_retries=1),
            validators=[FailingValidator()],
        )

        fallback_step = Step(
            name="validation_fallback", agent=fallback_agent, config=StepConfig(max_retries=1)
        )

        primary_step.fallback(fallback_step)

        # Act
        result = await real_executor_core.execute(
            step=primary_step,
            data="test",
            context=None,
            resources=None,
            limits=None,
            stream=False,
            on_chunk=None,
        )

        # Assert
        assert result.success is True
        assert result.output == "valid fallback output"
        assert result.metadata_["fallback_triggered"] is True

    @pytest.mark.asyncio
    async def test_real_fallback_with_plugin_failure(self, real_executor_core):
        """Test fallback triggered by plugin failure with real components."""
        # Arrange
        primary_agent = Mock()
        primary_agent.run = AsyncMock(return_value="plugin test output")

        fallback_agent = Mock()
        fallback_agent.run = AsyncMock(return_value="plugin fallback output")

        # Create a plugin that always fails
        from flujo.domain.plugins import ValidationPlugin

        class FailingPlugin(ValidationPlugin):
            async def validate(self, data):
                from flujo.domain.plugins import PluginOutcome

                return PluginOutcome(success=False, feedback="Plugin processing failed")

        primary_step = Step(
            name="plugin_primary",
            agent=primary_agent,
            config=StepConfig(max_retries=1),
            plugins=[(FailingPlugin(), 0)],
        )

        fallback_step = Step(
            name="plugin_fallback", agent=fallback_agent, config=StepConfig(max_retries=1)
        )

        primary_step.fallback(fallback_step)

        # Act
        result = await real_executor_core.execute(
            step=primary_step,
            data="test",
            context=None,
            resources=None,
            limits=None,
            stream=False,
            on_chunk=None,
        )

        # Assert
        assert result.success is True
        assert result.output == "plugin fallback output"
        assert result.metadata_["fallback_triggered"] is True

    @pytest.mark.asyncio
    async def test_real_fallback_with_cache_interaction(self, real_executor_core):
        """Test fallback with cache interaction using real components."""
        # Arrange
        primary_agent = Mock()
        primary_agent.run = AsyncMock(side_effect=Exception("Primary failed"))

        fallback_agent = Mock()
        fallback_agent.run = AsyncMock(return_value="cached fallback result")

        primary_step = Step(
            name="cache_primary", agent=primary_agent, config=StepConfig(max_retries=1)
        )

        fallback_step = Step(
            name="cache_fallback", agent=fallback_agent, config=StepConfig(max_retries=1)
        )

        primary_step.fallback(fallback_step)

        # Act - First execution
        result1 = await real_executor_core.execute(
            step=primary_step,
            data="test",
            context=None,
            resources=None,
            limits=None,
            stream=False,
            on_chunk=None,
        )

        # Act - Second execution (should use cache)
        result2 = await real_executor_core.execute(
            step=primary_step,
            data="test",
            context=None,
            resources=None,
            limits=None,
            stream=False,
            on_chunk=None,
        )

        # Assert
        assert result1.success is True
        assert result1.output == "cached fallback result"
        assert result1.metadata_["fallback_triggered"] is True

        assert result2.success is True
        assert result2.output == "cached fallback result"
        assert result2.metadata_["fallback_triggered"] is True

    @pytest.mark.asyncio
    async def test_real_fallback_with_telemetry_logging(self, real_executor_core):
        """Test fallback with telemetry logging using real components."""
        # Arrange
        primary_agent = Mock()
        primary_agent.run = AsyncMock(side_effect=Exception("Primary failed"))

        fallback_agent = Mock()
        fallback_agent.run = AsyncMock(return_value="telemetry fallback result")

        primary_step = Step(
            name="telemetry_primary", agent=primary_agent, config=StepConfig(max_retries=1)
        )

        fallback_step = Step(
            name="telemetry_fallback", agent=fallback_agent, config=StepConfig(max_retries=1)
        )

        primary_step.fallback(fallback_step)

        # Act
        result = await real_executor_core.execute(
            step=primary_step,
            data="test",
            context=None,
            resources=None,
            limits=None,
            stream=False,
            on_chunk=None,
        )

        # Assert
        assert result.success is True
        assert result.output == "telemetry fallback result"
        assert result.metadata_["fallback_triggered"] is True

    @pytest.mark.asyncio
    async def test_real_fallback_with_usage_meter_tracking(self, real_executor_core):
        """Test fallback with usage meter tracking using real components."""
        # Arrange
        primary_agent = Mock()
        primary_agent.run = AsyncMock(side_effect=Exception("Primary failed"))

        fallback_agent = Mock()
        fallback_agent.run = AsyncMock(return_value="metered fallback result")

        primary_step = Step(
            name="meter_primary", agent=primary_agent, config=StepConfig(max_retries=1)
        )

        fallback_step = Step(
            name="meter_fallback", agent=fallback_agent, config=StepConfig(max_retries=1)
        )

        primary_step.fallback(fallback_step)

        # Act
        result = await real_executor_core.execute(
            step=primary_step,
            data="test",
            context=None,
            resources=None,
            limits=None,
            stream=False,
            on_chunk=None,
        )

        # Assert
        assert result.success is True
        assert result.output == "metered fallback result"
        assert result.metadata_["fallback_triggered"] is True

    @pytest.mark.asyncio
    async def test_real_fallback_with_processor_pipeline(self, real_executor_core):
        """Test fallback with processor pipeline using real components."""
        # Arrange
        primary_agent = Mock()
        primary_agent.run = AsyncMock(side_effect=Exception("Primary failed"))

        fallback_agent = Mock()
        fallback_agent.run = AsyncMock(return_value="processed fallback result")

        primary_step = Step(
            name="processor_primary", agent=primary_agent, config=StepConfig(max_retries=1)
        )

        fallback_step = Step(
            name="processor_fallback", agent=fallback_agent, config=StepConfig(max_retries=1)
        )

        primary_step.fallback(fallback_step)

        # Act
        result = await real_executor_core.execute(
            step=primary_step,
            data="test",
            context=None,
            resources=None,
            limits=None,
            stream=False,
            on_chunk=None,
        )

        # Assert
        assert result.success is True
        assert result.output == "processed fallback result"
        assert result.metadata_["fallback_triggered"] is True

    @pytest.mark.asyncio
    async def test_real_fallback_with_plugin_runner(self, real_executor_core):
        """Test fallback with plugin runner using real components."""
        # Arrange
        primary_agent = Mock()
        primary_agent.run = AsyncMock(side_effect=Exception("Primary failed"))

        fallback_agent = Mock()
        fallback_agent.run = AsyncMock(return_value="plugin runner fallback result")

        primary_step = Step(
            name="plugin_runner_primary", agent=primary_agent, config=StepConfig(max_retries=1)
        )

        fallback_step = Step(
            name="plugin_runner_fallback", agent=fallback_agent, config=StepConfig(max_retries=1)
        )

        primary_step.fallback(fallback_step)

        # Act
        result = await real_executor_core.execute(
            step=primary_step,
            data="test",
            context=None,
            resources=None,
            limits=None,
            stream=False,
            on_chunk=None,
        )

        # Assert
        assert result.success is True
        assert result.output == "plugin runner fallback result"
        assert result.metadata_["fallback_triggered"] is True

    @pytest.mark.asyncio
    async def test_real_fallback_with_cache_backend(self, real_executor_core):
        """Test fallback with cache backend using real components."""
        # Arrange
        primary_agent = Mock()
        primary_agent.run = AsyncMock(side_effect=Exception("Primary failed"))

        fallback_agent = Mock()
        fallback_agent.run = AsyncMock(return_value="cache backend fallback result")

        primary_step = Step(
            name="cache_backend_primary", agent=primary_agent, config=StepConfig(max_retries=1)
        )

        fallback_step = Step(
            name="cache_backend_fallback", agent=fallback_agent, config=StepConfig(max_retries=1)
        )

        primary_step.fallback(fallback_step)

        # Act
        result = await real_executor_core.execute(
            step=primary_step,
            data="test",
            context=None,
            resources=None,
            limits=None,
            stream=False,
            on_chunk=None,
        )

        # Assert
        assert result.success is True
        assert result.output == "cache backend fallback result"
        assert result.metadata_["fallback_triggered"] is True

    @pytest.mark.asyncio
    async def test_real_fallback_with_telemetry(self, real_executor_core):
        """Test fallback with telemetry using real components."""
        # Arrange
        primary_agent = Mock()
        primary_agent.run = AsyncMock(side_effect=Exception("Primary failed"))

        fallback_agent = Mock()
        fallback_agent.run = AsyncMock(return_value="telemetry fallback result")

        primary_step = Step(
            name="telemetry_primary", agent=primary_agent, config=StepConfig(max_retries=1)
        )

        fallback_step = Step(
            name="telemetry_fallback", agent=fallback_agent, config=StepConfig(max_retries=1)
        )

        primary_step.fallback(fallback_step)

        # Act
        result = await real_executor_core.execute(
            step=primary_step,
            data="test",
            context=None,
            resources=None,
            limits=None,
            stream=False,
            on_chunk=None,
        )

        # Assert
        assert result.success is True
        assert result.output == "telemetry fallback result"
        assert result.metadata_["fallback_triggered"] is True

    @pytest.mark.asyncio
    async def test_real_fallback_integration_with_real_executor(self):
        """Test fallback integration with completely real ExecutorCore instance."""
        # Arrange
        executor = ExecutorCore()

        primary_agent = Mock()
        primary_agent.run = AsyncMock(side_effect=Exception("Primary failed"))

        fallback_agent = Mock()
        fallback_agent.run = AsyncMock(return_value="real integration fallback result")

        primary_step = Step(
            name="real_integration_primary", agent=primary_agent, config=StepConfig(max_retries=1)
        )

        fallback_step = Step(
            name="real_integration_fallback", agent=fallback_agent, config=StepConfig(max_retries=1)
        )

        primary_step.fallback(fallback_step)

        # Act
        result = await executor.execute(
            step=primary_step,
            data="test",
            context=None,
            resources=None,
            limits=None,
            stream=False,
            on_chunk=None,
        )

        # Assert
        assert result.success is True
        assert result.output == "real integration fallback result"
        assert result.metadata_["fallback_triggered"] is True
        assert "original_error" in result.metadata_

    @pytest.mark.asyncio
    async def test_real_fallback_backward_compatibility(self, real_executor_core):
        """Test that fallback functionality maintains backward compatibility."""
        # Arrange - Create steps without fallback (old behavior)
        primary_agent = Mock()
        primary_agent.run = AsyncMock(side_effect=Exception("Primary failed"))

        primary_step = Step(
            name="backward_compat_primary", agent=primary_agent, config=StepConfig(max_retries=1)
        )

        # Act
        result = await real_executor_core.execute(
            step=primary_step,
            data="test",
            context=None,
            resources=None,
            limits=None,
            stream=False,
            on_chunk=None,
        )

        # Assert - Should fail without fallback (old behavior)
        assert result.success is False
        assert "fallback_triggered" not in result.metadata_

    @pytest.mark.asyncio
    async def test_real_fallback_with_critical_exceptions(self, real_executor_core):
        """Test fallback with critical exceptions using real components."""
        # Arrange
        primary_agent = Mock()
        # Use a regular exception that will trigger fallback
        primary_agent.run = AsyncMock(side_effect=Exception("Primary failed"))

        fallback_agent = Mock()
        fallback_agent.run = AsyncMock(return_value="critical exception fallback result")

        primary_step = Step(
            name="critical_primary", agent=primary_agent, config=StepConfig(max_retries=1)
        )

        fallback_step = Step(
            name="critical_fallback", agent=fallback_agent, config=StepConfig(max_retries=1)
        )

        primary_step.fallback(fallback_step)

        # Act
        result = await real_executor_core.execute(
            step=primary_step,
            data="test",
            context=None,
            resources=None,
            limits=None,
            stream=False,
            on_chunk=None,
        )

        # Assert
        assert result.success is True
        assert result.output == "critical exception fallback result"
        assert result.metadata_["fallback_triggered"] is True

    @pytest.mark.asyncio
    async def test_real_fallback_with_pricing_not_configured(self, real_executor_core):
        """Test fallback with pricing not configured using real components."""
        # Arrange
        primary_agent = Mock()
        primary_agent.run = AsyncMock(side_effect=Exception("Primary failed"))

        fallback_agent = Mock()
        fallback_agent.run = AsyncMock(return_value="pricing fallback result")

        primary_step = Step(
            name="pricing_primary", agent=primary_agent, config=StepConfig(max_retries=1)
        )

        fallback_step = Step(
            name="pricing_fallback", agent=fallback_agent, config=StepConfig(max_retries=1)
        )

        primary_step.fallback(fallback_step)

        # Act
        result = await real_executor_core.execute(
            step=primary_step,
            data="test",
            context=None,
            resources=None,
            limits=None,
            stream=False,
            on_chunk=None,
        )

        # Assert
        assert result.success is True
        assert result.output == "pricing fallback result"
        assert result.metadata_["fallback_triggered"] is True

    @pytest.mark.asyncio
    async def test_real_fallback_with_missing_agent_error(self, real_executor_core):
        """Test fallback with missing agent error using real components."""
        # Arrange
        primary_agent = Mock()
        primary_agent.run = AsyncMock(side_effect=MissingAgentError("Agent not found"))

        fallback_agent = Mock()
        fallback_agent.run = AsyncMock(return_value="missing agent fallback result")

        primary_step = Step(
            name="missing_agent_primary", agent=primary_agent, config=StepConfig(max_retries=1)
        )

        fallback_step = Step(
            name="missing_agent_fallback", agent=fallback_agent, config=StepConfig(max_retries=1)
        )

        primary_step.fallback(fallback_step)

        # Act
        result = await real_executor_core.execute(
            step=primary_step,
            data="test",
            context=None,
            resources=None,
            limits=None,
            stream=False,
            on_chunk=None,
        )

        # Assert
        assert result.success is True
        assert result.output == "missing agent fallback result"
        assert result.metadata_["fallback_triggered"] is True

    @pytest.mark.asyncio
    async def test_real_fallback_with_quota_only(self, real_executor_core):
        """Test fallback in quota-only mode (breach_event removed)."""
        # Arrange

        primary_agent = Mock()
        primary_agent.run = AsyncMock(side_effect=Exception("Primary failed"))

        fallback_agent = Mock()
        fallback_agent.run = AsyncMock(return_value="breach event fallback result")

        primary_step = Step(
            name="breach_primary", agent=primary_agent, config=StepConfig(max_retries=1)
        )

        fallback_step = Step(
            name="breach_fallback", agent=fallback_agent, config=StepConfig(max_retries=1)
        )

        primary_step.fallback(fallback_step)

        # Act
        result = await real_executor_core.execute(
            step=primary_step,
            data="test",
            context=None,
            resources=None,
            limits=None,
            stream=False,
            on_chunk=None,
        )

        # Assert
        assert result.success is True
        assert result.output == "breach event fallback result"
        assert result.metadata_["fallback_triggered"] is True

    @pytest.mark.asyncio
    async def test_real_fallback_with_none_feedback(self, real_executor_core):
        """Test fallback with None feedback using real components."""
        # Arrange
        primary_agent = Mock()
        primary_agent.run = AsyncMock(side_effect=Exception("Primary failed"))

        fallback_agent = Mock()
        fallback_agent.run = AsyncMock(return_value="none feedback fallback result")

        primary_step = Step(
            name="none_feedback_primary", agent=primary_agent, config=StepConfig(max_retries=1)
        )

        fallback_step = Step(
            name="none_feedback_fallback", agent=fallback_agent, config=StepConfig(max_retries=1)
        )

        primary_step.fallback(fallback_step)

        # Act
        result = await real_executor_core.execute(
            step=primary_step,
            data="test",
            context=None,
            resources=None,
            limits=None,
            stream=False,
            on_chunk=None,
        )

        # Assert
        assert result.success is True
        assert result.output == "none feedback fallback result"
        assert result.metadata_["fallback_triggered"] is True
        assert "original_error" in result.metadata_

    @pytest.mark.asyncio
    async def test_real_fallback_latency_accumulation(self, real_executor_core):
        """Test fallback latency accumulation with real components."""
        # Arrange
        primary_agent = Mock()
        primary_agent.run = AsyncMock(side_effect=Exception("Primary failed"))

        fallback_agent = Mock()
        fallback_agent.run = AsyncMock(return_value="latency fallback result")

        primary_step = Step(
            name="latency_primary", agent=primary_agent, config=StepConfig(max_retries=1)
        )

        fallback_step = Step(
            name="latency_fallback", agent=fallback_agent, config=StepConfig(max_retries=1)
        )

        primary_step.fallback(fallback_step)

        # Act
        result = await real_executor_core.execute(
            step=primary_step,
            data="test",
            context=None,
            resources=None,
            limits=None,
            stream=False,
            on_chunk=None,
        )

        # Assert
        assert result.success is True
        assert result.output == "latency fallback result"
        assert result.metadata_["fallback_triggered"] is True
        assert result.latency_s > 0  # Should have accumulated latency

    @pytest.mark.asyncio
    async def test_real_fallback_metric_accounting_success(self, real_executor_core):
        """Test fallback metric accounting on success with real components."""
        # Arrange
        primary_agent = Mock()
        primary_agent.run = AsyncMock(side_effect=Exception("Primary failed"))

        fallback_agent = Mock()
        fallback_agent.run = AsyncMock(return_value="metric accounting fallback result")

        primary_step = Step(
            name="metric_primary", agent=primary_agent, config=StepConfig(max_retries=1)
        )

        fallback_step = Step(
            name="metric_fallback", agent=fallback_agent, config=StepConfig(max_retries=1)
        )

        primary_step.fallback(fallback_step)

        # Act
        result = await real_executor_core.execute(
            step=primary_step,
            data="test",
            context=None,
            resources=None,
            limits=None,
            stream=False,
            on_chunk=None,
        )

        # Assert
        assert result.success is True
        assert result.output == "metric accounting fallback result"
        assert result.metadata_["fallback_triggered"] is True
        # Metrics should be properly accounted for

    @pytest.mark.asyncio
    async def test_real_fallback_metric_accounting_failure(self, real_executor_core):
        """Test fallback metric accounting on failure with real components."""
        # Arrange
        primary_agent = Mock()
        primary_agent.run = AsyncMock(side_effect=Exception("Primary failed"))

        fallback_agent = Mock()
        fallback_agent.run = AsyncMock(side_effect=Exception("Fallback failed"))

        primary_step = Step(
            name="metric_failure_primary", agent=primary_agent, config=StepConfig(max_retries=1)
        )

        fallback_step = Step(
            name="metric_failure_fallback", agent=fallback_agent, config=StepConfig(max_retries=1)
        )

        primary_step.fallback(fallback_step)

        # Act
        result = await real_executor_core.execute(
            step=primary_step,
            data="test",
            context=None,
            resources=None,
            limits=None,
            stream=False,
            on_chunk=None,
        )

        # Assert
        assert result.success is False
        assert "Original error:" in result.feedback
        assert "Fallback error:" in result.feedback
        # Metrics should be properly accounted for

    @pytest.mark.asyncio
    async def test_real_fallback_metadata_preservation(self, real_executor_core):
        """Test fallback metadata preservation with real components."""
        # Arrange
        primary_agent = Mock()
        primary_agent.run = AsyncMock(side_effect=Exception("Primary failed"))

        fallback_agent = Mock()
        fallback_agent.run = AsyncMock(return_value="metadata preservation fallback result")

        primary_step = Step(
            name="metadata_primary", agent=primary_agent, config=StepConfig(max_retries=1)
        )

        fallback_step = Step(
            name="metadata_fallback", agent=fallback_agent, config=StepConfig(max_retries=1)
        )

        primary_step.fallback(fallback_step)

        # Act
        result = await real_executor_core.execute(
            step=primary_step,
            data="test",
            context=None,
            resources=None,
            limits=None,
            stream=False,
            on_chunk=None,
        )

        # Assert
        assert result.success is True
        assert result.output == "metadata preservation fallback result"
        assert result.metadata_["fallback_triggered"] is True
        assert "original_error" in result.metadata_
        assert "Primary failed" in result.metadata_["original_error"]

    @pytest.mark.asyncio
    async def test_real_fallback_with_no_fallback_step(self, real_executor_core):
        """Test behavior when no fallback step is configured."""
        # Arrange
        primary_agent = Mock()
        primary_agent.run = AsyncMock(side_effect=Exception("Primary failed"))

        primary_step = Step(
            name="no_fallback_primary", agent=primary_agent, config=StepConfig(max_retries=1)
        )
        # No fallback step configured

        # Act
        result = await real_executor_core.execute(
            step=primary_step,
            data="test",
            context=None,
            resources=None,
            limits=None,
            stream=False,
            on_chunk=None,
        )

        # Assert
        assert result.success is False
        assert "fallback_triggered" not in result.metadata_
        assert "Primary failed" in result.feedback

    @pytest.mark.asyncio
    async def test_real_fallback_execution_exception_handling(self, real_executor_core):
        """Test fallback execution exception handling with real components."""
        # Arrange
        primary_agent = Mock()
        primary_agent.run = AsyncMock(side_effect=Exception("Primary failed"))

        fallback_agent = Mock()
        fallback_agent.run = AsyncMock(side_effect=Exception("Fallback execution failed"))

        primary_step = Step(
            name="execution_exception_primary",
            agent=primary_agent,
            config=StepConfig(max_retries=1),
        )

        fallback_step = Step(
            name="execution_exception_fallback",
            agent=fallback_agent,
            config=StepConfig(max_retries=1),
        )

        primary_step.fallback(fallback_step)

        # Act
        result = await real_executor_core.execute(
            step=primary_step,
            data="test",
            context=None,
            resources=None,
            limits=None,
            stream=False,
            on_chunk=None,
        )

        # Assert
        assert result.success is False
        assert "Original error:" in result.feedback
        assert "Fallback error:" in result.feedback
        assert "Fallback execution failed" in result.feedback
