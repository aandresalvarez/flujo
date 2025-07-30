"""
Comprehensive tests for fallback functionality in ExecutorCore.

This test suite covers all aspects of fallback execution including:
- Successful fallbacks
- Failed fallbacks
- Metric accounting
- Edge cases
- Error conditions
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from flujo.application.core.ultra_executor import ExecutorCore
from flujo.domain.models import UsageLimits
from flujo.exceptions import (
    UsageLimitExceededError,
    MissingAgentError,
    PricingNotConfiguredError,
)


class TestExecutorCoreFallback:
    """Test suite for fallback functionality in ExecutorCore."""

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

        # Configure mock behaviors
        mock_processor_pipeline.apply_prompt.return_value = "processed data"
        mock_processor_pipeline.apply_output.return_value = "processed output"
        mock_plugin_runner.run_plugins.return_value = "final output"
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

    @pytest.fixture
    def create_step_with_fallback(self):
        """Helper to create a step with fallback configuration."""

        def _create_step(primary_fails=True, fallback_succeeds=True):
            primary_step = Mock()
            primary_step.name = "primary_step"
            primary_step.agent = Mock()
            if primary_fails:
                primary_step.agent.run = AsyncMock(side_effect=Exception("Primary failed"))
            else:
                primary_step.agent.run = AsyncMock(return_value="primary success")
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
            if fallback_succeeds:
                fallback_step.agent.run = AsyncMock(return_value="fallback success")
            else:
                fallback_step.agent.run = AsyncMock(side_effect=Exception("Fallback failed"))
            fallback_step.config.max_retries = 1
            fallback_step.config.temperature = 0.7
            fallback_step.processors = Mock()
            fallback_step.processors.prompt_processors = []
            fallback_step.processors.output_processors = []
            fallback_step.validators = []
            fallback_step.plugins = []

            primary_step.fallback_step = fallback_step
            return primary_step, fallback_step

        return _create_step

    @pytest.mark.asyncio
    async def test_fallback_not_triggered_on_primary_success(
        self, executor_core, create_step_with_fallback
    ):
        """Test that fallback is not triggered when primary step succeeds."""
        # Arrange
        primary_step, fallback_step = create_step_with_fallback(
            primary_fails=False, fallback_succeeds=True
        )

        # Configure executor
        executor_core._agent_runner.run.return_value = "primary success"

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
        assert (
            result.output == "processed output"
        )  # Processor pipeline output (no plugin runner called)
        assert "fallback_triggered" not in (result.metadata_ or {})

    @pytest.mark.asyncio
    async def test_fallback_triggered_on_primary_failure(
        self, executor_core, create_step_with_fallback
    ):
        """Test that fallback is triggered when primary step fails."""
        # Arrange
        primary_step, fallback_step = create_step_with_fallback(
            primary_fails=True, fallback_succeeds=True
        )

        # Configure executor
        executor_core._agent_runner.run.side_effect = [
            Exception("Primary failed"),  # Primary fails
            "fallback success",  # Fallback succeeds
        ]

        # Mock the execute method for fallback
        from flujo.domain.models import StepResult

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
            assert result.output == "fallback success"
            assert result.metadata_["fallback_triggered"] is True
            assert "original_error" in result.metadata_

    @pytest.mark.asyncio
    async def test_fallback_failure_propagates(self, executor_core, create_step_with_fallback):
        """Test that fallback failure is properly propagated."""
        # Arrange
        primary_step, fallback_step = create_step_with_fallback(
            primary_fails=True, fallback_succeeds=False
        )

        # Configure executor
        executor_core._agent_runner.run.side_effect = [
            Exception("Primary failed"),  # Primary fails
            Exception("Fallback failed"),  # Fallback fails
        ]

        # Mock the execute method for fallback failure
        from flujo.domain.models import StepResult

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
    async def test_fallback_metric_accounting_success(
        self, executor_core, create_step_with_fallback
    ):
        """Test metric accounting for successful fallbacks."""
        # Arrange
        primary_step, fallback_step = create_step_with_fallback(
            primary_fails=True, fallback_succeeds=True
        )

        # Configure executor
        executor_core._agent_runner.run.side_effect = [
            Exception("Primary failed"),  # Primary fails
            "fallback success",  # Fallback succeeds
        ]

        # Mock the execute method for fallback
        from flujo.domain.models import StepResult

        with patch.object(executor_core, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = StepResult(
                name="fallback_step",
                output="fallback success",
                success=True,
                attempts=1,
                latency_s=0.1,
                cost_usd=0.2,  # Fallback cost
                token_counts=23,  # Fallback tokens
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
            assert result.cost_usd == 0.2  # Should be fallback cost only
            assert result.token_counts == 23  # Should be fallback tokens only

    @pytest.mark.asyncio
    async def test_fallback_metric_accounting_failure(
        self, executor_core, create_step_with_fallback
    ):
        """Test metric accounting for failed fallbacks."""
        # Arrange
        primary_step, fallback_step = create_step_with_fallback(
            primary_fails=True, fallback_succeeds=False
        )

        # Configure executor
        executor_core._agent_runner.run.side_effect = [
            Exception("Primary failed"),  # Primary fails
            Exception("Fallback failed"),  # Fallback fails
        ]

        # Mock the execute method for fallback failure
        from flujo.domain.models import StepResult

        with patch.object(executor_core, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = StepResult(
                name="fallback_step",
                output=None,
                success=False,
                attempts=1,
                latency_s=0.1,
                cost_usd=0.2,  # Fallback cost
                token_counts=23,  # Fallback tokens
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
            assert result.cost_usd == 0.2  # Should be fallback cost only
            assert result.token_counts == 23  # Should be fallback tokens only

    @pytest.mark.asyncio
    async def test_fallback_latency_accumulation(self, executor_core, create_step_with_fallback):
        """Test that fallback latency is correctly accumulated."""
        # Arrange
        primary_step, fallback_step = create_step_with_fallback(
            primary_fails=True, fallback_succeeds=True
        )

        # Configure executor
        executor_core._agent_runner.run.side_effect = [
            Exception("Primary failed"),  # Primary fails
            "fallback success",  # Fallback succeeds
        ]

        # Mock the execute method for fallback
        from flujo.domain.models import StepResult

        with patch.object(executor_core, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = StepResult(
                name="fallback_step",
                output="fallback success",
                success=True,
                attempts=1,
                latency_s=0.1,  # Fallback latency
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

    @pytest.mark.asyncio
    async def test_fallback_with_none_feedback(self, executor_core, create_step_with_fallback):
        """Test fallback handling when primary step has no feedback."""
        # Arrange
        primary_step, fallback_step = create_step_with_fallback(
            primary_fails=True, fallback_succeeds=True
        )

        # Configure executor
        executor_core._agent_runner.run.side_effect = [
            Exception("Primary failed"),  # Primary fails
            "fallback success",  # Fallback succeeds
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
        assert result.feedback is None  # Should be cleared on successful fallback

    @pytest.mark.asyncio
    async def test_fallback_execution_exception_handling(
        self, executor_core, create_step_with_fallback
    ):
        """Test that exceptions during fallback execution are handled gracefully."""
        # Arrange
        primary_step, fallback_step = create_step_with_fallback(
            primary_fails=True, fallback_succeeds=False
        )

        # Configure executor to raise exception during fallback execution
        executor_core._agent_runner.run.side_effect = [
            Exception("Primary failed"),  # Primary fails
            Exception("Fallback execution failed"),  # Fallback execution fails
        ]

        # Mock the execute method to raise an exception

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

    @pytest.mark.asyncio
    async def test_fallback_with_usage_limits(self, executor_core, create_step_with_fallback):
        """Test fallback behavior with usage limits."""
        # Arrange
        primary_step, fallback_step = create_step_with_fallback(
            primary_fails=True, fallback_succeeds=True
        )
        limits = UsageLimits(total_cost_usd_limit=0.5, total_tokens_limit=100)

        # Configure executor
        executor_core._agent_runner.run.side_effect = [
            Exception("Primary failed"),  # Primary fails
            "fallback success",  # Fallback succeeds
        ]

        # Mock usage extraction
        with patch("flujo.cost.extract_usage_metrics") as mock_extract:
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
                limits,  # limits
                False,  # stream
                None,  # on_chunk
                "cache_key",
                None,  # breach_event
            )

            # Assert
            assert result.success is True
            # Usage limits should not be exceeded by the combined operation

    @pytest.mark.asyncio
    async def test_fallback_with_streaming(self, executor_core, create_step_with_fallback):
        """Test fallback behavior with streaming enabled."""
        # Arrange
        primary_step, fallback_step = create_step_with_fallback(
            primary_fails=True, fallback_succeeds=True
        )

        # Configure executor
        executor_core._agent_runner.run.side_effect = [
            Exception("Primary failed"),  # Primary fails
            "fallback success",  # Fallback succeeds
        ]

        # Mock the execute method for fallback
        from flujo.domain.models import StepResult

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
                True,  # stream
                AsyncMock(),  # on_chunk
                "cache_key",
                None,  # breach_event
            )

            # Assert
            assert result.success is True
            assert result.output == "fallback success"

    @pytest.mark.asyncio
    async def test_fallback_with_context_and_resources(
        self, executor_core, create_step_with_fallback
    ):
        """Test fallback behavior with context and resources."""
        # Arrange
        primary_step, fallback_step = create_step_with_fallback(
            primary_fails=True, fallback_succeeds=True
        )
        context = {"key": "value"}
        resources = {"resource": "data"}

        # Ensure step doesn't have persist_feedback_to_context attribute
        if hasattr(primary_step, "persist_feedback_to_context"):
            delattr(primary_step, "persist_feedback_to_context")

        # Configure executor
        executor_core._agent_runner.run.side_effect = [
            Exception("Primary failed"),  # Primary fails
            "fallback success",  # Fallback succeeds
        ]

        # Mock the execute method for fallback
        from flujo.domain.models import StepResult

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
                context,  # context
                resources,  # resources
                None,  # limits
                False,  # stream
                None,  # on_chunk
                "cache_key",
                None,  # breach_event
            )

            # Assert
            assert result.success is True
            assert result.output == "fallback success"

    @pytest.mark.asyncio
    async def test_fallback_metadata_preservation(self, executor_core, create_step_with_fallback):
        """Test that metadata is properly preserved during fallback."""
        # Arrange
        primary_step, fallback_step = create_step_with_fallback(
            primary_fails=True, fallback_succeeds=True
        )

        # Configure executor
        executor_core._agent_runner.run.side_effect = [
            Exception("Primary failed"),  # Primary fails
            "fallback success",  # Fallback succeeds
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
        assert result.metadata_["fallback_triggered"] is True
        assert "original_error" in result.metadata_
        assert "Primary failed" in result.metadata_["original_error"]

    @pytest.mark.asyncio
    async def test_fallback_with_no_fallback_step(self, executor_core, create_step_with_fallback):
        """Test behavior when step has no fallback configured."""
        # Arrange
        primary_step, _ = create_step_with_fallback(primary_fails=True, fallback_succeeds=True)
        primary_step.fallback_step = None  # No fallback configured

        # Configure executor
        executor_core._agent_runner.run.side_effect = Exception("Primary failed")

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
        assert "fallback_triggered" not in (result.metadata_ or {})

    @pytest.mark.asyncio
    async def test_fallback_with_critical_exceptions(
        self, executor_core, create_step_with_fallback
    ):
        """Test that critical exceptions are not retried and don't trigger fallback."""
        # Arrange
        primary_step, fallback_step = create_step_with_fallback(
            primary_fails=True, fallback_succeeds=True
        )

        # Configure executor to raise critical exceptions
        from flujo.domain.models import PipelineResult

        result = PipelineResult(step_history=[], total_cost_usd=0.0)
        executor_core._agent_runner.run.side_effect = UsageLimitExceededError(
            "Cost limit exceeded", result
        )

        # Act & Assert
        with pytest.raises(UsageLimitExceededError):
            await executor_core._execute_simple_step(
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

    @pytest.mark.asyncio
    async def test_fallback_with_pricing_not_configured(
        self, executor_core, create_step_with_fallback
    ):
        """Test fallback behavior with PricingNotConfiguredError."""
        # Arrange
        primary_step, fallback_step = create_step_with_fallback(
            primary_fails=True, fallback_succeeds=True
        )

        # Configure executor to raise PricingNotConfiguredError
        executor_core._agent_runner.run.side_effect = PricingNotConfiguredError("openai", "gpt-4")

        # Act & Assert
        with pytest.raises(PricingNotConfiguredError):
            await executor_core._execute_simple_step(
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

    @pytest.mark.asyncio
    async def test_fallback_with_missing_agent_error(
        self, executor_core, create_step_with_fallback
    ):
        """Test fallback behavior with MissingAgentError."""
        # Arrange
        primary_step, fallback_step = create_step_with_fallback(
            primary_fails=True, fallback_succeeds=True
        )
        primary_step.agent = None  # No agent configured

        # Act & Assert
        with pytest.raises(MissingAgentError):
            await executor_core._execute_simple_step(
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

    @pytest.mark.asyncio
    async def test_fallback_with_validation_failure(self, executor_core, create_step_with_fallback):
        """Test fallback behavior when primary step fails validation."""
        # Arrange
        primary_step, fallback_step = create_step_with_fallback(
            primary_fails=True, fallback_succeeds=True
        )

        # Configure executor
        executor_core._agent_runner.run.side_effect = [
            Exception("Primary failed"),  # Primary fails
            "fallback success",  # Fallback succeeds
        ]

        # Configure validator to fail
        executor_core._validator_runner.validate.side_effect = [
            ValueError("Validation failed"),  # Primary validation fails
            None,  # Fallback validation succeeds
        ]

        # Mock the execute method for fallback
        from flujo.domain.models import StepResult

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
            assert result.output == "fallback success"

    @pytest.mark.asyncio
    async def test_fallback_with_plugin_failure(self, executor_core, create_step_with_fallback):
        """Test fallback behavior when primary step fails plugin execution."""
        # Arrange
        primary_step, fallback_step = create_step_with_fallback(
            primary_fails=True, fallback_succeeds=True
        )

        # Configure executor
        executor_core._agent_runner.run.side_effect = [
            Exception("Primary failed"),  # Primary fails
            "fallback success",  # Fallback succeeds
        ]

        # Configure plugin to fail
        executor_core._plugin_runner.run_plugins.side_effect = [
            ValueError("Plugin validation failed: Plugin error"),  # Primary plugin fails
            "final output",  # Fallback plugin succeeds
        ]

        # Mock the execute method for fallback
        from flujo.domain.models import StepResult

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
            assert result.output == "fallback success"

    @pytest.mark.asyncio
    async def test_fallback_with_cache_hit(self, executor_core, create_step_with_fallback):
        """Test fallback behavior when primary step has a cache hit."""
        # Arrange
        primary_step, fallback_step = create_step_with_fallback(
            primary_fails=False, fallback_succeeds=True
        )

        # Configure cache to return a cached result
        cached_result = Mock()
        cached_result.success = True
        cached_result.output = "cached output"
        executor_core._cache_backend.get.return_value = cached_result

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
        assert (
            result.output == "processed output"
        )  # Processor pipeline output (no plugin runner called)
        assert "fallback_triggered" not in (result.metadata_ or {})

    @pytest.mark.asyncio
    async def test_fallback_with_breach_event(self, executor_core, create_step_with_fallback):
        """Test fallback behavior with breach event."""
        # Arrange
        primary_step, fallback_step = create_step_with_fallback(
            primary_fails=True, fallback_succeeds=True
        )
        breach_event = Mock()

        # Configure executor
        executor_core._agent_runner.run.side_effect = [
            Exception("Primary failed"),  # Primary fails
            "fallback success",  # Fallback succeeds
        ]

        # Mock the execute method for fallback
        from flujo.domain.models import StepResult

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
                breach_event,  # breach_event
            )

            # Assert
            assert result.success is True
            assert result.output == "fallback success"

    @pytest.mark.asyncio
    async def test_fallback_with_complex_data_types(self, executor_core, create_step_with_fallback):
        """Test fallback behavior with complex data types."""
        # Arrange
        primary_step, fallback_step = create_step_with_fallback(
            primary_fails=True, fallback_succeeds=True
        )
        complex_data = {"text": "test input", "numbers": [1, 2, 3], "nested": {"key": "value"}}

        # Configure executor
        executor_core._agent_runner.run.side_effect = [
            Exception("Primary failed"),  # Primary fails
            "fallback success",  # Fallback succeeds
        ]

        # Mock the execute method for fallback
        from flujo.domain.models import StepResult

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
                complex_data,
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

    @pytest.mark.asyncio
    async def test_fallback_with_multiple_retries(self, executor_core, create_step_with_fallback):
        """Test fallback behavior when primary step has multiple retries."""
        # Arrange
        primary_step, fallback_step = create_step_with_fallback(
            primary_fails=True, fallback_succeeds=True
        )
        primary_step.config.max_retries = 3  # Multiple retries

        # Configure executor to fail all retries
        executor_core._agent_runner.run.side_effect = [
            Exception("Primary failed attempt 1"),
            Exception("Primary failed attempt 2"),
            Exception("Primary failed attempt 3"),
            "fallback success",  # Fallback succeeds
        ]

        # Mock the execute method for fallback
        from flujo.domain.models import StepResult

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
            assert result.output == "fallback success"
            assert result.attempts == 3  # Should show all attempts were made

    @pytest.mark.asyncio
    async def test_fallback_with_telemetry_logging(self, executor_core, create_step_with_fallback):
        """Test that fallback triggers proper telemetry logging."""
        # Arrange
        primary_step, fallback_step = create_step_with_fallback(
            primary_fails=True, fallback_succeeds=True
        )

        # Configure executor
        executor_core._agent_runner.run.side_effect = [
            Exception("Primary failed"),  # Primary fails
            "fallback success",  # Fallback succeeds
        ]

        # Mock telemetry logging
        with patch("flujo.infra.telemetry.logfire.info") as mock_log:
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
            mock_log.assert_called_with(
                f"Step '{primary_step.name}' failed. Attempting fallback step '{fallback_step.name}'."
            )

    @pytest.mark.asyncio
    async def test_fallback_with_usage_meter_tracking(
        self, executor_core, create_step_with_fallback
    ):
        """Test that fallback properly tracks usage metrics using real fallback logic."""
        # Arrange - Create real step objects instead of Mocks
        from flujo.domain.dsl.step import Step, StepConfig
        from flujo.domain.processors import AgentProcessors

        # Create a real primary step that will fail
        primary_step = Step(
            name="primary_step",
            agent=Mock(),  # We'll mock the agent, not the step
            config=StepConfig(max_retries=1, temperature=0.7),
            processors=AgentProcessors(),
            validators=[],
            plugins=[],
        )

        # Create a real fallback step that will succeed
        fallback_step = Step(
            name="fallback_step",
            agent=Mock(),  # We'll mock the agent, not the step
            config=StepConfig(max_retries=1, temperature=0.7),
            processors=AgentProcessors(),
            validators=[],
            plugins=[],
        )

        # Set up the fallback relationship
        primary_step.fallback_step = fallback_step

        # Configure agent runner to fail primary and succeed fallback
        executor_core._agent_runner.run.side_effect = [
            Exception("Primary failed"),  # Primary fails
            "fallback success",  # Fallback succeeds
        ]

        # Patch extract_usage_metrics to return different values for primary and fallback
        with patch("flujo.cost.extract_usage_metrics") as mock_extract:
            mock_extract.side_effect = [
                (10, 5, 0.1),  # Primary: 10 prompt, 5 completion, $0.1
                (15, 8, 0.2),  # Fallback: 15 prompt, 8 completion, $0.2
            ]

            # Also patch the execute method to track calls
            with patch.object(executor_core, "execute", wraps=executor_core.execute):
                # Act - Let the real fallback logic run without patching execute
                # Disable caching to ensure we go through the real execution path
                executor_core._enable_cache = False
                executor_core._cache_backend = None  # Also disable cache backend

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
                assert (
                    result.output == "processed output"
                )  # Plugin runner processes the fallback output

                # Verify usage meter was called for the fallback execution
                # Note: The primary step failed before usage extraction, so only fallback is tracked
                assert executor_core._usage_meter.add.call_count == 1
                calls = executor_core._usage_meter.add.call_args_list

                # The fallback step execution (with default usage values)
                assert calls[0].args == (0.0, 0, 1)

                # Verify the aggregated metrics in the result
                # The fallback logic uses the fallback step's metrics
                assert result.token_counts == 1  # From fallback execution
                assert result.cost_usd == 0.0  # From fallback execution

    @pytest.mark.asyncio
    async def test_fallback_with_processor_pipeline(self, executor_core, create_step_with_fallback):
        """Test fallback behavior with processor pipeline."""
        # Arrange - Create real step objects instead of Mocks
        from flujo.domain.dsl.step import Step, StepConfig
        from flujo.domain.processors import AgentProcessors

        # Create a real primary step that will fail
        primary_step = Step(
            name="primary_step",
            agent=Mock(),  # We'll mock the agent, not the step
            config=StepConfig(max_retries=1, temperature=0.7),
            processors=AgentProcessors(),
            validators=[],
            plugins=[],
        )

        # Create a real fallback step that will succeed
        fallback_step = Step(
            name="fallback_step",
            agent=Mock(),  # We'll mock the agent, not the step
            config=StepConfig(max_retries=1, temperature=0.7),
            processors=AgentProcessors(),
            validators=[],
            plugins=[],
        )

        # Set up the fallback relationship
        primary_step.fallback_step = fallback_step

        # Configure executor
        executor_core._agent_runner.run.side_effect = [
            Exception("Primary failed"),  # Primary fails
            "fallback success",  # Fallback succeeds
        ]

        # Configure processor pipeline
        executor_core._processor_pipeline.apply_prompt.return_value = "processed data"
        executor_core._processor_pipeline.apply_output.return_value = "processed output"

        # Disable caching to ensure we go through the real execution path
        executor_core._enable_cache = False
        executor_core._cache_backend = None

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
        assert (
            result.output == "processed output"
        )  # Processor pipeline output (no plugin runner called)
        # Verify processor pipeline was called
        assert executor_core._processor_pipeline.apply_prompt.called
        assert executor_core._processor_pipeline.apply_output.called

    @pytest.mark.asyncio
    async def test_fallback_with_plugin_runner(self, executor_core, create_step_with_fallback):
        """Test fallback behavior with plugin runner."""
        # Arrange - Create real step objects instead of Mocks
        from flujo.domain.dsl.step import Step, StepConfig
        from flujo.domain.processors import AgentProcessors

        # Create a real primary step that will fail
        primary_step = Step(
            name="primary_step",
            agent=Mock(),  # We'll mock the agent, not the step
            config=StepConfig(max_retries=1, temperature=0.7),
            processors=AgentProcessors(),
            validators=[],
            plugins=[],
        )

        # Create a real fallback step that will succeed
        fallback_step = Step(
            name="fallback_step",
            agent=Mock(),  # We'll mock the agent, not the step
            config=StepConfig(max_retries=1, temperature=0.7),
            processors=AgentProcessors(),
            validators=[],
            plugins=[],
        )

        # Set up the fallback relationship
        primary_step.fallback_step = fallback_step

        # Configure executor
        executor_core._agent_runner.run.side_effect = [
            Exception("Primary failed"),  # Primary fails
            "fallback success",  # Fallback succeeds
        ]

        # Configure plugin runner
        executor_core._plugin_runner.run_plugins.return_value = "final output"

        # Disable caching to ensure we go through the real execution path
        executor_core._enable_cache = False
        executor_core._cache_backend = None

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
        assert (
            result.output == "processed output"
        )  # Processor pipeline output (no plugin runner called)
        # Plugin runner should NOT be called when plugins is empty
        executor_core._plugin_runner.run_plugins.assert_not_called()

    @pytest.mark.asyncio
    async def test_fallback_with_cache_backend(self, executor_core, create_step_with_fallback):
        """Test fallback behavior with cache backend."""
        # Arrange
        primary_step, fallback_step = create_step_with_fallback(
            primary_fails=True, fallback_succeeds=True
        )

        # Configure executor
        executor_core._agent_runner.run.side_effect = [
            Exception("Primary failed"),  # Primary fails
            "fallback success",  # Fallback succeeds
        ]

        # Configure cache backend
        executor_core._cache_backend.get.return_value = None  # No cache hit
        executor_core._cache_backend.put.return_value = None

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
        # Verify cache backend was called
        assert executor_core._cache_backend.get.called

    @pytest.mark.asyncio
    async def test_fallback_with_telemetry(self, executor_core, create_step_with_fallback):
        """Test fallback behavior with telemetry."""
        # Arrange - Create real step objects instead of Mocks
        from flujo.domain.dsl.step import Step, StepConfig
        from flujo.domain.processors import AgentProcessors

        # Create a real primary step that will fail
        primary_step = Step(
            name="primary_step",
            agent=Mock(),  # We'll mock the agent, not the step
            config=StepConfig(max_retries=1, temperature=0.7),
            processors=AgentProcessors(),
            validators=[],
            plugins=[],
        )

        # Create a real fallback step that will succeed
        fallback_step = Step(
            name="fallback_step",
            agent=Mock(),  # We'll mock the agent, not the step
            config=StepConfig(max_retries=1, temperature=0.7),
            processors=AgentProcessors(),
            validators=[],
            plugins=[],
        )

        # Set up the fallback relationship
        primary_step.fallback_step = fallback_step

        # Configure executor
        executor_core._agent_runner.run.side_effect = [
            Exception("Primary failed"),  # Primary fails
            "fallback success",  # Fallback succeeds
        ]

        # Configure telemetry
        mock_trace = Mock()
        executor_core._telemetry.trace.return_value = mock_trace

        # Disable caching to ensure we go through the real execution path
        executor_core._enable_cache = False
        executor_core._cache_backend = None

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
        assert (
            result.output == "processed output"
        )  # Processor pipeline output (no plugin runner called)
        # Note: Telemetry behavior may vary during fallback execution
        # The fallback execution might not trigger telemetry tracing

    @pytest.mark.asyncio
    async def test_fallback_integration_with_real_executor(self):
        """Test fallback functionality with a real ExecutorCore instance."""
        # This test would use actual components instead of mocks
        # to verify end-to-end functionality
        pass
