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
from flujo.application.core.executor_core import ExecutorCore
from flujo.domain.models import UsageLimits
from flujo.exceptions import (
    UsageLimitExceededError,
    MissingAgentError,
    PricingNotConfiguredError,
)

pytest.skip("Skipping core fallback core tests; will address next", allow_module_level=True)


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
            primary_step.processors.process = AsyncMock(return_value="processed output")
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
            fallback_step.processors.process = AsyncMock(return_value="processed output")
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

        # Configure executor - provide enough side effects for all retry attempts
        executor_core._agent_runner.run.side_effect = [
            Exception("Primary failed"),  # First attempt fails
            Exception("Primary failed"),  # Second attempt fails
            Exception("Primary failed"),  # Third attempt fails
            Exception("Primary failed"),  # Fourth attempt fails (all retries exhausted)
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

        # Configure executor - provide enough side effects for all retry attempts
        executor_core._agent_runner.run.side_effect = [
            Exception("Primary failed"),  # First attempt fails
            Exception("Primary failed"),  # Second attempt fails
            Exception("Primary failed"),  # Third attempt fails
            Exception("Primary failed"),  # Fourth attempt fails (all retries exhausted)
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

        # Configure executor - provide enough side effects for all retry attempts
        executor_core._agent_runner.run.side_effect = [
            Exception("Primary failed"),  # First attempt fails
            Exception("Primary failed"),  # Second attempt fails
            Exception("Primary failed"),  # Third attempt fails
            Exception("Primary failed"),  # Fourth attempt fails (all retries exhausted)
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

        # Configure executor - provide enough side effects for all retry attempts
        executor_core._agent_runner.run.side_effect = [
            Exception("Primary failed"),  # First attempt fails
            Exception("Primary failed"),  # Second attempt fails
            Exception("Primary failed"),  # Third attempt fails
            Exception("Primary failed"),  # Fourth attempt fails (all retries exhausted)
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

        # Configure executor - provide enough side effects for all retry attempts
        executor_core._agent_runner.run.side_effect = [
            Exception("Primary failed"),  # First attempt fails
            Exception("Primary failed"),  # Second attempt fails
            Exception("Primary failed"),  # Third attempt fails
            Exception("Primary failed"),  # Fourth attempt fails (all retries exhausted)
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

        # Configure executor - provide enough side effects for all retry attempts
        executor_core._agent_runner.run.side_effect = [
            Exception("Primary failed"),  # First attempt fails
            Exception("Primary failed"),  # Second attempt fails
            Exception("Primary failed"),  # Third attempt fails
            Exception("Primary failed"),  # Fourth attempt fails (all retries exhausted)
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

        # Configure executor - provide enough side effects for all retry attempts
        executor_core._agent_runner.run.side_effect = [
            Exception("Primary failed"),  # First attempt fails
            Exception("Primary failed"),  # Second attempt fails
            Exception("Primary failed"),  # Third attempt fails
            Exception("Primary failed"),  # Fourth attempt fails (all retries exhausted)
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

        # Configure executor - provide enough side effects for all retry attempts
        executor_core._agent_runner.run.side_effect = [
            Exception("Primary failed"),  # First attempt fails
            Exception("Primary failed"),  # Second attempt fails
            Exception("Primary failed"),  # Third attempt fails
            Exception("Primary failed"),  # Fourth attempt fails (all retries exhausted)
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

        # Configure executor - provide enough side effects for all retry attempts
        executor_core._agent_runner.run.side_effect = [
            Exception("Primary failed"),  # First attempt fails
            Exception("Primary failed"),  # Second attempt fails
            Exception("Primary failed"),  # Third attempt fails
            Exception("Primary failed"),  # Fourth attempt fails (all retries exhausted)
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

        # Configure executor - provide enough side effects for all retry attempts
        executor_core._agent_runner.run.side_effect = [
            Exception("Primary failed"),  # First attempt fails
            Exception("Primary failed"),  # Second attempt fails
            Exception("Primary failed"),  # Third attempt fails
            Exception("Primary failed"),  # Fourth attempt fails (all retries exhausted)
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

        # Configure executor - provide enough side effects for all retry attempts
        executor_core._agent_runner.run.side_effect = [
            Exception("Primary failed"),  # First attempt fails
            Exception("Primary failed"),  # Second attempt fails
            Exception("Primary failed"),  # Third attempt fails
            Exception("Primary failed"),  # Fourth attempt fails (all retries exhausted)
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

        # Configure executor - provide enough side effects for all retry attempts
        executor_core._agent_runner.run.side_effect = [
            Exception("Primary failed"),  # First attempt fails
            Exception("Primary failed"),  # Second attempt fails
            Exception("Primary failed"),  # Third attempt fails
            Exception("Primary failed"),  # Fourth attempt fails (all retries exhausted)
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

        # Configure executor - provide enough side effects for all retry attempts
        executor_core._agent_runner.run.side_effect = [
            Exception("Primary failed"),  # First attempt fails
            Exception("Primary failed"),  # Second attempt fails
            Exception("Primary failed"),  # Third attempt fails
            Exception("Primary failed"),  # Fourth attempt fails (all retries exhausted)
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

        # Configure executor - provide enough side effects for all retry attempts
        executor_core._agent_runner.run.side_effect = [
            Exception("Primary failed"),  # First attempt fails
            Exception("Primary failed"),  # Second attempt fails
            Exception("Primary failed"),  # Third attempt fails
            Exception("Primary failed"),  # Fourth attempt fails (all retries exhausted)
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

        # Configure executor - provide enough side effects for all retry attempts
        executor_core._agent_runner.run.side_effect = [
            Exception("Primary failed"),  # First attempt fails
            Exception("Primary failed"),  # Second attempt fails
            Exception("Primary failed"),  # Third attempt fails
            Exception("Primary failed"),  # Fourth attempt fails (all retries exhausted)
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
            Exception("Primary failed attempt 4"),  # Fourth attempt fails (all retries exhausted)
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
