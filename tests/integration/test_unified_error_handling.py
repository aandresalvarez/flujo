"""Integration tests for unified error handling contract."""

import pytest
from flujo import Step, Flujo
from flujo.testing.utils import gather_result


class FailingAgent:
    """Agent that always fails for testing error handling."""

    async def run(self, data, **kwargs):
        raise RuntimeError("Test failure")

    async def stream(self, data, **kwargs):
        yield "partial"
        raise RuntimeError("Test failure")


class MockPlugin:
    """Mock plugin that can be properly awaited."""

    async def validate(self, data, **kwargs):
        # Return a successful validation result
        from flujo.domain.plugins import PluginOutcome

        return PluginOutcome(success=True)


class TestUnifiedErrorHandling:
    """Test that all step types have consistent error handling."""

    @pytest.mark.asyncio
    async def test_simple_step_returns_stepresult_on_failure(self):
        """Test that simple non-streaming steps return StepResult on failure."""
        step = Step.model_validate(
            {"name": "simple", "agent": FailingAgent(), "config": {"max_retries": 1}}
        )
        runner = Flujo(step)

        # Should return PipelineResult with failed StepResult, not raise exception
        result = await gather_result(runner, "test_data")

        assert len(result.step_history) == 1
        step_result = result.step_history[0]
        assert not step_result.success
        assert "Test failure" in step_result.feedback

    @pytest.mark.asyncio
    async def test_streaming_step_returns_stepresult_on_failure(self):
        """Test that streaming steps return StepResult on failure."""
        step = Step.model_validate(
            {"name": "streaming", "agent": FailingAgent(), "config": {"max_retries": 1}}
        )
        runner = Flujo(step)

        # Should return failed StepResult, not raise exception
        async for item in runner.stream_async("test_data"):
            if hasattr(item, "step_history"):
                step_result = item.step_history[0]
                assert not step_result.success
                assert "Test failure" in step_result.feedback
                break

    @pytest.mark.asyncio
    async def test_complex_step_returns_stepresult_on_failure(self):
        """Test that complex steps return StepResult on failure."""
        step = Step.model_validate(
            {
                "name": "complex",
                "agent": FailingAgent(),
                "config": {"max_retries": 1},
                "plugins": [(MockPlugin(), 1)],  # Add plugin to make it complex
            }
        )
        runner = Flujo(step)

        # Should return PipelineResult with failed StepResult, not raise exception
        result = await gather_result(runner, "test_data")

        assert len(result.step_history) == 1
        step_result = result.step_history[0]
        assert not step_result.success
        assert "Test failure" in step_result.feedback

    @pytest.mark.asyncio
    async def test_consistent_api_contract(self):
        """Test that all step types have the same error handling contract."""
        # Create different step types
        simple_step = Step.model_validate(
            {"name": "simple", "agent": FailingAgent(), "config": {"max_retries": 1}}
        )
        streaming_step = Step.model_validate(
            {"name": "streaming", "agent": FailingAgent(), "config": {"max_retries": 1}}
        )
        complex_step = Step.model_validate(
            {
                "name": "complex",
                "agent": FailingAgent(),
                "config": {"max_retries": 1},
                "plugins": [(MockPlugin(), 1)],
            }
        )

        # All should return StepResult, never raise exceptions
        runner = Flujo(simple_step)
        result = await gather_result(runner, "test_data")
        assert not result.step_history[0].success

        # Test with streaming
        runner = Flujo(streaming_step)
        async for item in runner.stream_async("test_data"):
            if hasattr(item, "step_history"):
                assert not item.step_history[0].success
                break

        # Test with complex step
        runner = Flujo(complex_step)
        result = await gather_result(runner, "test_data")
        assert not result.step_history[0].success

    @pytest.mark.asyncio
    async def test_error_information_preservation(self):
        """Test that error information is preserved in StepResult."""
        step = Step.model_validate(
            {"name": "error_test", "agent": FailingAgent(), "config": {"max_retries": 1}}
        )
        runner = Flujo(step)

        result = await gather_result(runner, "test_data")
        step_result = result.step_history[0]

        # Verify error information is preserved
        assert not step_result.success
        assert step_result.feedback is not None
        assert "Test failure" in step_result.feedback
        assert step_result.attempts == 1  # Should reflect actual attempts
        assert step_result.latency_s == 0.0  # Should be set for failed steps

    @pytest.mark.asyncio
    async def test_pipeline_continuation_behavior(self):
        """Test that failed steps don't break pipeline execution flow."""
        # Create a pipeline with a failing step
        failing_step = Step.model_validate(
            {"name": "failing", "agent": FailingAgent(), "config": {"max_retries": 1}}
        )

        # This should complete with the failing step result, not crash
        runner = Flujo(failing_step)
        result = await gather_result(runner, "test_data")

        # Pipeline should complete with failed step
        assert len(result.step_history) == 1
        assert not result.step_history[0].success
        assert result.step_history[0].name == "failing"
