"""Integration tests for unified error handling contract."""

import pytest
from flujo import Step
from flujo.testing.utils import gather_result
from flujo.domain.models import StepResult
from flujo.domain.plugins import PluginOutcome
from tests.test_types.fixtures import execute_simple_step


class FailingAgent:
    async def run(self, data, **kwargs):
        raise RuntimeError("Test failure")

    async def stream(self, data, **kwargs):
        yield "partial"
        raise RuntimeError("Test failure")


class CriticalFailingAgent:
    async def run(self, data, **kwargs):
        from flujo.exceptions import PausedException

        raise PausedException("Test pause")

    async def stream(self, data, **kwargs):
        from flujo.exceptions import InfiniteFallbackError

        yield "partial"
        raise InfiniteFallbackError("Test infinite fallback")


class MockPlugin:
    async def validate(self, data: dict) -> PluginOutcome:
        return PluginOutcome(success=True, feedback="Mock validation passed")


@pytest.mark.asyncio
async def test_simple_step_returns_stepresult_on_failure():
    """Test that simple steps return StepResult on regular failures."""
    step = Step.model_validate(
        {"name": "failing", "agent": FailingAgent(), "config": {"max_retries": 1}}
    )

    from flujo.application.core.executor_core import ExecutorCore as UltraStepExecutor

    executor = UltraStepExecutor()

    result = await execute_simple_step(executor, step, "test_data")

    assert isinstance(result, StepResult)
    assert not result.success
    assert "Test failure" in result.feedback
    assert result.latency_s > 0.0  # Timing should be preserved


@pytest.mark.asyncio
async def test_streaming_step_returns_stepresult_on_failure():
    """Test that streaming steps return StepResult on regular failures."""
    step = Step.model_validate(
        {"name": "failing", "agent": FailingAgent(), "config": {"max_retries": 1}}
    )

    from flujo.application.core.executor_core import ExecutorCore as UltraStepExecutor

    executor = UltraStepExecutor()

    result = await execute_simple_step(executor, step, "test_data", stream=True)

    assert isinstance(result, StepResult)
    assert not result.success
    assert "Test failure" in result.feedback
    assert result.latency_s > 0.0  # Timing should be preserved


@pytest.mark.asyncio
async def test_complex_step_returns_stepresult_on_failure():
    """Test that complex steps return StepResult on regular failures."""
    step = Step.model_validate(
        {
            "name": "failing",
            "agent": FailingAgent(),
            "config": {"max_retries": 1},
            "plugins": [(MockPlugin(), 1)],
        }
    )

    from flujo.application.core.executor_core import ExecutorCore as UltraStepExecutor

    executor = UltraStepExecutor()

    result = await execute_simple_step(executor, step, "test_data")

    assert isinstance(result, StepResult)
    assert not result.success
    assert "Test failure" in result.feedback
    assert result.latency_s > 0.0  # Timing should be preserved


@pytest.mark.asyncio
async def test_critical_exceptions_are_re_raised():
    """Test that critical exceptions are re-raised for proper control flow."""
    from flujo.exceptions import PausedException, InfiniteFallbackError

    # Test PausedException
    paused_step = Step.model_validate(
        {"name": "paused", "agent": CriticalFailingAgent(), "config": {"max_retries": 1}}
    )

    from flujo.application.core.executor_core import ExecutorCore as UltraStepExecutor

    executor = UltraStepExecutor()

    with pytest.raises(PausedException, match="Test pause"):
        await execute_simple_step(executor, paused_step, "test_data")

    # Test streaming with InfiniteFallbackError
    with pytest.raises(InfiniteFallbackError, match="Test infinite fallback"):
        await execute_simple_step(executor, paused_step, "test_data", stream=True)


@pytest.mark.asyncio
async def test_consistent_api_contract():
    """Test that the API contract is consistent across different step types."""
    from flujo.application.core.executor_core import ExecutorCore as UltraStepExecutor

    executor = UltraStepExecutor()

    # All regular failures should return StepResult(success=False)
    step = Step.model_validate(
        {"name": "failing", "agent": FailingAgent(), "config": {"max_retries": 1}}
    )

    # Simple step
    result1 = await execute_simple_step(executor, step, "test_data")
    assert isinstance(result1, StepResult)
    assert not result1.success

    # Streaming step
    result2 = await execute_simple_step(executor, step, "test_data", stream=True)
    assert isinstance(result2, StepResult)
    assert not result2.success

    # Complex step
    complex_step = Step.model_validate(
        {
            "name": "failing",
            "agent": FailingAgent(),
            "config": {"max_retries": 1},
            "plugins": [(MockPlugin(), 1)],
        }
    )
    result3 = await execute_simple_step(executor, complex_step, "test_data")
    assert isinstance(result3, StepResult)
    assert not result3.success


@pytest.mark.asyncio
async def test_error_information_preservation():
    """Test that error information is preserved in StepResult.feedback."""
    step = Step.model_validate(
        {"name": "failing", "agent": FailingAgent(), "config": {"max_retries": 1}}
    )

    from flujo.application.core.executor_core import ExecutorCore as UltraStepExecutor

    executor = UltraStepExecutor()

    result = await execute_simple_step(executor, step, "test_data")

    assert "RuntimeError" in result.feedback
    assert "Test failure" in result.feedback
    assert result.attempts == 2  # 1 initial + 1 retry (max_retries=1)


@pytest.mark.asyncio
async def test_pipeline_continuation_behavior():
    """Test that pipelines handle failures appropriately."""
    from flujo import Flujo
    from flujo.testing.utils import StubAgent
    from flujo.domain.dsl import Pipeline

    # Create a pipeline with a failing step followed by a working step
    failing_step = Step.model_validate(
        {"name": "failing", "agent": FailingAgent(), "config": {"max_retries": 1}}
    )
    working_step = Step.model_validate(
        {"name": "working", "agent": StubAgent(["Success"]), "config": {"max_retries": 1}}
    )

    # Create pipeline properly using Pipeline constructor
    pipeline = Pipeline(steps=[failing_step, working_step])
    runner = Flujo(pipeline)
    result = await gather_result(runner, "test_data")

    # The pipeline should fail at the first step
    assert len(result.step_history) == 1
    assert not result.step_history[0].success
    assert result.step_history[0].name == "failing"
    assert "Test failure" in result.step_history[0].feedback


@pytest.mark.asyncio
async def test_timing_preservation_for_failed_steps():
    """Test that timing data is preserved for failed steps."""
    import time

    class SlowFailingAgent:
        async def run(self, data, **kwargs):
            time.sleep(0.1)  # Simulate some work
            raise RuntimeError("Test failure")

    slow_step = Step.model_validate(
        {"name": "slow", "agent": SlowFailingAgent(), "config": {"max_retries": 1}}
    )

    from flujo.application.core.executor_core import ExecutorCore as UltraStepExecutor

    executor = UltraStepExecutor()

    result = await execute_simple_step(executor, slow_step, "test_data")

    # Should preserve actual execution time
    assert isinstance(result, StepResult)
    assert not result.success
    assert result.latency_s >= 0.1  # Should reflect actual execution time
