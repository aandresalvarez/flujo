"""Typed mock factories for Flujo tests.

This module provides type-safe factory functions for creating mock objects
used in tests. These factories ensure mocks have proper type annotations
and can be verified with type checkers.

Usage:
    from tests.test_types.mocks import create_mock_executor_core

    def test_execution():
        executor = create_mock_executor_core(agent_output="test")
        result = await executor.execute(step, data)
"""

from typing import Any, Optional
from unittest.mock import AsyncMock, Mock
from flujo.application.core.executor_core import ExecutorCore
from flujo.domain.models import StepResult, UsageLimits
from tests.test_types.fakes import FakeCacheBackend


def create_mock_executor_core(
    agent_output: Any = "mock_output",
    processor_output: Any = None,
    cache_hit: bool = False,
    **overrides: Any,
) -> ExecutorCore[Any]:
    """Create a typed mock ExecutorCore for tests.

    Args:
        agent_output: Output to return from agent runner
        processor_output: Output to return from processor pipeline
        cache_hit: Whether cache should return a hit
        **overrides: Additional ExecutorCore parameters to override

    Returns:
        An ExecutorCore instance with mocked dependencies
    """
    mock_agent_runner = AsyncMock()
    mock_agent_runner.run = AsyncMock(return_value=agent_output)

    mock_processor = AsyncMock()
    mock_processor.apply_prompt = AsyncMock(return_value="processed_prompt")
    mock_processor.apply_output = AsyncMock(return_value=processor_output or agent_output)

    cache_backend = FakeCacheBackend()
    if cache_hit:
        cached_result = StepResult(
            name="cached_step",
            output={"cached": "result"},
            success=True,
            metadata_={"cache_hit": True},
        )
        cache_backend.store["cached_step"] = cached_result

    class _FastAgentStepExecutor:
        async def execute(
            self,
            core: Any,
            step: Any,
            data: Any,
            context: Any,
            resources: Any,
            limits: Optional[UsageLimits],
            stream: bool,
            on_chunk: Any,
            cache_key: Any,
            _fallback_depth: int = 0,
        ) -> StepResult:
            output = await step.agent.run(data, context=context, resources=resources)
            return StepResult(name=getattr(step, "name", "step"), output=output, success=True)

    executor = ExecutorCore(
        agent_runner=mock_agent_runner,
        processor_pipeline=mock_processor,
        validator_runner=AsyncMock(),
        plugin_runner=AsyncMock(),
        usage_meter=AsyncMock(),
        cache_backend=cache_backend,
        telemetry=Mock(),
        agent_step_executor=_FastAgentStepExecutor(),
        concurrency_limit=overrides.pop("concurrency_limit", 256),
        **overrides,
    )
    orig_execute = executor.execute

    async def delayed_execute(step: Any, data: Any, *args: Any, **kwargs: Any) -> StepResult:
        result = await orig_execute(step, data, *args, **kwargs)
        try:
            mock_agent_runner.reset_mock()
            mock_processor.reset_mock()
            try:
                step.agent.reset_mock()  # type: ignore[attr-defined]
            except Exception:
                pass
        except Exception:
            pass
        return result

    executor.execute = delayed_execute  # type: ignore[assignment]
    return executor


__all__ = [
    "create_mock_executor_core",
]
