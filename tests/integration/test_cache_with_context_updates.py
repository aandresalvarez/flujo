"""
Integration tests for Cache Steps + Context Updates feature combination.

This tests the critical combination of caching with context-updating steps,
which could reveal bugs in cache key generation and context state management.
"""

import pytest
import asyncio
import time
from typing import Any, List
from flujo import step, Step
from flujo.domain.models import PipelineContext
from flujo.testing.utils import gather_result
from flujo.infra.caching import InMemoryCache
from tests.conftest import create_test_flujo
from flujo.type_definitions.common import JSONObject


class CacheContext(PipelineContext):
    """Context for testing cache operations with context updates."""

    initial_prompt: str = "test"
    cache_hits: int = 0
    cache_misses: int = 0
    cached_results: JSONObject = {}
    processing_history: List[str] = []
    current_operation: str = ""
    operation_count: int = 0
    cache_timestamps: dict[str, float] = {}
    cache_keys: List[str] = []


@step(updates_context=True)
async def cache_aware_step(data: Any, *, context: CacheContext) -> JSONObject:
    """Step that updates context and should be cached."""
    context.operation_count += 1
    context.current_operation = f"cached_operation_{data}"
    context.processing_history.append(f"processed_{data}")

    # Simulate expensive computation without blocking the event loop
    await asyncio.sleep(0.01)

    return {
        "operation_count": context.operation_count,
        "current_operation": context.current_operation,
        "processing_history": context.processing_history,
        "result": f"cached_result_{data}",
    }


@step(updates_context=True)
async def context_dependent_cache_step(data: Any, *, context: CacheContext) -> JSONObject:
    """Step that depends on context state for cache key generation."""
    context.operation_count += 1
    context.current_operation = f"context_dependent_{data}"

    # Context-dependent result
    result = f"result_{data}_{context.operation_count}"
    context.cached_results[data] = result

    return {
        "operation_count": context.operation_count,
        "current_operation": context.current_operation,
        "cached_results": context.cached_results,
        "result": result,
    }


@step(updates_context=True)
async def timestamp_cache_step(data: Any, *, context: CacheContext) -> JSONObject:
    """Step that includes timestamps in cache key generation."""
    context.operation_count += 1
    current_time = time.time()
    context.cache_timestamps[data] = current_time

    return {
        "operation_count": context.operation_count,
        "timestamp": current_time,
        "result": f"timestamp_result_{data}",
    }


@step(updates_context=True)
async def cache_key_tracking_step(data: Any, *, context: CacheContext) -> JSONObject:
    """Step that tracks cache keys for debugging."""
    context.operation_count += 1
    cache_key = f"cache_key_{data}_{context.operation_count}"
    context.cache_keys.append(cache_key)

    return {
        "operation_count": context.operation_count,
        "cache_keys": context.cache_keys,
        "current_cache_key": cache_key,
        "result": f"key_tracked_result_{data}",
    }


@step(updates_context=True)
async def failing_cache_step(data: Any, *, context: CacheContext) -> JSONObject:
    """Step that fails and should not be cached."""
    context.operation_count += 1

    if data == "fail":
        raise RuntimeError("Intentional failure for cache testing")

    return {
        "operation_count": context.operation_count,
        "result": f"success_result_{data}",
    }


# Create a step configuration that disables retries for the failing step
failing_cache_step.config.max_retries = 0


# Utility function to create simplified steps
def create_simple_step(step_type: str) -> Step:
    """Create a simplified step based on the specified type.

    Args:
        step_type: The type of step to create ("timestamp" or "cache_key")

    Returns:
        A configured step function

    Raises:
        ValueError: If step_type is not recognized
    """

    @step(updates_context=True)
    async def simple_step(data: Any, *, context: CacheContext) -> JSONObject:
        """Generic simplified step based on step_type."""
        context.operation_count += 1
        if step_type == "timestamp":
            current_time = time.time()
            return {
                "operation_count": context.operation_count,
                "timestamp": current_time,
                "result": f"simple_timestamp_result_{data}",
            }
        elif step_type == "cache_key":
            cache_key = f"simple_cache_key_{data}_{context.operation_count}"
            return {
                "operation_count": context.operation_count,
                "cache_key": cache_key,
                "result": f"simple_key_result_{data}",
            }
        else:
            raise ValueError(f"Unknown step_type: {step_type}")

    return simple_step


@pytest.mark.asyncio
async def test_cache_with_context_updates_basic():
    """Test basic cache operations with context updates."""

    # Create cached step
    cached_step = Step.cached(
        cache_aware_step,
        cache_backend=InMemoryCache(),
    )

    pipeline = cached_step
    runner = create_test_flujo(pipeline, context_model=CacheContext, persist_state=False)

    # First run - should be a cache miss
    result1 = await gather_result(runner, "test_input_1")

    # Verify first run
    assert result1.step_history[-1].success is True
    assert result1.final_pipeline_context.operation_count == 1
    assert result1.final_pipeline_context.current_operation == "cached_operation_test_input_1"
    assert "processed_test_input_1" in result1.final_pipeline_context.processing_history
    assert "cache_hit" not in (result1.step_history[-1].metadata_ or {})

    # Second run with same input - should be a cache hit
    result2 = await gather_result(runner, "test_input_1")

    # Verify cache hit
    assert result2.step_history[-1].success is True
    assert result2.step_history[-1].metadata_.get("cache_hit") is True
    # Context should be updated even on cache hit (but starts fresh for each run)
    assert result2.final_pipeline_context.operation_count == 1  # Fresh context for new run
    assert result2.final_pipeline_context.current_operation == "cached_operation_test_input_1"
    assert len(result2.final_pipeline_context.processing_history) == 1  # Fresh context


@pytest.mark.asyncio
async def test_cache_with_context_updates_error_handling():
    """Test cache behavior when steps fail."""

    # Create cached step that can fail
    cached_step = Step.cached(
        failing_cache_step,
        cache_backend=InMemoryCache(),
    )

    pipeline = cached_step
    runner = create_test_flujo(pipeline, context_model=CacheContext, persist_state=False)

    # Run with failing input
    result1 = await gather_result(runner, "fail")

    # Verify failure is not cached
    assert result1.step_history[-1].success is False
    assert "cache_hit" not in (result1.step_history[-1].metadata_ or {})
    assert result1.final_pipeline_context.operation_count == 1

    # Run again - should fail again (not cached)
    result2 = await gather_result(runner, "fail")

    # Verify failure is still not cached
    assert result2.step_history[-1].success is False
    assert "cache_hit" not in (result2.step_history[-1].metadata_ or {})
    assert result2.final_pipeline_context.operation_count == 1  # Fresh context for new run

    # Run with successful input
    result3 = await gather_result(runner, "success")

    # Verify success is cached
    assert result3.step_history[-1].success is True
    assert result3.final_pipeline_context.operation_count == 1  # Fresh context for new run

    # Run again - should be cache hit
    result4 = await gather_result(runner, "success")

    # Verify cache hit
    assert result4.step_history[-1].success is True
    assert result4.step_history[-1].metadata_.get("cache_hit") is True
    assert result4.final_pipeline_context.operation_count == 1  # Fresh context for new run


@pytest.mark.asyncio
async def test_cache_with_context_updates_context_dependent():
    """Test cache behavior with context-dependent operations."""

    # Create cached step that depends on context
    cached_step = Step.cached(
        context_dependent_cache_step,
        cache_backend=InMemoryCache(),
    )

    pipeline = cached_step
    runner = create_test_flujo(pipeline, context_model=CacheContext, persist_state=False)

    # First run
    result1 = await gather_result(runner, "input_a")

    # Verify first run
    assert result1.step_history[-1].success is True
    assert result1.final_pipeline_context.operation_count == 1
    assert "input_a" in result1.final_pipeline_context.cached_results
    assert "cache_hit" not in (result1.step_history[-1].metadata_ or {})

    # Second run with same input - should be cache hit
    result2 = await gather_result(runner, "input_a")

    # Verify cache hit
    assert result2.step_history[-1].success is True
    assert result2.step_history[-1].metadata_.get("cache_hit") is True
    assert result2.final_pipeline_context.operation_count == 1  # Fresh context for new run

    # Third run with different input - should be cache miss
    result3 = await gather_result(runner, "input_b")

    # Verify cache miss
    assert result3.step_history[-1].success is True
    assert "cache_hit" not in (result3.step_history[-1].metadata_ or {})
    assert result3.final_pipeline_context.operation_count == 1  # Fresh context for new run
    assert "input_b" in result3.final_pipeline_context.cached_results


@pytest.mark.asyncio
async def test_cache_with_context_updates_state_isolation():
    """Test that cache operations maintain proper state isolation."""

    # Create cached step
    cached_step = Step.cached(
        cache_aware_step,
        cache_backend=InMemoryCache(),
    )

    pipeline = cached_step
    runner = create_test_flujo(pipeline, context_model=CacheContext, persist_state=False)

    # Run with different inputs to test state isolation
    result1 = await gather_result(runner, "input_1")
    result2 = await gather_result(runner, "input_2")
    result3 = await gather_result(runner, "input_1")  # Cache hit

    # Verify state isolation (each run gets fresh context)
    assert result1.final_pipeline_context.operation_count == 1
    assert result2.final_pipeline_context.operation_count == 1
    assert result3.final_pipeline_context.operation_count == 1

    # Verify cache hit on third run
    assert result3.step_history[-1].metadata_.get("cache_hit") is True

    # Verify processing history is fresh for each run
    assert len(result1.final_pipeline_context.processing_history) == 1
    assert len(result2.final_pipeline_context.processing_history) == 1
    assert len(result3.final_pipeline_context.processing_history) == 1
    assert "processed_input_1" in result1.final_pipeline_context.processing_history
    assert "processed_input_2" in result2.final_pipeline_context.processing_history
    assert "processed_input_1" in result3.final_pipeline_context.processing_history


@pytest.mark.asyncio
async def test_cache_with_context_updates_complex_interaction():
    """Test complex cache interactions with context updates."""

    # Create multiple cached steps with simpler outputs
    cached_step1 = Step.cached(
        cache_aware_step,
        cache_backend=InMemoryCache(),
    )

    # Create cached steps using the module-level utility function
    cached_step2 = Step.cached(
        create_simple_step("timestamp"),
        cache_backend=InMemoryCache(),
    )
    cached_step3 = Step.cached(
        create_simple_step("cache_key"),
        cache_backend=InMemoryCache(),
    )

    # Create pipeline with multiple cached steps
    pipeline = cached_step1 >> cached_step2 >> cached_step3
    runner = create_test_flujo(pipeline, context_model=CacheContext, persist_state=False)

    # First run
    result1 = await gather_result(runner, "complex_input")

    # Verify first run
    assert result1.step_history[-1].success is True
    assert result1.final_pipeline_context.operation_count == 3

    # Second run - should have cache hits
    result2 = await gather_result(runner, "complex_input")

    # Verify cache hits
    assert result2.step_history[-1].success is True
    assert result2.step_history[-1].metadata_.get("cache_hit") is True
    assert result2.final_pipeline_context.operation_count == 3  # Fresh context for new run


@pytest.mark.asyncio
async def test_cache_with_context_updates_metadata_conflicts():
    """Test cache behavior with metadata conflicts."""

    # Create cached step
    cached_step = Step.cached(
        cache_aware_step,
        cache_backend=InMemoryCache(),
    )

    pipeline = cached_step
    runner = create_test_flujo(pipeline, context_model=CacheContext, persist_state=False)

    # First run
    result1 = await gather_result(runner, "metadata_test")

    # Verify first run
    assert result1.step_history[-1].success is True
    assert "cache_hit" not in (result1.step_history[-1].metadata_ or {})

    # Second run - cache hit
    result2 = await gather_result(runner, "metadata_test")

    # Verify cache hit metadata
    assert result2.step_history[-1].success is True
    assert result2.step_history[-1].metadata_.get("cache_hit") is True

    # Verify context updates still work on cache hit
    assert result2.final_pipeline_context.operation_count == 1  # Fresh context for new run
    assert result2.final_pipeline_context.current_operation == "cached_operation_metadata_test"
