"""Tests for the ultra-optimized step executor."""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock

from flujo.application.core.ultra_executor import (
    UltraStepExecutor,
    _Frame,
    _State,
    _LRUCache,
    _UsageTracker,
)
from flujo.domain.dsl.step import Step
from flujo.domain.models import StepResult, UsageLimits
from flujo.exceptions import UsageLimitExceededError


class TestLRUCache:
    """Test the LRU cache functionality."""

    def test_cache_set_get(self):
        """Test basic cache set and get operations."""
        cache = _LRUCache(max_size=10, ttl=3600)
        result = StepResult(name="test", output="test_output", success=True)

        cache.set("test_key", result)
        retrieved = cache.get("test_key")

        assert retrieved is not None
        assert retrieved.name == "test"
        assert retrieved.output == "test_output"

    def test_cache_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = _LRUCache(max_size=2, ttl=3600)

        # Add 3 items to trigger eviction
        cache.set("key1", StepResult(name="test1", success=True))
        cache.set("key2", StepResult(name="test2", success=True))
        cache.set("key3", StepResult(name="test3", success=True))

        # First item should be evicted
        assert cache.get("key1") is None
        assert cache.get("key2") is not None
        assert cache.get("key3") is not None

    def test_cache_ttl(self):
        """Test TTL expiration."""
        cache = _LRUCache(max_size=10, ttl=0)  # Immediate expiration
        result = StepResult(name="test", success=True)

        cache.set("test_key", result)
        retrieved = cache.get("test_key")

        assert retrieved is None  # Should be expired immediately

    def test_cache_lru_promotion(self):
        """Test that accessed items are promoted to the end."""
        cache = _LRUCache(max_size=3, ttl=3600)

        cache.set("key1", StepResult(name="test1", success=True))
        cache.set("key2", StepResult(name="test2", success=True))
        cache.set("key3", StepResult(name="test3", success=True))

        # Access key1 to promote it
        cache.get("key1")

        # Add a new item - key2 should be evicted (oldest after promotion)
        cache.set("key4", StepResult(name="test4", success=True))

        assert cache.get("key1") is not None  # Still there
        assert cache.get("key2") is None  # Evicted
        assert cache.get("key3") is not None  # Still there
        assert cache.get("key4") is not None  # New item


class TestUsageTracker:
    """Test the usage tracker functionality."""

    @pytest.mark.asyncio
    async def test_usage_tracking(self):
        """Test usage tracking and limit checking."""
        tracker = _UsageTracker()

        # Add usage
        await tracker.add(1.0, 100)
        await tracker.add(2.0, 200)

        # Check limits
        limits = UsageLimits(total_cost_usd_limit=5.0, total_tokens_limit=500)
        await tracker.guard(limits)  # Should not raise

        # Test limit exceeded
        limits = UsageLimits(total_cost_usd_limit=1.0, total_tokens_limit=100)
        with pytest.raises(UsageLimitExceededError):
            await tracker.guard(limits)

    @pytest.mark.asyncio
    async def test_concurrent_usage_tracking(self):
        """Test usage tracking under concurrent access."""
        tracker = _UsageTracker()

        async def add_usage(cost: float, tokens: int):
            await tracker.add(cost, tokens)

        # Add usage concurrently
        tasks = [
            add_usage(1.0, 100),
            add_usage(2.0, 200),
            add_usage(3.0, 300),
        ]

        await asyncio.gather(*tasks)

        # Check total
        limits = UsageLimits(total_cost_usd_limit=10.0, total_tokens_limit=1000)
        await tracker.guard(limits)  # Should not raise


class TestUltraStepExecutor:
    """Test the ultra step executor."""

    @pytest.fixture
    def executor(self):
        """Create an ultra executor for testing."""
        return UltraStepExecutor(
            enable_cache=True,
            cache_size=100,
            cache_ttl=3600,
            concurrency_limit=4,
        )

    @pytest.fixture
    def mock_step(self):
        """Create a mock step for testing."""
        step = Mock(spec=Step)
        step.name = "test_step"
        step.config = Mock()
        step.config.max_retries = 3
        step.config.temperature = None
        step.agent = AsyncMock()
        step.agent.run.return_value = "test_output"
        step.validators = []

        # Add missing attributes that the step logic expects
        step.plugins = []
        step.fallback_step = None
        step.processors = Mock()
        step.processors.prompt_processors = []
        step.processors.output_processors = []
        step.failure_handlers = []
        step.persist_validation_results_to = None
        step.meta = {}
        step.persist_feedback_to_context = False

        return step

    @pytest.mark.asyncio
    async def test_simple_step_execution(self, executor, mock_step):
        """Test execution of a simple step with an agent."""
        result = await executor.execute_step(
            step=mock_step,
            data="test_input",
            context=None,
            resources=None,
        )

        assert result is not None
        assert result.name == "test_step"
        assert result.success is True
        assert result.output == "test_output"
        assert result.attempts == 1

    @pytest.mark.asyncio
    async def test_cached_step_execution(self, executor, mock_step):
        """Test that cached results are returned."""
        # First execution
        result1 = await executor.execute_step(
            step=mock_step,
            data="test_input",
            context=None,
            resources=None,
        )

        # Second execution with same input (should be cached)
        result2 = await executor.execute_step(
            step=mock_step,
            data="test_input",
            context=None,
            resources=None,
        )

        assert result1.output == result2.output
        # The second execution should be much faster due to caching

    @pytest.mark.asyncio
    async def test_step_with_retries(self, executor, mock_step):
        """Test step execution with retries."""
        # Make the agent fail twice, then succeed
        mock_step.agent.run.side_effect = [
            Exception("First failure"),
            Exception("Second failure"),
            "success_output",
        ]

        result = await executor.execute_step(
            step=mock_step,
            data="test_input",
            context=None,
            resources=None,
        )

        assert result.success is True
        assert result.output == "success_output"
        assert result.attempts == 3

    @pytest.mark.asyncio
    async def test_step_with_validation(self, executor, mock_step):
        """Test step execution with validators."""
        # Add a mock validator
        mock_validator = Mock()
        mock_validator.validate = AsyncMock(return_value=Mock(is_valid=True))
        mock_step.validators = [mock_validator]

        result = await executor.execute_step(
            step=mock_step,
            data="test_input",
            context=None,
            resources=None,
        )

        assert result.success is True
        mock_validator.validate.assert_called_once()

    @pytest.mark.asyncio
    async def test_step_with_failed_validation(self, executor, mock_step):
        """Test step execution with failed validation."""
        # Add a mock validator that fails
        mock_validator = Mock()
        mock_validator.validate = AsyncMock(
            return_value=Mock(is_valid=False, feedback="Validation failed")
        )
        mock_step.validators = [mock_validator]

        result = await executor.execute_step(
            step=mock_step,
            data="test_input",
            context=None,
            resources=None,
        )

        assert result.success is False
        assert "Validation failed" in result.feedback

    @pytest.mark.asyncio
    async def test_usage_limits(self, executor, mock_step):
        """Test usage limit enforcement."""

        # Create output with cost information using a real object
        class CostOutput:
            def __init__(self):
                self.cost_usd = 5.0
                self.token_counts = 1000

        cost_output = CostOutput()
        mock_step.agent.run.return_value = cost_output

        # Set usage limits
        usage_limits = UsageLimits(total_cost_usd_limit=3.0, total_tokens_limit=500)

        with pytest.raises(UsageLimitExceededError):
            await executor.execute_step(
                step=mock_step,
                data="test_input",
                context=None,
                resources=None,
                usage_limits=usage_limits,
            )

    @pytest.mark.asyncio
    async def test_streaming_execution(self, executor, mock_step):
        """Test streaming execution."""

        # Mock streaming agent
        async def mock_stream(data, **kwargs):
            yield "chunk1"
            yield "chunk2"
            yield "chunk3"

        mock_step.agent.stream = mock_stream
        chunks_received = []

        async def on_chunk(chunk):
            chunks_received.append(chunk)

        await executor.execute_step(
            step=mock_step,
            data="test_input",
            context=None,
            resources=None,
            stream=True,
            on_chunk=on_chunk,
        )

        assert len(chunks_received) == 3
        assert chunks_received == ["chunk1", "chunk2", "chunk3"]

    @pytest.mark.asyncio
    async def test_context_and_resources_passing(self, executor, mock_step):
        """Test that context and resources are passed correctly."""
        context = Mock()
        resources = Mock()

        await executor.execute_step(
            step=mock_step,
            data="test_input",
            context=context,
            resources=resources,
        )

        # Verify that context and resources were passed to the agent
        mock_step.agent.run.assert_called_once()
        call_args = mock_step.agent.run.call_args
        assert call_args[1]["context"] == context
        assert call_args[1]["resources"] == resources

    @pytest.mark.asyncio
    async def test_temperature_passing(self, executor, mock_step):
        """Test that temperature is passed correctly."""
        mock_step.config.temperature = 0.7

        await executor.execute_step(
            step=mock_step,
            data="test_input",
            context=None,
            resources=None,
        )

        # Verify that temperature was passed to the agent
        mock_step.agent.run.assert_called_once()
        call_args = mock_step.agent.run.call_args
        assert call_args[1]["temperature"] == 0.7

    def test_cache_key_generation(self, executor):
        """Test cache key generation."""
        step = Mock(spec=Step)
        step.name = "test_step"
        step.agent = Mock()

        frame = _Frame(
            step=step,
            data="test_data",
            context=None,
            resources=None,
        )

        cache_key = executor._cache_key(frame)
        assert isinstance(cache_key, str)
        assert len(cache_key) > 0

    def test_hash_obj(self, executor):
        """Test object hashing."""
        # Test basic types
        assert executor._hash_obj("test") == executor._hash_obj("test")
        assert executor._hash_obj(123) == executor._hash_obj(123)
        assert executor._hash_obj(1.23) == executor._hash_obj(1.23)

        # Test different objects
        assert executor._hash_obj("test") != executor._hash_obj("different")
        assert executor._hash_obj(123) != executor._hash_obj(456)

    @pytest.mark.asyncio
    async def test_concurrency_limiting(self, executor):
        """Test that concurrency limiting works."""
        # Create a step that takes time to execute
        slow_step = Mock(spec=Step)
        slow_step.name = "slow_step"
        slow_step.config = Mock()
        slow_step.config.max_retries = 1
        slow_step.agent = AsyncMock()

        async def slow_run(data, **kwargs):
            await asyncio.sleep(0.1)  # Simulate slow execution
            return "slow_output"

        slow_step.agent.run = slow_run

        # Execute multiple steps concurrently
        start_time = time.time()
        tasks = [executor.execute_step(slow_step, f"input_{i}", None, None) for i in range(10)]

        results = await asyncio.gather(*tasks)
        end_time = time.time()

        # All should succeed
        assert all(r.success for r in results)
        assert all(r.output == "slow_output" for r in results)

        # Should take longer than sequential execution due to concurrency limiting
        # (but less than full sequential execution)
        execution_time = end_time - start_time
        assert 0.1 < execution_time < 1.0  # Reasonable bounds

    def test_cache_property(self, executor):
        """Test that cache property is accessible."""
        assert executor.cache is not None
        assert isinstance(executor.cache, _LRUCache)

    def test_clear_cache(self, executor):
        """Test cache clearing."""
        # Add some items to cache
        result = StepResult(name="test", success=True)
        executor.cache.set("test_key", result)

        # Verify item is in cache
        assert executor.cache.get("test_key") is not None

        # Clear cache
        executor.clear_cache()

        # Verify cache is empty
        assert executor.cache.get("test_key") is None


class TestFrame:
    """Test the execution frame."""

    def test_frame_creation(self):
        """Test frame creation and basic properties."""
        step = Mock(spec=Step)
        step.name = "test_step"

        frame = _Frame(
            step=step,
            data="test_data",
            context=None,
            resources=None,
        )

        assert frame.step == step
        assert frame.data == "test_data"
        assert frame.context is None
        assert frame.resources is None
        assert frame.state == _State.PENDING
        assert frame.result is None
        assert frame.attempt == 1
        assert frame.max_retries == 3

    def test_frame_state_transitions(self):
        """Test frame state transitions."""
        step = Mock(spec=Step)
        frame = _Frame(step=step, data=None, context=None, resources=None)

        assert frame.state == _State.PENDING

        frame.state = _State.RUNNING
        assert frame.state == _State.RUNNING

        frame.state = _State.COMPLETED
        assert frame.state == _State.COMPLETED

    def test_frame_with_result(self):
        """Test frame with result."""
        step = Mock(spec=Step)
        result = StepResult(name="test", success=True)

        frame = _Frame(
            step=step,
            data=None,
            context=None,
            resources=None,
            result=result,
        )

        assert frame.result == result
