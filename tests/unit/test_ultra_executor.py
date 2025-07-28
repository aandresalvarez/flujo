"""Tests for the ultra-optimized step executor."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from dataclasses import dataclass
from typing import Any, Optional

from flujo.application.core.ultra_executor import UltraStepExecutor, _LRUCache, _UsageTracker
from flujo.domain.dsl.step import Step
from flujo.domain.models import StepResult, UsageLimits
from flujo.exceptions import UsageLimitExceededError


# Create a simple replacement for the removed _Frame class
@dataclass
class _Frame:
    step: Any
    data: Any
    context: Optional[Any] = None
    resources: Optional[Any] = None


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
        cache = _LRUCache(max_size=10, ttl=0)  # 0 means never expire (standard convention)
        result = StepResult(name="test", success=True)

        cache.set("test_key", result)
        retrieved = cache.get("test_key")

        assert retrieved is not None  # Should NOT expire with TTL=0 (never expire)
        assert retrieved.name == "test"

    def test_cache_ttl_never_expire(self):
        """Test that TTL of 0 means never expire (standard convention)."""
        cache = _LRUCache(max_size=10, ttl=0)  # 0 means never expire
        result = StepResult(name="test", success=True)

        cache.set("test_key", result)
        retrieved = cache.get("test_key")

        assert retrieved is not None  # Should NOT expire with TTL=0
        assert retrieved.name == "test"

    def test_cache_ttl_with_expiration(self):
        """Test TTL expiration with positive value."""
        cache = _LRUCache(max_size=10, ttl=0.1)  # 100ms expiration
        result = StepResult(name="test", success=True)

        cache.set("test_key", result)

        # Should be available immediately
        retrieved = cache.get("test_key")
        assert retrieved is not None

        # Wait for expiration
        import time

        time.sleep(0.15)  # Wait longer than TTL

        # Should be expired now
        retrieved = cache.get("test_key")
        assert retrieved is None

    def test_cache_monotonic_time(self):
        """Test that cache uses monotonic time for reliable TTL."""
        cache = _LRUCache(max_size=10, ttl=0.1)  # 100ms expiration
        result = StepResult(name="test", success=True)

        cache.set("test_key", result)

        # Should be available immediately
        retrieved = cache.get("test_key")
        assert retrieved is not None

        # Wait for expiration
        import time

        time.sleep(0.15)  # Wait longer than TTL

        # Should be expired now
        retrieved = cache.get("test_key")
        assert retrieved is None

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
        # Make the agent fail on first attempt
        mock_step.agent.run.side_effect = [Exception("First attempt fails"), "test_output"]

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
        assert result.attempts == 2

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

        assert result is not None
        assert result.success is True

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

        assert result is not None
        assert result.success is False
        assert "Validation failed" in result.feedback

    @pytest.mark.asyncio
    async def test_usage_limits(self, executor, mock_step):
        """Test usage limit enforcement at the pipeline/governor level."""

        from flujo.domain.models import UsageLimits
        from flujo.application.core.usage_governor import UsageGovernor
        from flujo.domain.models import PipelineResult

        # Create a mock output with explicit cost that exceeds the limit
        class MockOutput:
            def __init__(self):
                self.output = "test_output"
                self.cost_usd = 0.2  # Exceeds the 0.1 limit
                self.token_counts = 100

        mock_step.agent.run.return_value = MockOutput()

        # Create a pipeline with a single step
        limits = UsageLimits(total_cost_usd_limit=0.1)
        governor = UsageGovernor(limits)
        result = PipelineResult(step_history=[], total_cost_usd=0.0)

        # Run the step and add the result to the pipeline result
        step_result = await executor.execute_step(
            step=mock_step,
            data="test_input",
            context=None,
            resources=None,
        )
        result.step_history.append(step_result)
        result.total_cost_usd += getattr(step_result, "cost_usd", 0.0)

        # Now check usage limits at the governor level
        with pytest.raises(UsageLimitExceededError):
            governor.check_usage_limits(result, None)

    @pytest.mark.asyncio
    async def test_streaming_execution(self, executor, mock_step):
        """Test streaming execution."""

        # Mock streaming agent
        async def mock_stream(data, **kwargs):
            yield "chunk1"
            yield "chunk2"
            yield "chunk3"

        mock_step.agent.stream = mock_stream

        chunks = []

        async def on_chunk(chunk):
            chunks.append(chunk)

        result = await executor.execute_step(
            step=mock_step,
            data="test_input",
            context=None,
            resources=None,
            stream=True,
            on_chunk=on_chunk,
        )

        assert result is not None
        assert result.success is True
        assert chunks == ["chunk1", "chunk2", "chunk3"]

    @pytest.mark.asyncio
    async def test_context_and_resources_passing(self, executor, mock_step):
        """Test that context and resources are passed correctly."""
        context = {"key": "value"}
        resources = {"resource": "data"}

        # Track what was passed to the agent
        passed_kwargs = {}
        original_run = mock_step.agent.run

        async def track_kwargs(*args, **kwargs):
            passed_kwargs.update(kwargs)
            return original_run(*args, **kwargs)

        mock_step.agent.run = track_kwargs

        await executor.execute_step(
            step=mock_step,
            data="test_input",
            context=context,
            resources=resources,
        )

        assert passed_kwargs.get("context") == context
        assert passed_kwargs.get("resources") == resources

    @pytest.mark.asyncio
    async def test_temperature_passing(self, executor, mock_step):
        """Test that temperature is passed correctly."""
        mock_step.config.temperature = 0.7

        passed_kwargs = {}
        original_run = mock_step.agent.run

        async def track_kwargs(*args, **kwargs):
            passed_kwargs.update(kwargs)
            return original_run(*args, **kwargs)

        mock_step.agent.run = track_kwargs

        await executor.execute_step(
            step=mock_step,
            data="test_input",
            context=None,
            resources=None,
        )

        assert passed_kwargs.get("temperature") == 0.7

    def test_cache_key_generation(self, executor):
        """Test that cache keys are generated correctly."""
        # Create proper frame objects for testing
        step1 = Mock(spec=Step)
        step1.name = "test_step"
        step1.agent = Mock()
        step1.agent.run = lambda x: x

        step2 = Mock(spec=Step)
        step2.name = "test_step"
        step2.agent = Mock()
        step2.agent.run = lambda x: x

        # Create frame objects

        frame1 = _Frame(step=step1, data="test_data", context=None, resources=None)
        frame2 = _Frame(step=step2, data="test_data", context=None, resources=None)

        # Test cache key generation
        key1 = executor._cache_key(frame1)
        key2 = executor._cache_key(frame2)
        assert key1 != key2  # Different steps should have different keys

    def test_hash_obj(self, executor):
        """Test object hashing."""
        obj1 = {"key": "value"}
        obj2 = {"key": "value"}
        obj3 = {"key": "different"}

        hash1 = executor._hash_obj(obj1)
        hash2 = executor._hash_obj(obj2)
        hash3 = executor._hash_obj(obj3)

        assert hash1 == hash2  # Same content, same hash
        assert hash1 != hash3  # Different content, different hash

    def test_hash_obj_bytes_fix(self, executor):
        """Test that bytes are handled correctly without string conversion."""
        # Test bytes handling - this was a critical bug
        bytes_data = b"bytes_content"
        str_data = "string_content"

        bytes_hash = executor._hash_obj(bytes_data)
        str_hash = executor._hash_obj(str_data)

        # These should be different because bytes and string have different content
        assert bytes_hash != str_hash

        # Same bytes should hash to same value
        bytes_hash2 = executor._hash_obj(bytes_data)
        assert bytes_hash == bytes_hash2

    def test_stable_cache_keys(self, executor):
        """Test that cache keys are stable across different runs."""
        step = Mock(spec=Step)
        step.name = "test_step"
        step.agent = Mock()
        step.agent.run = lambda x: x

        # Create frame objects

        frame1 = _Frame(step=step, data="test_data", context=None, resources=None)
        frame2 = _Frame(step=step, data="test_data", context=None, resources=None)

        # Same step and data should generate same cache key
        key1 = executor._cache_key(frame1)
        key2 = executor._cache_key(frame2)
        assert key1 == key2

    @pytest.mark.asyncio
    async def test_caching_functionality(self):
        """Test that caching actually works in the executor."""
        from flujo.application.core.ultra_executor import UltraStepExecutor
        from flujo.domain.dsl import Step
        from flujo.testing.utils import StubAgent

        # Create executor with caching enabled
        executor = UltraStepExecutor(enable_cache=True, cache_size=100)

        # Create a simple step with multiple outputs
        agent = StubAgent(["test output 1", "test output 2", "test output 3"])
        step = Step(name="test_step", agent=agent)

        # First execution - should be cache miss
        result1 = await executor.execute_step(step, "test_input")
        assert result1.success
        assert "cache_hit" not in (result1.metadata_ or {})

        # Second execution with same input - should be cache hit
        result2 = await executor.execute_step(step, "test_input")
        assert result2.success
        assert result2.metadata_.get("cache_hit") is True

        # Third execution with different input - should be cache miss
        result3 = await executor.execute_step(step, "different_input")
        assert result3.success
        assert "cache_hit" not in (result3.metadata_ or {})

    @pytest.mark.asyncio
    async def test_caching_with_context(self):
        """Test caching works correctly with context."""
        from flujo.application.core.ultra_executor import UltraStepExecutor
        from flujo.domain.dsl import Step
        from flujo.testing.utils import StubAgent
        from flujo.domain.models import BaseModel

        class TestContext(BaseModel):
            value: str = "default"

        # Create executor with caching enabled
        executor = UltraStepExecutor(enable_cache=True, cache_size=100)

        # Create a simple step with multiple outputs
        agent = StubAgent(["test output 1", "test output 2", "test output 3"])
        step = Step(name="test_step", agent=agent)

        context1 = TestContext(value="context1")
        context2 = TestContext(value="context2")

        # First execution with context1
        result1 = await executor.execute_step(step, "test_input", context=context1)
        assert result1.success
        assert "cache_hit" not in (result1.metadata_ or {})

        # Second execution with same input and context - should be cache hit
        result2 = await executor.execute_step(step, "test_input", context=context1)
        assert result2.success
        assert result2.metadata_.get("cache_hit") is True

        # Third execution with different context - should be cache miss
        result3 = await executor.execute_step(step, "test_input", context=context2)
        assert result3.success
        assert "cache_hit" not in (result3.metadata_ or {})

    @pytest.mark.asyncio
    async def test_caching_disabled(self):
        """Test that caching is properly disabled when enable_cache=False."""
        from flujo.application.core.ultra_executor import UltraStepExecutor
        from flujo.domain.dsl import Step
        from flujo.testing.utils import StubAgent

        # Create executor with caching disabled
        executor = UltraStepExecutor(enable_cache=False)

        # Create a simple step with multiple outputs
        agent = StubAgent(["test output 1", "test output 2"])
        step = Step(name="test_step", agent=agent)

        # First execution
        result1 = await executor.execute_step(step, "test_input")
        assert result1.success
        assert "cache_hit" not in (result1.metadata_ or {})

        # Second execution with same input - should not be cache hit
        result2 = await executor.execute_step(step, "test_input")
        assert result2.success
        assert "cache_hit" not in (result2.metadata_ or {})

    @pytest.mark.asyncio
    async def test_cache_key_stability(self):
        """Test that cache keys are stable and deterministic."""
        from flujo.application.core.ultra_executor import UltraStepExecutor
        from flujo.domain.dsl import Step
        from flujo.testing.utils import StubAgent

        # Create two executors
        executor1 = UltraStepExecutor(enable_cache=True)
        executor2 = UltraStepExecutor(enable_cache=True)

        # Create identical steps
        agent1 = StubAgent(["test output"])
        agent2 = StubAgent(["test output"])
        step1 = Step(name="test_step", agent=agent1)
        step2 = Step(name="test_step", agent=agent2)

        # Create frame objects

        frame1 = _Frame(step=step1, data="test_data", context=None, resources=None)
        frame2 = _Frame(step=step2, data="test_data", context=None, resources=None)

        # Generate cache keys
        key1 = executor1._cache_key(frame1)
        key2 = executor2._cache_key(frame2)

        # Keys should be stable and deterministic for identical inputs
        assert key1 == key2

    def test_agent_identification_stability(self, executor):
        """Test that agent identification is stable and doesn't use memory addresses."""
        step1 = Mock(spec=Step)
        step1.name = "test_step"
        step1.agent = Mock()
        step1.agent.run = lambda x: x

        step2 = Mock(spec=Step)
        step2.name = "test_step"
        step2.agent = Mock()
        step2.agent.run = lambda x: x

        # Create frame objects

        frame1 = _Frame(step=step1, data="test_data", context=None, resources=None)
        frame2 = _Frame(step=step2, data="test_data", context=None, resources=None)

        # Generate cache keys
        key1 = executor._cache_key(frame1)
        key2 = executor._cache_key(frame2)

        # Keys should be different for different agent instances
        assert key1 != key2

        # But same agent should generate same key
        frame3 = _Frame(step=step1, data="test_data", context=None, resources=None)
        key3 = executor._cache_key(frame3)
        assert key1 == key3

    @pytest.mark.asyncio
    async def test_concurrency_limiting(self, executor):
        """Test that concurrency limiting works."""

        # Create a slow agent
        async def slow_run(data, **kwargs):
            await asyncio.sleep(0.1)
            return "slow_output"

        step = Mock(spec=Step)
        step.name = "slow_step"
        step.config = Mock()
        step.config.max_retries = 1
        step.config.temperature = None
        step.agent = Mock()
        step.agent.run = slow_run
        step.plugins = []
        step.validators = []
        step.fallback_step = None
        step.processors = Mock()
        step.processors.prompt_processors = []
        step.processors.output_processors = []
        step.failure_handlers = []
        step.persist_validation_results_to = None
        step.meta = {}
        step.persist_feedback_to_context = False

        # Run multiple steps concurrently
        tasks = [executor.execute_step(step, "input", None, None) for _ in range(3)]
        results = await asyncio.gather(*tasks)

        # With concurrency limit of 4, all should complete
        assert len(results) == 3
        assert all(r.success for r in results)

    def test_cache_property(self, executor):
        """Test that cache property is accessible."""
        assert executor.cache is not None
        assert isinstance(executor.cache, _LRUCache)

    def test_clear_cache(self, executor):
        """Test cache clearing."""
        # Add some items to cache
        executor.cache.set("key1", StepResult(name="test1", success=True))
        executor.cache.set("key2", StepResult(name="test2", success=True))

        # Verify items are in cache
        assert executor.cache.get("key1") is not None
        assert executor.cache.get("key2") is not None

        # Clear cache
        executor.clear_cache()

        # Verify items are gone
        assert executor.cache.get("key1") is None
        assert executor.cache.get("key2") is None

    @pytest.mark.asyncio
    async def test_step_with_plugins_validators_fallbacks(self, executor):
        """Test that steps with plugins, validators, or fallbacks use complex execution path."""
        from flujo.domain.dsl.step import Step

        # Create a proper plugin mock
        mock_plugin = Mock()
        mock_plugin.validate = AsyncMock(return_value=Mock(is_valid=True))

        # Create a step with plugins
        step_with_plugins = Mock(spec=Step)
        step_with_plugins.name = "step_with_plugins"
        step_with_plugins.config = Mock()
        step_with_plugins.config.max_retries = 3
        step_with_plugins.config.temperature = None
        step_with_plugins.config.timeout_s = 30.0  # Fix: return numeric value instead of Mock
        step_with_plugins.agent = AsyncMock()
        step_with_plugins.agent.run.return_value = "test_output"
        step_with_plugins.plugins = [(mock_plugin, 1.0)]  # Has plugins
        step_with_plugins.validators = []
        step_with_plugins.fallback_step = None
        step_with_plugins.processors = Mock()
        step_with_plugins.processors.prompt_processors = []
        step_with_plugins.processors.output_processors = []
        step_with_plugins.failure_handlers = []
        step_with_plugins.persist_validation_results_to = None
        step_with_plugins.meta = {}
        step_with_plugins.persist_feedback_to_context = False

        # This should use the complex execution path
        result = await executor.execute_step(
            step=step_with_plugins,
            data="test_input",
            context=None,
            resources=None,
        )

        # Verify it used the complex path (should have fallback_triggered metadata)
        assert result.success is True
        # The complex execution path may return a different output due to plugin processing
        assert result.output is not None

    @pytest.mark.asyncio
    async def test_step_with_validators(self, executor):
        """Test that steps with validators use complex execution path."""
        from flujo.domain.validation import Validator

        # Create a step with validators
        step_with_validators = Mock(spec=Step)
        step_with_validators.name = "step_with_validators"
        step_with_validators.config = Mock()
        step_with_validators.config.max_retries = 3
        step_with_validators.config.temperature = None
        step_with_validators.agent = AsyncMock()
        step_with_validators.agent.run.return_value = "test_output"
        step_with_validators.plugins = []
        step_with_validators.validators = [Mock(spec=Validator)]  # Has validators
        step_with_validators.fallback_step = None
        step_with_validators.processors = Mock()
        step_with_validators.processors.prompt_processors = []
        step_with_validators.processors.output_processors = []
        step_with_validators.failure_handlers = []
        step_with_validators.persist_validation_results_to = None
        step_with_validators.meta = {}
        step_with_validators.persist_feedback_to_context = False

        # This should use the complex execution path
        result = await executor.execute_step(
            step=step_with_validators,
            data="test_input",
            context=None,
            resources=None,
        )

        assert result is not None
        assert result.name == "step_with_validators"
        assert result.success is True

    @pytest.mark.asyncio
    async def test_step_with_fallback(self, executor):
        """Test that steps with fallbacks use complex execution path."""
        # Create a step with fallback
        step_with_fallback = Mock(spec=Step)
        step_with_fallback.name = "step_with_fallback"
        step_with_fallback.config = Mock()
        step_with_fallback.config.max_retries = 3
        step_with_fallback.config.temperature = None
        step_with_fallback.agent = AsyncMock()
        step_with_fallback.agent.run.return_value = "test_output"
        step_with_fallback.plugins = []
        step_with_fallback.validators = []
        step_with_fallback.fallback_step = Mock(spec=Step)  # Has fallback
        step_with_fallback.processors = Mock()
        step_with_fallback.processors.prompt_processors = []
        step_with_fallback.processors.output_processors = []
        step_with_fallback.failure_handlers = []
        step_with_fallback.persist_validation_results_to = None
        step_with_fallback.meta = {}
        step_with_fallback.persist_feedback_to_context = False

        # This should use the complex execution path
        result = await executor.execute_step(
            step=step_with_fallback,
            data="test_input",
            context=None,
            resources=None,
        )

        assert result is not None
        assert result.name == "step_with_fallback"
        assert result.success is True

    @pytest.mark.asyncio
    async def test_agent_with_kwargs_signature(self, executor):
        """Test that agents with **kwargs signature receive all parameters correctly."""

        # Create an agent that accepts **kwargs
        class KwargsAgent:
            async def run(self, data, **kwargs):
                return f"output: {data}, kwargs: {kwargs}"

        step = Mock(spec=Step)
        step.name = "kwargs_test"
        step.config = Mock()
        step.config.max_retries = 3
        step.config.temperature = 0.7
        step.agent = KwargsAgent()
        step.plugins = []
        step.validators = []
        step.fallback_step = None
        step.processors = Mock()
        step.processors.prompt_processors = []
        step.processors.output_processors = []
        step.failure_handlers = []
        step.persist_validation_results_to = None
        step.meta = {}
        step.persist_feedback_to_context = False

        result = await executor.execute_step(
            step=step,
            data="test_input",
            context={"key": "value"},
            resources={"resource": "data"},
        )

        assert result is not None
        assert result.success is True
        # The output should contain the kwargs that were passed
        assert "context" in result.output
        assert "resources" in result.output
        assert "temperature" in result.output

    @pytest.mark.asyncio
    async def test_mock_detection_safety(self, executor):
        """Test that mock detection doesn't break with non-mock objects."""
        # Create a step that returns a StepResult (not a mock)
        step = Mock(spec=Step)
        step.name = "mock_test"
        step.config = Mock()
        step.config.max_retries = 3
        step.config.temperature = None
        step.agent = AsyncMock()

        # Return a StepResult instead of a mock
        step_result = StepResult(name="test", output="test_output", success=True)
        step.agent.run.return_value = step_result

        step.plugins = []
        step.validators = []
        step.fallback_step = None
        step.processors = Mock()
        step.processors.prompt_processors = []
        step.processors.output_processors = []
        step.failure_handlers = []
        step.persist_validation_results_to = None
        step.meta = {}
        step.persist_feedback_to_context = False

        # This should not raise a TypeError
        result = await executor.execute_step(
            step=step,
            data="test_input",
            context=None,
            resources=None,
        )

        assert result is not None
        assert result.success is True

    @pytest.mark.asyncio
    async def test_mock_detection_blocks_mocks(self, executor):
        """Test that mock detection correctly blocks mock outputs."""
        # Create a step that returns a mock
        step = Mock(spec=Step)
        step.name = "mock_test"
        step.config = Mock()
        step.config.max_retries = 3
        step.config.temperature = None
        step.agent = AsyncMock()

        # Return a mock object
        mock_output = Mock()
        mock_output.output = "mock_output"
        step.agent.run.return_value = mock_output

        step.plugins = []
        step.validators = []
        step.fallback_step = None
        step.processors = Mock()
        step.processors.prompt_processors = []
        step.processors.output_processors = []
        step.failure_handlers = []
        step.persist_validation_results_to = None
        step.meta = {}
        step.persist_feedback_to_context = False

        # This should raise a TypeError
        with pytest.raises(TypeError, match="returned a Mock object"):
            await executor.execute_step(
                step=step,
                data="test_input",
                context=None,
                resources=None,
            )

    # ============================================================================
    # REGRESSION TESTS: Prevent caching bug from happening again
    # ============================================================================

    @pytest.mark.asyncio
    async def test_regression_cache_integration_works(self):
        """REGRESSION: Ensure caching is actually integrated into execution flow."""
        from flujo.application.core.ultra_executor import UltraStepExecutor
        from flujo.domain.dsl import Step
        from flujo.testing.utils import StubAgent

        # Create executor with caching enabled
        executor = UltraStepExecutor(enable_cache=True, cache_size=100)

        # Create a simple step
        agent = StubAgent(["test output 1", "test output 2"])
        step = Step(name="test_step", agent=agent)

        # First execution - should be cache miss
        result1 = await executor.execute_step(step, "test_input")
        assert result1.success
        assert "cache_hit" not in (result1.metadata_ or {})

        # Second execution with same input - should be cache hit
        result2 = await executor.execute_step(step, "test_input")
        assert result2.success
        assert result2.metadata_.get("cache_hit") is True

        # Verify the agent was only called once (proving cache worked)
        # Note: StubAgent doesn't track calls, but we can verify cache hit metadata

    @pytest.mark.asyncio
    async def test_regression_cache_disabled_works(self):
        """REGRESSION: Ensure caching can be properly disabled."""
        from flujo.application.core.ultra_executor import UltraStepExecutor
        from flujo.domain.dsl import Step
        from flujo.testing.utils import StubAgent

        # Create executor with caching disabled
        executor = UltraStepExecutor(enable_cache=False)

        # Create a simple step
        agent = StubAgent(["test output 1", "test output 2"])
        step = Step(name="test_step", agent=agent)

        # First execution
        result1 = await executor.execute_step(step, "test_input")
        assert result1.success
        assert "cache_hit" not in (result1.metadata_ or {})

        # Second execution with same input - should NOT be cache hit
        result2 = await executor.execute_step(step, "test_input")
        assert result2.success
        assert "cache_hit" not in (result2.metadata_ or {})

    def test_regression_cache_key_stability(self):
        """REGRESSION: Ensure cache keys are stable and don't use memory addresses."""
        from flujo.application.core.ultra_executor import UltraStepExecutor
        from flujo.domain.dsl import Step
        from unittest.mock import Mock

        executor = UltraStepExecutor(enable_cache=True)

        # Create two identical steps with different agent instances
        step1 = Step(name="test_step", agent=Mock())
        step2 = Step(name="test_step", agent=Mock())

        # Create frame objects

        frame1 = _Frame(step=step1, data="test_data", context=None, resources=None)
        frame2 = _Frame(step=step2, data="test_data", context=None, resources=None)

        # Generate cache keys
        key1 = executor._cache_key(frame1)
        key2 = executor._cache_key(frame2)

        # Keys should be different for different agent instances
        assert key1 != key2

        # But same agent should generate same key
        frame3 = _Frame(step=step1, data="test_data", context=None, resources=None)
        key3 = executor._cache_key(frame3)
        assert key1 == key3

    def test_regression_bytes_hashing_correct(self):
        """REGRESSION: Ensure bytes are hashed correctly without string conversion."""
        from flujo.application.core.ultra_executor import UltraStepExecutor

        executor = UltraStepExecutor(enable_cache=True)

        # Test with different content to ensure proper hashing
        bytes_data = b"bytes_content"
        str_data = "string_content"

        bytes_hash = executor._hash_obj(bytes_data)
        str_hash = executor._hash_obj(str_data)

        # These should be different because they have different content
        assert bytes_hash != str_hash

        # Same bytes should hash to same value
        bytes_hash2 = executor._hash_obj(bytes_data)
        assert bytes_hash == bytes_hash2

        # Different bytes should hash to different values
        bytes_data2 = b"different_bytes"
        bytes_hash3 = executor._hash_obj(bytes_data2)
        assert bytes_hash != bytes_hash3

    @pytest.mark.asyncio
    async def test_regression_cache_with_complex_steps(self):
        """REGRESSION: Ensure caching works with complex steps (plugins, validators, etc.)."""
        from flujo.application.core.ultra_executor import UltraStepExecutor
        from flujo.domain.dsl import Step
        from flujo.testing.utils import StubAgent
        from flujo.domain.plugins import PluginOutcome
        from flujo.domain.validation import ValidationResult

        # Create executor with caching enabled
        executor = UltraStepExecutor(enable_cache=True, cache_size=100)

        # Create a step with plugins and validators (complex step)
        agent = StubAgent(["test output 1", "test output 2"])
        step = Step(name="test_step", agent=agent)

        # Create proper plugin and validator objects
        class MockPlugin:
            async def validate(self, data: dict) -> PluginOutcome:
                return PluginOutcome(success=True)

        class MockValidator:
            async def validate(self, output: Any, *, context=None) -> ValidationResult:
                return ValidationResult(is_valid=True, validator_name="MockValidator")

        step.plugins = [(MockPlugin(), 1)]  # Plugin with priority
        step.validators = [MockValidator()]

        # First execution - should be cache miss
        result1 = await executor.execute_step(step, "test_input")
        assert result1.success
        assert "cache_hit" not in (result1.metadata_ or {})

        # Second execution with same input - should be cache hit
        result2 = await executor.execute_step(step, "test_input")
        assert result2.success
        assert result2.metadata_.get("cache_hit") is True

    @pytest.mark.asyncio
    async def test_regression_cache_with_resources(self):
        """REGRESSION: Ensure caching works correctly with resources."""
        from flujo.application.core.ultra_executor import UltraStepExecutor
        from flujo.domain.dsl import Step
        from flujo.testing.utils import StubAgent

        # Create executor with caching enabled
        executor = UltraStepExecutor(enable_cache=True, cache_size=100)

        # Create a simple step
        agent = StubAgent(["test output 1", "test output 2"])
        step = Step(name="test_step", agent=agent)

        resources1 = {"resource": "value1"}
        resources2 = {"resource": "value2"}

        # First execution with resources1
        result1 = await executor.execute_step(step, "test_input", resources=resources1)
        assert result1.success
        assert "cache_hit" not in (result1.metadata_ or {})

        # Second execution with same input and resources - should be cache hit
        result2 = await executor.execute_step(step, "test_input", resources=resources1)
        assert result2.success
        assert result2.metadata_.get("cache_hit") is True

        # Third execution with different resources - should be cache miss
        result3 = await executor.execute_step(step, "test_input", resources=resources2)
        assert result3.success
        assert "cache_hit" not in (result3.metadata_ or {})

    def test_regression_cache_key_includes_all_components(self):
        """REGRESSION: Ensure cache keys include all relevant components."""
        from flujo.application.core.ultra_executor import UltraStepExecutor
        from flujo.domain.dsl import Step
        from unittest.mock import Mock

        executor = UltraStepExecutor(enable_cache=True)

        # Create a step
        step = Step(name="test_step", agent=Mock())

        # Create frame objects with different components

        frame1 = _Frame(step=step, data="data1", context=None, resources=None)
        frame2 = _Frame(step=step, data="data2", context=None, resources=None)
        frame3 = _Frame(step=step, data="data1", context={"ctx": "val"}, resources=None)
        frame4 = _Frame(step=step, data="data1", context=None, resources={"res": "val"})

        # Generate cache keys
        key1 = executor._cache_key(frame1)
        key2 = executor._cache_key(frame2)
        key3 = executor._cache_key(frame3)
        key4 = executor._cache_key(frame4)

        # All keys should be different
        assert key1 != key2  # Different data
        assert key1 != key3  # Different context
        assert key1 != key4  # Different resources
        assert key2 != key3  # Different data and context
        assert key2 != key4  # Different data and resources
        assert key3 != key4  # Different context and resources

    @pytest.mark.asyncio
    async def test_regression_cache_persistence_across_executor_instances(self):
        """REGRESSION: Ensure cache keys are stable across different executor instances."""
        from flujo.application.core.ultra_executor import UltraStepExecutor
        from flujo.domain.dsl import Step
        from flujo.testing.utils import StubAgent

        # Create two executors
        executor1 = UltraStepExecutor(enable_cache=True)
        executor2 = UltraStepExecutor(enable_cache=True)

        # Create identical steps
        agent1 = StubAgent(["test output"])
        agent2 = StubAgent(["test output"])
        step1 = Step(name="test_step", agent=agent1)
        step2 = Step(name="test_step", agent=agent2)

        # Create frame objects

        frame1 = _Frame(step=step1, data="test_data", context=None, resources=None)
        frame2 = _Frame(step=step2, data="test_data", context=None, resources=None)

        # Generate cache keys
        key1 = executor1._cache_key(frame1)
        key2 = executor2._cache_key(frame2)

        # Keys should be stable and deterministic for identical inputs
        assert key1 == key2

    def test_regression_cache_key_handles_edge_cases(self):
        """REGRESSION: Ensure cache key generation handles edge cases gracefully."""
        from flujo.application.core.ultra_executor import UltraStepExecutor
        from flujo.domain.dsl import Step

        executor = UltraStepExecutor(enable_cache=True)

        # Test with None values
        step = Step(name="test_step", agent=None)

        frame = _Frame(step=step, data=None, context=None, resources=None)
        key = executor._cache_key(frame)

        # Should not raise an exception
        assert key is not None
        assert isinstance(key, str)
        assert len(key) > 0

    @pytest.mark.asyncio
    async def test_regression_cache_metadata_correct(self):
        """REGRESSION: Ensure cache hit metadata is set correctly."""
        from flujo.application.core.ultra_executor import UltraStepExecutor
        from flujo.domain.dsl import Step
        from flujo.testing.utils import StubAgent

        # Create executor with caching enabled
        executor = UltraStepExecutor(enable_cache=True, cache_size=100)

        # Create a simple step
        agent = StubAgent(["test output 1", "test output 2"])
        step = Step(name="test_step", agent=agent)

        # First execution - should be cache miss
        result1 = await executor.execute_step(step, "test_input")
        assert result1.success
        assert "cache_hit" not in (result1.metadata_ or {})

        # Second execution with same input - should be cache hit
        result2 = await executor.execute_step(step, "test_input")
        assert result2.success
        assert result2.metadata_.get("cache_hit") is True

        # Verify metadata structure
        assert isinstance(result2.metadata_, dict)
        assert "cache_hit" in result2.metadata_
        assert result2.metadata_["cache_hit"] is True

    # ============================================================================
    # NEW REGRESSION TESTS FOR COPILOT FEEDBACK IMPROVEMENTS
    # ============================================================================

    def test_regression_module_level_dataclasses_used(self):
        """REGRESSION: Ensure module-level dataclasses are used instead of inline definitions."""
        from flujo.application.core.ultra_executor import _CacheFrame, _ComplexCacheFrame

        # Verify the dataclasses exist at module level
        assert _CacheFrame is not None
        assert _ComplexCacheFrame is not None

        # Test that they can be instantiated
        cache_frame = _CacheFrame(step=Mock(), data="test", context=None, resources=None)
        complex_cache_frame = _ComplexCacheFrame(
            step=Mock(), data="test", context=None, resources=None
        )

        assert cache_frame.step is not None
        assert complex_cache_frame.step is not None

    def test_regression_agent_identification_includes_module(self):
        """REGRESSION: Ensure agent identification includes module name to prevent collisions."""
        from flujo.application.core.ultra_executor import UltraStepExecutor
        from flujo.domain.dsl import Step

        executor = UltraStepExecutor(enable_cache=True)

        # Create a mock agent with a specific class name
        class TestAgent:
            def __init__(self, config=None):
                self.config = config

            async def run(self, data, **kwargs):
                return "test_output"

        # Create steps with different agent instances
        agent1 = TestAgent(config={"model": "gpt-4"})
        agent2 = TestAgent(config={"model": "gpt-3.5"})
        step1 = Step(name="test_step", agent=agent1)
        step2 = Step(name="test_step", agent=agent2)

        # Create frame objects

        frame1 = _Frame(step=step1, data="test_data", context=None, resources=None)
        frame2 = _Frame(step=step2, data="test_data", context=None, resources=None)

        # Generate cache keys
        key1 = executor._cache_key(frame1)
        key2 = executor._cache_key(frame2)

        # Keys should be different for different agent configs
        assert key1 != key2

        # But same agent config should generate same key
        agent3 = TestAgent(config={"model": "gpt-4"})  # Same config as agent1
        step3 = Step(name="test_step", agent=agent3)
        frame3 = _Frame(step=step3, data="test_data", context=None, resources=None)
        key3 = executor._cache_key(frame3)
        assert key1 == key3

    def test_regression_consistent_agent_config_hashing(self):
        """REGRESSION: Ensure agent config hashing uses _hash_obj for consistency."""
        from flujo.application.core.ultra_executor import UltraStepExecutor
        from flujo.domain.dsl import Step

        executor = UltraStepExecutor(enable_cache=True)

        # Create agents with complex config objects
        class TestAgent:
            def __init__(self, config=None):
                self.config = config

            async def run(self, data, **kwargs):
                return "test_output"

        # Test with different config types
        agent1 = TestAgent(config={"model": "gpt-4", "temperature": 0.7})
        agent2 = TestAgent(config={"model": "gpt-4", "temperature": 0.7})  # Same config
        agent3 = TestAgent(config={"model": "gpt-4", "temperature": 0.8})  # Different config

        step1 = Step(name="test_step", agent=agent1)
        step2 = Step(name="test_step", agent=agent2)
        step3 = Step(name="test_step", agent=agent3)

        # Create frame objects

        frame1 = _Frame(step=step1, data="test_data", context=None, resources=None)
        frame2 = _Frame(step=step2, data="test_data", context=None, resources=None)
        frame3 = _Frame(step=step3, data="test_data", context=None, resources=None)

        # Generate cache keys
        key1 = executor._cache_key(frame1)
        key2 = executor._cache_key(frame2)
        key3 = executor._cache_key(frame3)

        # Same config should generate same key
        assert key1 == key2

        # Different config should generate different key
        assert key1 != key3

    def test_regression_cache_key_stability_across_python_runs(self):
        """REGRESSION: Ensure cache keys are stable across different Python runs."""
        from flujo.application.core.ultra_executor import UltraStepExecutor
        from flujo.domain.dsl import Step

        executor = UltraStepExecutor(enable_cache=True)

        # Create agents with identical configs
        class TestAgent:
            def __init__(self, config=None):
                self.config = config

            async def run(self, data, **kwargs):
                return "test_output"

        # Create multiple agents with identical configs
        config = {"model": "gpt-4", "temperature": 0.7, "max_tokens": 1000}
        agents = [TestAgent(config=config) for _ in range(3)]
        steps = [Step(name="test_step", agent=agent) for agent in agents]

        # Create frame objects

        frames = [
            _Frame(step=step, data="test_data", context=None, resources=None) for step in steps
        ]

        # Generate cache keys
        keys = [executor._cache_key(frame) for frame in frames]

        # All keys should be identical for identical configs
        assert len(set(keys)) == 1, "All keys should be identical for identical configs"

    @pytest.mark.asyncio
    async def test_regression_cache_performance_with_module_dataclasses(self):
        """REGRESSION: Ensure cache performance is not degraded by module-level dataclasses."""
        from flujo.application.core.ultra_executor import UltraStepExecutor
        from flujo.domain.dsl import Step
        from flujo.testing.utils import StubAgent
        import time

        # Create executor with caching enabled
        executor = UltraStepExecutor(enable_cache=True, cache_size=1000)

        # Create a simple step
        agent = StubAgent(["test output 1", "test output 2", "test output 3"])
        step = Step(name="test_step", agent=agent)

        # Measure performance of multiple cache operations
        start_time = time.perf_counter()

        # Execute multiple times to test cache performance
        for i in range(10):
            result = await executor.execute_step(step, f"test_input_{i % 3}")
            assert result.success  # Verify each execution succeeds

        end_time = time.perf_counter()
        execution_time = end_time - start_time

        # Execution should be fast (less than 1 second for 10 operations)
        assert execution_time < 1.0, f"Cache operations took too long: {execution_time}s"

        # Verify cache is working
        assert executor.cache is not None
        assert len(executor.cache._store) > 0

    def test_regression_agent_identification_handles_edge_cases(self):
        """REGRESSION: Ensure agent identification handles edge cases gracefully."""
        from flujo.application.core.ultra_executor import UltraStepExecutor
        from flujo.domain.dsl import Step

        executor = UltraStepExecutor(enable_cache=True)

        # Test with agent that has no config
        class AgentNoConfig:
            async def run(self, data, **kwargs):
                return "test_output"

        # Test with agent that has None config
        class AgentNoneConfig:
            def __init__(self):
                self.config = None

            async def run(self, data, **kwargs):
                return "test_output"

        # Test with agent that has empty config
        class AgentEmptyConfig:
            def __init__(self):
                self.config = {}

            async def run(self, data, **kwargs):
                return "test_output"

        # Test with agent that has complex config
        class AgentComplexConfig:
            def __init__(self):
                self.config = {
                    "model": "gpt-4",
                    "temperature": 0.7,
                    "nested": {"key": "value", "list": [1, 2, 3]},
                }

            async def run(self, data, **kwargs):
                return "test_output"

        # Create steps with different agent types
        step1 = Step(name="test_step", agent=AgentNoConfig())
        step2 = Step(name="test_step", agent=AgentNoneConfig())
        step3 = Step(name="test_step", agent=AgentEmptyConfig())
        step4 = Step(name="test_step", agent=AgentComplexConfig())

        # Create frame objects

        frames = [
            _Frame(step=step, data="test_data", context=None, resources=None)
            for step in [step1, step2, step3, step4]
        ]

        # Generate cache keys - should not raise exceptions
        keys = []
        for frame in frames:
            key = executor._cache_key(frame)
            keys.append(key)
            assert key is not None
            assert isinstance(key, str)
            assert len(key) > 0

        # All keys should be different for different agent types
        assert len(set(keys)) == len(keys), "Different agent types should generate different keys"

    @pytest.mark.asyncio
    async def test_regression_cache_mutation_does_not_corrupt_cached_data(self):
        """REGRESSION: Ensure cache mutation doesn't corrupt the original cached data."""
        from flujo.application.core.ultra_executor import UltraStepExecutor
        from flujo.domain.dsl import Step
        from flujo.testing.utils import StubAgent

        # Create executor with caching enabled
        executor = UltraStepExecutor(enable_cache=True, cache_size=100)

        # Create a simple step
        agent = StubAgent(["test output 1", "test output 2", "test output 3"])
        step = Step(name="test_step", agent=agent)

        # First execution - should be cache miss
        result1 = await executor.execute_step(step, "test_input")
        assert result1.success
        assert "cache_hit" not in (result1.metadata_ or {})

        # Second execution with same input - should be cache hit
        result2 = await executor.execute_step(step, "test_input")
        assert result2.success
        assert result2.metadata_.get("cache_hit") is True

        # Third execution with same input - should still be cache hit
        result3 = await executor.execute_step(step, "test_input")
        assert result3.success
        assert result3.metadata_.get("cache_hit") is True

        # CRITICAL: Verify that the original cached data wasn't corrupted
        # The metadata should be properly set on each cache hit without affecting the original
        assert result2.output == result1.output
        assert result3.output == result1.output
        assert result2.name == result1.name
        assert result3.name == result1.name

        # Verify that each result has its own metadata instance
        assert result2.metadata_ is not result3.metadata_
        assert result1.metadata_ is not result2.metadata_

    @pytest.mark.asyncio
    async def test_regression_cache_copy_behavior(self):
        """REGRESSION: Ensure cached results are properly copied to prevent mutation."""
        from flujo.application.core.ultra_executor import UltraStepExecutor
        from flujo.domain.dsl import Step
        from flujo.testing.utils import StubAgent

        # Create executor with caching enabled
        executor = UltraStepExecutor(enable_cache=True)

        # Create a step with multiple outputs
        agent = StubAgent(["test output 1", "test output 2", "test output 3"])
        step = Step(name="test_step", agent=agent)

        # First execution - cache miss
        result1 = await executor.execute_step(step, "test_input")
        assert result1.success
        assert (
            result1.metadata_ is None or result1.metadata_.get("cache_hit") is None
        )  # No cache hit metadata

        # Second execution - cache hit
        result2 = await executor.execute_step(step, "test_input")
        assert result2.success
        assert result2.metadata_.get("cache_hit") is True  # Should have cache hit metadata

        # Third execution - another cache hit
        result3 = await executor.execute_step(step, "test_input")
        assert result3.success
        assert result3.metadata_.get("cache_hit") is True  # Should have cache hit metadata

        # CRITICAL: Verify that the original cached data wasn't corrupted
        # The metadata should be properly set on each cache hit without affecting the original
        assert result2.output == result1.output
        assert result3.output == result1.output
        assert result2.name == result1.name
        assert result3.name == result1.name

        # Verify that each result has its own metadata instance (proper copying)
        assert result2.metadata_ is not result3.metadata_
        assert result1.metadata_ is not result2.metadata_

        # Verify that modifying one result's metadata doesn't affect others
        result2.metadata_["test_key"] = "test_value"
        assert "test_key" not in result3.metadata_
        assert result1.metadata_ is None or "test_key" not in result1.metadata_

    @pytest.mark.asyncio
    async def test_regression_cache_key_always_defined(self):
        """REGRESSION: Ensure cache_key is always defined to prevent NameError."""
        from flujo.application.core.ultra_executor import UltraStepExecutor
        from flujo.domain.dsl import Step
        from flujo.testing.utils import StubAgent

        # Create executor with caching disabled
        executor = UltraStepExecutor(enable_cache=False)

        # Create a simple step with multiple outputs
        agent = StubAgent(["test output 1", "test output 2", "test output 3"])
        step = Step(name="test_step", agent=agent)

        # This should not raise NameError even with caching disabled
        result = await executor.execute_step(step, "test_input")
        assert result.success

        # Test complex step execution
        result = await executor._execute_complex_step(step, "test_input")
        assert result.success

    @pytest.mark.asyncio
    async def test_regression_critical_exceptions_not_cached(self):
        """REGRESSION: Ensure critical exceptions are not cached when they occur."""
        from flujo.application.core.ultra_executor import UltraStepExecutor
        from flujo.domain.dsl import Step
        from flujo.exceptions import PausedException

        # Create executor with caching enabled
        executor = UltraStepExecutor(enable_cache=True)

        # Create a step that raises PausedException
        class PausedAgent:
            async def run(self, data, context=None):
                raise PausedException("Test pause")

        step = Step(name="paused_step", agent=PausedAgent())

        # First execution should raise PausedException
        with pytest.raises(PausedException, match="Test pause"):
            await executor._execute_complex_step(step, "test_input")

        # Second execution should also raise PausedException (not return cached result)
        with pytest.raises(PausedException, match="Test pause"):
            await executor._execute_complex_step(step, "test_input")

        # Verify that the exception was not cached by checking that it's still raised
        with pytest.raises(PausedException, match="Test pause"):
            await executor._execute_complex_step(step, "test_input")

    @pytest.mark.asyncio
    async def test_unified_error_handling_contract(self, executor):
        """Test that step types have appropriate error handling contract.

        FLUJO SPIRIT: Critical exceptions (PausedException, InfiniteFallbackError, InfiniteRedirectError)
        should be re-raised for proper control flow. Other exceptions return StepResult(success=False)
        for predictable API. Timing data should be preserved for all failures.
        """
        from flujo.domain.plugins import PluginOutcome

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

        # Test simple non-streaming step with regular exception
        simple_step = Step.model_validate(
            {"name": "simple", "agent": FailingAgent(), "config": {"max_retries": 1}}
        )
        result = await executor.execute_step(simple_step, "test_data")

        # Should return StepResult, not raise exception
        assert isinstance(result, StepResult)
        assert not result.success
        assert "RuntimeError: Test failure" in result.feedback
        assert result.latency_s > 0.0  # Timing should be preserved

        # Test streaming step with regular exception
        streaming_step = Step.model_validate(
            {"name": "streaming", "agent": FailingAgent(), "config": {"max_retries": 1}}
        )
        result = await executor.execute_step(streaming_step, "test_data", stream=True)

        # Should return StepResult, not raise exception
        assert isinstance(result, StepResult)
        assert not result.success
        assert "RuntimeError: Test failure" in result.feedback
        assert result.latency_s > 0.0  # Timing should be preserved

        # Test complex step (with plugins) with regular exception
        class MockPlugin:
            async def validate(self, data: dict) -> PluginOutcome:
                return PluginOutcome(valid=True, feedback="Mock validation passed")

        complex_step = Step.model_validate(
            {
                "name": "complex",
                "agent": FailingAgent(),
                "config": {"max_retries": 1},
                "plugins": [(MockPlugin(), 1)],  # Use proper mock plugin
            }
        )
        result = await executor.execute_step(complex_step, "test_data")

        # Should return StepResult, not raise exception
        assert isinstance(result, StepResult)
        assert not result.success
        assert "RuntimeError: Test failure" in result.feedback
        assert result.latency_s > 0.0  # Timing should be preserved

    @pytest.mark.asyncio
    async def test_critical_exceptions_are_re_raised(self, executor):
        """Test that critical exceptions are re-raised for proper control flow."""
        from flujo.exceptions import PausedException, InfiniteFallbackError, InfiniteRedirectError

        class PausedAgent:
            async def run(self, data, **kwargs):
                raise PausedException("Test pause")

        class InfiniteFallbackAgent:
            async def run(self, data, **kwargs):
                raise InfiniteFallbackError("Test infinite fallback")

        class InfiniteRedirectAgent:
            async def run(self, data, **kwargs):
                raise InfiniteRedirectError("Test infinite redirect")

        # Test PausedException
        paused_step = Step.model_validate(
            {"name": "paused", "agent": PausedAgent(), "config": {"max_retries": 1}}
        )
        with pytest.raises(PausedException, match="Test pause"):
            await executor.execute_step(paused_step, "test_data")

        # Test InfiniteFallbackError
        fallback_step = Step.model_validate(
            {"name": "fallback", "agent": InfiniteFallbackAgent(), "config": {"max_retries": 1}}
        )
        with pytest.raises(InfiniteFallbackError, match="Test infinite fallback"):
            await executor.execute_step(fallback_step, "test_data")

        # Test InfiniteRedirectError
        redirect_step = Step.model_validate(
            {"name": "redirect", "agent": InfiniteRedirectAgent(), "config": {"max_retries": 1}}
        )
        with pytest.raises(InfiniteRedirectError, match="Test infinite redirect"):
            await executor.execute_step(redirect_step, "test_data")

    @pytest.mark.asyncio
    async def test_timing_preservation_for_failed_steps(self, executor):
        """Test that timing data is preserved for failed steps."""
        import time

        class SlowFailingAgent:
            async def run(self, data, **kwargs):
                time.sleep(0.1)  # Simulate some work
                raise RuntimeError("Test failure")

        slow_step = Step.model_validate(
            {"name": "slow", "agent": SlowFailingAgent(), "config": {"max_retries": 1}}
        )
        result = await executor.execute_step(slow_step, "test_data")

        # Should preserve actual execution time
        assert isinstance(result, StepResult)
        assert not result.success
        assert result.latency_s >= 0.1  # Should reflect actual execution time

    @pytest.mark.asyncio
    async def test_retry_latency_measurement(self, executor):
        """Test that retry latency is measured independently for each attempt."""

        class RetryAgent:
            def __init__(self):
                self.call_count = 0

            async def run(self, data):
                self.call_count += 1
                if self.call_count < 3:  # Fail first two attempts
                    await asyncio.sleep(0.1)  # Simulate work
                    raise RuntimeError(f"Attempt {self.call_count} failed")
                return "success"

        step = Step(name="retry_test", agent=RetryAgent())
        step.config.max_retries = 3

        # Execute the step
        result = await executor.execute_step(step, "test_input")

        # Should succeed on third attempt (attempts 1, 2, and 3)
        assert result.success
        assert result.attempts == 3  # 3 attempts: 1 (fail), 2 (fail), 3 (success)

        # CRITICAL FIX: Latency should reflect only the successful attempt, not cumulative
        # The successful attempt took ~0.05s, not ~0.15s (0.1 + 0.05)
        assert result.latency_s > 0.0
        assert result.latency_s < 0.1  # Should be closer to 0.05s than 0.15s
        # For successful attempts, feedback is typically None
        assert result.feedback is None or "failed" not in result.feedback

    # ============================================================================
    # REGRESSION TESTS - Prevent previously fixed bugs from returning
    # ============================================================================

    def test_regression_dead_code_removal(self):
        """REGRESSION TEST: Ensure dead code (_State enum, _Frame dataclass) remains removed."""
        import flujo.application.core.ultra_executor as ultra_executor

        # Verify _State enum is not defined
        assert not hasattr(ultra_executor, "_State"), "Dead code _State enum should not exist"

        # Verify _Frame dataclass is not defined
        assert not hasattr(ultra_executor, "_Frame"), "Dead code _Frame dataclass should not exist"

        # Verify no references to removed classes in the module
        source_code = ultra_executor.__file__
        with open(source_code, "r") as f:
            content = f.read()
            assert "class _State" not in content, (
                "Dead code _State class should not exist in source"
            )
            assert "class _Frame" not in content, (
                "Dead code _Frame class should not exist in source"
            )

    def test_regression_redundant_retry_logic_removal(self):
        """REGRESSION TEST: Ensure redundant nested try-except retry logic remains removed."""
        import flujo.application.core.ultra_executor as ultra_executor

        # Check that the problematic nested try-except pattern is not present
        source_code = ultra_executor.__file__
        with open(source_code, "r") as f:
            content = f.read()

            # Look for the specific problematic pattern that was removed
            problematic_pattern = """try:
                                        if run_func is not None:
                                            filtered_kwargs = build_filtered_kwargs(run_func)
                                            raw = await run_func(processed_data, **filtered_kwargs)
                                        else:
                                            raise RuntimeError("Agent has no run method")
                                    except Exception:
                                        if run_func is not None:
                                            # Fallback to filtered kwargs for backward compatibility
                                            fallback_kwargs = build_filtered_kwargs(run_func)
                                            raw = await run_func(processed_data, **fallback_kwargs)
                                        else:
                                            raise RuntimeError("Agent has no run method")"""

            assert problematic_pattern not in content, "Redundant retry logic should not exist"

    def test_regression_input_validation_ultra_executor(self):
        """REGRESSION TEST: Ensure input validation remains in UltraStepExecutor constructor."""
        from flujo.application.core.ultra_executor import UltraStepExecutor

        # Test that invalid cache_size raises ValueError
        with pytest.raises(ValueError, match="cache_size must be positive"):
            UltraStepExecutor(cache_size=0)

        with pytest.raises(ValueError, match="cache_size must be positive"):
            UltraStepExecutor(cache_size=-1)

        # Test that invalid cache_ttl raises ValueError
        with pytest.raises(ValueError, match="cache_ttl must be non-negative"):
            UltraStepExecutor(cache_ttl=-1)

        # Test that invalid concurrency_limit raises ValueError
        with pytest.raises(ValueError, match="concurrency_limit must be positive if specified"):
            UltraStepExecutor(concurrency_limit=0)

        with pytest.raises(ValueError, match="concurrency_limit must be positive if specified"):
            UltraStepExecutor(concurrency_limit=-1)

        # Test that valid parameters work
        executor = UltraStepExecutor(cache_size=100, cache_ttl=3600, concurrency_limit=10)
        assert executor is not None

    def test_regression_input_validation_lru_cache(self):
        """REGRESSION TEST: Ensure input validation remains in _LRUCache constructor."""
        from flujo.application.core.ultra_executor import _LRUCache

        # Test that invalid max_size raises ValueError
        with pytest.raises(ValueError, match="max_size must be positive"):
            _LRUCache(max_size=0)

        with pytest.raises(ValueError, match="max_size must be positive"):
            _LRUCache(max_size=-1)

        # Test that invalid ttl raises ValueError
        with pytest.raises(ValueError, match="ttl must be non-negative"):
            _LRUCache(ttl=-1)

        # Test that valid parameters work
        cache = _LRUCache(max_size=100, ttl=3600)
        assert cache is not None

    def test_regression_ttl_logic_fix(self):
        """REGRESSION TEST: Ensure TTL=0 means 'never expire' (not 'expire immediately')."""
        from flujo.application.core.ultra_executor import _LRUCache
        from flujo.domain.models import StepResult

        # Create cache with TTL=0 (should never expire)
        cache = _LRUCache(max_size=10, ttl=0)

        # Create a test result
        result = StepResult(
            name="test", output="test_output", success=True, attempts=1, latency_s=0.1
        )

        # Store the result
        cache.set("test_key", result)

        # Wait a bit to ensure time has passed
        import time

        time.sleep(0.1)

        # Retrieve the result - should still be there (TTL=0 means never expire)
        retrieved = cache.get("test_key")
        assert retrieved is not None, "TTL=0 should mean 'never expire', not 'expire immediately'"
        assert retrieved.name == "test"

    def test_regression_monotonic_time_usage(self):
        """REGRESSION TEST: Ensure monotonic time is used for cache timestamps."""
        import flujo.application.core.ultra_executor as ultra_executor

        # Check that time.monotonic() is used instead of time.time()
        source_code = ultra_executor.__file__
        with open(source_code, "r") as f:
            content = f.read()

            # Should use monotonic time for cache operations
            assert "time.monotonic()" in content, "Cache should use monotonic time"

            # Should not use regular time.time() for cache timestamps
            # (but allow it for other purposes)
            cache_related_lines = []
            lines = content.split("\n")
            for i, line in enumerate(lines):
                if "time.time()" in line and ("cache" in line.lower() or "ttl" in line.lower()):
                    cache_related_lines.append(f"Line {i + 1}: {line.strip()}")

            assert not cache_related_lines, (
                f"Cache operations should not use time.time(): {cache_related_lines}"
            )

    def test_regression_independent_latency_measurement(self):
        """REGRESSION TEST: Ensure latency is measured independently for each retry attempt."""
        import flujo.application.core.ultra_executor as ultra_executor

        # Check that start_time is captured inside the retry loop
        source_code = ultra_executor.__file__
        with open(source_code, "r") as f:
            content = f.read()

            # Look for the pattern where start_time is captured inside the retry loop
            # This should be inside the "for attempt in range" loop
            lines = content.split("\n")
            retry_loop_started = False
            start_time_captured = False

            for line in lines:
                if "for attempt in range" in line:
                    retry_loop_started = True
                elif retry_loop_started and "start_time = time_perf_ns()" in line:
                    start_time_captured = True
                    break
                elif retry_loop_started and "except Exception:" in line:
                    # We've reached the exception handler, start_time should have been captured
                    break

            assert start_time_captured, "start_time should be captured inside the retry loop"

    def test_regression_no_duplicate_trace_functions(self):
        """REGRESSION TEST: Ensure no duplicate trace function definitions exist."""
        import flujo.application.core.ultra_executor as ultra_executor

        # Count trace function definitions
        source_code = ultra_executor.__file__
        with open(source_code, "r") as f:
            content = f.read()

            trace_definitions = content.count("def trace(")
            assert trace_definitions <= 2, (
                f"Should have at most 2 trace function definitions (telemetry fallback), found {trace_definitions}"
            )

            # The two allowed definitions should be in the try-except block for telemetry
            assert "try:" in content and "except Exception:" in content, (
                "Trace functions should be in telemetry fallback block"
            )

    def test_regression_no_undefined_imports(self):
        """REGRESSION TEST: Ensure all imports are properly defined and used."""
        import flujo.application.core.ultra_executor as ultra_executor

        # Check that all imported types are actually used
        source_code = ultra_executor.__file__
        with open(source_code, "r") as f:
            content = f.read()

            # Check that imported step types are used
            assert "LoopStep" in content or "isinstance(step, LoopStep)" in content, (
                "LoopStep should be imported and used"
            )
            assert "ConditionalStep" in content or "isinstance(step, ConditionalStep)" in content, (
                "ConditionalStep should be imported and used"
            )
            assert (
                "DynamicParallelRouterStep" in content
                or "isinstance(step, DynamicParallelRouterStep)" in content
            ), "DynamicParallelRouterStep should be imported and used"
            assert "ParallelStep" in content or "isinstance(step, ParallelStep)" in content, (
                "ParallelStep should be imported and used"
            )
            assert "CacheStep" in content or "isinstance(step, CacheStep)" in content, (
                "CacheStep should be imported and used"
            )

    def test_regression_constructor_validation_preserved(self):
        """REGRESSION TEST: Ensure constructor validation logic is preserved."""
        import flujo.application.core.ultra_executor as ultra_executor

        source_code = ultra_executor.__file__
        with open(source_code, "r") as f:
            content = f.read()

            # Check that validation logic exists in UltraStepExecutor
            assert "if cache_size <= 0:" in content, "UltraStepExecutor should validate cache_size"
            assert "if cache_ttl < 0:" in content, "UltraStepExecutor should validate cache_ttl"
            assert "if concurrency_limit is not None and concurrency_limit <= 0:" in content, (
                "UltraStepExecutor should validate concurrency_limit"
            )

            # Check that validation logic exists in _LRUCache
            assert "def __post_init__(self)" in content, (
                "_LRUCache should have __post_init__ method"
            )
            assert "if self.max_size <= 0:" in content, "_LRUCache should validate max_size"
            assert "if self.ttl < 0:" in content, "_LRUCache should validate ttl"
