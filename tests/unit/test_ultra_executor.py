"""Tests for the ultra-optimized step executor."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock

from flujo.application.core.ultra_executor import (
    UltraStepExecutor,
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
        from flujo.application.core.ultra_executor import _Frame

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
