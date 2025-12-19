"""
Comprehensive tests for UltraStepExecutor v2 modular architecture.

Tests cover:
- All interfaces and their default implementations
- ExecutorCore functionality
- Backward compatibility with UltraStepExecutor wrapper
- Cache key determinism
- Usage tracking and limits
- Error handling and retry logic
- Complex step handling
"""

import asyncio
import pytest
from unittest.mock import Mock
from typing import Any

from flujo.application.core.executor_core import (
    # Interfaces
    OrjsonSerializer,
    Blake3Hasher,
    InMemoryLRUBackend,
    ThreadSafeMeter,
    DefaultProcessorPipeline,
    DefaultValidatorRunner,
    ExecutorCore,
    DefaultCacheKeyGenerator as CacheKeyGenerator,
)
from flujo.domain.models import StepResult, UsageLimits, BaseModel
from flujo.domain.dsl.step import Step, StepConfig
from flujo.domain.validation import ValidationResult
from flujo.exceptions import (
    PausedException,
)
from tests.test_types.fixtures import execute_simple_step

# Alias for backward compatibility
UltraStepExecutor = ExecutorCore


class UltraTestContext(BaseModel):
    """Test context model."""

    value: str = "test"
    count: int = 0


class MockAgent:
    """Mock agent for testing."""

    def __init__(self, output: Any = "test output", should_fail: bool = False):
        self.output = output
        self.should_fail = should_fail
        self.call_count = 0
        self.model_id = "gpt-4o"  # Add model_id for cost extraction

    async def run(self, data: Any, **kwargs) -> Any:
        self.call_count += 1
        if self.should_fail and self.call_count == 1:
            raise RuntimeError("Agent failed")
        # Ensure we return the actual output, not a Mock object
        if hasattr(self.output, "_mock_name"):
            return "test output"  # Fallback for Mock objects
        return self.output


class MockProcessor:
    """Mock processor for testing."""

    def __init__(self, transform_func=None):
        self.transform_func = transform_func or (lambda x: f"processed_{x}")

    async def process(self, data: Any, context: Any = None) -> Any:
        return self.transform_func(data)


class MockValidator:
    """Mock validator for testing."""

    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail

    async def validate(self, data: Any, context: Any = None) -> ValidationResult:
        if self.should_fail:
            return ValidationResult(
                is_valid=False,
                feedback="Validation failed",
                validator_name="MockValidator",  # Add required field
            )
        return ValidationResult(
            is_valid=True,
            feedback="Validation passed",
            validator_name="MockValidator",  # Add required field
        )


# --------------------------------------------------------------------------- #
# ★ Interface Tests
# --------------------------------------------------------------------------- #


class TestSerializerInterface:
    """Test serializer implementations."""

    def test_orjson_serializer(self):
        """Test OrjsonSerializer implementation."""
        serializer = OrjsonSerializer()

        # Test basic types
        data = {"key": "value", "number": 42, "list": [1, 2, 3]}
        serialized = serializer.serialize(data)
        assert isinstance(serialized, bytes)

        deserialized = serializer.deserialize(serialized)
        assert deserialized == data

    def test_serializer_determinism(self):
        """Test that serialization is deterministic."""
        serializer = OrjsonSerializer()

        data = {"b": 2, "a": 1, "c": 3}  # Unordered dict
        result1 = serializer.serialize(data)
        result2 = serializer.serialize(data)

        assert result1 == result2  # Should be identical due to sort_keys


class TestHasherInterface:
    """Test hasher implementations."""

    def test_blake3_hasher(self):
        """Test Blake3Hasher implementation."""
        hasher = Blake3Hasher()

        data = b"test data"
        digest1 = hasher.digest(data)
        digest2 = hasher.digest(data)

        assert digest1 == digest2  # Deterministic
        assert isinstance(digest1, str)
        assert len(digest1) >= 32  # Reasonable length

    def test_hasher_determinism(self):
        """Test that hashing is deterministic across instances."""
        hasher1 = Blake3Hasher()
        hasher2 = Blake3Hasher()

        data = b"test data"
        assert hasher1.digest(data) == hasher2.digest(data)


class TestCacheBackendInterface:
    """Test cache backend implementations."""

    @pytest.mark.asyncio
    async def test_in_memory_lru_backend(self):
        """Test InMemoryLRUBackend implementation."""
        cache = InMemoryLRUBackend(max_size=2, ttl_s=1)

        result1 = StepResult(name="test1", output="output1", success=True)
        result2 = StepResult(name="test2", output="output2", success=True)
        result3 = StepResult(name="test3", output="output3", success=True)

        # Test basic put/get
        await cache.put("key1", result1, ttl_s=1)
        retrieved = await cache.get("key1")
        assert retrieved is not None
        assert retrieved.name == "test1"
        assert retrieved.output == "output1"

        # Test LRU eviction
        await cache.put("key2", result2, ttl_s=1)
        await cache.put("key3", result3, ttl_s=1)  # Should evict key1

        assert await cache.get("key1") is None  # Evicted
        assert await cache.get("key2") is not None
        assert await cache.get("key3") is not None

    @pytest.mark.asyncio
    async def test_cache_ttl(self):
        """Test cache TTL functionality."""
        cache = InMemoryLRUBackend(max_size=10, ttl_s=0.1)  # Very short TTL

        result = StepResult(name="test", output="output", success=True)
        await cache.put("key", result, ttl_s=0.1)

        # Should be available immediately
        assert await cache.get("key") is not None

        # Wait for TTL to expire
        await asyncio.sleep(0.2)
        assert await cache.get("key") is None

    @pytest.mark.asyncio
    async def test_cache_mutation_protection(self):
        """Test that cache returns copies to prevent mutation."""
        cache = InMemoryLRUBackend(max_size=10, ttl_s=3600)

        original = StepResult(name="test", output="original", success=True)
        await cache.put("key", original, ttl_s=3600)

        # Get cached result and modify it
        retrieved = await cache.get("key")
        assert retrieved is not None
        retrieved.metadata_ = {"modified": True}

        # Get again - should not have the modification
        retrieved2 = await cache.get("key")
        assert retrieved2 is not None
        assert retrieved2.metadata_ != {"modified": True}


class TestUsageMeterInterface:
    """Test usage meter implementations."""

    @pytest.mark.asyncio
    async def test_thread_safe_meter(self):
        """Test ThreadSafeMeter implementation."""
        meter = ThreadSafeMeter()

        # Test basic addition
        await meter.add(1.0, 100, 50)
        cost, prompt_tokens, completion_tokens = await meter.snapshot()

        assert cost == 1.0
        assert prompt_tokens == 100
        assert completion_tokens == 50

        # Test cumulative addition
        await meter.add(0.5, 25, 25)
        cost, prompt_tokens, completion_tokens = await meter.snapshot()

        assert cost == 1.5
        assert prompt_tokens == 125
        assert completion_tokens == 75

    @pytest.mark.asyncio
    async def test_usage_limit_enforcement(self):
        """Guard is compatibility-only; quota enforces limits elsewhere."""
        meter = ThreadSafeMeter()

        # Add usage that exceeds limits
        await meter.add(10.0, 1000, 500)

        limits = UsageLimits(total_cost_usd_limit=5.0, total_tokens_limit=1000)
        assert await meter.guard(limits) is None
        # Ensure tracked usage remains intact for reconciliation/telemetry
        cost, prompt_tokens, completion_tokens = await meter.snapshot()
        assert cost == 10.0
        assert prompt_tokens == 1000
        assert completion_tokens == 500

    @pytest.mark.asyncio
    async def test_concurrent_usage_tracking(self):
        """Test thread safety of usage meter."""
        meter = ThreadSafeMeter()

        # Run concurrent operations
        tasks = []
        for i in range(10):
            task = meter.add(0.1, 10, 5)
            tasks.append(task)

        await asyncio.gather(*tasks)

        cost, prompt_tokens, completion_tokens = await meter.snapshot()
        assert abs(cost - 1.0) < 1e-10  # 10 * 0.1 (handle floating point precision)
        assert prompt_tokens == 100  # 10 * 10
        assert completion_tokens == 50  # 10 * 5


# --------------------------------------------------------------------------- #
# ★ Cache Key Generation Tests
# --------------------------------------------------------------------------- #


class TestCacheKeyGeneration:
    """Test deterministic cache key generation."""

    def test_cache_key_determinism(self):
        """Test that cache keys are deterministic."""
        hasher = Blake3Hasher()
        generator = CacheKeyGenerator(hasher)

        # Create a test step
        step = Mock()
        step.name = "test_step"
        step.agent = None

        data = {"input": "test"}
        context = UltraTestContext(value="test")

        # Generate key multiple times
        key1 = generator.generate_key(step, data, context, None)
        key2 = generator.generate_key(step, data, context, None)

        assert key1 == key2

    def test_cache_key_uniqueness(self):
        """Test that different inputs produce different keys."""
        hasher = Blake3Hasher()
        generator = CacheKeyGenerator(hasher)

        step = Mock()
        step.name = "test_step"
        step.agent = None

        # Different data should produce different keys
        key1 = generator.generate_key(step, {"input": "test1"}, None, None)
        key2 = generator.generate_key(step, {"input": "test2"}, None, None)

        assert key1 != key2

    def test_agent_id_stability(self):
        """Test that agent IDs are stable across instances."""
        hasher = Blake3Hasher()
        generator = CacheKeyGenerator(hasher)

        # Create agents with same configuration
        agent1 = Mock()
        agent1.config = {"model": "gpt-4", "temperature": 0.7}

        agent2 = Mock()
        agent2.config = {"model": "gpt-4", "temperature": 0.7}

        step1 = Mock()
        step1.name = "test"
        step1.agent = agent1

        step2 = Mock()
        step2.name = "test"
        step2.agent = agent2

        key1 = generator.generate_key(step1, "data", None, None)
        key2 = generator.generate_key(step2, "data", None, None)

        assert key1 == key2  # Same config should produce same key


# --------------------------------------------------------------------------- #
# ★ ExecutorCore Tests
# --------------------------------------------------------------------------- #


class TestExecutorCore:
    """Test the core executor functionality."""

    @pytest.mark.asyncio
    async def test_basic_step_execution(self):
        """Test basic step execution."""
        executor = ExecutorCore(enable_cache=False)

        agent = MockAgent("test output")

        # Create a proper step object instead of using Mock
        class TestStep:
            def __init__(self):
                self.name = "test_step"
                self.agent = agent
                self.config = type("Config", (), {"max_retries": 3, "temperature": None})()
                self.processors = type(
                    "Processors", (), {"prompt_processors": [], "output_processors": []}
                )()
                self.validators = []
                self.plugins = []
                self.fallback_step = None
                self.meta = {}

        step = TestStep()

        result = await executor.execute(step, "input")

        assert result.success is True
        assert result.output == "test output"
        assert result.name == "test_step"

    @pytest.mark.asyncio
    async def test_step_execution_with_retry(self):
        """Test step execution with retry logic."""
        executor = ExecutorCore(enable_cache=False)

        # Create an agent that fails once then succeeds
        class FailingThenSucceedingAgent:
            def __init__(self):
                self.attempts = 0

            async def run(self, data: Any, **kwargs) -> Any:
                self.attempts += 1
                if self.attempts == 1:
                    raise ValueError("First attempt fails")
                return "success on retry"

        agent = FailingThenSucceedingAgent()

        # Create a proper step object instead of using Mock
        class TestStep:
            def __init__(self):
                self.name = "test_step"
                self.agent = agent
                self.config = type("Config", (), {"max_retries": 3, "temperature": None})()
                self.processors = type(
                    "Processors", (), {"prompt_processors": [], "output_processors": []}
                )()
                self.validators = []
                self.plugins = []
                self.fallback_step = None
                self.meta = {}

        step = TestStep()

        result = await executor.execute(step, "input")

        assert result.success is True
        assert result.output == "success on retry"
        assert result.attempts == 2  # Should have retried once (1 initial + 1 retry)

    @pytest.mark.asyncio
    async def test_caching_functionality(self):
        """Test caching functionality."""
        cache_backend = InMemoryLRUBackend()
        executor = ExecutorCore(cache_backend=cache_backend, enable_cache=True)

        agent = MockAgent("cached output")

        # Create a proper step object instead of using Mock
        class TestStep:
            def __init__(self):
                self.name = "test_step"
                self.agent = agent
                self.config = type("Config", (), {"max_retries": 3, "temperature": None})()
                self.processors = type(
                    "Processors", (), {"prompt_processors": [], "output_processors": []}
                )()
                self.validators = []
                self.plugins = []
                self.fallback_step = None
                self.meta = {}

        step = TestStep()

        # First execution should cache the result
        result1 = await executor.execute(step, "input")
        assert result1.success is True
        assert result1.output == "cached output"

        # Second execution should use cached result
        result2 = await executor.execute(step, "input")
        assert result2.success is True
        assert result2.output == "cached output"

        # Verify agent was only called once (cached on second call)
        assert agent.call_count == 1

    @pytest.mark.asyncio
    async def test_usage_tracking(self):
        """Test usage tracking functionality."""
        usage_meter = ThreadSafeMeter()
        executor = ExecutorCore(usage_meter=usage_meter, enable_cache=False)

        # Create a proper mock agent with model_id and usage info
        class MockAgentWithUsage:
            def __init__(self, output: str = "test output"):
                self.output = output
                self.model_id = "openai:gpt-4o"  # Add model_id for cost calculation

            async def run(self, data: Any, **kwargs) -> Any:
                class MockResponse:
                    def __init__(self, output: str):
                        self.output = output

                    def usage(self):
                        class MockUsage:
                            def __init__(self):
                                self.request_tokens = 100
                                self.response_tokens = 50
                                self.total_tokens = 150

                        return MockUsage()

                return MockResponse(self.output)

        agent = MockAgentWithUsage("output")

        # Create a proper step object instead of using Mock
        class TestStep:
            def __init__(self):
                self.name = "test_step"
                self.agent = agent
                self.config = type("Config", (), {"max_retries": 3, "temperature": None})()
                self.processors = type(
                    "Processors", (), {"prompt_processors": [], "output_processors": []}
                )()
                self.validators = []
                self.plugins = []
                self.fallback_step = None
                self.meta = {}

        step = TestStep()

        result = await executor.execute(step, "input")

        assert result.success is True
        assert result.cost_usd > 0.0  # Should have calculated cost
        assert result.token_counts == 150  # Should have token counts

    @pytest.mark.asyncio
    async def test_processor_pipeline(self):
        """Test processor pipeline functionality."""
        processor_pipeline = DefaultProcessorPipeline()
        executor = ExecutorCore(processor_pipeline=processor_pipeline, enable_cache=False)

        agent = MockAgent("agent output")

        # Create a proper step object instead of using Mock
        class TestStep:
            def __init__(self):
                self.name = "test_step"
                self.agent = agent
                self.config = type("Config", (), {"max_retries": 3, "temperature": None})()
                self.processors = type(
                    "Processors",
                    (),
                    {
                        "prompt_processors": [MockProcessor(lambda x: f"prompt_{x}")],
                        "output_processors": [MockProcessor(lambda x: f"output_{x}")],
                    },
                )()
                self.validators = []
                self.plugins = []
                self.fallback_step = None
                self.meta = {}

        step = TestStep()

        result = await executor.execute(step, "input")

        assert result.success is True
        # Output should be processed
        assert result.output == "output_agent output"

    @pytest.mark.asyncio
    async def test_validation_failure(self):
        """Test handling of validation failures."""
        validator_runner = DefaultValidatorRunner()
        executor = ExecutorCore(validator_runner=validator_runner, enable_cache=False)

        agent = MockAgent("output")

        # Create a proper step object instead of using Mock
        class TestStep:
            def __init__(self):
                self.name = "test_step"
                self.agent = agent
                self.config = type("Config", (), {"max_retries": 3, "temperature": None})()
                self.processors = type(
                    "Processors", (), {"prompt_processors": [], "output_processors": []}
                )()
                self.validators = [MockValidator(should_fail=True)]
                self.plugins = []
                self.fallback_step = None
                self.meta = {}
                self.is_complex = False  # Explicitly set to avoid Mock defaults

        step = TestStep()

        result = await executor.execute(step, "input")

        assert result.success is False
        assert "Validation failed" in result.feedback


# --------------------------------------------------------------------------- #
# ★ Backward Compatibility Tests
# --------------------------------------------------------------------------- #


class TestBackwardCompatibility:
    """Test backward compatibility with original UltraStepExecutor API."""

    @pytest.mark.asyncio
    async def test_constructor_compatibility(self):
        """Test that constructor accepts the same parameters."""
        # Should accept original parameters without error
        executor = UltraStepExecutor(
            enable_cache=True,
            cache_size=512,
            cache_ttl=1800,
            concurrency_limit=4,
        )

        assert executor is not None

    @pytest.mark.asyncio
    async def test_execute_signature(self):
        """Test that execute can run a simple step with legacy kwargs."""
        executor = UltraStepExecutor(enable_cache=False)

        agent = MockAgent("output")
        step = Mock()
        step.name = "test_step"
        step.agent = agent
        step.config = Mock()
        step.config.max_retries = 3
        step.config.temperature = None
        step.processors = Mock()
        step.processors.prompt_processors = []
        step.processors.output_processors = []
        step.validators = []
        step.plugins = []  # Add missing plugins attribute

        outcome = await executor.execute(
            step,
            "input",
            context=None,
            resources=None,
            usage_limits=None,
        )
        if isinstance(outcome, StepResult):
            result = outcome
        else:
            result = executor._unwrap_outcome_to_step_result(
                outcome, executor._safe_step_name(step)
            )

        assert result.success is True
        assert result.output == "output"

    @pytest.mark.slow  # Mark as slow due to cache property access issues
    def test_cache_property_compatibility(self):
        """Test that cache property is available for inspection."""
        executor = UltraStepExecutor(enable_cache=True)

        # Should expose cache property
        cache = executor.cache
        assert cache is not None

    @pytest.mark.slow  # Mark as slow due to cache operations
    @pytest.mark.asyncio
    async def test_clear_cache_compatibility(self):
        """Test that clear_cache method exists."""
        executor = UltraStepExecutor(enable_cache=True)

        # Should not raise an error
        await executor.clear_cache()


# --------------------------------------------------------------------------- #
# ★ Error Handling Tests
# --------------------------------------------------------------------------- #


class TestErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_critical_exception_propagation(self):
        """Test that critical exceptions are propagated."""
        executor = ExecutorCore(enable_cache=False)

        # Create an agent that raises a critical exception
        class CriticalAgent:
            async def run(self, data: Any, **kwargs) -> Any:
                raise PausedException("Critical error")

        # Create a proper step object instead of using Mock
        class TestStep:
            def __init__(self):
                self.name = "test_step"
                self.agent = CriticalAgent()
                self.config = type("Config", (), {"max_retries": 3, "temperature": None})()
                self.processors = type(
                    "Processors", (), {"prompt_processors": [], "output_processors": []}
                )()
                self.validators = []
                self.plugins = []
                self.fallback_step = None
                self.meta = {}

        step = TestStep()

        with pytest.raises(PausedException, match="Critical error"):
            await executor.execute(step, "input")

    @pytest.mark.asyncio
    async def test_usage_limit_exception_propagation(self):
        """Guard no longer raises; usage is tracked and execution succeeds in quota mode."""
        usage_meter = ThreadSafeMeter()
        executor = ExecutorCore(usage_meter=usage_meter, enable_cache=False)

        # Pre-load usage meter to exceed limits
        await usage_meter.add(10.0, 1000, 500)

        # Create a proper mock agent with model_id and usage info
        class MockAgentWithUsage:
            def __init__(self, output: str = "test output"):
                self.output = output
                self.model_id = "openai:gpt-4o"  # Add model_id for cost calculation

            async def run(self, data: Any, **kwargs) -> Any:
                class MockResponse:
                    def __init__(self, output: str):
                        self.output = output

                    def usage(self):
                        class MockUsage:
                            def __init__(self):
                                self.request_tokens = 100
                                self.response_tokens = 50

                        return MockUsage()

                return MockResponse(self.output)

        agent = MockAgentWithUsage("output")

        # Create a proper step object instead of using Mock
        class TestStep:
            def __init__(self):
                self.name = "test_step"
                self.agent = agent
                self.config = type("Config", (), {"max_retries": 3, "temperature": None})()
                self.processors = type(
                    "Processors", (), {"prompt_processors": [], "output_processors": []}
                )()
                self.validators = []
                self.plugins = []
                self.fallback_step = None
                self.meta = {}
                self.is_complex = False  # Explicitly set to avoid Mock defaults

        step = TestStep()

        limits = UsageLimits(total_cost_usd_limit=5.0)
        result = await executor.execute(step, "input", limits=limits)
        assert isinstance(result, StepResult)
        assert result.success is True
        # Usage meter retains preloaded values plus measured usage (~0.00125 cost)
        cost, prompt_tokens, completion_tokens = await usage_meter.snapshot()
        assert cost == pytest.approx(10.00125, rel=1e-6)
        assert prompt_tokens == 1100  # preloaded 1000 + measured 100
        assert completion_tokens == 550  # preloaded 500 + measured 50

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self):
        """Test that max retries are respected."""
        executor = ExecutorCore(enable_cache=False)

        # Create an agent that always fails
        class AlwaysFailingAgent:
            async def run(self, data: Any, **kwargs) -> Any:
                raise ValueError("Always fails")

        # Create a proper step object instead of using Mock
        class TestStep:
            def __init__(self):
                self.name = "test_step"
                self.agent = AlwaysFailingAgent()
                self.config = type("Config", (), {"max_retries": 2, "temperature": None})()
                self.processors = type(
                    "Processors", (), {"prompt_processors": [], "output_processors": []}
                )()
                self.validators = []
                self.plugins = []
                self.fallback_step = None
                self.meta = {}

        step = TestStep()

        result = await executor.execute(step, "input")

        assert result.success is False
        assert result.attempts == 3  # Should retry max_retries times (1 initial + 2 retries)
        assert "Always fails" in result.feedback


# --------------------------------------------------------------------------- #
# ★ Integration Tests
# --------------------------------------------------------------------------- #


class TestIntegration:
    """Integration tests with real step types."""

    @pytest.mark.asyncio
    async def test_real_step_execution(self):
        """Test execution with real Step objects."""
        executor = UltraStepExecutor(enable_cache=False)

        # Create a real Step object
        agent = MockAgent("real output")
        step = Step(
            name="real_test_step",
            agent=agent,
            config=StepConfig(max_retries=2),
        )

        result = await execute_simple_step(executor, step, "input data")

        assert result.success is True
        assert result.name == "real_test_step"
        assert result.output == "real output"

    @pytest.mark.asyncio
    async def test_complex_step_delegation(self):
        """Test that complex steps are properly delegated."""
        executor = ExecutorCore(enable_cache=False)

        # Create a CacheStep (which should be handled by step logic)
        from flujo.domain.dsl.cache_step import CacheStep

        # Create a wrapped step first
        wrapped_agent = MockAgent("wrapped output")
        wrapped_step = Step(
            name="wrapped_step",
            agent=wrapped_agent,
            config=StepConfig(max_retries=1),
        )

        cache_step = CacheStep(
            name="cache_test",
            wrapped_step=wrapped_step,
        )

        # This should delegate to step logic helpers
        result = await executor.execute(cache_step, "input")

        # The exact behavior depends on CacheStep implementation,
        # but it shouldn't crash
        assert isinstance(result, StepResult)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
