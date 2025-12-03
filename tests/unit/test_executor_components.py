"""
Unit tests for executor components and dependency injection.

Tests cover:
- All new default component implementations
- Dependency injection wiring in Flujo runner
- Component isolation and contract compliance
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from typing import Any

from flujo.application.core.executor_core import (
    # Default implementations
    OrjsonSerializer,
    Blake3Hasher,
    InMemoryLRUBackend,
    ThreadSafeMeter,
    DefaultAgentRunner,
    DefaultProcessorPipeline,
    DefaultValidatorRunner,
    DefaultPluginRunner,
    DefaultTelemetry,
    # Core executor
    ExecutorCore,
)
from flujo.domain.models import StepResult, UsageLimits
from flujo.domain.validation import ValidationResult


# --------------------------------------------------------------------------- #
# ★ Component Tests
# --------------------------------------------------------------------------- #


class TestOrjsonSerializer:
    """Test the OrjsonSerializer component."""

    def test_serialize_deserialize_roundtrip(self):
        """Test that serialize and deserialize work correctly."""
        serializer = OrjsonSerializer()

        # Test with simple data
        data = {"key": "value", "number": 42, "boolean": True}
        serialized = serializer.serialize(data)
        deserialized = serializer.deserialize(serialized)

        assert deserialized == data

    def test_serialize_deserialize_complex_data(self):
        """Test with complex nested data structures."""
        serializer = OrjsonSerializer()

        data = {
            "list": [1, 2, 3],
            "nested": {"a": {"b": {"c": "d"}}},
            "mixed": [{"key": "value"}, None, True, False, 123.45],
        }
        serialized = serializer.serialize(data)
        deserialized = serializer.deserialize(serialized)

        assert deserialized == data


class TestBlake3Hasher:
    """Test the Blake3Hasher component."""

    def test_digest_consistency(self):
        """Test that the same input produces the same hash."""
        hasher = Blake3Hasher()

        data = b"test data"
        hash1 = hasher.digest(data)
        hash2 = hasher.digest(data)

        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) > 0

    def test_digest_uniqueness(self):
        """Test that different inputs produce different hashes."""
        hasher = Blake3Hasher()

        hash1 = hasher.digest(b"data1")
        hash2 = hasher.digest(b"data2")

        assert hash1 != hash2


class TestInMemoryLRUBackend:
    """Test the InMemoryLRUBackend component."""

    @pytest.mark.asyncio
    async def test_get_put_basic(self):
        """Test basic get and put operations."""
        backend = InMemoryLRUBackend(max_size=10)

        result = StepResult(name="test", output="data", success=True)

        # Put and get
        await backend.put("key1", result, ttl_s=3600)
        retrieved = await backend.get("key1")

        assert retrieved is not None
        assert retrieved.name == "test"
        assert retrieved.output == "data"

    @pytest.mark.asyncio
    @pytest.mark.slow  # Mark as slow due to sleep timing operation
    async def test_cache_ttl_expiration(self):
        """Test TTL expiration."""
        backend = InMemoryLRUBackend(max_size=10, ttl_s=0.1)  # Very short TTL

        result = StepResult(name="test", output="data", success=True)
        await backend.put("key1", result, ttl_s=0.1)

        # Should be available immediately
        retrieved = await backend.get("key1")
        assert retrieved is not None

        # Wait for expiration
        await asyncio.sleep(0.2)

        # Should be expired
        retrieved = await backend.get("key1")
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        backend = InMemoryLRUBackend(max_size=2)

        result1 = StepResult(name="test1", output="data1", success=True)
        result2 = StepResult(name="test2", output="data2", success=True)
        result3 = StepResult(name="test3", output="data3", success=True)

        # Fill cache
        await backend.put("key1", result1, ttl_s=3600)
        await backend.put("key2", result2, ttl_s=3600)

        # Add third item - should evict first
        await backend.put("key3", result3, ttl_s=3600)

        # First item should be evicted
        assert await backend.get("key1") is None

        # Second and third should still be available
        assert await backend.get("key2") is not None
        assert await backend.get("key3") is not None

    @pytest.mark.asyncio
    async def test_cache_mutation_protection(self):
        """Test that cached objects are protected from mutation."""
        backend = InMemoryLRUBackend(max_size=10)

        result = StepResult(name="test", output="data", success=True)
        await backend.put("key1", result, ttl_s=3600)

        # Get the cached result
        cached = await backend.get("key1")
        assert cached is not None

        # Modify the cached result
        cached.output = "modified"

        # Get again - should be unchanged
        cached2 = await backend.get("key1")
        assert cached2 is not None
        assert cached2.output == "data"  # Should be original value


class TestThreadSafeMeter:
    """Test the ThreadSafeMeter component."""

    @pytest.mark.asyncio
    async def test_add_and_snapshot(self):
        """Test adding usage and getting snapshots."""
        meter = ThreadSafeMeter()

        # Add usage
        await meter.add(0.1, 50, 25)
        await meter.add(0.2, 75, 50)

        # Get snapshot
        cost, prompt_tokens, completion_tokens = await meter.snapshot()

        # Use approximate comparison for floating point
        assert abs(cost - 0.3) < 1e-10  # 0.1 + 0.2
        assert prompt_tokens == 125  # 50 + 75
        assert completion_tokens == 75  # 25 + 50

    @pytest.mark.asyncio
    async def test_guard_usage_limits(self):
        """Guard is a compatibility no-op in quota-only mode."""
        meter = ThreadSafeMeter()

        # Add some usage
        await meter.add(0.1, 50, 25)

        # Guard should not raise; quota enforcement happens elsewhere
        limits = UsageLimits(total_cost_usd_limit=0.05, total_tokens_limit=50)
        assert await meter.guard(limits) is None
        # Totals remain unchanged
        cost, prompt_tokens, completion_tokens = await meter.snapshot()
        assert abs(cost - 0.1) < 1e-10
        assert prompt_tokens == 50
        assert completion_tokens == 25

    @pytest.mark.asyncio
    @pytest.mark.slow  # Mark as slow due to sleep timing operation
    async def test_concurrent_usage_tracking(self):
        """Test thread-safe usage tracking."""
        meter = ThreadSafeMeter()

        # Simulate concurrent usage
        async def add_usage():
            for i in range(10):
                await meter.add(0.01, 1, 1)
                await asyncio.sleep(0.001)

        # Run multiple concurrent tasks
        tasks = [add_usage() for _ in range(5)]
        await asyncio.gather(*tasks)

        # Check final totals
        cost, prompt_tokens, completion_tokens = await meter.snapshot()
        # Use approximate comparison for floating point
        assert abs(cost - 0.5) < 1e-10  # 10 * 5 * 0.01
        assert prompt_tokens == 50  # 10 * 5 * 1
        assert completion_tokens == 50  # 10 * 5 * 1


class TestDefaultAgentRunner:
    """Test the DefaultAgentRunner component."""

    @pytest.mark.asyncio
    async def test_basic_agent_execution(self):
        """Test basic agent execution."""
        runner = DefaultAgentRunner()

        # Create a simple agent
        agent = Mock()
        agent.run = AsyncMock(return_value="test output")

        result = await runner.run(
            agent=agent,
            payload="test input",
            context=None,
            resources=None,
            options={},
        )

        assert result == "test output"
        agent.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_agent_with_context_and_resources(self):
        """Test agent execution with context and resources."""
        runner = DefaultAgentRunner()

        agent = Mock()
        agent.run = AsyncMock(return_value="test output")

        context = {"key": "value"}
        resources = {"resource": "data"}

        result = await runner.run(
            agent=agent,
            payload="test input",
            context=context,
            resources=resources,
            options={"temperature": 0.5},
        )

        assert result == "test output"
        # Check that context and resources were passed
        call_args = agent.run.call_args
        assert call_args[1]["context"] == context
        assert call_args[1]["resources"] == resources
        assert call_args[1]["temperature"] == 0.5


class TestDefaultProcessorPipeline:
    """Test the DefaultProcessorPipeline component."""

    @pytest.mark.asyncio
    async def test_apply_prompt_processors(self):
        """Test prompt processor application."""
        pipeline = DefaultProcessorPipeline()

        # Create a mock processor
        processor = Mock()
        processor.process = AsyncMock(return_value="processed data")

        processors = [processor]
        data = "original data"
        context = {"key": "value"}

        result = await pipeline.apply_prompt(processors, data, context=context)

        assert result == "processed data"
        processor.process.assert_called_once_with("original data", context=context)

    @pytest.mark.asyncio
    async def test_apply_output_processors(self):
        """Test output processor application."""
        pipeline = DefaultProcessorPipeline()

        # Create a mock processor
        processor = Mock()
        processor.process = AsyncMock(return_value="processed output")

        processors = [processor]
        data = "original output"
        context = {"key": "value"}

        result = await pipeline.apply_output(processors, data, context=context)

        assert result == "processed output"
        processor.process.assert_called_once_with("original output", context=context)


class TestDefaultValidatorRunner:
    """Test the DefaultValidatorRunner component."""

    @pytest.mark.asyncio
    async def test_validation_success(self):
        """Test successful validation."""
        runner = DefaultValidatorRunner()

        # Create a mock validator
        validator = Mock()
        validator.validate = AsyncMock(
            return_value=ValidationResult(is_valid=True, validator_name="TestValidator")
        )

        validators = [validator]
        data = "test data"
        context = {"key": "value"}

        # Should not raise an exception
        await runner.validate(validators, data, context=context)

        validator.validate.assert_called_once_with(data, context=context)

    @pytest.mark.asyncio
    async def test_validation_failure(self):
        """Test validation failure."""
        runner = DefaultValidatorRunner()

        # Create a mock validator that fails
        validator = Mock()
        validator.validate = AsyncMock(
            return_value=ValidationResult(
                is_valid=False, feedback="Validation failed", validator_name="TestValidator"
            )
        )

        validators = [validator]
        data = "test data"
        context = {"key": "value"}

        # Enhanced: Returns validation result instead of raising exception
        result = await runner.validate(validators, data, context=context)
        # Enhanced: Validation runner returns list of ValidationResult objects
        assert isinstance(result, list) and len(result) > 0 and not result[0].is_valid


class TestDefaultPluginRunner:
    """Test the DefaultPluginRunner component."""

    @pytest.mark.asyncio
    async def test_plugin_execution(self):
        """Test plugin execution."""
        runner = DefaultPluginRunner()

        # Create a proper ValidationPlugin instance
        from flujo.domain.plugins import ValidationPlugin, PluginOutcome

        class TestPlugin(ValidationPlugin):
            async def validate(self, data: dict[str, Any], *, context: Any = None) -> PluginOutcome:
                return PluginOutcome(success=True, new_solution="processed data")

        plugin = TestPlugin()
        plugins = [(plugin, 1)]  # (plugin, priority)
        data = "original data"
        context = {"key": "value"}

        result = await runner.run_plugins(plugins, data, context=context)

        assert result == "processed data"

    @pytest.mark.asyncio
    async def test_plugin_with_priority_ordering(self):
        """Test that plugins are executed in priority order."""
        runner = DefaultPluginRunner()

        # Create proper ValidationPlugin instances
        from flujo.domain.plugins import ValidationPlugin, PluginOutcome

        class Plugin1(ValidationPlugin):
            async def validate(self, data: dict[str, Any], *, context: Any = None) -> PluginOutcome:
                return PluginOutcome(success=True, new_solution="processed by plugin1")

        class Plugin2(ValidationPlugin):
            async def validate(self, data: dict[str, Any], *, context: Any = None) -> PluginOutcome:
                return PluginOutcome(success=True, new_solution="processed by plugin2")

        plugin1 = Plugin1()
        plugin2 = Plugin2()
        plugins = [(plugin1, 1), (plugin2, 2)]  # plugin2 has higher priority

        data = "original data"
        context = {"key": "value"}

        result = await runner.run_plugins(plugins, data, context=context)

        # Should be processed by plugin1 (lower priority) - higher priority plugins execute first,
        # but since plugin2 returns "processed by plugin2", the final result should be from plugin1
        # because plugin2 executes first and returns "processed by plugin2", then plugin1 executes
        # and returns "processed by plugin1", which becomes the final result
        assert result == "processed by plugin1"


class TestDefaultTelemetry:
    """Test the DefaultTelemetry component."""

    def test_trace_decorator(self):
        """Test the trace decorator."""
        telemetry = DefaultTelemetry()

        # Test that the decorator doesn't break function execution
        @telemetry.trace("test_trace")
        def test_function():
            return "test result"

        result = test_function()
        assert result == "test result"


# --------------------------------------------------------------------------- #
# ★ Dependency Injection Tests
# --------------------------------------------------------------------------- #


class TestFlujoCompositionRoot:
    """Test the Flujo runner's composition root functionality."""

    def test_create_default_backend(self):
        """Test that Flujo creates a properly wired backend."""
        from flujo.application.runner import Flujo

        # Create a Flujo instance without specifying a backend
        flujo = Flujo(pipeline=None)

        # The _create_default_backend method should create a LocalBackend
        # with an ExecutorCore that has all dependencies properly wired
        backend = flujo._create_default_backend()

        # Verify it's a LocalBackend
        from flujo.infra.backends import LocalBackend

        assert isinstance(backend, LocalBackend)

        # Verify it has an ExecutorCore
        assert hasattr(backend, "_executor")

        # Use type check instead of isinstance to avoid import path issues
        assert type(backend._executor).__name__ == "ExecutorCore"
        # Also verify it's actually an ExecutorCore by checking the module
        assert "executor_core" in type(backend._executor).__module__

    def test_executor_core_dependency_injection(self):
        """Test that ExecutorCore accepts all dependencies via DI."""
        # Test with all explicit dependencies
        serializer = OrjsonSerializer()
        hasher = Blake3Hasher()
        cache_backend = InMemoryLRUBackend()
        usage_meter = ThreadSafeMeter()
        agent_runner = DefaultAgentRunner()
        processor_pipeline = DefaultProcessorPipeline()
        validator_runner = DefaultValidatorRunner()
        plugin_runner = DefaultPluginRunner()
        telemetry = DefaultTelemetry()

        executor = ExecutorCore(
            serializer=serializer,
            hasher=hasher,
            cache_backend=cache_backend,
            usage_meter=usage_meter,
            agent_runner=agent_runner,
            processor_pipeline=processor_pipeline,
            validator_runner=validator_runner,
            plugin_runner=plugin_runner,
            telemetry=telemetry,
        )

        # Verify all dependencies are properly stored
        assert executor._serializer is serializer
        assert executor._hasher is hasher
        assert executor._cache_backend is cache_backend
        assert executor._usage_meter is usage_meter
        assert executor._agent_runner is agent_runner
        assert executor._processor_pipeline is processor_pipeline
        assert executor._validator_runner is validator_runner
        assert executor._plugin_runner is plugin_runner
        assert executor._telemetry is telemetry

    def test_executor_core_default_dependencies(self):
        """Test that ExecutorCore uses sensible defaults."""
        executor = ExecutorCore()

        # Verify default dependencies are created
        assert isinstance(executor._serializer, OrjsonSerializer)
        assert isinstance(executor._hasher, Blake3Hasher)
        assert isinstance(executor._usage_meter, ThreadSafeMeter)
        assert isinstance(executor._agent_runner, DefaultAgentRunner)
        assert isinstance(executor._processor_pipeline, DefaultProcessorPipeline)
        assert isinstance(executor._validator_runner, DefaultValidatorRunner)
        assert isinstance(executor._plugin_runner, DefaultPluginRunner)
        assert isinstance(executor._telemetry, DefaultTelemetry)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
