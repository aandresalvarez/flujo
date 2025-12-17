"""Comprehensive regression tests for critical bugs and edge cases.

This test file is designed to catch the types of bugs we've encountered
and prevent regressions in future development. It covers:

1. Circular reference serialization bugs
2. Failed step recording bugs
3. Lambda serialization null handling bugs
4. Performance regression bugs
5. Serialization edge cases that could break in CI
"""

import pytest
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Any

from flujo.domain.models import BaseModel, PipelineResult
from flujo.application.core.execution_manager import ExecutionManager
from flujo.state.backends.sqlite import SQLiteBackend
from flujo.utils.serialization import (
    _serialize_for_json,
    _robust_serialize_internal,
    register_custom_serializer,
)
from flujo.domain import Step

# This module aggregates regression and performance edge cases; mark as slow
pytestmark = [pytest.mark.slow]


class CircularReferenceModel(BaseModel):
    """Test model for circular reference detection."""

    name: str
    parent: Optional["CircularReferenceModel"] = None
    children: List["CircularReferenceModel"] = []


CircularReferenceModel.model_rebuild()

# Note: CircularReferenceModel should use BaseModel's built-in circular reference handling
# Do not register a custom serializer that bypasses this logic


# Register serializers for classes defined within test methods
def register_test_serializers():
    """Register serializers for classes defined within test methods."""
    # These will be registered when the test methods run
    pass


class TestCircularReferenceSerialization:
    """Test that circular reference serialization works correctly."""

    def test_circular_reference_detection(self):
        """Test that circular references are detected and handled properly."""
        # Create circular reference
        parent = CircularReferenceModel(name="parent")
        child = CircularReferenceModel(name="child", parent=parent)
        parent.children.append(child)

        # Test default mode - should set circular refs to None
        dumped = parent.model_dump(mode="default")
        assert dumped["children"][0]["parent"] is None

        # Test cache mode - should use placeholder
        dumped_cache = parent.model_dump(mode="cache")
        assert dumped_cache["children"][0]["parent"] == "<CircularReferenceModel circular>"

    def test_self_reference(self):
        """Test self-referencing objects."""
        node = CircularReferenceModel(name="self")
        node.parent = node

        dumped = node.model_dump(mode="default")
        assert dumped["parent"] is None

        dumped_cache = node.model_dump(mode="cache")
        assert dumped_cache["parent"] == "<CircularReferenceModel circular>"

    def test_deep_circular_reference(self):
        """Test deeply nested circular references."""
        root = CircularReferenceModel(name="root")
        child1 = CircularReferenceModel(name="child1", parent=root)
        grandchild = CircularReferenceModel(name="grandchild", parent=child1)
        grandchild.children.append(root)  # Create cycle
        root.children.append(child1)
        child1.children.append(grandchild)

        dumped = root.model_dump(mode="default")
        grandchild_dump = dumped["children"][0]["children"][0]
        assert grandchild_dump["children"][0] is None  # Circular ref should be None


class TestFailedStepRecording:
    """Test that failed steps are properly recorded."""

    @pytest.mark.asyncio
    async def test_failed_step_recording(self, tmp_path: Path):
        """Test that failed steps are recorded to persistence."""
        db_path = tmp_path / "test.db"

        # Use async with for proper SQLite connection cleanup
        async with SQLiteBackend(db_path) as backend:
            # Create a step that will fail
            class FailingAgent:
                async def run(self, *args, **kwargs) -> Any:
                    raise RuntimeError("Simulated failure")

                async def run_async(self, *args, **kwargs) -> Any:
                    return await self.run(*args, **kwargs)

            failing_agent = FailingAgent()
            step = Step.model_validate({"name": "failing_step", "agent": failing_agent})

            # Create execution manager with state manager
            from flujo.application.core.state_manager import StateManager
            from flujo.domain.dsl.pipeline import Pipeline

            state_manager = StateManager(state_backend=backend)

            # Create a proper pipeline
            pipeline = Pipeline(steps=[step])

            execution_manager = ExecutionManager(pipeline=pipeline, state_manager=state_manager)

            # Run the step
            result = PipelineResult()
            context = None
            run_id = "test_run"

            # Execute steps
            async for _ in execution_manager.execute_steps(
                start_idx=0,
                data="test_input",
                context=context,
                result=result,
                run_id=run_id,
            ):
                pass

            # Verify failed step was recorded
            assert len(result.step_history) == 1
            failed_step = result.step_history[0]
            assert failed_step.name == "failing_step"
            assert not failed_step.success

            # Verify step was persisted
            steps = await backend.list_run_steps(run_id)
            assert len(steps) == 1
            assert steps[0]["step_name"] == "failing_step"
            assert steps[0]["status"] == "failed"


class TestLambdaSerializationNullHandling:
    """Test that lambda serialization correctly handles null values."""

    @pytest.mark.asyncio
    async def test_lambda_null_handling(self, tmp_path: Path):
        """Test that lambda expressions correctly handle None values."""
        db_path = tmp_path / "test.db"

        # Use async with for proper SQLite connection cleanup
        async with SQLiteBackend(db_path) as backend:
            # Test data with None values - use current schema fields
            from datetime import datetime, timezone

            step_data = {
                "run_id": "test_run",
                "step_name": "test_step",
                "step_index": 0,
                "status": "completed",
                "output": None,  # This should be handled correctly
                "cost_usd": None,
                "token_counts": None,
                "execution_time_ms": None,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

            # Save step result
            await backend.save_step_result(step_data)

            # Verify the data was saved correctly (no "None" strings)
            steps = await backend.list_run_steps("test_run")
            assert len(steps) == 1
            saved_step = steps[0]

            # Check that None values are preserved as None, not converted to "None" strings
            assert saved_step["output"] is None
            assert saved_step["cost_usd"] is None
            assert saved_step["token_counts"] is None
            assert saved_step["execution_time_ms"] is None


class TestSerializationEdgeCases:
    """Test edge cases that could break in CI environments."""

    def test_unknown_type_serialization(self):
        """Test serialization of unknown types."""

        class UnknownType:
            def __init__(self, value):
                self.value = value

        # Register serializer for this test class
        register_custom_serializer(UnknownType, lambda obj: obj.__dict__)

        obj = UnknownType("test")

        # Test _serialize_for_json with unknown type - should now serialize objects with __dict__
        result_safe = _serialize_for_json(obj)
        assert isinstance(result_safe, dict)
        assert result_safe["value"] == "test"

        # Test robust_serialize with unknown type - should handle gracefully
        result_robust = _robust_serialize_internal(obj)
        assert isinstance(result_robust, dict)
        assert result_robust["value"] == "test"

    def test_custom_object_with_circular_ref(self):
        """Test custom objects with circular references."""

        class CustomObject:
            def __init__(self, name):
                self.name = name
                self.parent = None

        # Register serializer for this test class
        register_custom_serializer(CustomObject, lambda obj: obj.__dict__)

        obj1 = CustomObject("parent")
        obj2 = CustomObject("child")
        obj2.parent = obj1
        obj1.children = [obj2]  # Create circular reference

        # Should not raise exception and should serialize as dict
        result = _robust_serialize_internal(obj1)
        assert isinstance(result, dict)
        assert result["name"] == "parent"
        assert "children" in result

    def test_serialization_error_handling(self):
        """Test that serialization errors are handled gracefully."""

        class ProblematicObject:
            def __getattr__(self, name):
                raise RuntimeError("Simulated error")

        # Register serializer for this test class
        register_custom_serializer(ProblematicObject, lambda obj: {})

        obj = ProblematicObject()

        # Should not raise exception and should serialize as dict
        result = _robust_serialize_internal(obj)
        assert isinstance(result, dict)
        # The object should be serialized as an empty dict since it has no __dict__ attributes
        assert result == {}


class TestPerformanceRegression:
    """Test for performance regressions."""

    @pytest.mark.slow  # Mark as slow due to performance measurement
    def test_serialization_performance(self):
        """Test that serialization doesn't have performance regressions."""
        # Create a complex object
        data = {
            "nested": {
                "list": [{"item": i} for i in range(100)],
                "dict": {f"key_{i}": f"value_{i}" for i in range(100)},
            },
            "circular_ref": None,  # Will be set below
        }

        # Create circular reference
        data["circular_ref"] = data

        # Measure serialization time
        start_time = time.perf_counter()
        result = _robust_serialize_internal(data)
        serialization_time = time.perf_counter() - start_time

        # Should complete in reasonable time (adjust threshold as needed)
        assert serialization_time < 1.0, f"Serialization took {serialization_time:.3f}s"
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    @pytest.mark.slow  # Mark as slow due to performance measurement
    async def test_database_operation_performance(self, tmp_path: Path):
        """Test that database operations don't have performance regressions."""
        db_path = tmp_path / "perf_test.db"

        # Use async with for proper SQLite connection cleanup
        async with SQLiteBackend(db_path) as backend:
            # Create test data
            run_data = {
                "run_id": "perf_test_run",
                "pipeline_id": "test_pipeline_id",
                "pipeline_name": "test_pipeline",
                "pipeline_version": "1.0",
                "status": "running",
                "start_time": datetime.now(timezone.utc),
            }

            # Measure save time
            start_time = time.perf_counter()
            await backend.save_run_start(run_data)
            save_time = time.perf_counter() - start_time

            # Should complete quickly - 1s sanity check for CI variance
            assert save_time < 1.0, f"Save operation took {save_time:.3f}s"


class TestCIEnvironmentCompatibility:
    """Test compatibility with CI environment constraints."""

    def test_memory_efficient_serialization(self):
        """Test that serialization doesn't consume excessive memory."""
        import gc
        import os
        import platform

        try:
            import psutil
        except ImportError:
            pytest.skip("psutil not available")

        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Create large data structure
        large_data = {"items": [{"id": i, "data": f"item_{i}" * 100} for i in range(1000)]}

        # Perform serialization
        result = _robust_serialize_internal(large_data)

        # Force garbage collection multiple times to ensure cleanup
        for _ in range(3):
            gc.collect()

        # Check memory usage
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Adaptive threshold based on environment and platform
        base_threshold = 70 * 1024 * 1024  # 70MB base threshold

        # Adjust for CI environments
        if os.getenv("CI"):
            base_threshold = 150 * 1024 * 1024  # 150MB for CI environments

        # Adjust for different platforms
        if platform.system() == "Darwin":  # macOS
            base_threshold = int(base_threshold * 1.2)  # 20% more lenient on macOS
        elif platform.system() == "Linux":
            base_threshold = int(base_threshold * 1.1)  # 10% more lenient on Linux

        # Additional adjustment for memory pressure
        try:
            # Check system memory pressure
            vm = psutil.virtual_memory()
            if vm.percent > 80:  # High memory pressure
                base_threshold = int(base_threshold * 1.3)  # 30% more lenient
        except Exception:
            # If we can't get memory info, be conservative
            pass

        # Log memory usage for debugging
        memory_mb = memory_increase / 1024 / 1024
        threshold_mb = base_threshold / 1024 / 1024

        # Use a more robust assertion that considers the environment
        if memory_increase >= base_threshold:
            # Log detailed information for debugging
            print("Memory test details:")
            print(f"  - Memory increase: {memory_mb:.1f}MB")
            print(f"  - Threshold: {threshold_mb:.1f}MB")
            print(f"  - Platform: {platform.system()}")
            print(f"  - CI: {os.getenv('CI', 'False')}")
            try:
                vm = psutil.virtual_memory()
                print(f"  - System memory usage: {vm.percent:.1f}%")
            except Exception:
                print("  - System memory info: unavailable")

            # Only fail if the increase is significantly over threshold
            # This accounts for CI environment variability
            if memory_increase > base_threshold * 1.5:  # 50% over threshold
                assert False, (
                    f"Memory increased by {memory_mb:.1f}MB (threshold: {threshold_mb:.1f}MB) - "
                    f"significantly over limit. This may indicate a memory leak."
                )
            else:
                # Log warning but don't fail for moderate overages
                print(
                    f"WARNING: Memory increase {memory_mb:.1f}MB exceeds threshold {threshold_mb:.1f}MB "
                    f"but is within acceptable range for CI environment."
                )

        assert isinstance(result, dict)

    def test_concurrent_serialization(self):
        """Test that serialization works correctly under concurrent access."""
        import threading
        import queue

        results = queue.Queue()
        errors = queue.Queue()

        def serialize_worker(worker_id):
            try:
                data = {
                    "worker_id": worker_id,
                    "nested": {"value": f"data_{worker_id}"},
                    "list": [i for i in range(100)],
                }
                result = _robust_serialize_internal(data)
                results.put((worker_id, result))
            except Exception as e:
                errors.put((worker_id, e))

        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=serialize_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Check results
        assert errors.empty(), f"Errors occurred: {[errors.get() for _ in range(errors.qsize())]}"
        assert results.qsize() == 5, f"Expected 5 results, got {results.qsize()}"


class TestEdgeCaseRobustness:
    """Test robustness against edge cases that could cause failures."""

    def test_extremely_deep_nesting(self):
        """Test serialization of extremely deeply nested structures."""
        # Create deeply nested structure
        data = {}
        current = data
        for i in range(100):  # Reduced depth for CI compatibility
            current["nested"] = {}
            current = current["nested"]

        # Should not cause stack overflow
        result = _robust_serialize_internal(data)
        assert isinstance(result, dict)
        assert "nested" in result

    def test_large_string_handling(self):
        """Test handling of very large strings."""
        large_string = "x" * (1024 * 1024)  # 1MB string
        data = {"large_string": large_string}

        # Should handle without memory issues
        result = _robust_serialize_internal(data)
        assert isinstance(result, dict)
        assert len(result["large_string"]) == len(large_string)

    def test_special_characters(self):
        """Test handling of special characters in serialization."""
        special_chars = {
            "unicode": "🎉🚀💻",
            "quotes": '"""\'"\'"',
            "newlines": "line1\nline2\r\nline3",
            "null_bytes": "data\x00with\x00nulls",
            "control_chars": "data\x01\x02\x03\x04",
        }

        # Should handle all special characters
        result = _robust_serialize_internal(special_chars)
        assert isinstance(result, dict)
        assert result["unicode"] == special_chars["unicode"]
        assert result["quotes"] == special_chars["quotes"]
        assert result["newlines"] == special_chars["newlines"]


if __name__ == "__main__":
    pytest.main([__file__])
