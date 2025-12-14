"""Memory leak detection tests.

These tests detect memory leaks and ensure proper resource cleanup
in long-running Flujo applications.
"""
# ruff: noqa

from __future__ import annotations

import asyncio
import gc
import weakref
from typing import Any

import pytest

from flujo.domain.models import PipelineContext, StepResult
from flujo.domain.dsl.step import Step
from tests.test_types.fakes import FakeAgent
from tests.test_types.mocks import create_mock_executor_core


pytestmark = [
    pytest.mark.slow,
    pytest.mark.memory,
    pytest.mark.stress,
]


class TestMemoryLeakDetection:
    """Test suite for memory leak detection."""

    def test_no_executor_memory_leak_on_repeated_execution(self):
        """Test that ExecutorCore doesn't leak memory on repeated executions."""
        executor = create_mock_executor_core()

        def create_objects():
            """Create objects that should be garbage collected."""
            step = Step(name="memory_test", agent=FakeAgent("ok"))
            data = {"input": f"test_data_{id(step)}"}  # Unique data
            context = PipelineContext()
            return step, data, context

        # Warm up executor so one-time allocations (policy registry, caches, etc.)
        # don't count as leaks in the measured loop.
        async def _warmup() -> None:
            step, data, _ = create_objects()
            await executor.execute(step, data)

        asyncio.run(_warmup())
        gc.collect()

        async def _run_iterations() -> None:
            # Track StepResult retention across many iterations.
            # Counting *all* GC-tracked objects is too noisy across Python versions and event-loop
            # implementations (it includes asyncio internals, caches, etc.). This focuses on the
            # concrete regression we care about: unbounded retention of execution results.
            def _count_step_results() -> int:
                # Avoid `isinstance(..., StepResult)` because Pydantic's metaclass `__instancecheck__`
                # can trigger attribute access on arbitrary objects returned by `gc.get_objects()`.
                return sum(1 for obj in gc.get_objects() if type(obj) is StepResult)

            initial_step_results = _count_step_results()

            for i in range(100):
                step, data, _ = create_objects()
                result = await executor.execute(step, data)
                assert isinstance(result, StepResult)

                # Force garbage collection periodically
                if i % 20 == 0:
                    gc.collect()
                    current_step_results = _count_step_results()

                    # Use the first checkpoint after warmup to reset the baseline,
                    # then measure growth incrementally.
                    if i == 0:
                        initial_step_results = current_step_results
                        continue

                    step_result_growth = current_step_results - initial_step_results
                    initial_step_results = current_step_results

                    # Allow some growth but not unbounded (max 10% growth per 20 iterations)
                    max_growth = int(100 * 0.1)  # 10% of 100 = 10 objects
                    assert step_result_growth <= max_growth, (
                        f"Memory leak detected: {step_result_growth} retained StepResults after {i + 1} iterations "
                        f"(initial: {initial_step_results}, current: {current_step_results})"
                    )

        asyncio.run(_run_iterations())

    def test_context_cleanup_after_execution(self):
        """Test that PipelineContext objects are properly cleaned up."""
        from flujo.application.core.context_manager import ContextManager

        # Create context with nested data structures
        context = PipelineContext()
        context.step_outputs = {
            "nested": {"deeply": {"nested": ["data"] * 100}},
            "large_list": list(range(1000)),
            "metadata": [{"key": "value"} for _ in range(50)],
        }

        # Create weak reference to track cleanup
        context_ref = weakref.ref(context)

        # Use context in isolation
        async def use_context():
            isolated = ContextManager.isolate(context)
            # Modify isolated context
            isolated.step_outputs["new_key"] = "new_value"
            return isolated

        isolated_context = asyncio.run(use_context())

        # Delete references
        del isolated_context
        del context

        # Force garbage collection
        gc.collect()

        # Context should be garbage collected
        assert context_ref() is None, "PipelineContext was not garbage collected"

    def test_weak_reference_cleanup_for_circular_references(self):
        """Test that circular references are properly cleaned up."""
        # Create objects that might form circular references
        executor = create_mock_executor_core()

        # Create a step that holds references
        step = Step(name="circular_test", agent=FakeAgent("ok"))

        # Create context that references the step
        context = PipelineContext()
        context.step_outputs = {"step_ref": step}

        # Step agent references context (potential circular reference)
        step.agent.context_ref = context

        # Create weak references
        step_ref = weakref.ref(step)
        context_ref = weakref.ref(context)
        executor_ref = weakref.ref(executor)

        async def execute_and_cleanup():
            # Execute step
            result = await executor.execute(step, {"input": "test"})
            assert isinstance(result, StepResult)

            # Clear references
            del result
            return step, context, executor

        # Execute and get references
        step, context, executor = asyncio.run(execute_and_cleanup())

        # Clear local references
        del step
        del context
        del executor

        # Force garbage collection
        gc.collect()
        gc.collect()  # Second collection for circular references

        # All objects should be cleaned up
        assert step_ref() is None, "Step with circular reference not cleaned up"
        assert context_ref() is None, "Context with circular reference not cleaned up"
        assert executor_ref() is None, "Executor with circular reference not cleaned up"

    def test_large_data_processing_memory_efficiency(self):
        """Test that processing large data doesn't cause memory bloat."""
        executor = create_mock_executor_core()

        # Create large data sets
        large_data_sets = []
        for i in range(10):
            large_data = {
                "id": i,
                "payload": "x" * 10000,  # 10KB string
                "metadata": [{"key": "value"} for _ in range(100)],
                "nested": {"deep": {"structure": ["item"] * 1000}},
            }
            large_data_sets.append(large_data)

        initial_memory = get_memory_usage_mb()

        async def process_large_data():
            results = []
            for i, data in enumerate(large_data_sets):
                step = Step(name=f"large_step_{i}", agent=FakeAgent("ok"))
                result = await executor.execute(step, data)
                results.append(result)

                # Process in batches to test memory cleanup
                if (i + 1) % 3 == 0:
                    current_memory = get_memory_usage_mb()
                    memory_growth = current_memory - initial_memory

                    # Allow some memory growth but not unbounded
                    max_growth_mb = 50  # 50MB max growth for processing
                    assert memory_growth <= max_growth_mb, (
                        f"Memory growth {memory_growth:.1f}MB exceeds threshold {max_growth_mb}MB "
                        f"during large data processing (iteration {i + 1})"
                    )

            return results

        results = asyncio.run(process_large_data())

        # Verify all results are valid
        assert len(results) == len(large_data_sets)
        assert all(isinstance(r, StepResult) for r in results)

        # Final memory check
        final_memory = get_memory_usage_mb()
        final_growth = final_memory - initial_memory

        # After processing, memory should not be excessively high
        max_final_growth_mb = 20  # Allow 20MB final growth
        assert final_growth <= max_final_growth_mb, (
            f"Final memory growth {final_growth:.1f}MB exceeds threshold {max_final_growth_mb}MB"
        )

    def test_async_task_cleanup_in_background_execution(self):
        """Test that background tasks are properly cleaned up."""
        from flujo.application.core.background_task_manager import BackgroundTaskManager

        manager = BackgroundTaskManager()

        async def create_background_tasks():
            # Create several background tasks
            tasks = []
            for i in range(5):
                task = asyncio.create_task(asyncio.sleep(0.1))
                manager.add_task(task)
                tasks.append(task)

            # Wait for tasks to complete
            await asyncio.gather(*tasks)

            return len(manager._background_tasks)

        # Run tasks and check cleanup
        active_tasks = asyncio.run(create_background_tasks())

        # Tasks should be cleaned up automatically
        assert active_tasks == 0, f"{active_tasks} background tasks not cleaned up"

    def test_cache_memory_bounds(self):
        """Test that caching doesn't cause unbounded memory growth."""
        from flujo.infrastructure.caching import InMemoryLRUCache

        cache = InMemoryLRUCache(max_size=100, ttl=3600)

        # Add many items to cache
        for i in range(200):  # More than max_size
            key = f"key_{i}"
            value = f"value_{i}_" + "x" * 1000  # 1KB values
            cache.set(key, value)

        # Cache should respect size limits
        assert len(cache._store) <= cache.max_size, (
            f"Cache size {len(cache._store)} exceeds max_size {cache.max_size}"
        )

        # Memory usage should be bounded
        # (This is a basic check - in production you'd want more sophisticated monitoring)
        cache_memory_estimate = sum(len(str(k)) + len(str(v)) for k, (v, _) in cache._store.items())

        max_memory_kb = 1024  # 1MB max for cache
        assert cache_memory_estimate <= max_memory_kb * 1024, (
            f"Cache memory usage {cache_memory_estimate} bytes exceeds {max_memory_kb}KB limit"
        )


class TestResourceLeakDetection:
    """Test suite for resource leak detection."""

    def test_file_handle_cleanup(self):
        """Test that file handles are properly closed."""
        import tempfile
        import os

        # This test is more relevant for components that handle file I/O
        # For now, it's a placeholder for future file handling components

        # Create temporary files
        temp_files = []
        for i in range(5):
            fd, path = tempfile.mkstemp()
            temp_files.append((fd, path))

            # Close file descriptor
            os.close(fd)

        # Clean up temp files
        for _, path in temp_files:
            os.unlink(path)

        # Test passes if no exceptions
        assert True

    def test_network_connection_cleanup(self):
        """Test that network connections are properly closed."""
        # Placeholder for network connection testing
        # This would test HTTP clients, database connections, etc.
        assert True

    def test_thread_pool_cleanup(self):
        """Test that thread pools are properly cleaned up."""
        import concurrent.futures

        # Create and use thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(lambda: "test") for _ in range(5)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # Thread pool should be automatically cleaned up by context manager
        assert len(results) == 5
        assert all(r == "test" for r in results)


def get_memory_usage_mb() -> float:
    """Get current memory usage in MB."""
    psutil = pytest.importorskip("psutil", reason="psutil not available")
    import os

    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
