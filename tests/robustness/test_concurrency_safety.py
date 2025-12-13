"""Concurrency and thread safety tests.

These tests ensure that Flujo components are safe for concurrent use
and handle race conditions properly.
"""
# ruff: noqa

import asyncio
import threading
import time
from typing import List, Dict, Any, Optional
import pytest
from unittest.mock import AsyncMock

pytestmark = [
    pytest.mark.slow,
    pytest.mark.serial,  # Thread safety tests should run serially
]
from concurrent.futures import ThreadPoolExecutor, as_completed

from flujo.application.core.executor_core import ExecutorCore
from flujo.domain.models import PipelineContext, StepResult
from flujo.domain.dsl.step import Step
from flujo.application.core.context_manager import ContextManager
from tests.test_types.mocks import create_mock_executor_core


class TestConcurrencySafety:
    """Test suite for concurrency and thread safety."""

    def test_executor_thread_safety(self):
        """Test that ExecutorCore is thread-safe under concurrent access."""
        executor = create_mock_executor_core()
        results = []
        errors = []

        def execute_in_thread(thread_id: int):
            """Execute operations in a separate thread."""
            try:

                async def run_execution():
                    step = Step(name=f"thread_{thread_id}_step", agent=AsyncMock())
                    data = {"thread_id": thread_id, "input": f"data_{thread_id}"}
                    result = await executor.execute(step, data)
                    return result

                result = asyncio.run(run_execution())
                results.append((thread_id, result))
            except Exception as e:
                errors.append((thread_id, str(e)))

        # Run concurrent executions
        threads = []
        num_threads = 10

        for i in range(num_threads):
            thread = threading.Thread(target=execute_in_thread, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify results
        assert len(errors) == 0, f"Thread execution errors: {errors}"
        assert len(results) == num_threads, f"Expected {num_threads} results, got {len(results)}"

        # Verify each thread got a valid result
        for thread_id, result in results:
            assert isinstance(result, StepResult), (
                f"Thread {thread_id} got invalid result: {result}"
            )
            assert result.success, f"Thread {thread_id} execution failed: {result.feedback}"

    def test_context_isolation_under_concurrency(self):
        """Test that context isolation works correctly under concurrent access."""
        base_context = PipelineContext()
        base_context.step_outputs = {"counter": 0, "shared_data": []}

        results = []
        errors = []

        def isolate_and_modify_context(thread_id: int):
            """Isolate context and modify it in a separate thread."""
            try:
                # Isolate context
                isolated = ContextManager.isolate(base_context)

                # Modify isolated context
                isolated.step_outputs["thread_id"] = thread_id
                isolated.step_outputs["counter"] = thread_id * 10

                # Simulate some work
                time.sleep(0.01)

                # Store result
                results.append((thread_id, isolated.step_outputs.copy()))

                # Try to merge back (should not affect base context during concurrent access)
                if thread_id % 2 == 0:  # Only some threads merge
                    ContextManager.merge(base_context, isolated)

            except Exception as e:
                errors.append((thread_id, str(e)))

        # Run concurrent context operations
        threads = []
        num_threads = 20

        for i in range(num_threads):
            thread = threading.Thread(target=isolate_and_modify_context, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Verify no errors occurred
        assert len(errors) == 0, f"Context isolation errors: {errors}"
        assert len(results) == num_threads, f"Expected {num_threads} results, got {len(results)}"

        # Verify isolation worked - each thread should have its own data
        thread_ids = set()
        for thread_id, per_thread_data in results:
            assert per_thread_data["thread_id"] == thread_id, f"Thread {thread_id} data corruption"
            assert per_thread_data["counter"] == thread_id * 10, (
                f"Thread {thread_id} counter corruption"
            )
            thread_ids.add(thread_id)

        assert len(thread_ids) == num_threads, "Thread ID collision detected"

        # Verify base context was updated by concurrent merges (as expected)
        # Even-numbered threads (0, 2, 4, ..., 18) merge back, so counter should be
        # one of the even thread_id * 10 values (depends on merge order)
        # The important thing is that the counter is a valid value from one of the merges
        valid_counters = {i * 10 for i in range(0, num_threads, 2)}
        assert base_context.step_outputs["counter"] in valid_counters, (
            f"Base context counter has unexpected value: {base_context.step_outputs['counter']}"
        )

    def test_async_task_safety(self):
        """Test that async operations are safe under high concurrency."""
        executor = create_mock_executor_core()

        async def execute_async_operation(task_id: int):
            """Execute an async operation safely."""
            step = Step(name=f"async_{task_id}", agent=AsyncMock())
            data = {"task_id": task_id, "payload": f"data_{task_id}"}

            # Add some async work
            await asyncio.sleep(0.001)

            result = await executor.execute(step, data)
            return task_id, result

        async def run_concurrent_async_operations():
            """Run many async operations concurrently."""
            num_tasks = 50
            tasks = [execute_async_operation(i) for i in range(num_tasks)]

            # Execute all concurrently
            results = await asyncio.gather(*tasks)

            return results

        # Run the concurrent operations
        results = asyncio.run(run_concurrent_async_operations())

        # Verify all operations completed successfully
        assert len(results) == 50, f"Expected 50 results, got {len(results)}"

        for task_id, result in results:
            assert isinstance(result, StepResult), f"Task {task_id} got invalid result"
            assert result.success, f"Task {task_id} failed: {result.feedback}"

        # Verify no task ID collisions or data corruption
        task_ids = {task_id for task_id, _ in results}
        assert len(task_ids) == 50, "Task ID collision detected"

    def test_cache_thread_safety(self):
        """Test that caching is thread-safe under concurrent access."""
        from flujo.infrastructure.caching import InMemoryLRUCache

        cache = InMemoryLRUCache(max_size=100, ttl=3600)

        results = []
        errors = []

        def cache_operation_thread(thread_id: int):
            """Perform cache operations in a thread."""
            try:
                # Mix of set and get operations
                for i in range(10):
                    key = f"key_{thread_id}_{i}"
                    value = f"value_{thread_id}_{i}"

                    # Set operation
                    cache.set(key, value)

                    # Get operation
                    retrieved = cache.get(key)

                    # Verify consistency
                    assert retrieved == value, f"Cache inconsistency in thread {thread_id}"

                    # Some threads do more operations
                    if thread_id % 3 == 0:
                        # Test cache eviction under concurrent access
                        cache.set(f"extra_{thread_id}_{i}", f"extra_value_{i}")

                results.append(thread_id)

            except Exception as e:
                errors.append((thread_id, str(e)))

        # Run concurrent cache operations
        threads = []
        num_threads = 15

        for i in range(num_threads):
            thread = threading.Thread(target=cache_operation_thread, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Verify no errors
        assert len(errors) == 0, f"Cache thread safety errors: {errors}"
        assert len(results) == num_threads, (
            f"Expected {num_threads} successful threads, got {len(results)}"
        )

    def test_shared_resource_access_safety(self):
        """Test that shared resources are accessed safely."""
        # This test would be more relevant with actual shared resources
        # For now, test with mock shared state

        shared_counter = {"value": 0}
        lock = threading.Lock()
        results = []
        errors = []

        def access_shared_resource(thread_id: int):
            """Access shared resource with proper synchronization."""
            try:
                # Simulate some work
                time.sleep(0.001)

                # Safely access shared resource
                with lock:
                    current_value = shared_counter["value"]
                    shared_counter["value"] = current_value + 1

                results.append((thread_id, shared_counter["value"]))

            except Exception as e:
                errors.append((thread_id, str(e)))

        # Run concurrent shared resource access
        threads = []
        num_threads = 20

        for i in range(num_threads):
            thread = threading.Thread(target=access_shared_resource, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Verify no errors and proper synchronization
        assert len(errors) == 0, f"Shared resource access errors: {errors}"
        assert len(results) == num_threads

        # Verify final counter value (should be num_threads)
        final_values = [value for _, value in results]
        max_value = max(final_values)
        assert max_value == num_threads, f"Shared counter final value {max_value} != {num_threads}"

    def test_async_context_manager_safety(self):
        """Test that async context managers work safely under concurrency."""

        class AsyncTestManager:
            """Test async context manager."""

            def __init__(self):
                self.active = False
                self.usage_count = 0

            async def __aenter__(self):
                self.active = True
                self.usage_count += 1
                await asyncio.sleep(0.001)  # Simulate async setup
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                self.active = False
                await asyncio.sleep(0.001)  # Simulate async cleanup

        manager = AsyncTestManager()
        results = []
        errors = []

        async def use_async_context_manager(task_id: int):
            """Use async context manager safely."""
            try:
                async with manager:
                    # Verify context manager state
                    assert manager.active, f"Context manager not active in task {task_id}"
                    assert manager.usage_count > 0, f"Invalid usage count in task {task_id}"

                    # Simulate work
                    await asyncio.sleep(0.001)

                    results.append(task_id)

            except Exception as e:
                errors.append((task_id, str(e)))

        async def run_concurrent_context_usage():
            """Run concurrent async context manager usage."""
            num_tasks = 25
            tasks = [use_async_context_manager(i) for i in range(num_tasks)]
            await asyncio.gather(*tasks)

        # Run concurrent context manager usage
        asyncio.run(run_concurrent_context_usage())

        # Verify results
        assert len(errors) == 0, f"Async context manager errors: {errors}"
        assert len(results) == 25, f"Expected 25 results, got {len(results)}"

        # Verify context manager final state
        assert not manager.active, "Context manager should not be active after all usage"
        assert manager.usage_count == 25, f"Expected 25 usages, got {manager.usage_count}"

    def test_event_loop_thread_safety(self):
        """Test that operations work correctly across different event loops."""
        # This is challenging to test directly, but we can test basic async operation isolation

        async def isolated_async_operation(task_id: int):
            """Run an isolated async operation."""
            step = Step(name=f"isolated_{task_id}", agent=AsyncMock())
            data = {"task_id": task_id}

            # Create new executor for isolation
            executor = create_mock_executor_core()
            result = await executor.execute(step, data)

            return task_id, result

        def run_in_thread(thread_id: int):
            """Run async operations in a separate thread with its own event loop."""
            try:

                async def run_operations():
                    results = []
                    for i in range(3):
                        task_id = thread_id * 10 + i
                        result = await isolated_async_operation(task_id)
                        results.append(result)
                    return results

                # Each thread gets its own event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                try:
                    results = loop.run_until_complete(run_operations())
                    return thread_id, results
                finally:
                    loop.close()

            except Exception as e:
                return thread_id, str(e)

        # Run operations in multiple threads
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(run_in_thread, i) for i in range(3)]
            thread_results = [f.result() for f in as_completed(futures)]

        # Verify all threads completed successfully
        successful_threads = 0
        for thread_id, result in thread_results:
            if isinstance(result, list):
                successful_threads += 1
                assert len(result) == 3, f"Thread {thread_id} got {len(result)} results, expected 3"
                for task_id, step_result in result:
                    assert isinstance(step_result, StepResult), (
                        f"Invalid result in thread {thread_id}"
                    )
            else:
                pytest.fail(f"Thread {thread_id} failed: {result}")

        assert successful_threads == 3, (
            f"Only {successful_threads}/3 threads completed successfully"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
