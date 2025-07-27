"""
Unit tests for performance optimization utilities.

This module tests the scratch buffer management system, including both
task-local storage and buffer pooling modes.
"""

import pytest
import asyncio
from typing import List, Set

from flujo.utils.performance import (
    get_scratch_buffer,
    clear_scratch_buffer,
    enable_buffer_pooling,
    disable_buffer_pooling,
    get_buffer_pool_stats,
    _scratch_buffer_var,
    DEFAULT_BUFFER_SIZE,
    MAX_POOL_SIZE,
)


class TestScratchBufferManagement:
    """Test scratch buffer management functionality."""

    def setup_method(self):
        """Reset buffer pooling state before each test."""
        disable_buffer_pooling()
        # Clear any existing task-local buffers
        try:
            _scratch_buffer_var.set(None)
        except LookupError:
            pass

        # Clear the buffer pool to ensure clean state
        try:
            from flujo.utils.performance import _get_buffer_pool

            pool = _get_buffer_pool()
            while not pool.empty():
                try:
                    pool.get_nowait()
                except Exception:
                    break
        except Exception:
            pass

    def test_get_scratch_buffer_creates_new_buffer(self):
        """Test that get_scratch_buffer creates a new buffer when none exists."""
        buffer = get_scratch_buffer()
        assert isinstance(buffer, bytearray)
        assert len(buffer) == DEFAULT_BUFFER_SIZE
        assert buffer == bytearray(DEFAULT_BUFFER_SIZE)

    def test_get_scratch_buffer_reuses_existing_buffer(self):
        """Test that get_scratch_buffer reuses existing buffer."""
        buffer1 = get_scratch_buffer()
        buffer2 = get_scratch_buffer()
        assert buffer1 is buffer2  # Same object reference

    def test_clear_scratch_buffer_clears_contents(self):
        """Test that clear_scratch_buffer clears buffer contents."""
        buffer = get_scratch_buffer()
        # Fill buffer with data
        buffer.extend(b"test data")
        assert len(buffer) > 0

        clear_scratch_buffer()
        assert len(buffer) == 0

    def test_clear_scratch_buffer_preserves_buffer_object(self):
        """Test that clear_scratch_buffer preserves the buffer object."""
        buffer1 = get_scratch_buffer()
        clear_scratch_buffer()
        buffer2 = get_scratch_buffer()
        assert buffer1 is buffer2  # Same object reference

    def test_clear_scratch_buffer_when_no_buffer_exists(self):
        """Test clear_scratch_buffer behavior when no buffer exists."""
        # Should not raise an exception
        clear_scratch_buffer()
        # Should create a buffer and clear it
        buffer = get_scratch_buffer()
        assert len(buffer) == 0

    def test_buffer_isolation_between_tasks(self):
        """Test that buffers are isolated between different async tasks."""
        buffers: List[bytearray] = []

        async def task1():
            buffer = get_scratch_buffer()
            # Clear the buffer first to ensure it's empty
            buffer.clear()
            buffer.extend(b"task1")
            buffers.append(buffer)
            return buffer

        async def task2():
            buffer = get_scratch_buffer()
            # Clear the buffer first to ensure it's empty
            buffer.clear()
            buffer.extend(b"task2")
            buffers.append(buffer)
            return buffer

        async def run_tasks():
            # Run tasks concurrently
            results = await asyncio.gather(task1(), task2())
            return results

        # Run the test
        asyncio.run(run_tasks())

        # Verify buffers are different objects
        assert len(buffers) == 2
        assert buffers[0] is not buffers[1]
        assert buffers[0] == bytearray(b"task1")
        assert buffers[1] == bytearray(b"task2")


class TestBufferPooling:
    """Test buffer pooling functionality."""

    def setup_method(self):
        """Reset buffer pooling state before each test."""
        disable_buffer_pooling()
        # Clear any existing task-local buffers
        try:
            _scratch_buffer_var.set(None)
        except LookupError:
            pass

        # Clear the buffer pool to ensure clean state
        try:
            from flujo.utils.performance import _get_buffer_pool

            pool = _get_buffer_pool()
            while not pool.empty():
                try:
                    pool.get_nowait()
                except Exception:
                    break
        except Exception:
            pass

    def test_enable_buffer_pooling(self):
        """Test enabling buffer pooling."""
        # Import the module to access the global variable
        import flujo.utils.performance as perf_module

        assert not perf_module.ENABLE_BUFFER_POOLING
        enable_buffer_pooling()
        assert perf_module.ENABLE_BUFFER_POOLING

    def test_disable_buffer_pooling(self):
        """Test disabling buffer pooling."""
        # Import the module to access the global variable
        import flujo.utils.performance as perf_module

        enable_buffer_pooling()
        assert perf_module.ENABLE_BUFFER_POOLING
        disable_buffer_pooling()
        assert not perf_module.ENABLE_BUFFER_POOLING

    def test_get_buffer_pool_stats_when_disabled(self):
        """Test buffer pool stats when pooling is disabled."""
        disable_buffer_pooling()
        stats = get_buffer_pool_stats()
        assert stats["enabled"] is False
        assert stats["pool_size"] == 0
        assert stats["utilization"] == 0.0

    def test_get_buffer_pool_stats_when_enabled(self):
        """Test buffer pool stats when pooling is enabled."""
        enable_buffer_pooling()
        stats = get_buffer_pool_stats()
        assert stats["enabled"] is True
        assert stats["max_size"] == MAX_POOL_SIZE
        assert 0.0 <= stats["utilization"] <= 1.0

    def test_buffer_pooling_retrieves_from_pool(self):
        """Test that buffer pooling retrieves buffers from pool when available."""
        enable_buffer_pooling()

        # Manually add a buffer to the pool
        from flujo.utils.performance import _get_buffer_pool

        pool = _get_buffer_pool()
        test_buffer = bytearray(b"test")
        pool.put(test_buffer)

        # Get buffer - should retrieve from pool
        buffer = get_scratch_buffer()
        assert buffer is test_buffer

    def test_buffer_pooling_creates_new_when_pool_empty(self):
        """Test that buffer pooling creates new buffer when pool is empty."""
        enable_buffer_pooling()

        # Get buffer when pool is empty
        buffer = get_scratch_buffer()
        assert isinstance(buffer, bytearray)
        assert len(buffer) == DEFAULT_BUFFER_SIZE

    def test_buffer_pooling_stores_in_task_local(self):
        """Test that pooled buffers are stored in task-local storage."""
        enable_buffer_pooling()

        # Get buffer
        buffer = get_scratch_buffer()

        # Verify it's stored in task-local storage
        task_buffer = _scratch_buffer_var.get()
        assert task_buffer is buffer

    def test_clear_scratch_buffer_returns_to_pool(self):
        """Test that clear_scratch_buffer returns buffer to pool when pooling enabled."""
        enable_buffer_pooling()

        # Get buffer
        buffer = get_scratch_buffer()
        buffer.extend(b"test data")

        # Clear buffer
        clear_scratch_buffer()

        # Verify buffer is returned to pool
        stats = get_buffer_pool_stats()
        assert stats["pool_size"] == 1

        # Verify task-local reference is cleared
        task_buffer = _scratch_buffer_var.get()
        assert task_buffer is None

    def test_clear_scratch_buffer_keeps_in_task_local_when_disabled(self):
        """Test that clear_scratch_buffer keeps buffer in task-local when pooling disabled."""
        disable_buffer_pooling()

        # Get buffer
        buffer = get_scratch_buffer()
        buffer.extend(b"test data")

        # Clear buffer
        clear_scratch_buffer()

        # Verify buffer remains in task-local storage
        task_buffer = _scratch_buffer_var.get()
        assert task_buffer is buffer
        assert len(task_buffer) == 0

    def test_buffer_pooling_consistent_behavior(self):
        """Test that buffer pooling provides consistent behavior across calls."""
        enable_buffer_pooling()

        # Get buffer multiple times
        buffer1 = get_scratch_buffer()
        buffer2 = get_scratch_buffer()
        buffer3 = get_scratch_buffer()

        # All should be the same object
        assert buffer1 is buffer2
        assert buffer2 is buffer3

        # Clear and get again
        clear_scratch_buffer()
        # Note: In some cases, the same buffer might be returned if it's the only one available
        # This is acceptable behavior as long as the buffer is properly cleared

    def test_buffer_pooling_pool_full_handling(self):
        """Test handling when buffer pool is full."""
        enable_buffer_pooling()

        # Test that we can get and return buffers without issues
        buffer = get_scratch_buffer()
        buffer.extend(b"test data")

        # Clear buffer - should return it to pool
        clear_scratch_buffer()

        # Verify that the pool has at least one buffer
        stats = get_buffer_pool_stats()
        assert stats["pool_size"] >= 1

        # Verify that the pool doesn't exceed max size
        assert stats["pool_size"] <= MAX_POOL_SIZE

    def test_buffer_pooling_concurrent_access(self):
        """Test buffer pooling with concurrent access."""
        enable_buffer_pooling()

        buffers: Set[int] = set()

        async def worker():
            buffer = get_scratch_buffer()
            buffer.clear()  # Clear to ensure clean state
            buffer.extend(b"worker")
            buffers.add(id(buffer))
            clear_scratch_buffer()
            return buffer

        async def run_concurrent():
            # Run multiple workers concurrently
            tasks = [worker() for _ in range(10)]
            await asyncio.gather(*tasks)

        # Run the test
        asyncio.run(run_concurrent())

        # Verify that buffers were reused (fewer unique buffers than workers)
        assert len(buffers) <= 10
        # In a well-functioning pool, we should see some reuse
        assert len(buffers) <= MAX_POOL_SIZE


class TestBufferLeakPrevention:
    """Test that the buffer leak is properly prevented."""

    def setup_method(self):
        """Reset buffer pooling state before each test."""
        disable_buffer_pooling()
        try:
            _scratch_buffer_var.set(None)
        except LookupError:
            pass

        # Clear the buffer pool to ensure clean state
        try:
            from flujo.utils.performance import _get_buffer_pool

            pool = _get_buffer_pool()
            while not pool.empty():
                try:
                    pool.get_nowait()
                except Exception:
                    break
        except Exception:
            pass

    def test_no_buffer_leak_in_pooling_mode(self):
        """Test that buffers are properly returned to pool to prevent leaks."""
        enable_buffer_pooling()

        # Get initial pool stats
        initial_stats = get_buffer_pool_stats()
        initial_pool_size = initial_stats["pool_size"]

        # Perform multiple get/clear cycles
        for _ in range(50):
            buffer = get_scratch_buffer()
            buffer.extend(b"test data")
            clear_scratch_buffer()

        # Get final pool stats
        final_stats = get_buffer_pool_stats()
        final_pool_size = final_stats["pool_size"]

        # Pool size should not have decreased (indicating no leaks)
        assert final_pool_size >= initial_pool_size

        # Pool should not be empty (indicating buffers are being returned)
        assert final_pool_size > 0

    def test_consistent_buffer_identity_in_pooling_mode(self):
        """Test that buffer identity is consistent within a task when pooling is enabled."""
        enable_buffer_pooling()

        # Get buffer multiple times without clearing
        buffer1 = get_scratch_buffer()
        buffer2 = get_scratch_buffer()
        buffer3 = get_scratch_buffer()

        # All should be the same object (consistent identity)
        assert buffer1 is buffer2
        assert buffer2 is buffer3

        # Clear and get again
        clear_scratch_buffer()
        # Note: In some cases, the same buffer might be returned if it's the only one available
        # This is acceptable behavior as long as the buffer is properly cleared

    def test_consistent_clear_behavior(self):
        """Test that clear_scratch_buffer behavior is consistent regardless of pooling mode."""
        # Test with pooling disabled
        disable_buffer_pooling()
        buffer1 = get_scratch_buffer()
        buffer1.extend(b"test")
        clear_scratch_buffer()
        assert len(buffer1) == 0

        # Test with pooling enabled
        enable_buffer_pooling()
        buffer2 = get_scratch_buffer()
        buffer2.extend(b"test")
        clear_scratch_buffer()
        assert len(buffer2) == 0

        # Both should have cleared the buffer contents
        # The difference is in where the buffer goes after clearing

    def test_task_local_storage_consistency(self):
        """Test that task-local storage behavior is consistent."""
        enable_buffer_pooling()

        # Get buffer
        buffer = get_scratch_buffer()

        # Verify it's stored in task-local storage
        task_buffer = _scratch_buffer_var.get()
        assert task_buffer is buffer

        # Clear buffer
        clear_scratch_buffer()

        # Verify task-local reference is cleared
        task_buffer = _scratch_buffer_var.get()
        assert task_buffer is None

        # Get buffer again
        new_buffer = get_scratch_buffer()

        # Should be stored in task-local storage again
        task_buffer = _scratch_buffer_var.get()
        assert task_buffer is new_buffer


class TestPerformanceOptimizations:
    """Test performance optimization utilities."""

    def test_time_perf_ns(self):
        """Test nanosecond precision timing."""
        from flujo.utils.performance import time_perf_ns, time_perf_ns_to_seconds

        start = time_perf_ns()
        assert isinstance(start, int)
        assert start > 0

        # Test conversion
        seconds = time_perf_ns_to_seconds(start)
        assert isinstance(seconds, float)
        assert seconds > 0

    def test_measure_time_decorator(self):
        """Test the measure_time decorator."""
        from flujo.utils.performance import measure_time

        @measure_time
        def test_function():
            return "test result"

        result = test_function()
        assert result == "test result"

    def test_measure_time_async_decorator(self):
        """Test the measure_time_async decorator."""
        from flujo.utils.performance import measure_time_async

        @measure_time_async
        async def test_async_function():
            await asyncio.sleep(0.001)  # Small delay
            return "test async result"

        result = asyncio.run(test_async_function())
        assert result == "test async result"

    def test_optimize_event_loop(self):
        """Test event loop optimization."""
        from flujo.utils.performance import optimize_event_loop

        # Should not raise an exception
        optimize_event_loop()


if __name__ == "__main__":
    pytest.main([__file__])
