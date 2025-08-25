"""
Performance optimization utilities for Flujo.

This module provides high-performance utilities and optimizations that can be
applied throughout the Flujo codebase to improve throughput, reduce latency,
and minimize memory usage.

## Scratch Buffer Management

The module implements a sophisticated scratch buffer management system that addresses
the trade-offs between memory usage and performance in async contexts:

### Default Behavior (Task-Local Storage)
- Uses `contextvars.ContextVar` for thread-safe, task-local buffer storage
- Each async task gets its own 4KB buffer to avoid race conditions
- Buffers are reused within the same task context
- Optimal for most applications with moderate concurrency

### High-Concurrency Mode (Buffer Pooling)
- Optional buffer pooling mechanism for high-concurrency scenarios
- Reduces memory usage by sharing buffers across tasks
- Configurable pool size with bounds checking
- Small performance overhead due to synchronization

### Memory Usage Considerations

**Task-Local Storage (Default):**
- Memory usage: ~4KB per concurrent async task
- Best performance with minimal synchronization overhead
- Recommended for most applications

**Buffer Pooling (Optional):**
- Memory usage: ~4KB × pool_size (default: 100 buffers = 400KB)
- Reduces memory usage in high-concurrency scenarios
- Small performance overhead due to queue synchronization
- Use only when memory usage is critical and concurrency is high

### Usage Guidelines

1. **Default Applications**: Use the default task-local storage
2. **High-Concurrency Applications**: Consider enabling buffer pooling
3. **Memory-Critical Applications**: Enable buffer pooling and monitor pool stats
4. **Performance-Critical Applications**: Stick with task-local storage

### Configuration

```python
from flujo.utils.performance import enable_buffer_pooling, disable_buffer_pooling

# Enable for high-concurrency scenarios
enable_buffer_pooling()

# Disable to return to default behavior
disable_buffer_pooling()

# Monitor pool utilization
stats = get_buffer_pool_stats()
print(f"Pool utilization: {stats['utilization']:.2%}")
```

This implementation provides a robust solution that balances memory efficiency
with performance, offering both task-local safety and optional pooling for
high-concurrency scenarios.
"""

import contextvars
import logging
import time
from functools import wraps
from queue import Queue
from typing import Any, Awaitable, Callable, Optional, TypeVar

# Configure logging
logger = logging.getLogger(__name__)

# Type variable for generic functions
T = TypeVar("T")

# Buffer pooling configuration
DEFAULT_BUFFER_SIZE = 4096  # 4KB initial size
MAX_POOL_SIZE = 100  # Maximum number of buffers in the pool
ENABLE_BUFFER_POOLING = False  # Can be enabled for high-concurrency scenarios

# Thread-safe pool for reusable scratch buffers to minimize memory usage
# Only used when ENABLE_BUFFER_POOLING is True
_buffer_pool: Optional[Queue[bytearray]] = None


def _get_buffer_pool() -> Queue[bytearray]:
    """Get or create the buffer pool with proper initialization."""
    global _buffer_pool
    if _buffer_pool is None:
        _buffer_pool = Queue(maxsize=MAX_POOL_SIZE)
    return _buffer_pool


# Task-local storage for scratch buffers to avoid race conditions in async contexts
_scratch_buffer_var: contextvars.ContextVar[Optional[bytearray]] = contextvars.ContextVar(
    "scratch_buffer", default=None
)
# Task-local flag to track if buffer has been returned to pool
_buffer_returned_to_pool_var: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "buffer_returned_to_pool", default=False
)


def _get_thread_scratch_buffer() -> bytearray:
    """Get task-local scratch buffer, creating it if necessary.

    This function provides a thread-safe way to get a scratch buffer for temporary
    operations. The buffer is task-local, meaning each async task gets its own
    buffer to avoid race conditions.

    When ENABLE_BUFFER_POOLING is True, the function will attempt to reuse
    buffers from a global pool to reduce memory usage in high-concurrency
    scenarios. However, this comes with a small performance cost due to
    synchronization overhead.

    Returns:
        A bytearray buffer ready for temporary operations
    """
    # Always check task-local storage first for consistency
    task_buffer: Optional[bytearray] = _scratch_buffer_var.get()
    buffer_returned_to_pool: bool = _buffer_returned_to_pool_var.get()

    if task_buffer is not None and not buffer_returned_to_pool:
        # Ensure callers always receive an empty buffer at retrieval time
        # to avoid leaking content across test/task boundaries.
        if len(task_buffer) != 0:
            task_buffer.clear()
        return task_buffer

    if ENABLE_BUFFER_POOLING:
        # Try to get a buffer from the pool first
        pool = _get_buffer_pool()
        try:
            pooled_buffer = pool.get_nowait()
            # Clear the pooled buffer before use
            pooled_buffer.clear()
            # Store the pooled buffer in task-local storage to prevent leaks
            _scratch_buffer_var.set(pooled_buffer)
            _buffer_returned_to_pool_var.set(False)  # Mark as not returned to pool
            return pooled_buffer
        except Exception:
            # Pool is empty or not available, create new buffer
            pass

    # Create new buffer and store in task-local storage
    # Create an empty bytearray that can grow efficiently
    new_buffer = bytearray()  # Start empty, will grow as needed
    _scratch_buffer_var.set(new_buffer)
    _buffer_returned_to_pool_var.set(False)  # Mark as not returned to pool
    return new_buffer


def _return_buffer_to_pool_sync(buffer: bytearray) -> bool:
    """Return a buffer to the pool synchronously with proper error handling.

    Returns:
        True if the buffer was successfully returned to the pool, False otherwise
    """
    if not ENABLE_BUFFER_POOLING:
        return False

    pool = _get_buffer_pool()
    if pool.full():
        # Pool is full, discard the buffer
        logger.debug("Buffer pool is full, discarding buffer")
        return False

    try:
        # Clear the buffer before returning to pool
        buffer.clear()
        # Use put_nowait to avoid blocking
        pool.put_nowait(buffer)
        # Mark that this buffer has been returned to the pool
        _buffer_returned_to_pool_var.set(True)
        return True
    except Exception as e:
        # Pool is full or another error occurred, discard the buffer
        logger.debug("Failed to return buffer to pool: %s", e, exc_info=True)
        return False


def clear_scratch_buffer() -> None:
    """Clear the task-local scratch buffer for reuse.

    This function clears the contents of the current task's scratch buffer,
    making it ready for reuse. The buffer object itself is preserved to
    avoid allocation overhead.

    When buffer pooling is enabled, this function will clear the buffer contents
    but keep the buffer in task-local storage for immediate reuse within the same
    task. This maintains buffer identity while still enabling efficient memory usage.

    Behavior is consistent regardless of pooling mode:
    - Always clears the buffer contents
    - Always maintains buffer identity within the same task
    - When pooling is enabled, buffers can be explicitly released to the pool

    This ensures consistent API semantics and prevents buffer leaks.
    """
    # Get current buffer from task-local storage
    task_buffer: Optional[bytearray] = _scratch_buffer_var.get()

    if task_buffer is None:
        # No buffer exists, create one and clear it for consistency
        buffer = _get_thread_scratch_buffer()
        buffer.clear()
        return

    # Clear the buffer contents
    task_buffer.clear()

    # When pooling is enabled, we keep the buffer in task-local storage
    # for immediate reuse within the same task. This maintains buffer identity
    # while still allowing cross-task pooling when explicitly released.
    # The buffer will only be returned to the pool when release_scratch_buffer()
    # is explicitly called.


def get_scratch_buffer() -> bytearray:
    """Get the task-local scratch buffer for temporary operations.

    This is the main public API for obtaining a scratch buffer. The buffer
    is guaranteed to be thread-safe and task-local, meaning each async task
    will get its own buffer to avoid race conditions.

    The buffer starts empty and can grow as needed to minimize allocation overhead
    during performance-critical operations.

    Returns:
        A bytearray buffer ready for temporary operations
    """
    return _get_thread_scratch_buffer()


def release_scratch_buffer() -> None:
    """Release the task-local scratch buffer back to the pool (when pooling is enabled).

    This function explicitly releases the current task's scratch buffer back to
    the global pool when buffer pooling is enabled. This is useful for:
    - Explicitly managing buffer lifecycle
    - Ensuring buffers are returned to the pool before task completion
    - Optimizing memory usage in high-concurrency scenarios

    When buffer pooling is disabled, this function is a no-op since buffers
    are managed entirely within task-local storage.

    Note: This function is optional - buffers will be automatically cleaned up
    when the task context is destroyed, but explicit release can be beneficial
    for memory optimization in long-running tasks.
    """
    if not ENABLE_BUFFER_POOLING:
        # No-op when pooling is disabled
        return

    task_buffer: Optional[bytearray] = _scratch_buffer_var.get()
    buffer_returned_to_pool: bool = _buffer_returned_to_pool_var.get()

    if task_buffer is None or buffer_returned_to_pool:
        # No buffer to release or buffer already returned
        return

    # Return buffer to pool
    if _return_buffer_to_pool_sync(task_buffer):
        # Clear the task-local reference and mark as returned
        _scratch_buffer_var.set(None)
        _buffer_returned_to_pool_var.set(True)


def enable_buffer_pooling() -> None:
    """Enable buffer pooling for high-concurrency scenarios.

    This function enables the buffer pooling mechanism, which can significantly
    reduce memory usage in scenarios with many concurrent async tasks. However,
    it comes with a small performance cost due to synchronization overhead.

    WARNING: This changes the behavior of scratch buffers. When enabled:
    - Buffers may be shared between different async contexts
    - The same buffer object may be returned to different tasks
    - There is a small performance overhead due to synchronization

    This should only be enabled when memory usage is a critical concern
    and the application has many concurrent async tasks.
    """
    global ENABLE_BUFFER_POOLING
    ENABLE_BUFFER_POOLING = True
    logger.info("Buffer pooling enabled for high-concurrency scenarios")


def disable_buffer_pooling() -> None:
    """Disable buffer pooling and return to task-local storage.

    This function disables the buffer pooling mechanism and returns to the
    default task-local storage behavior. This is the recommended setting
    for most applications as it provides the best balance of performance
    and safety.
    """
    global ENABLE_BUFFER_POOLING
    ENABLE_BUFFER_POOLING = False
    logger.info("Buffer pooling disabled, using task-local storage")


def get_buffer_pool_stats() -> dict[str, Any]:
    """Get statistics about the buffer pool (when enabled).

    Returns:
        Dictionary with pool statistics including size and utilization
    """
    if not ENABLE_BUFFER_POOLING:
        return {"enabled": False, "pool_size": 0, "utilization": 0.0}

    pool = _get_buffer_pool()
    return {
        "enabled": True,
        "pool_size": pool.qsize(),
        "max_size": MAX_POOL_SIZE,
        "utilization": pool.qsize() / MAX_POOL_SIZE,
    }


def time_perf_ns() -> int:
    """Get current time in nanoseconds using perf_counter_ns for maximum precision."""
    return time.perf_counter_ns()


def time_perf_ns_to_seconds(ns: int) -> float:
    """Convert nanoseconds to seconds."""
    return ns / 1_000_000_000.0


def measure_time(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to measure execution time with nanosecond precision.

    This decorator provides high-precision timing measurements for performance
    critical functions. It uses time.perf_counter_ns() for maximum accuracy.

    Args:
        func: The function to measure

    Returns:
        Wrapped function that measures and logs execution time
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        start_ns = time_perf_ns()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_ns = time_perf_ns()
            duration_ns = end_ns - start_ns
            duration_s = time_perf_ns_to_seconds(duration_ns)
            logger.info(f"{func.__name__}: {duration_s:.6f}s ({duration_ns}ns)")

    return wrapper


def measure_time_async(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
    """Async decorator to measure execution time with nanosecond precision.

    This decorator provides high-precision timing measurements for async
    performance critical functions.

    Args:
        func: The async function to measure

    Returns:
        Wrapped async function that measures and logs execution time
    """

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> T:
        start_ns = time_perf_ns()
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            end_ns = time_perf_ns()
            duration_ns = end_ns - start_ns
            duration_s = time_perf_ns_to_seconds(duration_ns)
            logger.info(f"{func.__name__}: {duration_s:.6f}s ({duration_ns}ns)")

    return wrapper


def optimize_event_loop() -> None:
    """Optimize the event loop for better performance.

    This function applies various optimizations to the current event loop
    to improve performance for async operations. It should be called
    early in the application lifecycle, before any async operations begin.

    Optimizations include:
    - Enabling uvloop if available (significant performance improvement)
    - Configuring the event loop for better throughput
    - Setting appropriate limits for concurrent operations
    """
    try:
        import uvloop
        import asyncio

        # uvloop provides significant performance improvements over the default event loop
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        logger.info("✅ Using uvloop for enhanced async performance")
    except ImportError:
        # uvloop not available, use default event loop
        logger.info("ℹ️ uvloop not available, using default event loop")
        pass

    # Configure the event loop for better performance
    try:
        import asyncio

        # Use get_running_loop() to avoid RuntimeError when no loop is active
        loop = asyncio.get_running_loop()

        # Set a reasonable limit for concurrent operations
        # This prevents resource exhaustion in high-concurrency scenarios
        if hasattr(loop, "set_default_executor"):
            import concurrent.futures
            import os

            # Shutdown existing executor if it exists to prevent resource leaks
            if hasattr(loop, "_default_executor") and loop._default_executor is not None:
                loop._default_executor.shutdown(wait=True)

            executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=min(32, (os.cpu_count() or 1) * 4)
            )
            loop.set_default_executor(executor)
            logger.info("✅ Configured thread pool executor for better performance")
    except RuntimeError:
        # No event loop is running, which is fine for early initialization
        logger.info("ℹ️ No event loop running, optimizations will be applied when loop starts")
    except Exception as e:
        logger.warning(f"⚠️ Failed to optimize event loop: {e}")
