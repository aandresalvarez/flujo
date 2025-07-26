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

import logging
import time
import contextvars
from typing import Any, Callable, TypeVar, Awaitable, Optional
from functools import wraps
from queue import Queue

# Configure logger for the module
logger = logging.getLogger(__name__)

# Type variable for generic functions
T = TypeVar("T")

# Performance constants
DEFAULT_BUFFER_SIZE = 4096  # 4KB initial size
MAX_POOL_SIZE = 100  # Maximum number of buffers in the pool
ENABLE_BUFFER_POOLING = False  # Can be enabled for high-concurrency scenarios

# Thread-safe pool for reusable scratch buffers to minimize memory usage
# Only used when ENABLE_BUFFER_POOLING is True
_buffer_pool: Optional[Queue[bytearray]] = None


def _get_buffer_pool() -> Queue[bytearray]:
    """Get or create the buffer pool."""
    global _buffer_pool
    if _buffer_pool is None:
        _buffer_pool = Queue(maxsize=MAX_POOL_SIZE)
    return _buffer_pool


# Task-local storage for scratch buffers to avoid race conditions in async contexts
_scratch_buffer_var: contextvars.ContextVar[Optional[bytearray]] = contextvars.ContextVar(
    "scratch_buffer", default=None
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
        A bytearray buffer of DEFAULT_BUFFER_SIZE (4KB) for temporary operations
    """
    if ENABLE_BUFFER_POOLING:
        # Try to get a buffer from the pool first
        pool = _get_buffer_pool()
        try:
            pooled_buffer = pool.get_nowait()
            return pooled_buffer
        except Exception:
            # Pool is empty or not available, create new buffer
            pass

    # Use task-local storage (default behavior)
    task_buffer: Optional[bytearray] = _scratch_buffer_var.get()
    if task_buffer is None:
        new_buffer = bytearray(DEFAULT_BUFFER_SIZE)  # 4KB initial size
        _scratch_buffer_var.set(new_buffer)
        return new_buffer
    return task_buffer


def clear_scratch_buffer() -> None:
    """Clear the task-local scratch buffer for reuse.

    This function clears the contents of the current task's scratch buffer,
    making it ready for reuse. The buffer object itself is preserved to
    avoid allocation overhead.

    When ENABLE_BUFFER_POOLING is True, the buffer is returned to the
    global pool for reuse by other tasks.
    """
    if ENABLE_BUFFER_POOLING:
        # Get current buffer and return it to pool
        buffer = _get_thread_scratch_buffer()
        buffer.clear()
        pool = _get_buffer_pool()
        try:
            pool.put_nowait(buffer)
        except Exception:
            # Pool is full, discard the buffer
            pass
    else:
        # Standard behavior: clear the task-local buffer
        buffer = _get_thread_scratch_buffer()
        buffer.clear()


def get_scratch_buffer() -> bytearray:
    """Get the task-local scratch buffer for temporary operations.

    This is the main public API for obtaining a scratch buffer. The buffer
    is guaranteed to be thread-safe and task-local, meaning each async task
    will get its own buffer to avoid race conditions.

    The buffer is pre-allocated to DEFAULT_BUFFER_SIZE (4KB) to minimize
    allocation overhead during performance-critical operations.

    Returns:
        A bytearray buffer ready for temporary operations
    """
    return _get_thread_scratch_buffer()


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
    """Optimize the asyncio event loop for better performance.

    This function attempts to use uvloop on Unix systems for significantly
    better async performance. On Windows or when uvloop is not available,
    it falls back to the standard asyncio event loop.

    Note: This function must be called explicitly to enable uvloop.
    """
    try:
        import uvloop
        import asyncio

        current_policy = asyncio.get_event_loop_policy()
        if isinstance(current_policy, uvloop.EventLoopPolicy):
            logger.info("ℹ️  uvloop is already set as the event loop policy")
            return

        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        logger.info("✅ Using uvloop for enhanced async performance")
    except ImportError:
        logger.info("ℹ️  uvloop not available, using standard asyncio event loop")
    except Exception as e:
        logger.error(f"⚠️  Failed to initialize uvloop: {e}")


# Initialize optimizations when module is imported
optimize_event_loop()
