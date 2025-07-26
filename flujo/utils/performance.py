"""
Performance optimization utilities for Flujo.

This module provides high-performance utilities and optimizations that can be
applied throughout the Flujo codebase to improve throughput, reduce latency,
and minimize memory usage.
"""

import logging
import time
import threading
from typing import Any, Callable, TypeVar, Awaitable
from functools import wraps

# Configure logger for the module
logger = logging.getLogger(__name__)

# Type variable for generic functions
T = TypeVar("T")

# Performance constants
DEFAULT_BUFFER_SIZE = 4096  # 4KB initial size

# Thread-local storage for scratch buffers to avoid race conditions
_thread_local = threading.local()


def _get_thread_scratch_buffer() -> bytearray:
    """Get thread-local scratch buffer, creating it if necessary."""
    if not hasattr(_thread_local, "scratch_buffer"):
        _thread_local.scratch_buffer = bytearray(DEFAULT_BUFFER_SIZE)  # 4KB initial size
    return _thread_local.scratch_buffer  # type: ignore[no-any-return]


def clear_scratch_buffer() -> None:
    """Clear the thread-local scratch buffer for reuse."""
    buffer = _get_thread_scratch_buffer()
    buffer.clear()


def get_scratch_buffer() -> bytearray:
    """Get the thread-local scratch buffer for temporary operations."""
    return _get_thread_scratch_buffer()


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
    """
    try:
        import uvloop
        import asyncio

        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        logger.info("✅ Using uvloop for enhanced async performance")
    except ImportError:
        logger.info("ℹ️  uvloop not available, using standard asyncio event loop")
    except Exception as e:
        logger.error(f"⚠️  Failed to initialize uvloop: {e}")


# Initialize optimizations when module is imported
optimize_event_loop()
