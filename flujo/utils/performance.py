"""
Performance optimization utilities for Flujo.

This module provides high-performance utilities and optimizations that can be
applied throughout the Flujo codebase to improve throughput, reduce latency,
and minimize memory usage.
"""

import time
from typing import Any, Callable, TypeVar
from functools import wraps

# Type variable for generic functions
T = TypeVar('T')

# Module-level scratch buffer for performance optimization
# This reduces memory allocations by reusing a single bytearray
_SCRATCH_BUFFER = bytearray(4096)  # 4KB initial size

def clear_scratch_buffer() -> None:
    """Clear the scratch buffer for reuse."""
    _SCRATCH_BUFFER.clear()

def get_scratch_buffer() -> bytearray:
    """Get the scratch buffer for temporary operations."""
    return _SCRATCH_BUFFER

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
            print(f"{func.__name__}: {duration_s:.6f}s ({duration_ns}ns)")
    
    return wrapper

def measure_time_async(func: Callable[..., T]) -> Callable[..., T]:
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
            print(f"{func.__name__}: {duration_s:.6f}s ({duration_ns}ns)")
    
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
        if hasattr(asyncio, 'set_event_loop_policy'):
            asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
            print("✅ Using uvloop for enhanced async performance")
        else:
            print("⚠️  uvloop available but set_event_loop_policy not found")
    except ImportError:
        print("ℹ️  uvloop not available, using standard asyncio event loop")
    except Exception as e:
        print(f"⚠️  Failed to initialize uvloop: {e}")

# Initialize optimizations when module is imported
optimize_event_loop() 