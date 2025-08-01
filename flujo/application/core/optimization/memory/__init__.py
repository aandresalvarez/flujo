"""
Memory optimization components.

This module contains components for optimizing memory usage:
- OptimizedObjectPool: High-performance object pooling
- OptimizedContextManager: Context management with copy-on-write
- MemoryOptimization: Memory allocation optimization utilities
"""

from .object_pool import OptimizedObjectPool
from .context_manager import OptimizedContextManager
from .memory_utils import MemoryOptimization

__all__ = [
    'OptimizedObjectPool',
    'OptimizedContextManager', 
    'MemoryOptimization',
]