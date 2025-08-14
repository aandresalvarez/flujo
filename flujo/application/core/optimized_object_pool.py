"""
Shim module: re-export optimized object pool from the canonical optimization package.

This avoids duplication and ensures a single source of truth for the implementation.
"""

from .optimization.memory.object_pool import (  # noqa: F401
    PoolStats,
    OptimizedObjectPool,
    TypedObjectPool,
    get_global_pool,
    get_pooled_object,
    return_pooled_object,
    create_typed_pool,
    PooledObject,
)

__all__ = [
    "PoolStats",
    "OptimizedObjectPool",
    "TypedObjectPool",
    "get_global_pool",
    "get_pooled_object",
    "return_pooled_object",
    "create_typed_pool",
    "PooledObject",
]
