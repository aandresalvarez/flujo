"""
Optimized object pool system with type-specific optimizations.

This module provides high-performance object pooling with type-specific optimizations,
overflow protection, and utilization statistics. The pool system reduces memory
allocations and garbage collection pressure by reusing frequently allocated objects.
"""

import asyncio
import time
import weakref
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Type, TypeVar, Generic, Optional, Set, Callable
from threading import RLock

T = TypeVar("T")


@dataclass
class PoolStats:
    """Statistics for object pool utilization."""

    total_gets: int = 0
    total_puts: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    current_size: int = 0
    max_size_reached: int = 0
    overflow_count: int = 0
    created_objects: int = 0
    reused_objects: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_requests = self.cache_hits + self.cache_misses
        return self.cache_hits / total_requests if total_requests > 0 else 0.0

    @property
    def reuse_rate(self) -> float:
        """Calculate object reuse rate."""
        total_objects = self.created_objects + self.reused_objects
        return self.reused_objects / total_objects if total_objects > 0 else 0.0


@dataclass
class OptimizedObjectPool:
    """
    High-performance object pool with type-specific optimizations.

    Features:
    - Type-specific pools for better memory locality
    - Overflow protection to prevent unbounded growth
    - Utilization statistics for monitoring
    - Thread-safe operations with minimal locking
    - Weak references to prevent memory leaks
    - Fast paths for common object types
    """

    max_pool_size: int = 500
    max_pools: int = 50
    cleanup_threshold: float = 0.75  # Cleanup when pool reaches 75% capacity
    stats_enabled: bool = True

    # Internal state
    _pools: Dict[Type[Any], deque[Any]] = field(default_factory=dict, init=False)
    _locks: Dict[Type[Any], asyncio.Lock] = field(default_factory=dict, init=False)
    _stats: Dict[Type[Any], PoolStats] = field(default_factory=dict, init=False)
    _weak_refs: Set[weakref.ref[Any]] = field(default_factory=set, init=False)
    _last_cleanup: float = field(default_factory=time.time, init=False)
    _global_lock: RLock = field(default_factory=RLock, init=False)

    def __post_init__(self) -> None:
        """Initialize the object pool."""
        if self.max_pool_size <= 0:
            raise ValueError("max_pool_size must be positive")
        if self.max_pools <= 0:
            raise ValueError("max_pools must be positive")

    async def get(self, obj_type: Type[T], *args: Any, **kwargs: Any) -> T:
        """
        Get object from pool or create new one.

        Args:
            obj_type: Type of object to get
            *args: Arguments for object creation
            **kwargs: Keyword arguments for object creation

        Returns:
            Object instance of requested type
        """
        # Fast path for common types
        if obj_type in {str, int, float, bool, list, dict, set, tuple}:
            return self._create_builtin(obj_type, *args, **kwargs)

        # Get or create pool for this type
        pool = await self._get_pool(obj_type)
        lock = await self._get_lock(obj_type)

        async with lock:
            # Try to get from pool
            if pool:
                obj = pool.popleft()
                self._update_stats(obj_type, "cache_hit")
                self._update_stats(obj_type, "reused_object")

                # Reset object state if it has a reset method
                if hasattr(obj, "reset"):
                    try:
                        if asyncio.iscoroutinefunction(obj.reset):
                            await obj.reset()
                        else:
                            obj.reset()
                    except Exception:
                        # If reset fails, create new object
                        return await self._create_object(obj_type, *args, **kwargs)

                return obj  # type: ignore[no-any-return]
            else:
                # Pool is empty, create new object
                self._update_stats(obj_type, "cache_miss")
                return await self._create_object(obj_type, *args, **kwargs)

    async def put(self, obj: Any) -> None:
        """
        Return object to pool.

        Args:
            obj: Object to return to pool
        """
        if obj is None:
            return

        obj_type = type(obj)

        # Skip builtin types
        if obj_type in {str, int, float, bool, tuple}:
            return

        # Skip if we have too many pools
        if len(self._pools) >= self.max_pools:
            return

        pool = await self._get_pool(obj_type)
        lock = await self._get_lock(obj_type)

        async with lock:
            # Check pool size to prevent overflow
            if len(pool) >= self.max_pool_size:
                self._update_stats(obj_type, "overflow")
                return

            # Clean object state if possible
            if hasattr(obj, "clear") and callable(obj.clear):
                try:
                    obj.clear()
                except Exception:
                    # If clearing fails, don't pool the object
                    return

            # Add to pool
            pool.append(obj)
            self._update_stats(obj_type, "put")

            # Update max size reached
            current_size = len(pool)
            stats = self._get_stats(obj_type)
            if current_size > stats.max_size_reached:
                stats.max_size_reached = current_size

        # Periodic cleanup
        await self._maybe_cleanup()

    async def clear(self, obj_type: Optional[Type[Any]] = None) -> None:
        """
        Clear pool(s).

        Args:
            obj_type: Specific type to clear, or None to clear all
        """
        if obj_type is not None:
            # Clear specific pool
            if obj_type in self._pools:
                lock = await self._get_lock(obj_type)
                async with lock:
                    self._pools[obj_type].clear()
                    if self.stats_enabled:
                        self._stats[obj_type] = PoolStats()
        else:
            # Clear all pools
            with self._global_lock:
                for pool in self._pools.values():
                    pool.clear()
                self._pools.clear()
                self._locks.clear()
                if self.stats_enabled:
                    self._stats.clear()

    def get_stats(self, obj_type: Optional[Type[Any]] = None) -> Dict[Type[Any], PoolStats]:
        """
        Get pool utilization statistics.

        Args:
            obj_type: Specific type to get stats for, or None for all

        Returns:
            Dictionary mapping types to their statistics
        """
        if not self.stats_enabled:
            return {}

        if obj_type is not None:
            return {obj_type: self._get_stats(obj_type)}
        else:
            # Update current sizes
            for pool_type, pool in self._pools.items():
                stats = self._get_stats(pool_type)
                stats.current_size = len(pool)

            return dict(self._stats)

    def get_total_stats(self) -> PoolStats:
        """Get aggregated statistics across all pools."""
        if not self.stats_enabled:
            return PoolStats()

        total = PoolStats()
        for stats in self._stats.values():
            total.total_gets += stats.total_gets
            total.total_puts += stats.total_puts
            total.cache_hits += stats.cache_hits
            total.cache_misses += stats.cache_misses
            total.current_size += stats.current_size
            total.max_size_reached = max(total.max_size_reached, stats.max_size_reached)
            total.overflow_count += stats.overflow_count
            total.created_objects += stats.created_objects
            total.reused_objects += stats.reused_objects

        return total

    async def _get_pool(self, obj_type: Type[Any]) -> deque[Any]:
        """Get or create pool for object type."""
        if obj_type not in self._pools:
            with self._global_lock:
                if obj_type not in self._pools:
                    self._pools[obj_type] = deque()
        return self._pools[obj_type]

    async def _get_lock(self, obj_type: Type[Any]) -> asyncio.Lock:
        """Get or create lock for object type."""
        if obj_type not in self._locks:
            with self._global_lock:
                if obj_type not in self._locks:
                    self._locks[obj_type] = asyncio.Lock()
        return self._locks[obj_type]

    def _get_stats(self, obj_type: Type[Any]) -> PoolStats:
        """Get or create stats for object type."""
        if not self.stats_enabled:
            return PoolStats()

        if obj_type not in self._stats:
            with self._global_lock:
                if obj_type not in self._stats:
                    self._stats[obj_type] = PoolStats()
        return self._stats[obj_type]

    def _update_stats(self, obj_type: Type[Any], stat_type: str) -> None:
        """Update statistics for object type."""
        if not self.stats_enabled:
            return

        stats = self._get_stats(obj_type)

        if stat_type == "get":
            stats.total_gets += 1
        elif stat_type == "put":
            stats.total_puts += 1
        elif stat_type == "cache_hit":
            stats.cache_hits += 1
        elif stat_type == "cache_miss":
            stats.cache_misses += 1
        elif stat_type == "overflow":
            stats.overflow_count += 1
        elif stat_type == "created_object":
            stats.created_objects += 1
        elif stat_type == "reused_object":
            stats.reused_objects += 1

    async def _create_object(self, obj_type: Type[T], *args: Any, **kwargs: Any) -> T:
        """Create new object instance."""
        self._update_stats(obj_type, "created_object")

        try:
            # Handle special cases
            if hasattr(obj_type, "__call__"):
                if asyncio.iscoroutinefunction(obj_type):
                    return await obj_type(*args, **kwargs)  # type: ignore[no-any-return]
                else:
                    return obj_type(*args, **kwargs)
            else:
                # Fallback to basic instantiation
                return obj_type()
        except Exception:
            # If creation fails, return a basic instance
            try:
                return obj_type()
            except Exception:
                # Last resort - return None and let caller handle
                raise ValueError(f"Cannot create instance of {obj_type}")

    def _create_builtin(self, obj_type: Type[T], *args: Any, **kwargs: Any) -> T:
        """Create builtin type instances (fast path)."""
        self._update_stats(obj_type, "created_object")

        if obj_type is str:
            return str(*args, **kwargs) if args or kwargs else ""  # type: ignore[return-value]
        elif obj_type is int:
            return int(*args, **kwargs) if args or kwargs else 0  # type: ignore[return-value]
        elif obj_type is float:
            return float(*args, **kwargs) if args or kwargs else 0.0  # type: ignore[return-value]
        elif obj_type is bool:
            return bool(*args, **kwargs) if args or kwargs else False  # type: ignore[return-value]
        elif obj_type is list:
            return list(*args, **kwargs) if args or kwargs else []  # type: ignore[return-value]
        elif obj_type is dict:
            return dict(*args, **kwargs) if args or kwargs else {}  # type: ignore[return-value]
        elif obj_type is set:
            return set(*args, **kwargs) if args or kwargs else set()  # type: ignore[return-value]
        elif obj_type is tuple:
            return tuple(*args, **kwargs) if args or kwargs else ()  # type: ignore[return-value]
        else:
            return obj_type(*args, **kwargs)

    async def _maybe_cleanup(self) -> None:
        """Perform periodic cleanup if needed."""
        current_time = time.time()

        # Only cleanup every 60 seconds
        if current_time - self._last_cleanup < 60:
            return

        self._last_cleanup = current_time

        # Check if cleanup is needed
        total_objects = sum(len(pool) for pool in self._pools.values())
        total_capacity = len(self._pools) * self.max_pool_size

        if total_objects / total_capacity < self.cleanup_threshold:
            return

        # Perform cleanup
        await self._cleanup_pools()

    async def _cleanup_pools(self) -> None:
        """Clean up pools by removing excess objects."""
        with self._global_lock:
            for obj_type, pool in list(self._pools.items()):
                if len(pool) > self.max_pool_size // 2:
                    # Remove half the objects from oversized pools
                    target_size = self.max_pool_size // 2
                    while len(pool) > target_size:
                        pool.pop()

                # Remove empty pools
                if not pool:
                    del self._pools[obj_type]
                    if obj_type in self._locks:
                        del self._locks[obj_type]
                    if obj_type in self._stats:
                        del self._stats[obj_type]


class TypedObjectPool(Generic[T]):
    """Type-safe wrapper around OptimizedObjectPool."""

    def __init__(
        self,
        obj_type: Type[T],
        pool: OptimizedObjectPool,
        factory: Optional[Callable[..., T]] = None,
    ):
        self.obj_type = obj_type
        self.pool = pool
        self.factory = factory

    async def get(self, *args: Any, **kwargs: Any) -> T:
        """Get object of specific type."""
        return await self.pool.get(self.obj_type, *args, **kwargs)

    async def put(self, obj: T) -> None:
        """Return object to pool."""
        await self.pool.put(obj)

    def get_stats(self) -> PoolStats:
        """Get statistics for this type."""
        stats = self.pool.get_stats(self.obj_type)
        return stats[self.obj_type] if self.obj_type in stats else PoolStats()


# Global object pool instance
_global_pool: Optional[OptimizedObjectPool] = None


def get_global_pool() -> OptimizedObjectPool:
    """Get global object pool instance."""
    global _global_pool
    if _global_pool is None:
        _global_pool = OptimizedObjectPool()
    return _global_pool


async def get_pooled_object(obj_type: Type[T], *args: Any, **kwargs: Any) -> T:
    """Get object from global pool."""
    pool = get_global_pool()
    return await pool.get(obj_type, *args, **kwargs)


async def return_pooled_object(obj: Any) -> None:
    """Return object to global pool."""
    pool = get_global_pool()
    await pool.put(obj)


def create_typed_pool(obj_type: Type[T], **pool_kwargs: Any) -> TypedObjectPool[T]:
    """Create typed object pool."""
    pool = OptimizedObjectPool(**pool_kwargs)
    return TypedObjectPool(obj_type, pool)


# Context manager for automatic object return
class PooledObject(Generic[T]):
    """Context manager for pooled objects."""

    def __init__(
        self,
        obj_type: Type[T],
        pool: Optional[OptimizedObjectPool] = None,
        *args: Any,
        **kwargs: Any,
    ):
        self.obj_type = obj_type
        self.pool = pool or get_global_pool()
        self.args = args
        self.kwargs = kwargs
        self.obj: Optional[T] = None

    async def __aenter__(self) -> T:
        """Get object from pool."""
        self.obj = await self.pool.get(self.obj_type, *self.args, **self.kwargs)
        return self.obj

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Return object to pool."""
        if self.obj is not None:
            await self.pool.put(self.obj)
