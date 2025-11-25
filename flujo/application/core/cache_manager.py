"""Cache management for step execution results."""

from __future__ import annotations
from typing import Any, Optional
from .default_components import DefaultCacheKeyGenerator, _LRUCache


class CacheManager:
    """Manages caching of step execution results."""

    def __init__(
        self,
        backend: Any = None,
        key_generator: Optional[Any] = None,
        enable_cache: bool = True,
    ) -> None:
        self._backend = backend
        self._key_generator = key_generator or DefaultCacheKeyGenerator()
        self._enable_cache = enable_cache
        self._internal_cache: Optional[_LRUCache] = None

    @property
    def backend(self) -> Any:
        """Get the cache backend."""
        return self._backend

    def get_internal_cache(self) -> _LRUCache:
        """Get or create the internal LRU cache."""
        if self._internal_cache is None:
            # Create a reasonable default cache
            self._internal_cache = _LRUCache(max_size=1024, ttl=3600)
        return self._internal_cache

    def clear_cache(self) -> None:
        """Clear all cached data."""
        if self._internal_cache is not None:
            self._internal_cache.clear()
        if hasattr(self._backend, "clear"):
            self._backend.clear()

    def generate_cache_key(
        self, step: Any, data: Any, context: Optional[Any], resources: Optional[Any]
    ) -> str:
        """Generate a cache key for the given step execution parameters."""
        if not self._enable_cache:
            return ""
        return self._key_generator.generate_key(step, data, context, resources)

    def is_cache_enabled(self) -> bool:
        """Check if caching is enabled."""
        return self._enable_cache

    async def get_cached_result(self, key: str) -> Optional[Any]:
        """Retrieve a cached result by key."""
        if not self._enable_cache or not key:
            return None

        # Try backend first, then internal cache
        if hasattr(self._backend, "get"):
            try:
                result = await self._backend.get(key)
                if result is not None:
                    return result
            except Exception:
                pass

        if self._internal_cache is not None:
            return self._internal_cache.get(key)

        return None

    async def set_cached_result(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store a result in cache."""
        if not self._enable_cache or not key:
            return

        # Store in backend first, then internal cache
        if hasattr(self._backend, "set"):
            try:
                await self._backend.set(key, value, ttl=ttl)
            except Exception:
                pass
        elif hasattr(self._backend, "put"):
            try:
                ttl_s = ttl if ttl is not None else getattr(self._backend, "ttl_s", 0)
                await self._backend.put(key, value, ttl_s=ttl_s)
            except Exception:
                pass

        if self._internal_cache is not None:
            # _LRUCache manages TTL internally; no ttl kwarg is accepted
            self._internal_cache.set(key, value)
