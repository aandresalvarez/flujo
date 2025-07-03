from __future__ import annotations

from typing import Protocol, Any, Dict, Optional, runtime_checkable


@runtime_checkable
class CacheBackend(Protocol):
    async def get(self, key: str) -> Any: ...

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None: ...


class InMemoryCache(CacheBackend):
    """Simple in-memory cache for step results."""

    def __init__(self) -> None:
        # TTL handling could be added later
        self._cache: Dict[str, Any] = {}

    async def get(self, key: str) -> Any:
        return self._cache.get(key)

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        self._cache[key] = value
