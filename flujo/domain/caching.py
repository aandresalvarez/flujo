from __future__ import annotations

from typing import Protocol, Any, Optional, runtime_checkable
from pydantic import BaseModel, PrivateAttr
from flujo.type_definitions.common import JSONObject


@runtime_checkable
class CacheBackend(Protocol):
    async def get(self, key: str) -> Any: ...

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None: ...


class InMemoryCache(BaseModel):
    """Simple in-memory cache for step results."""

    _cache: JSONObject = PrivateAttr(default_factory=dict)

    async def get(self, key: str) -> Any:
        value = self._cache.get(key)
        if value is None:
            return None
        # Return a defensive copy so subsequent mutations (e.g., setting
        # metadata_['cache_hit'] = True) do not affect previously returned
        # references held by callers/tests.
        try:
            if hasattr(value, "model_copy"):
                return value.model_copy(deep=True)
        except Exception:
            pass
        try:
            import copy as _copy

            return _copy.deepcopy(value)
        except Exception as e:
            # CRITICAL: Do not return original reference to avoid cache poisoning.
            # Force recomputation by returning None and warn for observability.
            try:
                import logging as _logging

                _logging.warning(
                    "InMemoryCache: failed defensive copy for key '%s'. Returning None to avoid cache poisoning. Error: %s",
                    key,
                    e,
                )
            except Exception:
                pass
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        # Store a defensive copy to avoid later in-place mutations affecting
        # the cached baseline object.
        try:
            if hasattr(value, "model_copy"):
                self._cache[key] = value.model_copy(deep=True)
                return
        except Exception:
            pass
        try:
            import copy as _copy

            self._cache[key] = _copy.deepcopy(value)
        except Exception as e:
            # CRITICAL: Do not store original reference to avoid cache poisoning.
            # Skip caching and warn; caller will recompute next time.
            try:
                import logging as _logging

                _logging.warning(
                    "InMemoryCache: failed to store defensive copy for key '%s'. Skipping cache write. Error: %s",
                    key,
                    e,
                )
            except Exception:
                pass


__all__ = ["CacheBackend", "InMemoryCache"]
