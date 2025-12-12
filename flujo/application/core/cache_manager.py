"""Cache management for step execution results."""

from __future__ import annotations
import asyncio
from typing import TYPE_CHECKING, Any, Optional
from ...domain.models import StepOutcome, StepResult, Success
from ...infra import telemetry
from .default_cache_components import DefaultCacheKeyGenerator, _LRUCache

# Import the cache override context variable from shared module
from .context_vars import _CACHE_OVERRIDE

if TYPE_CHECKING:  # pragma: no cover
    from .types import ExecutionFrame


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

    async def clear_cache(self) -> None:
        """Clear all cached data."""
        if self._internal_cache is not None:
            self._internal_cache.clear()
        if hasattr(self._backend, "clear"):
            if asyncio.iscoroutinefunction(self._backend.clear):
                await self._backend.clear()
            else:
                self._backend.clear()

    def generate_cache_key(
        self, step: Any, data: Any, context: Optional[Any], resources: Optional[Any]
    ) -> str:
        """Generate a cache key for the given step execution parameters."""
        if not self.is_cache_enabled():
            return ""
        return self._key_generator.generate_key(step, data, context, resources)

    def is_cache_enabled(self) -> bool:
        """Check if caching is enabled, respecting task-local overrides."""
        # Check task-local override first (used by loop iteration runner)
        override = _CACHE_OVERRIDE.get(None)
        if override is not None:
            return bool(override)
        return self._enable_cache

    async def get_cached_result(self, key: str) -> Optional[Any]:
        """Retrieve a cached result by key."""
        if not self.is_cache_enabled() or not key:
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

    async def fetch_step_result(self, key: str) -> Optional[StepResult]:
        """Retrieve a cached StepResult and mark it as a cache hit."""
        cached = await self.get_cached_result(key)
        if not isinstance(cached, StepResult):
            return None
        md = getattr(cached, "metadata_", None)
        if md is None:
            cached.metadata_ = {"cache_hit": True}
        else:
            md["cache_hit"] = True
        return cached

    async def set_cached_result(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store a result in cache."""
        if not self.is_cache_enabled() or not key:
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

    async def persist_step_result(self, key: str, result: StepResult, ttl_s: int = 3600) -> None:
        """Persist a successful StepResult to the configured cache layers."""
        await self.set_cached_result(key, result, ttl=ttl_s)

    def _should_cache_step_result(self, step: Any, result: Optional[StepResult]) -> bool:
        """Determine if a result should be cached for the given step."""
        if not self.is_cache_enabled() or result is None or not getattr(result, "success", False):
            return False
        try:
            from ...domain.dsl.loop import LoopStep

            if isinstance(step, LoopStep):
                return False
        except Exception:
            pass
        try:
            meta_obj = getattr(step, "meta", None)
            is_adapter_step = (
                bool(meta_obj.get("is_adapter")) if isinstance(meta_obj, dict) else False
            )
            if is_adapter_step:
                return False
        except Exception:
            pass
        metadata = getattr(result, "metadata_", None)
        if isinstance(metadata, dict) and metadata.get("no_cache"):
            return False
        return True

    async def maybe_persist_step_result(
        self, step: Any, result: Optional[StepResult], key: Optional[str], ttl_s: int = 3600
    ) -> None:
        """Persist a StepResult when caching is enabled and allowed for the step/result."""
        if not key or not self._should_cache_step_result(step, result):
            return
        if result is None:
            return
        await self.persist_step_result(key, result, ttl_s=ttl_s)
        try:
            telemetry.logfire.debug(f"Cached result for step: {getattr(step, 'name', '<unnamed>')}")
        except Exception:
            pass

    async def maybe_fetch_step_result(self, frame: "ExecutionFrame[Any]") -> Optional[StepResult]:
        """Return a cached StepResult for the frame when enabled (skips loops/adapters)."""
        if not self.is_cache_enabled():
            return None
        step = getattr(frame, "step", None)
        try:
            from ...domain.dsl.loop import LoopStep

            if isinstance(step, LoopStep):
                return None
        except Exception:
            pass
        try:
            meta_obj = getattr(step, "meta", None)
            is_adapter_step = (
                bool(meta_obj.get("is_adapter")) if isinstance(meta_obj, dict) else False
            )
            if is_adapter_step:
                return None
        except Exception:
            pass
        key = self.generate_cache_key(
            step,
            getattr(frame, "data", None),
            getattr(frame, "context", None),
            getattr(frame, "resources", None),
        )
        if not key:
            return None
        return await self.fetch_step_result(key)

    async def maybe_return_cached(
        self, frame: "ExecutionFrame[Any]", *, called_with_frame: bool
    ) -> Optional[StepOutcome[StepResult] | StepResult]:
        """Return cached outcome or StepResult if present."""
        cached = await self.maybe_fetch_step_result(frame)
        if cached is None:
            return None
        if not isinstance(cached, StepResult):
            return None
        if called_with_frame:
            return Success(step_result=cached)
        return cached
