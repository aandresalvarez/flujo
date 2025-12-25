"""Cache management for step execution results."""

from __future__ import annotations
import asyncio
from typing import TYPE_CHECKING, Optional
from ....domain.models import StepOutcome, StepResult, Success
from ....infra import telemetry
from .default_cache_components import DefaultCacheKeyGenerator, _LRUCache

# Import the cache override context variable from shared module
from ..context.context_vars import _CACHE_OVERRIDE

if TYPE_CHECKING:  # pragma: no cover
    from ..types import ExecutionFrame, TContext


class CacheManager:
    """Manages caching of step execution results."""

    def __init__(
        self,
        backend: object | None = None,
        key_generator: object | None = None,
        enable_cache: bool = True,
    ) -> None:
        self._backend = backend
        self._key_generator = key_generator or DefaultCacheKeyGenerator()
        self._enable_cache = enable_cache
        self._internal_cache: Optional[_LRUCache] = None

    @property
    def backend(self) -> object | None:
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
        clear_fn = getattr(self._backend, "clear", None)
        if callable(clear_fn):
            if asyncio.iscoroutinefunction(clear_fn):
                await clear_fn()
            else:
                clear_fn()

    def generate_cache_key(
        self,
        step: object,
        data: object,
        context: object | None,
        resources: object | None,
    ) -> str:
        """Generate a cache key for the given step execution parameters."""
        if not self.is_cache_enabled():
            return ""
        gen_fn = getattr(self._key_generator, "generate_key", None)
        if not callable(gen_fn):
            raise TypeError("Cache key generator must provide generate_key()")
        return str(gen_fn(step, data, context, resources))

    def is_cache_enabled(self) -> bool:
        """Check if caching is enabled, respecting task-local overrides."""
        # Check task-local override first (used by loop iteration runner)
        override = _CACHE_OVERRIDE.get(None)
        if override is not None:
            return bool(override)
        return self._enable_cache

    async def get_cached_result(self, key: str) -> object | None:
        """Retrieve a cached result by key."""
        if not self.is_cache_enabled() or not key:
            return None

        # Try backend first, then internal cache
        backend_get = getattr(self._backend, "get", None)
        if callable(backend_get):
            try:
                result = await backend_get(key)
                if result is not None:
                    result_obj: object = result
                    return result_obj
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

    async def set_cached_result(self, key: str, value: object, ttl: Optional[int] = None) -> None:
        """Store a result in cache."""
        if not self.is_cache_enabled() or not key:
            return

        # Store in backend first, then internal cache
        backend_set = getattr(self._backend, "set", None)
        if callable(backend_set):
            try:
                await backend_set(key, value, ttl=ttl)
            except Exception:
                pass
        else:
            backend_put = getattr(self._backend, "put", None)
            if callable(backend_put):
                try:
                    ttl_s = ttl if ttl is not None else getattr(self._backend, "ttl_s", 0)
                    await backend_put(key, value, ttl_s=ttl_s)
                except Exception:
                    pass

        if self._internal_cache is not None and isinstance(value, StepResult):
            # _LRUCache manages TTL internally; no ttl kwarg is accepted
            self._internal_cache.set(key, value)

    async def persist_step_result(self, key: str, result: StepResult, ttl_s: int = 3600) -> None:
        """Persist a successful StepResult to the configured cache layers."""
        await self.set_cached_result(key, result, ttl=ttl_s)

    def _should_cache_step_result(self, step: object, result: Optional[StepResult]) -> bool:
        """Determine if a result should be cached for the given step."""
        if not self.is_cache_enabled() or result is None or not getattr(result, "success", False):
            return False
        try:
            from ....domain.dsl.loop import LoopStep

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
        self,
        step: object,
        result: Optional[StepResult],
        key: Optional[str],
        ttl_s: int = 3600,
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

    async def maybe_fetch_step_result(
        self, frame: "ExecutionFrame[TContext]"
    ) -> Optional[StepResult]:
        """Return a cached StepResult for the frame when enabled (skips loops/adapters)."""
        if not self.is_cache_enabled():
            return None
        step = getattr(frame, "step", None)
        try:
            from ....domain.dsl.loop import LoopStep

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
        self, frame: "ExecutionFrame[TContext]", *, called_with_frame: bool
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
