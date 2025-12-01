"""
Optimized context management with copy-on-write and caching.

This module provides optimized context handling with copy-on-write optimization,
caching, immutability detection, and efficient merge algorithms to reduce
context handling overhead.
"""

from flujo.type_definitions.common import JSONObject
import copy
import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional, Set, Tuple, Callable, TypeVar, List
from weakref import WeakKeyDictionary
from threading import RLock

from .....domain.models import BaseModel
from .....utils.context import safe_merge_context_updates


T = TypeVar("T")


@dataclass
class ContextStats:
    """Statistics for context operations."""

    copy_operations: int = 0
    merge_operations: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    immutable_detections: int = 0
    cow_optimizations: int = 0

    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0


@dataclass
class CachedContext:
    """Cached context with metadata."""

    context: Any
    hash_value: str
    timestamp: float
    access_count: int = 0
    is_immutable: bool = False

    def touch(self) -> None:
        """Update access information."""
        self.access_count += 1
        self.timestamp = time.time()


class OptimizedContextManager:
    """
    Context management with copy-on-write optimization and caching.

    Features:
    - Copy-on-write optimization for immutable contexts
    - LRU caching for frequently accessed contexts
    - Immutability detection to avoid unnecessary copying
    - Efficient merge algorithms with conflict resolution
    - Weak references to prevent memory leaks
    - Statistical tracking for performance monitoring
    """

    def __init__(
        self,
        cache_size: int = 512,
        merge_cache_size: int = 256,
        ttl_seconds: int = 1800,
        enable_cow: bool = True,
        enable_stats: bool = True,
    ):
        self.cache_size = cache_size
        self.merge_cache_size = merge_cache_size
        self.ttl_seconds = ttl_seconds
        self.enable_cow = enable_cow
        self.enable_stats = enable_stats

        # Context caches
        self._context_cache: OrderedDict[int, CachedContext] = OrderedDict()
        self._merge_cache: OrderedDict[Tuple[str, str], Any] = OrderedDict()
        self._immutable_cache: WeakKeyDictionary[Any, bool] = WeakKeyDictionary()

        # Copy-on-write tracking
        self._cow_contexts: WeakKeyDictionary[Any, Set[Any]] = WeakKeyDictionary()
        self._original_contexts: WeakKeyDictionary[Any, Any] = WeakKeyDictionary()

        # Thread safety
        self._lock = RLock()

        # Statistics
        self._stats = ContextStats() if enable_stats else None

    def optimized_copy(self, context: Any) -> Any:
        """
        Optimized context copying with caching and COW.

        Args:
            context: Context to copy

        Returns:
            Copied context (may be shared if immutable)
        """
        if context is None:
            return None

        with self._lock:
            self._update_stats("copy_operation")

            # Check if context is immutable
            if self._is_immutable(context):
                self._update_stats("immutable_detection")
                return context  # No need to copy immutable objects

            # Generate cache key
            context_id = id(context)

            # Check cache
            if context_id in self._context_cache:
                cached = self._context_cache[context_id]

                # Check TTL
                if time.time() - cached.timestamp < self.ttl_seconds:
                    cached.touch()
                    self._context_cache.move_to_end(context_id)
                    self._update_stats("cache_hit")

                    # COW optimization: return shared reference if possible
                    if self.enable_cow and cached.is_immutable:
                        self._update_stats("cow_optimization")
                        return cached.context

                    return self._deep_copy_optimized(cached.context)
                else:
                    # Expired, remove from cache
                    del self._context_cache[context_id]

            self._update_stats("cache_miss")

            # Create copy
            copied_context = self._deep_copy_optimized(context)

            # Cache the result
            self._cache_context(context_id, copied_context, context)

            return copied_context

    def optimized_merge(self, target: Any, source: Any) -> bool:
        """
        Optimized context merging with caching and conflict resolution.

        Args:
            target: Target context to merge into
            source: Source context to merge from

        Returns:
            True if merge was successful, False otherwise
        """
        if target is None or source is None:
            return False

        with self._lock:
            self._update_stats("merge_operation")

            # Generate cache key for merge operation
            target_hash = self._hash_context(target)
            source_hash = self._hash_context(source)
            merge_key = (target_hash, source_hash)

            # Check merge cache
            if merge_key in self._merge_cache:
                cached_result = self._merge_cache[merge_key]
                self._merge_cache.move_to_end(merge_key)
                self._update_stats("cache_hit")

                # Apply cached merge result
                return self._apply_cached_merge(target, cached_result)

            self._update_stats("cache_miss")

            # Perform merge
            try:
                # Use the existing safe_merge_context_updates function
                result = safe_merge_context_updates(target, source)

                # Cache the merge operation
                self._cache_merge_result(merge_key, target, source)

                return result
            except Exception:
                return False

    def is_immutable(self, context: Any) -> bool:
        """
        Check if context can be safely shared (immutable).

        Args:
            context: Context to check

        Returns:
            True if context is immutable, False otherwise
        """
        return self._is_immutable(context)

    def clear_cache(self) -> None:
        """Clear all caches."""
        with self._lock:
            self._context_cache.clear()
            self._merge_cache.clear()
            self._immutable_cache.clear()
            self._cow_contexts.clear()
            self._original_contexts.clear()

    def get_stats(self) -> Optional[ContextStats]:
        """Get context management statistics."""
        return self._stats

    def _is_immutable(self, context: Any) -> bool:
        """Check if context is immutable."""
        # Check cache first
        if context in self._immutable_cache:
            return self._immutable_cache[context]

        # Determine immutability
        immutable = self._determine_immutability(context)

        # Cache result
        try:
            self._immutable_cache[context] = immutable
        except (TypeError, KeyError):
            # Can't cache unhashable types
            pass

        return immutable

    def _determine_immutability(self, context: Any) -> bool:
        """Determine if a context is immutable."""
        # Basic immutable types
        if isinstance(context, (str, int, float, bool, bytes, tuple, frozenset, type(None))):
            return True

        # Mutable types
        if isinstance(context, (list, dict, set, bytearray)):
            return False

        # Check for __slots__ (often indicates immutability)
        if hasattr(context, "__slots__"):
            # Check if all slots are immutable
            try:
                for slot in context.__slots__:
                    if hasattr(context, slot):
                        value = getattr(context, slot)
                        if not self._is_immutable(value):
                            return False
                return True
            except (AttributeError, TypeError):
                pass

        # Check for BaseModel (Pydantic models can be immutable)
        if isinstance(context, BaseModel):
            # Check if model is configured as immutable
            try:
                config = getattr(context, "model_config", None)
                if config and getattr(config, "frozen", False):
                    return True
            except (AttributeError, TypeError):
                pass

        # Check for dataclass with frozen=True
        if hasattr(context, "__dataclass_fields__"):
            try:
                if getattr(context, "__dataclass_params__").frozen:
                    return True
            except (AttributeError, TypeError):
                pass

        # Check for custom immutable indicators
        if hasattr(context, "_immutable") and context._immutable:
            return True

        # Default to mutable for safety
        return False

    def _deep_copy_optimized(self, context: Any) -> Any:
        """Optimized deep copy with special handling for common types."""
        if context is None:
            return None

        # Fast path for immutable types
        if self._is_immutable(context):
            return context

        # Handle BaseModel efficiently
        if isinstance(context, BaseModel):
            try:
                return context.model_copy(deep=True)
            except Exception:
                pass

        # Handle dictionaries efficiently
        if isinstance(context, dict):
            return {k: self._deep_copy_optimized(v) for k, v in context.items()}

        # Handle lists efficiently
        if isinstance(context, list):
            return [self._deep_copy_optimized(item) for item in context]

        # Handle sets efficiently
        if isinstance(context, set):
            return {self._deep_copy_optimized(item) for item in context}

        # Handle tuples (create new tuple with copied elements)
        if isinstance(context, tuple):
            return tuple(self._deep_copy_optimized(item) for item in context)

        # Fallback to standard deep copy
        try:
            return copy.deepcopy(context)
        except Exception:
            # If deep copy fails, try shallow copy
            try:
                return copy.copy(context)
            except Exception:
                # Last resort: return original (risky but better than crashing)
                return context

    def _hash_context(self, context: Any) -> str:
        """Generate hash for context."""
        try:
            if isinstance(context, BaseModel):
                # Use model's JSON representation for hashing
                json_str = context.model_dump_json(sort_keys=True)
                return hashlib.md5(json_str.encode()).hexdigest()
            elif isinstance(context, dict):
                # Sort keys for consistent hashing
                items = sorted(context.items()) if context else []
                content = str(items)
                return hashlib.md5(content.encode()).hexdigest()
            elif isinstance(context, (list, tuple, set)):
                # Convert to string representation
                content = str(sorted(context) if isinstance(context, set) else context)
                return hashlib.md5(content.encode()).hexdigest()
            else:
                # Use string representation
                content = str(context)
                return hashlib.md5(content.encode()).hexdigest()
        except Exception:
            # Fallback to object id
            return str(id(context))

    def _cache_context(self, context_id: int, copied_context: Any, original_context: Any) -> None:
        """Cache a copied context."""
        # Remove oldest entries if cache is full
        while len(self._context_cache) >= self.cache_size:
            self._context_cache.popitem(last=False)

        # Create cached context
        cached = CachedContext(
            context=copied_context,
            hash_value=self._hash_context(copied_context),
            timestamp=time.time(),
            is_immutable=self._is_immutable(copied_context),
        )

        # Add to cache
        self._context_cache[context_id] = cached

        # Track COW relationship if enabled
        if self.enable_cow:
            try:
                self._original_contexts[copied_context] = original_context
                if original_context not in self._cow_contexts:
                    self._cow_contexts[original_context] = set()
                self._cow_contexts[original_context].add(copied_context)
            except (TypeError, KeyError):
                # Can't track unhashable types
                pass

    def _cache_merge_result(self, merge_key: Tuple[str, str], target: Any, source: Any) -> None:
        """Cache a merge operation result."""
        # Remove oldest entries if cache is full
        while len(self._merge_cache) >= self.merge_cache_size:
            self._merge_cache.popitem(last=False)

        # Create a snapshot of the merge result
        try:
            merge_result = self._create_merge_snapshot(target, source)
            self._merge_cache[merge_key] = merge_result
        except Exception:
            # If we can't cache the result, that's okay
            pass

    def _create_merge_snapshot(self, target: Any, source: Any) -> JSONObject:
        """Create a snapshot of merge operation for caching."""
        snapshot = {
            "target_hash": self._hash_context(target),
            "source_hash": self._hash_context(source),
            "timestamp": time.time(),
        }

        # Store field-level changes if possible
        if isinstance(target, BaseModel) and isinstance(source, BaseModel):
            try:
                target_dict = target.model_dump()
                source_dict = source.model_dump()

                changes = {}
                for key, value in source_dict.items():
                    if key not in target_dict or target_dict[key] != value:
                        changes[key] = value

                snapshot["changes"] = changes
            except Exception:
                pass
        elif isinstance(target, dict) and isinstance(source, dict):
            changes = {}
            for key, value in source.items():
                if key not in target or target[key] != value:
                    changes[key] = value
            snapshot["changes"] = changes

        return snapshot

    def _apply_cached_merge(self, target: Any, cached_result: JSONObject) -> bool:
        """Apply a cached merge result."""
        try:
            # Check if cached result is still valid
            current_hash = self._hash_context(target)
            if current_hash != cached_result.get("target_hash"):
                return False

            # Apply changes if available
            changes = cached_result.get("changes", {})
            if changes:
                if isinstance(target, BaseModel):
                    # Update BaseModel fields
                    for key, value in changes.items():
                        if hasattr(target, key):
                            setattr(target, key, value)
                elif isinstance(target, dict):
                    # Update dictionary
                    target.update(changes)
                else:
                    # Can't apply cached changes to this type
                    return False

            return True
        except Exception:
            return False

    def _update_stats(self, stat_type: str) -> None:
        """Update statistics."""
        if not self.enable_stats or self._stats is None:
            return

        if stat_type == "copy_operation":
            self._stats.copy_operations += 1
        elif stat_type == "merge_operation":
            self._stats.merge_operations += 1
        elif stat_type == "cache_hit":
            self._stats.cache_hits += 1
        elif stat_type == "cache_miss":
            self._stats.cache_misses += 1
        elif stat_type == "immutable_detection":
            self._stats.immutable_detections += 1
        elif stat_type == "cow_optimization":
            self._stats.cow_optimizations += 1


# Global context manager instance
_global_context_manager: Optional[OptimizedContextManager] = None


def get_global_context_manager() -> OptimizedContextManager:
    """Get the global context manager instance."""
    global _global_context_manager
    if _global_context_manager is None:
        _global_context_manager = OptimizedContextManager()
    return _global_context_manager


def optimized_copy_context(context: Any) -> Any:
    """Convenience function to copy context using global manager."""
    manager = get_global_context_manager()
    return manager.optimized_copy(context)


def optimized_merge_context(target: Any, source: Any) -> bool:
    """Convenience function to merge contexts using global manager."""
    manager = get_global_context_manager()
    return manager.optimized_merge(target, source)


def is_context_immutable(context: Any) -> bool:
    """Convenience function to check context immutability."""
    manager = get_global_context_manager()
    return manager.is_immutable(context)


# Context manager for automatic context handling
class ManagedContext:
    """Context manager for automatic context copying and cleanup."""

    def __init__(
        self,
        context: Any,
        manager: Optional[OptimizedContextManager] = None,
        auto_merge_back: bool = False,
    ):
        self.original_context = context
        self.manager = manager or get_global_context_manager()
        self.auto_merge_back = auto_merge_back
        self.copied_context: Optional[Any] = None

    def __enter__(self) -> Any:
        """Copy context on entry."""
        self.copied_context = self.manager.optimized_copy(self.original_context)
        return self.copied_context

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Optionally merge changes back on exit."""
        if self.auto_merge_back and self.copied_context is not None:
            self.manager.optimized_merge(self.original_context, self.copied_context)
        self.copied_context = None


class ContextPool:
    """Pool of pre-allocated contexts for high-frequency operations."""

    def __init__(self, context_factory: Callable[..., Any], pool_size: int = 100):
        self.context_factory = context_factory
        self.pool_size = pool_size
        self._pool: List[Any] = []
        self._lock = RLock()

        # Pre-allocate contexts
        self._fill_pool()

    def get_context(self) -> Any:
        """Get a context from the pool."""
        with self._lock:
            if self._pool:
                return self._pool.pop()
            else:
                # Pool is empty, create new context
                return self.context_factory()

    def return_context(self, context: Any) -> None:
        """Return a context to the pool."""
        with self._lock:
            if len(self._pool) < self.pool_size:
                # Reset context state if possible
                if hasattr(context, "reset"):
                    try:
                        context.reset()
                    except Exception:
                        # If reset fails, don't return to pool
                        return
                elif hasattr(context, "clear"):
                    try:
                        context.clear()
                    except Exception:
                        return

                self._pool.append(context)

    def _fill_pool(self) -> None:
        """Fill the pool with pre-allocated contexts."""
        for _ in range(self.pool_size):
            try:
                context = self.context_factory()
                self._pool.append(context)
            except Exception:
                # If context creation fails, stop filling
                break
