"""
Algorithm optimizations for cache keys, serialization, and hashing.

This module provides optimized algorithms for cache key generation, serialization
performance, and hash computation efficiency with caching and fast paths to
reduce computational overhead.
"""

import hashlib
import time
import weakref
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from threading import RLock

from .....domain.models import BaseModel

# Try to import high-performance libraries
try:
    import orjson

    HAS_ORJSON = True
except ImportError:
    HAS_ORJSON = False

try:
    import blake3

    HAS_BLAKE3 = True
except ImportError:
    HAS_BLAKE3 = False

try:
    import xxhash

    HAS_XXHASH = True
except ImportError:
    HAS_XXHASH = False


@dataclass
class HashStats:
    """Statistics for hash operations."""

    total_hashes: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_time_ns: int = 0
    hash_collisions: int = 0

    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_requests = self.cache_hits + self.cache_misses
        return self.cache_hits / total_requests if total_requests > 0 else 0.0

    @property
    def average_time_ns(self) -> float:
        """Calculate average hash time."""
        return self.total_time_ns / self.total_hashes if self.total_hashes > 0 else 0.0


@dataclass
class SerializationStats:
    """Statistics for serialization operations."""

    total_serializations: int = 0
    total_deserializations: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_serialize_time_ns: int = 0
    total_deserialize_time_ns: int = 0
    total_bytes_serialized: int = 0

    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_requests = self.cache_hits + self.cache_misses
        return self.cache_hits / total_requests if total_requests > 0 else 0.0

    @property
    def average_serialize_time_ns(self) -> float:
        """Calculate average serialization time."""
        return (
            self.total_serialize_time_ns / self.total_serializations
            if self.total_serializations > 0
            else 0.0
        )

    @property
    def compression_ratio(self) -> float:
        """Calculate average compression ratio."""
        if self.total_serializations == 0:
            return 1.0
        return self.total_bytes_serialized / self.total_serializations


class OptimizedHasher:
    """
    High-performance hasher with multiple algorithms and caching.

    Features:
    - Multiple hash algorithms (Blake3, xxHash, SHA256)
    - Hash result caching with LRU eviction
    - Fast paths for common data types
    - Collision detection and statistics
    - Weak reference tracking for memory efficiency
    """

    def __init__(self, algorithm: str = "auto", cache_size: int = 10000, enable_stats: bool = True):
        self.algorithm = algorithm
        self.cache_size = cache_size
        self.enable_stats = enable_stats

        # Choose optimal hash algorithm
        if algorithm == "auto":
            if HAS_BLAKE3:
                self.algorithm = "blake3"
            elif HAS_XXHASH:
                self.algorithm = "xxhash"
            else:
                self.algorithm = "sha256"

        # Hash cache with LRU eviction
        self._hash_cache: OrderedDict[int, str] = OrderedDict()
        # Use bounded weak-reference-based collision tracker to avoid memory growth
        self._collision_tracker: Dict[str, List[weakref.ref[Any]]] = defaultdict(list)
        self._weak_refs: Set[weakref.ref[Any]] = set()

        # Statistics
        self._stats = HashStats() if enable_stats else None

        # Thread safety
        self._lock = RLock()

    def hash_object(self, obj: Any) -> str:
        """
        Hash an object with caching and optimization.

        Args:
            obj: Object to hash

        Returns:
            Hexadecimal hash string
        """
        start_time = time.perf_counter_ns()

        try:
            # Fast path for None
            if obj is None:
                return self._hash_bytes(b"null")

            # Fast path for basic types
            if isinstance(obj, (str, int, float, bool)):
                return self._hash_basic_type(obj)

            # Check cache
            obj_id = id(obj)
            with self._lock:
                if obj_id in self._hash_cache:
                    # LRU promotion
                    hash_value = self._hash_cache[obj_id]
                    self._hash_cache.move_to_end(obj_id)

                    if self.enable_stats and self._stats is not None:
                        self._stats.cache_hits += 1
                        self._stats.total_hashes += 1
                        self._stats.total_time_ns += time.perf_counter_ns() - start_time

                    return hash_value

                # Cache miss
                if self.enable_stats and self._stats is not None:
                    self._stats.cache_misses += 1

            # Compute hash
            hash_value = self._compute_hash(obj)

            # Cache result
            self._cache_hash(obj_id, hash_value, obj)

            # Check for collisions
            self._check_collision(hash_value, obj)

            # Update statistics
            if self.enable_stats and self._stats is not None:
                self._stats.total_hashes += 1
                self._stats.total_time_ns += time.perf_counter_ns() - start_time

            return hash_value

        except Exception:
            # Fallback to simple hash
            fallback_hash = self._hash_bytes(str(obj).encode("utf-8", errors="ignore"))

            if self.enable_stats and self._stats is not None:
                self._stats.total_hashes += 1
                self._stats.total_time_ns += time.perf_counter_ns() - start_time

            return fallback_hash

    def _hash_basic_type(self, obj: Union[str, int, float, bool]) -> str:
        """Fast path for basic types."""
        if isinstance(obj, str):
            return self._hash_bytes(obj.encode("utf-8"))
        elif isinstance(obj, bool):
            return self._hash_bytes(b"true" if obj else b"false")
        else:
            return self._hash_bytes(str(obj).encode("utf-8"))

    def _compute_hash(self, obj: Any) -> str:
        """Compute hash for complex objects."""
        # Handle BaseModel efficiently
        if isinstance(obj, BaseModel):
            try:
                json_bytes = obj.model_dump_json(sort_keys=True).encode("utf-8")
                return self._hash_bytes(json_bytes)
            except Exception:
                pass

        # Handle dictionaries
        if isinstance(obj, dict):
            return self._hash_dict(obj)

        # Handle lists and tuples
        if isinstance(obj, (list, tuple)):
            return self._hash_sequence(obj)

        # Handle sets
        if isinstance(obj, set):
            return self._hash_set(obj)

        # Handle bytes
        if isinstance(obj, bytes):
            return self._hash_bytes(obj)

        # Fallback to string representation
        return self._hash_bytes(str(obj).encode("utf-8", errors="ignore"))

    def _hash_dict(self, d: Dict[Any, Any]) -> str:
        """Hash dictionary with sorted keys."""
        try:
            # Sort keys for deterministic hashing
            sorted_items = sorted(d.items(), key=lambda x: str(x[0]))
            content = []

            for key, value in sorted_items:
                key_hash = self.hash_object(key)
                value_hash = self.hash_object(value)
                content.append(f"{key_hash}:{value_hash}")

            combined = "|".join(content)
            return self._hash_bytes(combined.encode("utf-8"))

        except Exception:
            # Fallback
            return self._hash_bytes(str(d).encode("utf-8", errors="ignore"))

    def _hash_sequence(self, seq: Union[List[Any], Tuple[Any, ...]]) -> str:
        """Hash sequence (list or tuple)."""
        try:
            content = []
            for item in seq:
                item_hash = self.hash_object(item)
                content.append(item_hash)

            combined = "|".join(content)
            return self._hash_bytes(combined.encode("utf-8"))

        except Exception:
            # Fallback
            return self._hash_bytes(str(seq).encode("utf-8", errors="ignore"))

    def _hash_set(self, s: Set[Any]) -> str:
        """Hash set with sorted elements."""
        try:
            # Sort elements for deterministic hashing
            sorted_elements = sorted(s, key=lambda x: str(x))
            content = []

            for item in sorted_elements:
                item_hash = self.hash_object(item)
                content.append(item_hash)

            combined = "|".join(content)
            return self._hash_bytes(combined.encode("utf-8"))

        except Exception:
            # Fallback
            return str(self._hash_bytes(str(s).encode("utf-8", errors="ignore")))

    def _hash_bytes(self, data: bytes) -> str:
        """Hash bytes using the selected algorithm."""
        if self.algorithm == "blake3" and HAS_BLAKE3:
            return str(blake3.blake3(data).hexdigest())
        elif self.algorithm == "xxhash" and HAS_XXHASH:
            return str(xxhash.xxh64(data).hexdigest())
        elif self.algorithm == "sha256":
            return str(hashlib.sha256(data).hexdigest())
        else:
            # Fallback to MD5 for speed (not cryptographically secure)
            return str(hashlib.md5(data).hexdigest())

    def _cache_hash(self, obj_id: int, hash_value: str, obj: Any) -> None:
        """Cache hash result with LRU eviction."""
        with self._lock:
            # Remove oldest entries if cache is full
            while len(self._hash_cache) >= self.cache_size:
                self._hash_cache.popitem(last=False)

            # Add to cache
            self._hash_cache[obj_id] = hash_value

            # Track with weak reference
            try:
                weak_ref = weakref.ref(obj, lambda ref: self._cleanup_cache(obj_id))
                self._weak_refs.add(weak_ref)
            except TypeError:
                # Some objects can't be weakly referenced
                pass

    def _check_collision(self, hash_value: str, obj: Any) -> None:
        """Check for hash collisions without retaining strong references.

        Tracks up to a small bounded number of weak references per hash to avoid memory leaks.
        """
        if not self.enable_stats:
            return

        with self._lock:
            lst = self._collision_tracker.get(hash_value)
            if lst is None:
                self._collision_tracker[hash_value] = [weakref.ref(obj)]
                return

            # Prune dead refs
            alive: List[weakref.ref[Any]] = [r for r in lst if r() is not None]
            # Collision detection: compare against a few alive objects
            for r in alive:
                existing = r()
                if existing is not None and existing is not obj and existing != obj:
                    if self._stats is not None:
                        self._stats.hash_collisions += 1
                    break
            # Append current as weak ref
            alive.append(weakref.ref(obj))
            # Bound list size to prevent unbounded growth (keep last 4)
            if len(alive) > 4:
                alive = alive[-4:]
            self._collision_tracker[hash_value] = alive

    def _cleanup_cache(self, obj_id: int) -> None:
        """Clean up cache entry when object is garbage collected."""
        with self._lock:
            self._hash_cache.pop(obj_id, None)

    def get_stats(self) -> Optional[HashStats]:
        """Get hash statistics."""
        return self._stats

    def clear_cache(self) -> None:
        """Clear hash cache."""
        with self._lock:
            self._hash_cache.clear()
            self._collision_tracker.clear()
            self._weak_refs.clear()


class OptimizedSerializer:
    """
    High-performance serializer with caching and compression.

    Features:
    - Multiple serialization formats (orjson, json)
    - Serialization result caching
    - Compression for large objects
    - Fast paths for common types
    - Statistics tracking
    """

    def __init__(
        self,
        format: str = "auto",
        cache_size: int = 1000,
        compression_threshold: int = 1024,
        enable_stats: bool = True,
    ):
        self.format = format
        self.cache_size = cache_size
        self.compression_threshold = compression_threshold
        self.enable_stats = enable_stats

        # Choose optimal serialization format
        if format == "auto":
            self.format = "orjson" if HAS_ORJSON else "json"

        # Serialization cache
        self._serialize_cache: OrderedDict[int, bytes] = OrderedDict()
        self._deserialize_cache: OrderedDict[str, Any] = OrderedDict()
        self._weak_refs: Set[weakref.ref[Any]] = set()

        # Statistics
        self._stats = SerializationStats() if enable_stats else None

        # Thread safety
        self._lock = RLock()

    def serialize(self, obj: Any) -> bytes:
        """
        Serialize object to bytes with caching.

        Args:
            obj: Object to serialize

        Returns:
            Serialized bytes
        """
        start_time = time.perf_counter_ns()

        try:
            # Fast path for None
            if obj is None:
                return b"null"

            # Fast path for basic types
            if isinstance(obj, (str, int, float, bool)):
                return self._serialize_basic_type(obj)

            # Check cache
            obj_id = id(obj)
            with self._lock:
                if obj_id in self._serialize_cache:
                    # LRU promotion
                    result = self._serialize_cache[obj_id]
                    self._serialize_cache.move_to_end(obj_id)

                    if self.enable_stats and self._stats is not None:
                        self._stats.cache_hits += 1
                        self._stats.total_serializations += 1
                        self._stats.total_serialize_time_ns += time.perf_counter_ns() - start_time

                    return result

                # Cache miss
                if self.enable_stats and self._stats is not None:
                    self._stats.cache_misses += 1

            # Serialize object
            result = self._serialize_object(obj)

            # Cache result
            self._cache_serialization(obj_id, result, obj)

            # Update statistics
            if self.enable_stats and self._stats is not None:
                self._stats.total_serializations += 1
                self._stats.total_serialize_time_ns += time.perf_counter_ns() - start_time
                self._stats.total_bytes_serialized += len(result)

            return result

        except Exception:
            # Fallback serialization
            fallback_result = str(obj).encode("utf-8", errors="ignore")

            if self.enable_stats and self._stats is not None:
                self._stats.total_serializations += 1
                self._stats.total_serialize_time_ns += time.perf_counter_ns() - start_time
                self._stats.total_bytes_serialized += len(fallback_result)

            return fallback_result

    def deserialize(self, data: bytes) -> Any:
        """
        Deserialize bytes to object with caching.

        Args:
            data: Bytes to deserialize

        Returns:
            Deserialized object
        """
        start_time = time.perf_counter_ns()

        try:
            # Create cache key
            data_hash = hashlib.md5(data).hexdigest()

            # Check cache
            with self._lock:
                if data_hash in self._deserialize_cache:
                    # LRU promotion
                    result = self._deserialize_cache[data_hash]
                    self._deserialize_cache.move_to_end(data_hash)

                    if self.enable_stats and self._stats is not None:
                        self._stats.cache_hits += 1
                        self._stats.total_deserializations += 1
                        self._stats.total_deserialize_time_ns += time.perf_counter_ns() - start_time

                    return result

                # Cache miss
                if self.enable_stats and self._stats is not None:
                    self._stats.cache_misses += 1

            # Deserialize data
            result = self._deserialize_data(data)

            # Cache result
            with self._lock:
                # Remove oldest entries if cache is full
                while len(self._deserialize_cache) >= self.cache_size:
                    self._deserialize_cache.popitem(last=False)

                self._deserialize_cache[data_hash] = result

            # Update statistics
            if self.enable_stats and self._stats is not None:
                self._stats.total_deserializations += 1

            return result

        except Exception:
            # Fallback deserialization
            try:
                result = data.decode("utf-8", errors="ignore")

                if self.enable_stats and self._stats is not None:
                    self._stats.total_deserializations += 1

                return result
            except Exception:
                return str(data)

    def _serialize_basic_type(self, obj: Union[str, int, float, bool]) -> bytes:
        """Fast path for basic types."""
        if self.format == "orjson" and HAS_ORJSON:
            return orjson.dumps(obj)
        else:
            import json

            return json.dumps(obj, separators=(",", ":")).encode("utf-8")

    def _serialize_object(self, obj: Any) -> bytes:
        """Serialize complex objects using unified serialization."""
        from flujo.utils.serialization import safe_serialize

        try:
            # Use unified serialization logic first
            serialized_obj = safe_serialize(obj, mode="default")

            # Then encode to bytes using the appropriate library
            if self.format == "orjson" and HAS_ORJSON:
                return orjson.dumps(serialized_obj, option=orjson.OPT_SORT_KEYS)
            else:
                import json

                return json.dumps(serialized_obj, sort_keys=True, separators=(",", ":")).encode(
                    "utf-8"
                )
        except Exception:
            # Fallback to string representation
            return str(obj).encode("utf-8", errors="ignore")

    def _deserialize_data(self, data: bytes) -> Any:
        """Deserialize data using the selected format and unified deserialization."""
        from flujo.utils.serialization import safe_deserialize

        # First decode from bytes
        if self.format == "orjson" and HAS_ORJSON:
            raw_data = orjson.loads(data)
        else:
            import json

            raw_data = json.loads(data.decode("utf-8"))

        # Then apply unified deserialization logic if needed
        return safe_deserialize(raw_data)

    def _cache_serialization(self, obj_id: int, result: bytes, obj: Any) -> None:
        """Cache serialization result."""
        with self._lock:
            # Remove oldest entries if cache is full
            while len(self._serialize_cache) >= self.cache_size:
                self._serialize_cache.popitem(last=False)

            # Add to cache
            self._serialize_cache[obj_id] = result

            # Track with weak reference
            try:
                weak_ref = weakref.ref(obj, lambda ref: self._cleanup_serialize_cache(obj_id))
                self._weak_refs.add(weak_ref)
            except TypeError:
                # Some objects can't be weakly referenced
                pass

    def _cleanup_serialize_cache(self, obj_id: int) -> None:
        """Clean up serialization cache entry."""
        with self._lock:
            self._serialize_cache.pop(obj_id, None)

    def get_stats(self) -> Optional[SerializationStats]:
        """Get serialization statistics."""
        return self._stats

    def clear_cache(self) -> None:
        """Clear serialization caches."""
        with self._lock:
            self._serialize_cache.clear()
            self._deserialize_cache.clear()
            self._weak_refs.clear()


class OptimizedCacheKeyGenerator:
    """
    Optimized cache key generator with intelligent hashing.

    Features:
    - Hierarchical key generation
    - Component-based caching
    - Fast paths for common patterns
    - Collision avoidance
    - Statistics tracking
    """

    def __init__(
        self,
        hasher: Optional[OptimizedHasher] = None,
        serializer: Optional[OptimizedSerializer] = None,
        enable_component_caching: bool = True,
    ):
        self.hasher = hasher or OptimizedHasher()
        self.serializer = serializer or OptimizedSerializer()
        self.enable_component_caching = enable_component_caching

        # Component cache for partial key reuse
        self._component_cache: Dict[str, str] = {}
        self._lock = RLock()

    def generate_cache_key(
        self,
        step: Any,
        data: Any,
        context: Optional[Any] = None,
        resources: Optional[Any] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate optimized cache key.

        Args:
            step: Step object
            data: Input data
            context: Execution context
            resources: Available resources
            **kwargs: Additional parameters

        Returns:
            Cache key string
        """
        # Build key components
        components = []

        # Step component
        step_key = self._get_step_key(step)
        components.append(f"step:{step_key}")

        # Data component
        data_key = self._get_data_key(data)
        components.append(f"data:{data_key}")

        # Context component (if present)
        if context is not None:
            context_key = self._get_context_key(context)
            components.append(f"context:{context_key}")

        # Resources component (if present)
        if resources is not None:
            resources_key = self._get_resources_key(resources)
            components.append(f"resources:{resources_key}")

        # Additional parameters
        if kwargs:
            kwargs_key = self._get_kwargs_key(kwargs)
            components.append(f"kwargs:{kwargs_key}")

        # Combine components
        combined_key = "|".join(components)

        # Generate final hash
        return self.hasher.hash_object(combined_key)

    def _get_step_key(self, step: Any) -> str:
        """Get cache key component for step."""
        cache_key = "step_key"

        if self.enable_component_caching:
            step_id = id(step)
            cache_key = f"step_{step_id}"

            with self._lock:
                if cache_key in self._component_cache:
                    return self._component_cache[cache_key]

        # Generate step key
        step_components = []

        # Step name and type
        step_name = getattr(step, "name", "unknown")
        step_type = type(step).__name__
        step_components.append(f"{step_type}:{step_name}")

        # Agent hash (if present)
        if hasattr(step, "agent") and step.agent:
            agent_key = self._get_agent_key(step.agent)
            step_components.append(f"agent:{agent_key}")

        # Configuration hash (if present)
        if hasattr(step, "config") and step.config:
            config_key = self.hasher.hash_object(step.config)
            step_components.append(f"config:{config_key}")

        step_key = "|".join(step_components)
        step_hash = self.hasher.hash_object(step_key)

        # Cache result
        if self.enable_component_caching:
            with self._lock:
                self._component_cache[cache_key] = step_hash

        return step_hash

    def _get_agent_key(self, agent: Any) -> str:
        """Get cache key component for agent."""
        # Use agent type and configuration for stable identification
        agent_type = f"{type(agent).__module__}.{type(agent).__name__}"

        # Include agent configuration if available
        config_parts = [agent_type]

        if hasattr(agent, "config"):
            config_hash = self.hasher.hash_object(agent.config)
            config_parts.append(config_hash)

        return self.hasher.hash_object("|".join(config_parts))

    def _get_data_key(self, data: Any) -> str:
        """Get cache key component for data."""
        return self.hasher.hash_object(data)

    def _get_context_key(self, context: Any) -> str:
        """Get cache key component for context."""
        cache_key = "context_key"

        if self.enable_component_caching:
            context_id = id(context)
            cache_key = f"context_{context_id}"

            with self._lock:
                if cache_key in self._component_cache:
                    return self._component_cache[cache_key]

        # Generate context key
        context_hash = self.hasher.hash_object(context)

        # Cache result
        if self.enable_component_caching:
            with self._lock:
                self._component_cache[cache_key] = context_hash

        return context_hash

    def _get_resources_key(self, resources: Any) -> str:
        """Get cache key component for resources."""
        return self.hasher.hash_object(resources)

    def _get_kwargs_key(self, kwargs: Dict[str, Any]) -> str:
        """Get cache key component for additional parameters."""
        # Sort kwargs for deterministic key generation
        sorted_kwargs = sorted(kwargs.items(), key=lambda x: str(x[0]))
        return self.hasher.hash_object(sorted_kwargs)

    def clear_component_cache(self) -> None:
        """Clear component cache."""
        with self._lock:
            self._component_cache.clear()

    def get_component_cache_stats(self) -> Dict[str, Any]:
        """Get component cache statistics."""
        with self._lock:
            return {
                "cache_size": len(self._component_cache),
                "cache_enabled": self.enable_component_caching,
            }


class AlgorithmOptimizations:
    """
    Main algorithm optimizations coordinator.

    Provides a unified interface for all algorithm optimizations including
    hashing, serialization, and cache key generation.
    """

    def __init__(
        self,
        hasher: Optional[OptimizedHasher] = None,
        serializer: Optional[OptimizedSerializer] = None,
        cache_key_generator: Optional[OptimizedCacheKeyGenerator] = None,
    ):
        self.hasher = hasher or OptimizedHasher()
        self.serializer = serializer or OptimizedSerializer()
        self.cache_key_generator = cache_key_generator or OptimizedCacheKeyGenerator(
            hasher=self.hasher, serializer=self.serializer
        )

        # Weak references for cleanup
        self._weak_refs: Set[weakref.ref[Any]] = set()
        self._last_cleanup = time.time()

    def hash_object(self, obj: Any) -> str:
        """Hash an object with optimizations."""
        return self.hasher.hash_object(obj)

    def serialize(self, obj: Any) -> bytes:
        """Serialize an object with optimizations."""
        return self.serializer.serialize(obj)

    def deserialize(self, data: bytes) -> Any:
        """Deserialize data with optimizations."""
        return self.serializer.deserialize(data)

    def generate_cache_key(
        self,
        step: Any,
        data: Any,
        context: Optional[Any] = None,
        resources: Optional[Any] = None,
        **kwargs: Any,
    ) -> str:
        """Generate optimized cache key."""
        return self.cache_key_generator.generate_cache_key(step, data, context, resources, **kwargs)

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive algorithm optimization statistics."""
        return {
            "hasher": self.hasher.get_stats(),
            "serializer": self.serializer.get_stats(),
            "cache_key_generator": self.cache_key_generator.get_component_cache_stats(),
        }

    def clear_all_caches(self) -> None:
        """Clear all algorithm optimization caches."""
        self.hasher.clear_cache()
        self.serializer.clear_cache()
        self.cache_key_generator.clear_component_cache()


# Global instances
_global_hasher: Optional[OptimizedHasher] = None
_global_serializer: Optional[OptimizedSerializer] = None
_global_cache_key_generator: Optional[OptimizedCacheKeyGenerator] = None
_global_algorithm_optimizations: Optional[AlgorithmOptimizations] = None


def get_global_hasher() -> OptimizedHasher:
    """Get the global optimized hasher instance."""
    global _global_hasher
    if _global_hasher is None:
        _global_hasher = OptimizedHasher()
    return _global_hasher


def get_global_serializer() -> OptimizedSerializer:
    """Get the global optimized serializer instance."""
    global _global_serializer
    if _global_serializer is None:
        _global_serializer = OptimizedSerializer()
    return _global_serializer


def get_global_cache_key_generator() -> OptimizedCacheKeyGenerator:
    """Get the global optimized cache key generator instance."""
    global _global_cache_key_generator
    if _global_cache_key_generator is None:
        _global_cache_key_generator = OptimizedCacheKeyGenerator()
    return _global_cache_key_generator


def get_global_algorithm_optimizations() -> AlgorithmOptimizations:
    """Get the global algorithm optimizations instance."""
    global _global_algorithm_optimizations
    if _global_algorithm_optimizations is None:
        _global_algorithm_optimizations = AlgorithmOptimizations()
    return _global_algorithm_optimizations


# Convenience functions
def hash_object_optimized(obj: Any) -> str:
    """Convenience function to hash object with optimizations."""
    hasher = get_global_hasher()
    return hasher.hash_object(obj)


def serialize_optimized(obj: Any) -> bytes:
    """Convenience function to serialize object with optimizations."""
    serializer = get_global_serializer()
    return serializer.serialize(obj)


def deserialize_optimized(data: bytes) -> Any:
    """Convenience function to deserialize data with optimizations."""
    serializer = get_global_serializer()
    return serializer.deserialize(data)


def generate_cache_key_optimized(
    step: Any,
    data: Any,
    context: Optional[Any] = None,
    resources: Optional[Any] = None,
    **kwargs: Any,
) -> str:
    """Convenience function to generate optimized cache key."""
    generator = get_global_cache_key_generator()
    return generator.generate_cache_key(step, data, context, resources, **kwargs)
