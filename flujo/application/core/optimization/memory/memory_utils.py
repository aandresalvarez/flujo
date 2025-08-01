"""
Memory allocation optimization utilities.

This module provides pre-allocation strategies, string operation optimizations,
temporary object reduction, memory pressure detection, and automatic cleanup
mechanisms to reduce memory usage and improve performance.
"""

import gc
import os
import psutil
import sys
import threading
import time
import weakref
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Callable, TypeVar, Generic
from threading import RLock, Event
from concurrent.futures import ThreadPoolExecutor

T = TypeVar('T')


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    
    current_rss_mb: float = 0.0
    peak_rss_mb: float = 0.0
    allocated_objects: int = 0
    deallocated_objects: int = 0
    gc_collections: int = 0
    pressure_events: int = 0
    cleanup_operations: int = 0
    
    @property
    def net_objects(self) -> int:
        """Net number of allocated objects."""
        return self.allocated_objects - self.deallocated_objects


@dataclass
class MemoryPressureConfig:
    """Configuration for memory pressure detection."""
    
    warning_threshold_mb: float = 1024.0  # 1GB
    critical_threshold_mb: float = 2048.0  # 2GB
    check_interval_seconds: float = 30.0
    gc_threshold_ratio: float = 0.8  # Trigger GC at 80% of threshold
    cleanup_threshold_ratio: float = 0.9  # Trigger cleanup at 90% of threshold


class PreAllocationPool(Generic[T]):
    """
    Pre-allocation pool for frequently used objects.
    
    This pool pre-allocates objects to reduce allocation overhead
    and provides fast access to commonly used object types.
    """
    
    def __init__(
        self,
        factory: Callable[[], T],
        initial_size: int = 100,
        max_size: int = 1000,
        growth_factor: float = 1.5
    ):
        self.factory = factory
        self.initial_size = initial_size
        self.max_size = max_size
        self.growth_factor = growth_factor
        
        self._pool: deque[T] = deque()
        self._lock = RLock()
        self._allocated_count = 0
        self._returned_count = 0
        
        # Pre-allocate initial objects
        self._fill_pool(initial_size)
    
    def get(self) -> T:
        """Get an object from the pool."""
        with self._lock:
            if self._pool:
                obj = self._pool.popleft()
                self._allocated_count += 1
                return obj
            else:
                # Pool is empty, create new object
                obj = self.factory()
                self._allocated_count += 1
                return obj
    
    def put(self, obj: T) -> None:
        """Return an object to the pool."""
        with self._lock:
            if len(self._pool) < self.max_size:
                # Reset object state if possible
                if hasattr(obj, 'reset'):
                    try:
                        obj.reset()
                    except Exception:
                        # If reset fails, don't return to pool
                        return
                elif hasattr(obj, 'clear'):
                    try:
                        obj.clear()
                    except Exception:
                        return
                
                self._pool.append(obj)
                self._returned_count += 1
    
    def grow(self, additional_size: Optional[int] = None) -> None:
        """Grow the pool by adding more pre-allocated objects."""
        if additional_size is None:
            additional_size = int(len(self._pool) * (self.growth_factor - 1))
        
        additional_size = min(additional_size, self.max_size - len(self._pool))
        if additional_size > 0:
            self._fill_pool(additional_size)
    
    def shrink(self, target_size: Optional[int] = None) -> None:
        """Shrink the pool to reduce memory usage."""
        with self._lock:
            if target_size is None:
                target_size = max(self.initial_size, len(self._pool) // 2)
            
            while len(self._pool) > target_size:
                self._pool.pop()
    
    def get_stats(self) -> Dict[str, int]:
        """Get pool statistics."""
        with self._lock:
            return {
                'pool_size': len(self._pool),
                'allocated_count': self._allocated_count,
                'returned_count': self._returned_count,
                'hit_rate': self._returned_count / max(self._allocated_count, 1)
            }
    
    def _fill_pool(self, size: int) -> None:
        """Fill the pool with pre-allocated objects."""
        for _ in range(size):
            try:
                obj = self.factory()
                self._pool.append(obj)
            except Exception:
                # If object creation fails, stop filling
                break


class StringOptimizer:
    """
    String operation optimizations.
    
    Provides optimized string operations including interning,
    concatenation optimization, and format string caching.
    """
    
    def __init__(self, intern_threshold: int = 100, cache_size: int = 1000):
        self.intern_threshold = intern_threshold
        self.cache_size = cache_size
        
        self._intern_cache: Dict[str, str] = {}
        self._format_cache: Dict[str, str] = {}
        self._concat_cache: Dict[tuple, str] = {}
        self._lock = RLock()
    
    def optimized_intern(self, s: str) -> str:
        """Intern strings that are likely to be reused."""
        if len(s) > self.intern_threshold:
            return s
        
        with self._lock:
            if s in self._intern_cache:
                return self._intern_cache[s]
            
            # Limit cache size
            if len(self._intern_cache) >= self.cache_size:
                # Remove oldest entries (simple FIFO)
                oldest_keys = list(self._intern_cache.keys())[:self.cache_size // 4]
                for key in oldest_keys:
                    del self._intern_cache[key]
            
            interned = sys.intern(s) if hasattr(sys, 'intern') else s
            self._intern_cache[s] = interned
            return interned
    
    def optimized_format(self, template: str, *args, **kwargs) -> str:
        """Optimized string formatting with caching."""
        # Create cache key
        cache_key = (template, args, tuple(sorted(kwargs.items())))
        
        with self._lock:
            if cache_key in self._format_cache:
                return self._format_cache[cache_key]
            
            # Limit cache size
            if len(self._format_cache) >= self.cache_size:
                oldest_keys = list(self._format_cache.keys())[:self.cache_size // 4]
                for key in oldest_keys:
                    del self._format_cache[key]
            
            result = template.format(*args, **kwargs)
            self._format_cache[cache_key] = result
            return result
    
    def optimized_concat(self, *strings: str) -> str:
        """Optimized string concatenation."""
        if len(strings) <= 1:
            return strings[0] if strings else ""
        
        cache_key = tuple(strings)
        
        with self._lock:
            if cache_key in self._concat_cache:
                return self._concat_cache[cache_key]
            
            # Limit cache size
            if len(self._concat_cache) >= self.cache_size:
                oldest_keys = list(self._concat_cache.keys())[:self.cache_size // 4]
                for key in oldest_keys:
                    del self._concat_cache[key]
            
            # Use join for efficiency
            result = ''.join(strings)
            self._concat_cache[cache_key] = result
            return result
    
    def clear_caches(self) -> None:
        """Clear all string caches."""
        with self._lock:
            self._intern_cache.clear()
            self._format_cache.clear()
            self._concat_cache.clear()


class TemporaryObjectTracker:
    """
    Tracker for temporary objects to reduce allocations.
    
    Tracks temporary object creation patterns and provides
    optimization suggestions and automatic cleanup.
    """
    
    def __init__(self, tracking_enabled: bool = True):
        self.tracking_enabled = tracking_enabled
        
        self._object_counts: Dict[type, int] = defaultdict(int)
        self._creation_patterns: Dict[str, List[float]] = defaultdict(list)
        self._weak_refs: Set[weakref.ref] = set()
        self._lock = RLock()
    
    def track_creation(self, obj: Any, context: str = "unknown") -> None:
        """Track object creation."""
        if not self.tracking_enabled:
            return
        
        with self._lock:
            obj_type = type(obj)
            self._object_counts[obj_type] += 1
            self._creation_patterns[context].append(time.time())
            
            # Track with weak reference
            try:
                weak_ref = weakref.ref(obj, self._cleanup_ref)
                self._weak_refs.add(weak_ref)
            except TypeError:
                # Some objects can't be weakly referenced
                pass
    
    def get_creation_stats(self) -> Dict[str, Any]:
        """Get object creation statistics."""
        with self._lock:
            stats = {
                'object_counts': dict(self._object_counts),
                'total_objects': sum(self._object_counts.values()),
                'tracked_refs': len(self._weak_refs),
                'creation_patterns': {}
            }
            
            # Analyze creation patterns
            current_time = time.time()
            for context, timestamps in self._creation_patterns.items():
                recent_timestamps = [t for t in timestamps if current_time - t < 3600]  # Last hour
                stats['creation_patterns'][context] = {
                    'total_creations': len(timestamps),
                    'recent_creations': len(recent_timestamps),
                    'creation_rate': len(recent_timestamps) / 3600 if recent_timestamps else 0
                }
            
            return stats
    
    def suggest_optimizations(self) -> List[str]:
        """Suggest optimizations based on tracking data."""
        suggestions = []
        
        with self._lock:
            # Analyze object counts
            for obj_type, count in self._object_counts.items():
                if count > 1000:
                    suggestions.append(
                        f"Consider object pooling for {obj_type.__name__} (created {count} times)"
                    )
            
            # Analyze creation patterns
            current_time = time.time()
            for context, timestamps in self._creation_patterns.items():
                recent_timestamps = [t for t in timestamps if current_time - t < 300]  # Last 5 minutes
                if len(recent_timestamps) > 100:
                    suggestions.append(
                        f"High object creation rate in {context} ({len(recent_timestamps)} in 5 minutes)"
                    )
        
        return suggestions
    
    def _cleanup_ref(self, ref: weakref.ref) -> None:
        """Clean up dead weak reference."""
        self._weak_refs.discard(ref)


class MemoryPressureDetector:
    """
    Memory pressure detection and automatic cleanup.
    
    Monitors system memory usage and triggers cleanup operations
    when memory pressure is detected.
    """
    
    def __init__(self, config: Optional[MemoryPressureConfig] = None):
        self.config = config or MemoryPressureConfig()
        self.stats = MemoryStats()
        
        self._cleanup_callbacks: List[Callable[[], None]] = []
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_event = Event()
        self._lock = RLock()
        
        # Start monitoring
        self.start_monitoring()
    
    def start_monitoring(self) -> None:
        """Start memory pressure monitoring."""
        if self._monitoring_thread is None or not self._monitoring_thread.is_alive():
            self._stop_event.clear()
            self._monitoring_thread = threading.Thread(
                target=self._monitor_memory,
                daemon=True,
                name="MemoryPressureDetector"
            )
            self._monitoring_thread.start()
    
    def stop_monitoring(self) -> None:
        """Stop memory pressure monitoring."""
        self._stop_event.set()
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=5.0)
    
    def register_cleanup_callback(self, callback: Callable[[], None]) -> None:
        """Register a cleanup callback."""
        with self._lock:
            self._cleanup_callbacks.append(callback)
    
    def unregister_cleanup_callback(self, callback: Callable[[], None]) -> None:
        """Unregister a cleanup callback."""
        with self._lock:
            if callback in self._cleanup_callbacks:
                self._cleanup_callbacks.remove(callback)
    
    def check_memory_pressure(self) -> Dict[str, Any]:
        """Check current memory pressure."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            current_rss_mb = memory_info.rss / 1024 / 1024
            
            # Update stats
            self.stats.current_rss_mb = current_rss_mb
            if current_rss_mb > self.stats.peak_rss_mb:
                self.stats.peak_rss_mb = current_rss_mb
            
            # Determine pressure level
            pressure_level = "normal"
            if current_rss_mb > self.config.critical_threshold_mb:
                pressure_level = "critical"
            elif current_rss_mb > self.config.warning_threshold_mb:
                pressure_level = "warning"
            
            return {
                'current_rss_mb': current_rss_mb,
                'pressure_level': pressure_level,
                'warning_threshold_mb': self.config.warning_threshold_mb,
                'critical_threshold_mb': self.config.critical_threshold_mb,
                'gc_threshold_mb': self.config.warning_threshold_mb * self.config.gc_threshold_ratio,
                'cleanup_threshold_mb': self.config.warning_threshold_mb * self.config.cleanup_threshold_ratio
            }
        except Exception:
            return {
                'current_rss_mb': 0.0,
                'pressure_level': 'unknown',
                'error': 'Failed to check memory pressure'
            }
    
    def trigger_cleanup(self, force: bool = False) -> None:
        """Trigger cleanup operations."""
        pressure_info = self.check_memory_pressure()
        current_rss = pressure_info['current_rss_mb']
        
        # Trigger GC if needed
        gc_threshold = pressure_info.get('gc_threshold_mb', 0)
        if force or current_rss > gc_threshold:
            gc.collect()
            self.stats.gc_collections += 1
        
        # Trigger cleanup callbacks if needed
        cleanup_threshold = pressure_info.get('cleanup_threshold_mb', 0)
        if force or current_rss > cleanup_threshold:
            with self._lock:
                for callback in self._cleanup_callbacks:
                    try:
                        callback()
                        self.stats.cleanup_operations += 1
                    except Exception:
                        # Continue with other callbacks even if one fails
                        pass
        
        # Update pressure events
        if pressure_info['pressure_level'] in ['warning', 'critical']:
            self.stats.pressure_events += 1
    
    def get_stats(self) -> MemoryStats:
        """Get memory statistics."""
        # Update current memory usage
        pressure_info = self.check_memory_pressure()
        self.stats.current_rss_mb = pressure_info['current_rss_mb']
        return self.stats
    
    def _monitor_memory(self) -> None:
        """Memory monitoring loop."""
        while not self._stop_event.wait(self.config.check_interval_seconds):
            try:
                pressure_info = self.check_memory_pressure()
                
                # Trigger cleanup if pressure is high
                if pressure_info['pressure_level'] in ['warning', 'critical']:
                    self.trigger_cleanup()
                
            except Exception:
                # Continue monitoring even if check fails
                pass


class MemoryOptimization:
    """
    Main memory optimization coordinator.
    
    Coordinates all memory optimization components and provides
    a unified interface for memory management.
    """
    
    def __init__(
        self,
        enable_preallocation: bool = True,
        enable_string_optimization: bool = True,
        enable_tracking: bool = True,
        enable_pressure_detection: bool = True,
        pressure_config: Optional[MemoryPressureConfig] = None
    ):
        self.enable_preallocation = enable_preallocation
        self.enable_string_optimization = enable_string_optimization
        self.enable_tracking = enable_tracking
        self.enable_pressure_detection = enable_pressure_detection
        
        # Initialize components
        self._pools: Dict[str, PreAllocationPool] = {}
        self._string_optimizer = StringOptimizer() if enable_string_optimization else None
        self._object_tracker = TemporaryObjectTracker(enable_tracking)
        self._pressure_detector = MemoryPressureDetector(pressure_config) if enable_pressure_detection else None
        
        # Register cleanup callbacks
        if self._pressure_detector:
            self._pressure_detector.register_cleanup_callback(self._cleanup_pools)
            self._pressure_detector.register_cleanup_callback(self._cleanup_string_caches)
    
    def create_pool(self, name: str, factory: Callable[[], T], **kwargs) -> PreAllocationPool[T]:
        """Create a pre-allocation pool."""
        if not self.enable_preallocation:
            raise RuntimeError("Pre-allocation is disabled")
        
        pool = PreAllocationPool(factory, **kwargs)
        self._pools[name] = pool
        return pool
    
    def get_pool(self, name: str) -> Optional[PreAllocationPool]:
        """Get a pre-allocation pool by name."""
        return self._pools.get(name)
    
    def track_object(self, obj: Any, context: str = "unknown") -> None:
        """Track object creation."""
        if self.enable_tracking:
            self._object_tracker.track_creation(obj, context)
    
    def optimize_string(self, operation: str, *args, **kwargs) -> str:
        """Perform optimized string operation."""
        if not self.enable_string_optimization or not self._string_optimizer:
            # Fallback to standard operations
            if operation == "format":
                return args[0].format(*args[1:], **kwargs)
            elif operation == "concat":
                return ''.join(args)
            elif operation == "intern":
                return args[0]
            else:
                raise ValueError(f"Unknown string operation: {operation}")
        
        if operation == "format":
            return self._string_optimizer.optimized_format(*args, **kwargs)
        elif operation == "concat":
            return self._string_optimizer.optimized_concat(*args)
        elif operation == "intern":
            return self._string_optimizer.optimized_intern(args[0])
        else:
            raise ValueError(f"Unknown string operation: {operation}")
    
    def check_memory_pressure(self) -> Dict[str, Any]:
        """Check current memory pressure."""
        if self._pressure_detector:
            return self._pressure_detector.check_memory_pressure()
        else:
            return {'pressure_level': 'unknown', 'error': 'Pressure detection disabled'}
    
    def trigger_cleanup(self, force: bool = False) -> None:
        """Trigger cleanup operations."""
        if self._pressure_detector:
            self._pressure_detector.trigger_cleanup(force)
        else:
            # Manual cleanup
            self._cleanup_pools()
            self._cleanup_string_caches()
            gc.collect()
    
    def get_optimization_suggestions(self) -> List[str]:
        """Get optimization suggestions."""
        suggestions = []
        
        if self.enable_tracking:
            suggestions.extend(self._object_tracker.suggest_optimizations())
        
        # Add pool-specific suggestions
        for name, pool in self._pools.items():
            stats = pool.get_stats()
            if stats['hit_rate'] < 0.5:
                suggestions.append(f"Pool '{name}' has low hit rate ({stats['hit_rate']:.2f})")
        
        return suggestions
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory optimization statistics."""
        stats = {
            'pools': {name: pool.get_stats() for name, pool in self._pools.items()},
            'tracking': self._object_tracker.get_creation_stats() if self.enable_tracking else {},
            'memory_pressure': self.check_memory_pressure(),
            'suggestions': self.get_optimization_suggestions()
        }
        
        if self._pressure_detector:
            stats['pressure_stats'] = self._pressure_detector.get_stats()
        
        return stats
    
    def _cleanup_pools(self) -> None:
        """Clean up pre-allocation pools."""
        for pool in self._pools.values():
            pool.shrink()
    
    def _cleanup_string_caches(self) -> None:
        """Clean up string optimization caches."""
        if self._string_optimizer:
            self._string_optimizer.clear_caches()
    
    def __del__(self):
        """Cleanup on destruction."""
        if self._pressure_detector:
            self._pressure_detector.stop_monitoring()


# Global memory optimizer instance
_global_memory_optimizer: Optional[MemoryOptimization] = None


def get_global_memory_optimizer() -> MemoryOptimization:
    """Get the global memory optimizer instance."""
    global _global_memory_optimizer
    if _global_memory_optimizer is None:
        _global_memory_optimizer = MemoryOptimization()
    return _global_memory_optimizer


def create_object_pool(name: str, factory: Callable[[], T], **kwargs) -> PreAllocationPool[T]:
    """Convenience function to create an object pool."""
    optimizer = get_global_memory_optimizer()
    return optimizer.create_pool(name, factory, **kwargs)


def track_object_creation(obj: Any, context: str = "unknown") -> None:
    """Convenience function to track object creation."""
    optimizer = get_global_memory_optimizer()
    optimizer.track_object(obj, context)


def optimize_string_operation(operation: str, *args, **kwargs) -> str:
    """Convenience function for optimized string operations."""
    optimizer = get_global_memory_optimizer()
    return optimizer.optimize_string(operation, *args, **kwargs)


def check_memory_status() -> Dict[str, Any]:
    """Convenience function to check memory status."""
    optimizer = get_global_memory_optimizer()
    return optimizer.check_memory_pressure()


def trigger_memory_cleanup(force: bool = False) -> None:
    """Convenience function to trigger memory cleanup."""
    optimizer = get_global_memory_optimizer()
    optimizer.trigger_cleanup(force)