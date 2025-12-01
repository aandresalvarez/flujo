"""
Optimized telemetry system with minimal performance impact.

This module provides low-overhead telemetry with span pooling, metric buffering,
batch processing, fast tracing context managers, and efficient metric recording
to minimize performance impact while maintaining observability.
"""

import asyncio
import multiprocessing
import time
import threading
from collections import deque, defaultdict
from contextlib import contextmanager, asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Optional, Callable, Union, Set, Generator, AsyncGenerator
from asyncio import Task
from threading import RLock
from enum import Enum
import uuid
import weakref

from .optimization.memory.object_pool import get_global_pool
from flujo.type_definitions.common import JSONObject


class LogLevel(Enum):
    """Log levels for telemetry."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class Span:
    """Telemetry span for tracing."""

    span_id: str
    name: str
    start_time_ns: int
    end_time_ns: Optional[int] = None
    parent_span_id: Optional[str] = None
    tags: JSONObject = field(default_factory=dict)
    logs: list[JSONObject] = field(default_factory=list)
    status: str = "ok"

    @property
    def duration_ns(self) -> Optional[int]:
        """Get span duration in nanoseconds."""
        if self.end_time_ns is None:
            return None
        return self.end_time_ns - self.start_time_ns

    @property
    def duration_ms(self) -> Optional[float]:
        """Get span duration in milliseconds."""
        duration_ns = self.duration_ns
        if duration_ns is None:
            return None
        return duration_ns / 1_000_000

    def finish(self, status: str = "ok") -> None:
        """Finish the span."""
        self.end_time_ns = time.perf_counter_ns()
        self.status = status

    def add_tag(self, key: str, value: Any) -> None:
        """Add tag to span."""
        self.tags[key] = value

    def add_log(self, message: str, level: LogLevel = LogLevel.INFO, **kwargs: Any) -> None:
        """Add log entry to span."""
        log_entry = {
            "timestamp_ns": time.perf_counter_ns(),
            "message": message,
            "level": level.value,
            **kwargs,
        }
        self.logs.append(log_entry)

    def reset(self) -> None:
        """Reset span for reuse."""
        self.span_id = str(uuid.uuid4())
        self.name = ""
        self.start_time_ns = 0
        self.end_time_ns = None
        self.parent_span_id = None
        self.tags.clear()
        self.logs.clear()
        self.status = "ok"


@dataclass
class Metric:
    """Telemetry metric."""

    name: str
    metric_type: MetricType
    value: Union[int, float]
    timestamp_ns: int
    tags: JSONObject = field(default_factory=dict)

    def reset(self) -> None:
        """Reset metric for reuse."""
        self.name = ""
        self.metric_type = MetricType.COUNTER
        self.value = 0
        self.timestamp_ns = 0
        self.tags.clear()


@dataclass
class TelemetryStats:
    """Statistics for telemetry operations."""

    spans_created: int = 0
    spans_finished: int = 0
    metrics_recorded: int = 0
    batches_processed: int = 0

    # Performance statistics
    total_span_overhead_ns: int = 0
    total_metric_overhead_ns: int = 0
    max_span_overhead_ns: int = 0
    max_metric_overhead_ns: int = 0

    # Pool statistics
    span_pool_hits: int = 0
    span_pool_misses: int = 0
    metric_pool_hits: int = 0
    metric_pool_misses: int = 0

    # Performance tracking
    _execution_times: deque[int] = field(default_factory=lambda: deque(maxlen=100))
    _memory_usage: deque[int] = field(default_factory=lambda: deque(maxlen=100))
    _cache_hit_rates: deque[float] = field(default_factory=lambda: deque(maxlen=100))

    @property
    def average_span_overhead_ns(self) -> float:
        """Calculate average span overhead."""
        if self.spans_created == 0:
            return 0.0
        return self.total_span_overhead_ns / self.spans_created

    @property
    def average_metric_overhead_ns(self) -> float:
        """Calculate average metric overhead."""
        if self.metrics_recorded == 0:
            return 0.0
        return self.total_metric_overhead_ns / self.metrics_recorded

    @property
    def span_pool_hit_rate(self) -> float:
        """Calculate span pool hit rate."""
        total_requests = self.span_pool_hits + self.span_pool_misses
        if total_requests == 0:
            return 0.0
        return self.span_pool_hits / total_requests

    @property
    def metric_pool_hit_rate(self) -> float:
        """Calculate metric pool hit rate."""
        total_requests = self.metric_pool_hits + self.metric_pool_misses
        if total_requests == 0:
            return 0.0
        return self.metric_pool_hits / total_requests


class CircularBuffer:
    """High-performance circular buffer for metrics."""

    def __init__(self, size: int):
        self.size = size
        self.buffer = [None] * size
        self.head = 0
        self.tail = 0
        self.count = 0
        self._lock = RLock()

    def put(self, item: Any) -> bool:
        """Put item in buffer. Returns False if buffer is full."""
        with self._lock:
            if self.count >= self.size:
                return False

            self.buffer[self.tail] = item
            self.tail = (self.tail + 1) % self.size
            self.count += 1
            return True

    def get(self) -> Optional[Any]:
        """Get item from buffer."""
        with self._lock:
            if self.count == 0:
                return None

            item = self.buffer[self.head]
            self.buffer[self.head] = None  # Clear reference
            self.head = (self.head + 1) % self.size
            self.count -= 1
            return item

    def get_batch(self, max_items: int) -> list[Any]:
        """Get batch of items from buffer."""
        items: list[Any] = []
        with self._lock:
            for _ in range(min(max_items, self.count)):
                item = self.buffer[self.head]
                if item is None:
                    break

                items.append(item)
                self.buffer[self.head] = None
                self.head = (self.head + 1) % self.size
                self.count -= 1

        return items

    def is_full(self) -> bool:
        """Check if buffer is full."""
        with self._lock:
            return self.count >= self.size

    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        with self._lock:
            return self.count == 0

    def current_size(self) -> int:
        """Get current buffer size."""
        with self._lock:
            return self.count


class BatchProcessor:
    """Batch processor for efficient telemetry data processing."""

    def __init__(
        self, batch_size: int = 100, flush_interval: float = 5.0, max_buffer_size: int = 10000
    ):
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.max_buffer_size = max_buffer_size

        # Buffers
        self._span_buffer = CircularBuffer(max_buffer_size)
        self._metric_buffer = CircularBuffer(max_buffer_size)

        # Processing
        self._processors: list[Callable[[list[Any]], None]] = []
        self._last_flush = time.time()
        self._processing_task: Optional[Task[Any]] = None
        self._shutdown_event = asyncio.Event()

        # Thread safety
        self._lock = RLock()

    def add_processor(self, processor: Callable[[list[Any]], None]) -> None:
        """Add batch processor function."""
        with self._lock:
            self._processors.append(processor)

    def submit_span(self, span: Span) -> bool:
        """Submit span for batch processing."""
        return self._span_buffer.put(span)

    def submit_metric(self, metric: Metric) -> bool:
        """Submit metric for batch processing."""
        return self._metric_buffer.put(metric)

    async def start_processing(self) -> None:
        """Start batch processing task."""
        if self._processing_task is not None:
            return

        self._shutdown_event.clear()
        self._processing_task = asyncio.create_task(self._processing_loop())

    async def stop_processing(self) -> None:
        """Stop batch processing task."""
        if self._processing_task is None:
            return

        self._shutdown_event.set()
        await self._processing_task
        self._processing_task = None

    async def flush(self) -> None:
        """Flush all pending data."""
        await self._process_batches(force=True)

    async def _processing_loop(self) -> None:
        """Main processing loop."""
        while not self._shutdown_event.is_set():
            try:
                await self._process_batches()
                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
            except asyncio.CancelledError:
                break
            except Exception:
                # Continue processing even if batch processing fails
                continue

        # Final flush on shutdown
        await self._process_batches(force=True)

    async def _process_batches(self, force: bool = False) -> None:
        """Process batches if conditions are met."""
        current_time = time.time()
        should_flush = (
            force
            or current_time - self._last_flush >= self.flush_interval
            or self._span_buffer.current_size() >= self.batch_size
            or self._metric_buffer.current_size() >= self.batch_size
        )

        if not should_flush:
            return

        # Process spans
        spans = self._span_buffer.get_batch(self.batch_size)
        if spans:
            await self._process_span_batch(spans)

        # Process metrics
        metrics = self._metric_buffer.get_batch(self.batch_size)
        if metrics:
            await self._process_metric_batch(metrics)

        self._last_flush = current_time

    async def _process_span_batch(self, spans: list[Span]) -> None:
        """Process batch of spans."""
        for processor in self._processors:
            try:
                if asyncio.iscoroutinefunction(processor):
                    await processor(spans)
                else:
                    processor(spans)
            except Exception:
                # Continue with other processors
                continue

    async def _process_metric_batch(self, metrics: list[Metric]) -> None:
        """Process batch of metrics."""
        for processor in self._processors:
            try:
                if asyncio.iscoroutinefunction(processor):
                    await processor(metrics)
                else:
                    processor(metrics)
            except Exception:
                # Continue with other processors
                continue


class OptimizedTelemetry:
    """
    Optimized telemetry system with minimal performance impact.

    Features:
    - Object pooling for spans and metrics
    - Circular buffering for efficient data collection
    - Batch processing to reduce overhead
    - Fast tracing context managers
    - Sampling for high-frequency events
    - Statistics tracking
    """

    def __init__(
        self,
        enable_tracing: bool = True,
        enable_metrics: bool = True,
        enable_sampling: bool = True,
        sampling_rate: float = 0.5,
        batch_size: int = 200,
        flush_interval: float = 2.0,
        enable_stats: bool = True,
    ):
        self.enable_tracing = enable_tracing
        self.enable_metrics = enable_metrics
        self.enable_sampling = enable_sampling
        self.sampling_rate = sampling_rate
        self.enable_stats = enable_stats
        self.num_workers = multiprocessing.cpu_count()

        # Object pools
        self._object_pool = get_global_pool()
        self._span_pool: deque[Span] = deque(maxlen=1000)
        self._metric_pool: deque[Metric] = deque(maxlen=1000)

        # Batch processor
        self._batch_processor = BatchProcessor(batch_size=batch_size, flush_interval=flush_interval)

        # Current span tracking
        self._current_spans: dict[int, Span] = {}  # thread_id -> span
        self._span_stack: dict[int, list[Span]] = defaultdict(list)  # thread_id -> span stack

        # Statistics
        self._stats = TelemetryStats() if enable_stats else None

        # Thread safety
        self._lock = RLock()

        # Sampling state
        self._sample_counter = 0
        self._sample_threshold = int(1.0 / max(sampling_rate, 0.001))

        # Worker tasks
        self._worker_tasks: list[Optional[Task[Any]]] = [None] * self.num_workers
        self._worker_running = False

        # Weak references for cleanup
        self._weak_refs: Set[weakref.ref[Any]] = set()
        self._last_cleanup = time.time()

        # Callbacks
        self._callbacks: list[Callable[[str, JSONObject], None]] = []

    async def start(self) -> None:
        """Start telemetry system."""
        await self._batch_processor.start_processing()

    async def stop(self) -> None:
        """Stop telemetry system."""
        await self._batch_processor.stop_processing()

    def should_sample(self) -> bool:
        """Determine if current operation should be sampled."""
        if not self.enable_sampling or self.sampling_rate >= 1.0:
            return True

        with self._lock:
            self._sample_counter += 1
            return self._sample_counter % self._sample_threshold == 0

    def start_span(
        self, name: str, parent_span: Optional[Span] = None, tags: Optional[JSONObject] = None
    ) -> Optional[Span]:
        """Start a new span."""
        if not self.enable_tracing or not self.should_sample():
            return None

        start_time = time.perf_counter_ns()

        try:
            # Get span from pool
            span = self._get_span_from_pool()

            # Initialize span
            span.span_id = str(uuid.uuid4())
            span.name = name
            span.start_time_ns = time.perf_counter_ns()
            span.parent_span_id = parent_span.span_id if parent_span else None

            if tags:
                span.tags.update(tags)

            # Track current span
            thread_id = threading.get_ident()
            with self._lock:
                self._span_stack[thread_id].append(span)
                self._current_spans[thread_id] = span

            # Update statistics
            if self.enable_stats and self._stats is not None:
                overhead = time.perf_counter_ns() - start_time
                self._stats.spans_created += 1
                self._stats.total_span_overhead_ns += overhead
                self._stats.max_span_overhead_ns = max(self._stats.max_span_overhead_ns, overhead)

            return span

        except Exception:
            return None

    def finish_span(self, span: Optional[Span], status: str = "ok") -> None:
        """Finish a span."""
        if not span or not self.enable_tracing:
            return

        start_time = time.perf_counter_ns()

        try:
            # Finish span
            span.finish(status)

            # Remove from current tracking
            thread_id = threading.get_ident()
            with self._lock:
                span_stack = self._span_stack[thread_id]
                if span_stack and span_stack[-1] == span:
                    span_stack.pop()

                # Update current span
                if span_stack:
                    self._current_spans[thread_id] = span_stack[-1]
                else:
                    self._current_spans.pop(thread_id, None)

            # Submit for batch processing
            self._batch_processor.submit_span(span)

            # Update statistics
            if self.enable_stats and self._stats is not None:
                overhead = time.perf_counter_ns() - start_time
                self._stats.spans_finished += 1
                self._stats.total_span_overhead_ns += overhead
                self._stats.max_span_overhead_ns = max(self._stats.max_span_overhead_ns, overhead)

        except Exception:
            # Return span to pool on error
            self._return_span_to_pool(span)

    def get_current_span(self) -> Optional[Span]:
        """Get current span for this thread."""
        thread_id = threading.get_ident()
        return self._current_spans.get(thread_id)

    @contextmanager
    def trace(
        self, name: str, tags: Optional[JSONObject] = None
    ) -> Generator[Optional[Span], None, None]:
        """Context manager for tracing."""
        span: Optional[Span] = self.start_span(name, tags=tags)
        try:
            yield span
        finally:
            if span is not None:
                self.finish_span(span)

    @asynccontextmanager
    async def trace_async(
        self, name: str, tags: Optional[JSONObject] = None
    ) -> AsyncGenerator[Optional[Span], None]:
        """Async context manager for tracing."""
        span: Optional[Span] = self.start_span(name, tags=tags)
        try:
            yield span
        finally:
            if span is not None:
                self.finish_span(span)

    def record_metric(
        self,
        name: str,
        value: Union[int, float],
        metric_type: MetricType = MetricType.COUNTER,
        tags: Optional[JSONObject] = None,
    ) -> None:
        """Record a metric."""
        if not self.enable_metrics or not self.should_sample():
            return

        start_time = time.perf_counter_ns()

        try:
            # Get metric from pool
            metric = self._get_metric_from_pool()

            # Initialize metric
            metric.name = name
            metric.metric_type = metric_type
            metric.value = value
            metric.timestamp_ns = time.perf_counter_ns()

            if tags:
                metric.tags.update(tags)

            # Submit for batch processing
            self._batch_processor.submit_metric(metric)

            # Update statistics
            if self.enable_stats and self._stats is not None:
                overhead = time.perf_counter_ns() - start_time
                self._stats.metrics_recorded += 1
                self._stats.total_metric_overhead_ns += overhead
                self._stats.max_metric_overhead_ns = max(
                    self._stats.max_metric_overhead_ns, overhead
                )

        except Exception:
            pass  # Silently ignore metric recording errors

    def increment_counter(
        self, name: str, value: int = 1, tags: Optional[JSONObject] = None
    ) -> None:
        """Increment a counter metric."""
        self.record_metric(name, value, MetricType.COUNTER, tags)

    def set_gauge(
        self, name: str, value: Union[int, float], tags: Optional[JSONObject] = None
    ) -> None:
        """Set a gauge metric."""
        self.record_metric(name, value, MetricType.GAUGE, tags)

    def record_histogram(
        self, name: str, value: Union[int, float], tags: Optional[JSONObject] = None
    ) -> None:
        """Record a histogram metric."""
        self.record_metric(name, value, MetricType.HISTOGRAM, tags)

    def record_timer(self, name: str, duration_ns: int, tags: Optional[JSONObject] = None) -> None:
        """Record a timer metric."""
        duration_ms = duration_ns / 1_000_000
        self.record_metric(name, duration_ms, MetricType.TIMER, tags)

    def add_processor(self, processor: Callable[[list[Any]], None]) -> None:
        """Add batch processor function."""
        self._batch_processor.add_processor(processor)

    async def flush(self) -> None:
        """Flush all pending telemetry data."""
        await self._batch_processor.flush()

    def _get_span_from_pool(self) -> Span:
        """Get span from pool or create new one."""
        if self._stats is not None:
            self._stats.spans_created += 1
            self._stats.total_span_overhead_ns += 0  # Minimal overhead
            self._stats.max_span_overhead_ns = max(self._stats.max_span_overhead_ns, 0)

        # Try to get from pool first
        if self._span_pool:
            try:
                span = self._span_pool.popleft()
                if self._stats is not None:
                    self._stats.span_pool_hits += 1
                span.reset()
                return span
            except IndexError:
                pass

        # Create new span if pool is empty
        if self._stats is not None:
            self._stats.span_pool_misses += 1
        return Span(
            span_id=str(uuid.uuid4()),
            name="",
            start_time_ns=time.perf_counter_ns(),
        )

    def _return_span_to_pool(self, span: Span) -> None:
        """Return span to pool."""
        if self._stats is not None:
            self._stats.spans_finished += 1
            self._stats.total_span_overhead_ns += 0  # Minimal overhead
            self._stats.max_span_overhead_ns = max(self._stats.max_span_overhead_ns, 0)

        # Return to pool if available
        if (
            self._span_pool
            and self._span_pool.maxlen is not None
            and len(self._span_pool) < self._span_pool.maxlen
        ):
            self._span_pool.append(span)

    def _get_metric_from_pool(self) -> Metric:
        """Get metric from pool or create new one."""
        if self._stats is not None:
            self._stats.metrics_recorded += 1
            self._stats.total_metric_overhead_ns += 0  # Minimal overhead
            self._stats.max_metric_overhead_ns = max(self._stats.max_metric_overhead_ns, 0)

        # Try to get from pool first
        if self._metric_pool:
            try:
                metric = self._metric_pool.popleft()
                if self._stats is not None:
                    self._stats.metric_pool_hits += 1
                metric.reset()
                return metric
            except IndexError:
                pass

        # Create new metric if pool is empty
        if self._stats is not None:
            self._stats.metric_pool_misses += 1
        return Metric(
            name="",
            metric_type=MetricType.COUNTER,
            value=0,
            timestamp_ns=time.perf_counter_ns(),
        )

    def _return_metric_to_pool(self, metric: Metric) -> None:
        """Return metric to pool."""
        if self._stats is not None:
            self._stats.metrics_recorded += 1
            self._stats.total_metric_overhead_ns += 0  # Minimal overhead
            self._stats.max_metric_overhead_ns = max(self._stats.max_metric_overhead_ns, 0)

        # Return to pool if available
        if (
            self._metric_pool
            and self._metric_pool.maxlen is not None
            and len(self._metric_pool) < self._metric_pool.maxlen
        ):
            self._metric_pool.append(metric)

    def get_stats(self) -> Optional[TelemetryStats]:
        """Get telemetry statistics."""
        return self._stats

    def clear_stats(self) -> None:
        """Clear telemetry statistics."""
        if self._stats:
            self._stats = TelemetryStats()

    def register_callback(self, callback: Callable[[str, JSONObject], None]) -> None:
        """Register a callback for telemetry events."""
        self._callbacks.append(callback)

    def _notify_callbacks(self, event: str, data: JSONObject) -> None:
        """Notify all registered callbacks."""
        for callback in self._callbacks:
            try:
                callback(event, data)
            except Exception:
                # Log but don't fail
                pass


# Global telemetry instance
_global_telemetry: Optional[OptimizedTelemetry] = None


def get_global_telemetry() -> OptimizedTelemetry:
    """Get global telemetry instance."""
    global _global_telemetry
    if _global_telemetry is None:
        _global_telemetry = OptimizedTelemetry()
    return _global_telemetry


# Convenience functions
def trace_function(
    name: str, tags: Optional[JSONObject] = None
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator for tracing functions."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if asyncio.iscoroutinefunction(func):

            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                telemetry = get_global_telemetry()
                span = telemetry.start_span(name, tags=tags)
                try:
                    result = await func(*args, **kwargs)
                    if span:
                        span.add_tag("status", "success")
                    return result
                except Exception as e:
                    if span:
                        span.add_tag("status", "error")
                        span.add_tag("error_message", str(e))
                    raise
                finally:
                    if span:
                        telemetry.finish_span(span)

            return async_wrapper
        else:

            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                telemetry = get_global_telemetry()
                span = telemetry.start_span(name, tags=tags)
                try:
                    result = func(*args, **kwargs)
                    if span:
                        span.add_tag("status", "success")
                    return result
                except Exception as e:
                    if span:
                        span.add_tag("status", "error")
                        span.add_tag("error_message", str(e))
                    raise
                finally:
                    if span:
                        telemetry.finish_span(span)

            return sync_wrapper

    return decorator


def start_span(name: str, tags: Optional[JSONObject] = None) -> Optional[Span]:
    """Start a span using global telemetry."""
    telemetry = get_global_telemetry()
    return telemetry.start_span(name, tags=tags)


def finish_span(span: Optional[Span], status: str = "ok") -> None:
    """Finish a span using global telemetry."""
    telemetry = get_global_telemetry()
    telemetry.finish_span(span, status)


def record_metric(
    name: str,
    value: Union[int, float],
    metric_type: MetricType = MetricType.COUNTER,
    tags: Optional[JSONObject] = None,
) -> None:
    """Record a metric using global telemetry."""
    telemetry = get_global_telemetry()
    telemetry.record_metric(name, value, metric_type, tags)


def increment_counter(name: str, value: int = 1, tags: Optional[JSONObject] = None) -> None:
    """Increment a counter using global telemetry."""
    telemetry = get_global_telemetry()
    telemetry.increment_counter(name, value, tags)


def set_gauge(name: str, value: Union[int, float], tags: Optional[JSONObject] = None) -> None:
    """Set a gauge using global telemetry."""
    telemetry = get_global_telemetry()
    telemetry.set_gauge(name, value, tags)
