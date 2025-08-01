"""
Concurrency optimization components.

This module provides adaptive concurrency limits, work-stealing queues,
semaphore optimization, and contention reduction strategies to improve
concurrent execution performance.
"""

import asyncio
import multiprocessing
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Awaitable
from asyncio import Task
from threading import RLock
from enum import Enum


class LoadLevel(Enum):
    """System load levels for adaptive concurrency."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ConcurrencyStats:
    """Statistics for concurrency operations."""

    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    queued_tasks: int = 0
    active_tasks: int = 0

    # Timing statistics
    total_wait_time_ns: int = 0
    total_execution_time_ns: int = 0
    max_wait_time_ns: int = 0
    max_execution_time_ns: int = 0

    # Concurrency statistics
    max_concurrent_tasks: int = 0
    semaphore_contentions: int = 0
    work_steals: int = 0

    @property
    def completion_rate(self) -> float:
        """Calculate task completion rate."""
        if self.total_tasks == 0:
            return 0.0
        return self.completed_tasks / self.total_tasks

    @property
    def average_wait_time_ns(self) -> float:
        """Calculate average wait time."""
        if self.completed_tasks == 0:
            return 0.0
        return self.total_wait_time_ns / self.completed_tasks

    @property
    def average_execution_time_ns(self) -> float:
        """Calculate average execution time."""
        if self.completed_tasks == 0:
            return 0.0
        return self.total_execution_time_ns / self.completed_tasks


@dataclass
class WorkItem:
    """Work item for work-stealing queue."""

    task_id: str
    coro: Awaitable[Any]
    priority: int = 0
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    @property
    def age_seconds(self) -> float:
        """Get age of work item in seconds."""
        return time.time() - self.created_at

    @property
    def execution_time_seconds(self) -> Optional[float]:
        """Get execution time in seconds."""
        if self.started_at is None or self.completed_at is None:
            return None
        return self.completed_at - self.started_at


class AdaptiveSemaphore:
    """
    Adaptive semaphore that adjusts limits based on system load.

    Features:
    - Dynamic limit adjustment based on performance metrics
    - Load-based scaling
    - Contention detection and mitigation
    - Statistics tracking
    """

    def __init__(
        self,
        initial_limit: int,
        min_limit: int = 1,
        max_limit: Optional[int] = None,
        adaptation_interval: float = 30.0,
        enable_stats: bool = True,
    ):
        self.initial_limit = initial_limit
        self.min_limit = min_limit
        self.max_limit = max_limit or multiprocessing.cpu_count() * 4
        self.adaptation_interval = adaptation_interval
        self.enable_stats = enable_stats

        # Current state
        self.current_limit = initial_limit
        self._semaphore = asyncio.Semaphore(initial_limit)
        self._last_adaptation = time.time()

        # Performance tracking
        self._acquire_times: deque[int] = deque(maxlen=1000)
        self._contention_count = 0
        self._successful_acquisitions = 0

        # Statistics
        self._stats = ConcurrencyStats() if enable_stats else None

        # Thread safety
        self._lock = RLock()

    async def acquire(self) -> None:
        """Acquire semaphore with contention tracking."""
        start_time = time.perf_counter_ns()

        # Check if semaphore is immediately available
        # asyncio.Semaphore doesn't have acquire_nowait, so we use a different approach
        if self._semaphore._value > 0:
            # Semaphore is available, acquire it
            await self._semaphore.acquire()
            # Immediate acquisition
            if self.enable_stats and self._stats is not None:
                self._stats.semaphore_contentions += 0  # No contention
        else:
            # Contention detected
            with self._lock:
                self._contention_count += 1

            if self.enable_stats and self._stats is not None:
                self._stats.semaphore_contentions += 1

            # Wait for acquisition
            await self._semaphore.acquire()

        # Track acquisition time
        acquisition_time = time.perf_counter_ns() - start_time
        with self._lock:
            self._acquire_times.append(acquisition_time)
            self._successful_acquisitions += 1

        # Update statistics
        if self.enable_stats and self._stats is not None:
            self._stats.total_wait_time_ns += acquisition_time
            self._stats.max_wait_time_ns = max(self._stats.max_wait_time_ns, acquisition_time)

        # Check if adaptation is needed
        await self._maybe_adapt()

    def release(self) -> None:
        """Release semaphore."""
        self._semaphore.release()

    async def _maybe_adapt(self) -> None:
        """Adapt semaphore limit based on performance metrics."""
        current_time = time.time()

        if current_time - self._last_adaptation < self.adaptation_interval:
            return

        with self._lock:
            self._last_adaptation = current_time

            # Calculate metrics
            if len(self._acquire_times) < 10:
                return  # Not enough data

            avg_acquisition_time = sum(self._acquire_times) / len(self._acquire_times)
            contention_rate = self._contention_count / max(self._successful_acquisitions, 1)

            # Determine load level
            load_level = self._determine_load_level(avg_acquisition_time, contention_rate)

            # Adjust limit based on load
            new_limit = self._calculate_new_limit(load_level)

            if new_limit != self.current_limit:
                await self._adjust_limit(new_limit)

            # Reset counters
            self._contention_count = 0
            self._successful_acquisitions = 0
            self._acquire_times.clear()

    def _determine_load_level(
        self, avg_acquisition_time: float, contention_rate: float
    ) -> LoadLevel:
        """Determine current load level."""
        # Convert nanoseconds to milliseconds for easier thresholds
        avg_acquisition_ms = avg_acquisition_time / 1_000_000

        if avg_acquisition_ms > 100 or contention_rate > 0.8:
            return LoadLevel.CRITICAL
        elif avg_acquisition_ms > 50 or contention_rate > 0.5:
            return LoadLevel.HIGH
        elif avg_acquisition_ms > 10 or contention_rate > 0.2:
            return LoadLevel.MEDIUM
        else:
            return LoadLevel.LOW

    def _calculate_new_limit(self, load_level: LoadLevel) -> int:
        """Calculate new semaphore limit based on load level."""
        if load_level == LoadLevel.CRITICAL:
            # Reduce limit to reduce contention
            new_limit = max(self.min_limit, int(self.current_limit * 0.7))
        elif load_level == LoadLevel.HIGH:
            # Slightly reduce limit
            new_limit = max(self.min_limit, int(self.current_limit * 0.9))
        elif load_level == LoadLevel.LOW:
            # Increase limit to allow more concurrency
            new_limit = min(self.max_limit, int(self.current_limit * 1.2))
        else:  # MEDIUM
            # Keep current limit
            new_limit = self.current_limit

        return new_limit

    async def _adjust_limit(self, new_limit: int) -> None:
        """Adjust semaphore limit."""
        if new_limit > self.current_limit:
            # Increase limit by releasing additional permits
            for _ in range(new_limit - self.current_limit):
                self._semaphore.release()
        elif new_limit < self.current_limit:
            # Decrease limit by acquiring permits (non-blocking)
            permits_to_acquire = self.current_limit - new_limit
            for _ in range(permits_to_acquire):
                try:
                    self._semaphore.release()
                except ValueError:
                    # Can't release more permits, limit reached naturally
                    break

        self.current_limit = new_limit

    def get_stats(self) -> Optional[ConcurrencyStats]:
        """Get concurrency statistics."""
        return self._stats

    def get_current_limit(self) -> int:
        """Get current semaphore limit."""
        return self.current_limit


class WorkStealingQueue:
    """
    Work-stealing queue for load balancing.

    Features:
    - Priority-based task scheduling
    - Work stealing between workers
    - Load balancing
    - Statistics tracking
    """

    def __init__(
        self,
        num_workers: Optional[int] = None,
        enable_work_stealing: bool = True,
        enable_stats: bool = True,
    ):
        self.num_workers = num_workers or multiprocessing.cpu_count()
        self.enable_work_stealing = enable_work_stealing
        self.enable_stats = enable_stats

        # Worker queues (one per worker)
        self._worker_queues: List[deque[WorkItem]] = [deque() for _ in range(self.num_workers)]
        self._worker_locks: List[RLock] = [RLock() for _ in range(self.num_workers)]

        # Round-robin assignment
        self._next_worker = 0
        self._assignment_lock = RLock()

        # Statistics
        self._stats = ConcurrencyStats() if enable_stats else None

        # Work stealing tracking
        self._steal_attempts = 0
        self._successful_steals = 0

    def submit_work(self, work_item: WorkItem) -> None:
        """Submit work item to queue."""
        # Choose worker queue (round-robin with load balancing)
        worker_id = self._choose_worker_queue()

        # Add to worker queue
        with self._worker_locks[worker_id]:
            # Insert based on priority (higher priority first)
            queue = self._worker_queues[worker_id]

            # Simple insertion sort by priority
            inserted = False
            for i, existing_item in enumerate(queue):
                if work_item.priority > existing_item.priority:
                    queue.insert(i, work_item)
                    inserted = True
                    break

            if not inserted:
                queue.append(work_item)

        # Update statistics
        if self.enable_stats and self._stats is not None:
            self._stats.total_tasks += 1
            self._stats.queued_tasks += 1

    def get_work(self, worker_id: int) -> Optional[WorkItem]:
        """Get work item for specific worker."""
        # Try to get work from own queue first
        work_item = self._get_work_from_queue(worker_id)

        if work_item is None and self.enable_work_stealing:
            # Try to steal work from other workers
            work_item = self._steal_work(worker_id)

        if work_item is not None:
            work_item.started_at = time.time()

            # Update statistics
            if self.enable_stats and self._stats is not None:
                self._stats.queued_tasks -= 1
                self._stats.active_tasks += 1

        return work_item

    def complete_work(self, work_item: WorkItem, success: bool = True) -> None:
        """Mark work item as completed."""
        work_item.completed_at = time.time()

        # Update statistics
        if self.enable_stats and self._stats is not None:
            self._stats.active_tasks -= 1
            if success:
                self._stats.completed_tasks += 1
            else:
                self._stats.failed_tasks += 1

            # Update timing statistics
            if work_item.execution_time_seconds is not None:
                execution_time_ns = int(work_item.execution_time_seconds * 1_000_000_000)
                self._stats.total_execution_time_ns += execution_time_ns
                self._stats.max_execution_time_ns = max(
                    self._stats.max_execution_time_ns, execution_time_ns
                )

    def _choose_worker_queue(self) -> int:
        """Choose worker queue for new work item."""
        with self._assignment_lock:
            # Simple round-robin with load balancing
            best_worker = self._next_worker
            min_queue_size = len(self._worker_queues[best_worker])

            # Find worker with smallest queue
            for i in range(self.num_workers):
                queue_size = len(self._worker_queues[i])
                if queue_size < min_queue_size:
                    min_queue_size = queue_size
                    best_worker = i

            # Update round-robin counter
            self._next_worker = (self._next_worker + 1) % self.num_workers
            return best_worker

    def _get_work_from_queue(self, worker_id: int) -> Optional[WorkItem]:
        """Get work item from specific worker queue."""
        with self._worker_locks[worker_id]:
            queue = self._worker_queues[worker_id]
            if queue:
                return queue.popleft()
        return None

    def _steal_work(self, worker_id: int) -> Optional[WorkItem]:
        """Steal work from other workers."""
        if not self.enable_work_stealing:
            return None

        # Try to steal from other workers
        for other_worker in range(self.num_workers):
            if other_worker == worker_id:
                continue

            with self._worker_locks[other_worker]:
                queue = self._worker_queues[other_worker]
                if len(queue) > 1:  # Leave at least one item
                    stolen_item = queue.pop()
                    self._successful_steals += 1
                    return stolen_item

        return None

    def get_queue_sizes(self) -> List[int]:
        """Get current queue sizes for all workers."""
        return [len(queue) for queue in self._worker_queues]

    def get_total_queued_work(self) -> int:
        """Get total number of queued work items."""
        return sum(len(queue) for queue in self._worker_queues)

    def get_steal_stats(self) -> Tuple[int, int]:
        """Get work stealing statistics."""
        return self._steal_attempts, self._successful_steals

    def get_stats(self) -> Optional[ConcurrencyStats]:
        """Get concurrency statistics."""
        if not self.enable_stats:
            return None

        if self._stats is not None:
            # Update current queue sizes
            self._stats.queued_tasks = sum(len(queue) for queue in self._worker_queues)
            self._stats.active_tasks = (
                self._stats.total_tasks
                - self._stats.queued_tasks
                - self._stats.completed_tasks
                - self._stats.failed_tasks
            )

        return self._stats


class ConcurrencyOptimizer:
    """
    Main concurrency optimization coordinator.

    Features:
    - Adaptive concurrency management
    - Work-stealing task distribution
    - Load balancing
    - Performance monitoring
    """

    def __init__(
        self,
        initial_concurrency: Optional[int] = None,
        enable_work_stealing: bool = True,
        enable_adaptive_limits: bool = True,
        enable_stats: bool = True,
    ):
        self.initial_concurrency = initial_concurrency or multiprocessing.cpu_count() * 2
        self.num_workers = multiprocessing.cpu_count()
        self.enable_work_stealing = enable_work_stealing
        self.enable_adaptive_limits = enable_adaptive_limits
        self.enable_stats = enable_stats

        # Core components
        self._semaphore = (
            AdaptiveSemaphore(initial_limit=self.initial_concurrency, enable_stats=enable_stats)
            if enable_adaptive_limits
            else asyncio.Semaphore(self.initial_concurrency)
        )

        self._work_queue = (
            WorkStealingQueue(enable_work_stealing=enable_work_stealing, enable_stats=enable_stats)
            if enable_work_stealing
            else None
        )

        # Worker management
        self._workers: List[Task[Any]] = []
        self._worker_shutdown = asyncio.Event()
        self._running = False

        # Worker tasks
        self._worker_tasks: List[Optional[Task[Any]]] = [None] * self.num_workers
        self._worker_running = False

        # Monitoring task
        self._monitoring_task: Optional[Task[Any]] = None

        # Statistics
        self._global_stats = ConcurrencyStats() if enable_stats else None

    async def start_workers(self) -> None:
        """Start worker tasks."""
        if self._running:
            return

        self._running = True
        self._worker_shutdown.clear()

        if self._work_queue:
            # Start work-stealing workers
            for worker_id in range(self._work_queue.num_workers):
                worker_task = asyncio.create_task(
                    self._worker_loop(worker_id), name=f"concurrency_worker_{worker_id}"
                )
                self._workers.append(worker_task)

    async def stop_workers(self) -> None:
        """Stop worker tasks."""
        if not self._running:
            return

        self._running = False
        self._worker_shutdown.set()

        # Wait for workers to finish
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
            self._workers.clear()

    async def execute_with_concurrency(
        self, coro: Awaitable[Any], priority: int = 0, task_id: Optional[str] = None
    ) -> Any:
        """Execute coroutine with concurrency optimization."""
        if task_id is None:
            task_id = f"task_{id(coro)}"

        if self._work_queue and self._running:
            # Use work-stealing queue
            work_item = WorkItem(task_id=task_id, coro=coro, priority=priority)

            # Submit to work queue
            self._work_queue.submit_work(work_item)

            # Wait for completion (simplified - in practice would use futures)
            # For now, execute directly with semaphore
            return await self._execute_with_semaphore(coro)
        else:
            # Direct execution with semaphore
            return await self._execute_with_semaphore(coro)

    async def _execute_with_semaphore(self, coro: Awaitable[Any]) -> Any:
        """Execute coroutine with semaphore control."""
        if isinstance(self._semaphore, AdaptiveSemaphore):
            await self._semaphore.acquire()
        else:
            await self._semaphore.acquire()

        try:
            start_time = time.perf_counter_ns()
            result = await coro
            execution_time = time.perf_counter_ns() - start_time

            # Update statistics
            if self.enable_stats and self._global_stats is not None:
                self._global_stats.completed_tasks += 1
                self._global_stats.total_execution_time_ns += execution_time
                self._global_stats.max_execution_time_ns = max(
                    self._global_stats.max_execution_time_ns, execution_time
                )

            return result

        except Exception as e:
            # Update failure statistics
            if self.enable_stats and self._global_stats is not None:
                self._global_stats.failed_tasks += 1
            raise e

        finally:
            if isinstance(self._semaphore, AdaptiveSemaphore):
                self._semaphore.release()
            else:
                self._semaphore.release()

    async def _worker_loop(self, worker_id: int) -> None:
        """Worker loop for processing work items."""
        while not self._worker_shutdown.is_set():
            try:
                # Get work from queue
                if self._work_queue is None:
                    await asyncio.sleep(0.01)
                    continue
                work_item = self._work_queue.get_work(worker_id)

                if work_item is None:
                    # No work available, sleep briefly
                    await asyncio.sleep(0.01)
                    continue

                # Execute work item
                try:
                    await self._execute_with_semaphore(work_item.coro)
                    if self._work_queue is not None:
                        self._work_queue.complete_work(work_item, success=True)
                except Exception:
                    if self._work_queue is not None:
                        self._work_queue.complete_work(work_item, success=False)

            except asyncio.CancelledError:
                break
            except Exception:
                # Continue processing other work items
                continue

    def get_concurrency_stats(self) -> Dict[str, Any]:
        """Get comprehensive concurrency statistics."""
        stats: Dict[str, Any] = {
            "total_workers": self.num_workers,
            "work_stealing_enabled": self.enable_work_stealing,
        }

        if self._work_queue:
            queue_sizes = self._work_queue.get_queue_sizes()
            stats.update(
                {
                    "worker_queues": queue_sizes,
                    "total_queued_work": self._work_queue.get_total_queued_work(),
                    "steal_attempts": self._work_queue._steal_attempts,
                    "successful_steals": self._work_queue._successful_steals,
                }
            )

        if self._global_stats is not None:
            stats.update(
                {
                    "total_tasks": self._global_stats.total_tasks,
                    "completed_tasks": self._global_stats.completed_tasks,
                    "failed_tasks": self._global_stats.failed_tasks,
                    "active_tasks": self._global_stats.active_tasks,
                    "queued_tasks": self._global_stats.queued_tasks,
                    "completion_rate": float(self._global_stats.completion_rate),
                    "average_wait_time_ns": float(self._global_stats.average_wait_time_ns),
                    "average_execution_time_ns": float(
                        self._global_stats.average_execution_time_ns
                    ),
                }
            )

        return stats

    async def __aenter__(self) -> "ConcurrencyOptimizer":
        """Enter async context."""
        await self.start_workers()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context."""
        await self.stop_workers()


# Global concurrency optimizer instance
_global_concurrency_optimizer: Optional[ConcurrencyOptimizer] = None


def get_global_concurrency_optimizer() -> ConcurrencyOptimizer:
    """Get global concurrency optimizer instance."""
    global _global_concurrency_optimizer
    if _global_concurrency_optimizer is None:
        _global_concurrency_optimizer = ConcurrencyOptimizer()
    return _global_concurrency_optimizer


async def execute_with_optimized_concurrency(
    coro: Awaitable[Any], priority: int = 0, task_id: Optional[str] = None
) -> Any:
    """Execute coroutine with optimized concurrency."""
    optimizer = get_global_concurrency_optimizer()
    return await optimizer.execute_with_concurrency(coro, priority, task_id)


def get_concurrency_stats() -> Dict[str, Any]:
    """Get global concurrency statistics."""
    optimizer = get_global_concurrency_optimizer()
    return optimizer.get_concurrency_stats()
