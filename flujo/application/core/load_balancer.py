"""
Load balancer for work distribution optimization.

This module provides work distribution optimization with priority-based execution,
resource contention reduction, task scheduling optimization, work-stealing queue,
and load distribution algorithms for optimal performance under varying loads.
"""

import asyncio
import time
import random
import heapq
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Callable, Union, Tuple, Awaitable
from threading import RLock
from enum import Enum
import statistics
import uuid

from .optimized_telemetry import get_global_telemetry, MetricType
from .adaptive_resource_manager import ResourceType, get_global_adaptive_resource_manager


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_RESPONSE_TIME = "least_response_time"
    RESOURCE_BASED = "resource_based"
    PRIORITY_BASED = "priority_based"
    ADAPTIVE = "adaptive"


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class WorkerStatus(Enum):
    """Worker status states."""
    IDLE = "idle"
    BUSY = "busy"
    OVERLOADED = "overloaded"
    FAILED = "failed"
    MAINTENANCE = "maintenance"


@dataclass
class Task:
    """Task representation for load balancing."""
    
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    priority: TaskPriority = TaskPriority.NORMAL
    estimated_duration: float = 1.0  # seconds
    resource_requirements: Dict[ResourceType, float] = field(default_factory=dict)
    
    # Task execution
    coro: Optional[Awaitable[Any]] = None
    func: Optional[Callable[..., Any]] = None
    args: Tuple[Any, ...] = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    created_at: float = field(default_factory=time.time)
    scheduled_at: Optional[float] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    
    # Dependencies
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)
    
    @property
    def is_ready(self) -> bool:
        """Check if task is ready for execution (all dependencies met)."""
        return len(self.dependencies) == 0
    
    @property
    def wait_time(self) -> float:
        """Calculate task wait time."""
        if self.scheduled_at and self.started_at:
            return self.started_at - self.scheduled_at
        elif self.scheduled_at:
            return time.time() - self.scheduled_at
        return 0.0
    
    @property
    def execution_time(self) -> Optional[float]:
        """Calculate task execution time."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None


@dataclass
class Worker:
    """Worker representation for load balancing."""
    
    worker_id: str
    capacity: float = 1.0  # Maximum concurrent tasks
    current_load: float = 0.0  # Current load (0.0-1.0)
    status: WorkerStatus = WorkerStatus.IDLE
    
    # Performance metrics
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    average_response_time: float = 0.0
    
    # Resource capabilities
    resource_capacity: Dict[ResourceType, float] = field(default_factory=dict)
    resource_usage: Dict[ResourceType, float] = field(default_factory=dict)
    
    # Task tracking
    active_tasks: Set[str] = field(default_factory=set)
    task_history: deque = field(default_factory=lambda: deque(maxlen=100))
    
    # Health metrics
    last_heartbeat: float = field(default_factory=time.time)
    failure_count: int = 0
    recovery_time: Optional[float] = None
    
    @property
    def is_available(self) -> bool:
        """Check if worker is available for new tasks."""
        return (
            self.status in {WorkerStatus.IDLE, WorkerStatus.BUSY} and
            self.current_load < self.capacity and
            self.failure_count < 3
        )
    
    @property
    def success_rate(self) -> float:
        """Calculate worker success rate."""
        if self.total_tasks == 0:
            return 1.0
        return self.completed_tasks / self.total_tasks
    
    @property
    def utilization(self) -> float:
        """Calculate worker utilization."""
        return self.current_load / self.capacity if self.capacity > 0 else 0.0
    
    def can_handle_task(self, task: Task) -> bool:
        """Check if worker can handle the given task."""
        if not self.is_available:
            return False
        
        # Check resource requirements
        for resource_type, required in task.resource_requirements.items():
            available = self.resource_capacity.get(resource_type, 0.0)
            current_usage = self.resource_usage.get(resource_type, 0.0)
            
            if current_usage + required > available:
                return False
        
        return True
    
    def assign_task(self, task: Task) -> None:
        """Assign task to worker."""
        self.active_tasks.add(task.task_id)
        self.total_tasks += 1
        
        # Update resource usage
        for resource_type, required in task.resource_requirements.items():
            current = self.resource_usage.get(resource_type, 0.0)
            self.resource_usage[resource_type] = current + required
        
        # Update load
        task_load = task.estimated_duration / 10.0  # Simple heuristic
        self.current_load = min(self.capacity, self.current_load + task_load)
        
        # Update status
        if self.current_load >= self.capacity * 0.9:
            self.status = WorkerStatus.OVERLOADED
        elif self.current_load > 0:
            self.status = WorkerStatus.BUSY
    
    def complete_task(self, task: Task, success: bool, execution_time: float) -> None:
        """Mark task as completed."""
        if task.task_id in self.active_tasks:
            self.active_tasks.remove(task.task_id)
        
        # Update counters
        if success:
            self.completed_tasks += 1
        else:
            self.failed_tasks += 1
            self.failure_count += 1
        
        # Update resource usage
        for resource_type, required in task.resource_requirements.items():
            current = self.resource_usage.get(resource_type, 0.0)
            self.resource_usage[resource_type] = max(0.0, current - required)
        
        # Update load
        task_load = task.estimated_duration / 10.0
        self.current_load = max(0.0, self.current_load - task_load)
        
        # Update response time
        if self.completed_tasks > 0:
            self.average_response_time = (
                (self.average_response_time * (self.completed_tasks - 1) + execution_time) /
                self.completed_tasks
            )
        
        # Update status
        if self.current_load == 0:
            self.status = WorkerStatus.IDLE
        elif self.current_load < self.capacity * 0.9:
            self.status = WorkerStatus.BUSY
        
        # Add to history
        self.task_history.append({
            'task_id': task.task_id,
            'success': success,
            'execution_time': execution_time,
            'timestamp': time.time()
        })
        
        # Reset failure count on success
        if success and self.failure_count > 0:
            self.failure_count = max(0, self.failure_count - 1)


class TaskQueue:
    """Priority-based task queue with dependency management."""
    
    def __init__(self):
        self._priority_queues: Dict[TaskPriority, List[Task]] = {
            priority: [] for priority in TaskPriority
        }
        self._task_map: Dict[str, Task] = {}
        self._dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        self._lock = RLock()
    
    def add_task(self, task: Task) -> None:
        """Add task to queue."""
        with self._lock:
            # Add to task map
            self._task_map[task.task_id] = task
            
            # Add to dependency graph
            for dep_id in task.dependencies:
                self._dependency_graph[dep_id].add(task.task_id)
            
            # Add to priority queue if ready
            if task.is_ready:
                heapq.heappush(
                    self._priority_queues[task.priority],
                    (-task.priority.value, task.created_at, task)
                )
                task.scheduled_at = time.time()
    
    def get_next_task(self, worker: Worker) -> Optional[Task]:
        """Get next available task for worker."""
        with self._lock:
            # Check priority queues from highest to lowest
            for priority in sorted(TaskPriority, key=lambda p: p.value, reverse=True):
                queue = self._priority_queues[priority]
                
                # Find first task that worker can handle
                for i, (_, _, task) in enumerate(queue):
                    if worker.can_handle_task(task):
                        # Remove from queue
                        del queue[i]
                        heapq.heapify(queue)
                        return task
            
            return None
    
    def complete_task(self, task_id: str) -> None:
        """Mark task as completed and update dependencies."""
        with self._lock:
            if task_id in self._task_map:
                # Remove from task map
                task = self._task_map.pop(task_id)
                
                # Update dependent tasks
                for dependent_id in self._dependency_graph.get(task_id, set()):
                    if dependent_id in self._task_map:
                        dependent_task = self._task_map[dependent_id]
                        dependent_task.dependencies.discard(task_id)
                        
                        # Add to queue if now ready
                        if dependent_task.is_ready:
                            heapq.heappush(
                                self._priority_queues[dependent_task.priority],
                                (-dependent_task.priority.value, dependent_task.created_at, dependent_task)
                            )
                            dependent_task.scheduled_at = time.time()
                
                # Clean up dependency graph
                if task_id in self._dependency_graph:
                    del self._dependency_graph[task_id]
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        with self._lock:
            return {
                'total_tasks': len(self._task_map),
                'queued_by_priority': {
                    priority.name: len(queue)
                    for priority, queue in self._priority_queues.items()
                },
                'pending_dependencies': len([
                    task for task in self._task_map.values()
                    if not task.is_ready
                ])
            }


class LoadBalancer:
    """
    Load balancer for work distribution optimization.
    
    Features:
    - Multiple load balancing strategies
    - Priority-based task scheduling
    - Resource-aware task assignment
    - Worker health monitoring
    - Dependency management
    - Performance optimization
    - Adaptive load balancing
    """
    
    def __init__(
        self,
        strategy: LoadBalancingStrategy = LoadBalancingStrategy.ADAPTIVE,
        enable_telemetry: bool = True,
        health_check_interval: float = 5.0
    ):
        self.strategy = strategy
        self.enable_telemetry = enable_telemetry
        self.health_check_interval = health_check_interval
        
        # Core components
        self._task_queue = TaskQueue()
        self._workers: Dict[str, Worker] = {}
        self._telemetry = get_global_telemetry() if enable_telemetry else None
        self._resource_manager = get_global_adaptive_resource_manager()
        
        # Load balancing state
        self._round_robin_index = 0
        self._strategy_weights: Dict[str, float] = {}
        
        # Health monitoring
        self._health_check_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        # Statistics
        self._stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'total_execution_time': 0.0,
            'strategy_usage': defaultdict(int),
            'worker_assignments': defaultdict(int)
        }
        
        # Thread safety
        self._lock = RLock()
    
    async def start(self) -> None:
        """Start load balancer."""
        self._shutdown_event.clear()
        
        if self._health_check_task is None:
            self._health_check_task = asyncio.create_task(self._health_check_loop())
    
    async def stop(self) -> None:
        """Stop load balancer."""
        self._shutdown_event.set()
        
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            self._health_check_task = None
    
    def add_worker(
        self, 
        worker_id: str, 
        capacity: float = 1.0,
        resource_capacity: Optional[Dict[ResourceType, float]] = None
    ) -> None:
        """Add worker to load balancer."""
        with self._lock:
            worker = Worker(
                worker_id=worker_id,
                capacity=capacity,
                resource_capacity=resource_capacity or {}
            )
            self._workers[worker_id] = worker
    
    def remove_worker(self, worker_id: str) -> None:
        """Remove worker from load balancer."""
        with self._lock:
            if worker_id in self._workers:
                worker = self._workers[worker_id]
                worker.status = WorkerStatus.MAINTENANCE
                
                # Don't remove immediately if worker has active tasks
                if not worker.active_tasks:
                    del self._workers[worker_id]
    
    async def submit_task(self, task: Task) -> str:
        """Submit task for execution."""
        with self._lock:
            self._task_queue.add_task(task)
            self._stats['total_tasks'] += 1
        
        # Record telemetry
        if self._telemetry:
            self._telemetry.increment_counter(
                "load_balancer.tasks_submitted",
                tags={"priority": task.priority.name}
            )
        
        return task.task_id
    
    async def execute_task(self, task: Task) -> Any:
        """Execute task using load balancing."""
        # Select worker
        worker = self._select_worker(task)
        
        if not worker:
            raise RuntimeError("No available workers for task execution")
        
        # Assign task to worker
        worker.assign_task(task)
        task.started_at = time.time()
        
        # Record assignment
        with self._lock:
            self._stats['worker_assignments'][worker.worker_id] += 1
            self._stats['strategy_usage'][self.strategy.value] += 1
        
        try:
            # Execute task
            start_time = time.perf_counter()
            
            if task.coro:
                result = await task.coro
            elif task.func:
                if asyncio.iscoroutinefunction(task.func):
                    result = await task.func(*task.args, **task.kwargs)
                else:
                    result = task.func(*task.args, **task.kwargs)
            else:
                raise ValueError("Task has no executable function or coroutine")
            
            execution_time = time.perf_counter() - start_time
            task.completed_at = time.time()
            
            # Update worker and statistics
            worker.complete_task(task, success=True, execution_time=execution_time)
            self._task_queue.complete_task(task.task_id)
            
            with self._lock:
                self._stats['completed_tasks'] += 1
                self._stats['total_execution_time'] += execution_time
            
            # Record telemetry
            if self._telemetry:
                self._telemetry.record_histogram(
                    "load_balancer.task_execution_time",
                    execution_time * 1000,  # Convert to ms
                    tags={
                        "worker_id": worker.worker_id,
                        "priority": task.priority.name,
                        "strategy": self.strategy.value
                    }
                )
            
            return result
            
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            task.completed_at = time.time()
            
            # Update worker and statistics
            worker.complete_task(task, success=False, execution_time=execution_time)
            self._task_queue.complete_task(task.task_id)
            
            with self._lock:
                self._stats['failed_tasks'] += 1
            
            # Record telemetry
            if self._telemetry:
                self._telemetry.increment_counter(
                    "load_balancer.task_failures",
                    tags={
                        "worker_id": worker.worker_id,
                        "error_type": type(e).__name__
                    }
                )
            
            raise e
    
    def _select_worker(self, task: Task) -> Optional[Worker]:
        """Select best worker for task based on strategy."""
        available_workers = [
            worker for worker in self._workers.values()
            if worker.can_handle_task(task)
        ]
        
        if not available_workers:
            return None
        
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_selection(available_workers)
        
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_selection(available_workers)
        
        elif self.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            return self._least_response_time_selection(available_workers)
        
        elif self.strategy == LoadBalancingStrategy.RESOURCE_BASED:
            return self._resource_based_selection(available_workers, task)
        
        elif self.strategy == LoadBalancingStrategy.PRIORITY_BASED:
            return self._priority_based_selection(available_workers, task)
        
        elif self.strategy == LoadBalancingStrategy.ADAPTIVE:
            return self._adaptive_selection(available_workers, task)
        
        else:
            # Default to round robin
            return self._round_robin_selection(available_workers)
    
    def _round_robin_selection(self, workers: List[Worker]) -> Worker:
        """Round robin worker selection."""
        with self._lock:
            worker = workers[self._round_robin_index % len(workers)]
            self._round_robin_index += 1
            return worker
    
    def _least_connections_selection(self, workers: List[Worker]) -> Worker:
        """Select worker with least active connections."""
        return min(workers, key=lambda w: len(w.active_tasks))
    
    def _least_response_time_selection(self, workers: List[Worker]) -> Worker:
        """Select worker with lowest average response time."""
        return min(workers, key=lambda w: w.average_response_time)
    
    def _resource_based_selection(self, workers: List[Worker], task: Task) -> Worker:
        """Select worker based on resource availability."""
        def resource_score(worker: Worker) -> float:
            score = 0.0
            for resource_type, required in task.resource_requirements.items():
                available = worker.resource_capacity.get(resource_type, 0.0)
                current_usage = worker.resource_usage.get(resource_type, 0.0)
                
                if available > 0:
                    utilization = (current_usage + required) / available
                    score += 1.0 - utilization  # Higher score for lower utilization
            
            return score
        
        return max(workers, key=resource_score)
    
    def _priority_based_selection(self, workers: List[Worker], task: Task) -> Worker:
        """Select worker based on task priority and worker performance."""
        def priority_score(worker: Worker) -> float:
            # Base score from success rate
            base_score = worker.success_rate
            
            # Adjust for task priority
            if task.priority == TaskPriority.CRITICAL:
                # Prefer workers with lower current load for critical tasks
                base_score *= (1.0 - worker.utilization)
            
            # Penalty for overloaded workers
            if worker.status == WorkerStatus.OVERLOADED:
                base_score *= 0.5
            
            return base_score
        
        return max(workers, key=priority_score)
    
    def _adaptive_selection(self, workers: List[Worker], task: Task) -> Worker:
        """Adaptive worker selection based on current conditions."""
        # Get system metrics
        try:
            system_metrics = self._resource_manager.get_system_metrics()
            
            # Determine best strategy based on system state
            cpu_usage = system_metrics.get(ResourceType.CPU)
            memory_usage = system_metrics.get(ResourceType.MEMORY)
            
            if cpu_usage and cpu_usage.current_usage > 0.8:
                # High CPU usage - use least connections
                return self._least_connections_selection(workers)
            
            elif memory_usage and memory_usage.current_usage > 0.8:
                # High memory usage - use resource-based selection
                return self._resource_based_selection(workers, task)
            
            elif task.priority in {TaskPriority.HIGH, TaskPriority.CRITICAL}:
                # High priority task - use priority-based selection
                return self._priority_based_selection(workers, task)
            
            else:
                # Normal conditions - use least response time
                return self._least_response_time_selection(workers)
                
        except Exception:
            # Fallback to round robin if adaptive selection fails
            return self._round_robin_selection(workers)
    
    async def _health_check_loop(self) -> None:
        """Health check loop for workers."""
        while not self._shutdown_event.is_set():
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception:
                continue
    
    async def _perform_health_checks(self) -> None:
        """Perform health checks on all workers."""
        current_time = time.time()
        
        with self._lock:
            for worker in self._workers.values():
                # Check for stale workers
                if current_time - worker.last_heartbeat > self.health_check_interval * 3:
                    if worker.status != WorkerStatus.FAILED:
                        worker.status = WorkerStatus.FAILED
                        worker.failure_count += 1
                
                # Check for recovery
                elif worker.status == WorkerStatus.FAILED and worker.failure_count > 0:
                    # Simple recovery logic
                    if current_time - worker.last_heartbeat < self.health_check_interval:
                        worker.status = WorkerStatus.IDLE
                        worker.recovery_time = current_time
    
    def update_worker_heartbeat(self, worker_id: str) -> None:
        """Update worker heartbeat."""
        with self._lock:
            if worker_id in self._workers:
                self._workers[worker_id].last_heartbeat = time.time()
    
    def get_worker_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        with self._lock:
            return {
                'total_workers': len(self._workers),
                'available_workers': len([
                    w for w in self._workers.values() if w.is_available
                ]),
                'worker_details': {
                    worker_id: {
                        'status': worker.status.value,
                        'utilization': worker.utilization,
                        'success_rate': worker.success_rate,
                        'active_tasks': len(worker.active_tasks),
                        'total_tasks': worker.total_tasks
                    }
                    for worker_id, worker in self._workers.items()
                }
            }
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get task queue statistics."""
        return self._task_queue.get_queue_stats()
    
    def get_load_balancer_stats(self) -> Dict[str, Any]:
        """Get comprehensive load balancer statistics."""
        with self._lock:
            return {
                'execution_stats': self._stats.copy(),
                'worker_stats': self.get_worker_stats(),
                'queue_stats': self.get_queue_stats(),
                'current_strategy': self.strategy.value,
                'average_execution_time': (
                    self._stats['total_execution_time'] / max(self._stats['completed_tasks'], 1)
                ),
                'success_rate': (
                    self._stats['completed_tasks'] / max(self._stats['total_tasks'], 1)
                )
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics (alias for get_load_balancer_stats)."""
        return self.get_load_balancer_stats()


# Global load balancer instance
_global_load_balancer: Optional[LoadBalancer] = None


def get_global_load_balancer() -> LoadBalancer:
    """Get the global load balancer instance."""
    global _global_load_balancer
    if _global_load_balancer is None:
        _global_load_balancer = LoadBalancer()
    return _global_load_balancer


# Convenience functions
async def start_load_balancer() -> None:
    """Start load balancer."""
    balancer = get_global_load_balancer()
    await balancer.start()


async def stop_load_balancer() -> None:
    """Stop load balancer."""
    balancer = get_global_load_balancer()
    await balancer.stop()


async def submit_task(
    func: Callable[..., Any],
    *args,
    priority: TaskPriority = TaskPriority.NORMAL,
    estimated_duration: float = 1.0,
    resource_requirements: Optional[Dict[ResourceType, float]] = None,
    **kwargs
) -> Any:
    """Submit task for load-balanced execution."""
    balancer = get_global_load_balancer()
    
    task = Task(
        priority=priority,
        estimated_duration=estimated_duration,
        resource_requirements=resource_requirements or {},
        func=func,
        args=args,
        kwargs=kwargs
    )
    
    return await balancer.execute_task(task)


def add_worker(
    worker_id: str,
    capacity: float = 1.0,
    resource_capacity: Optional[Dict[ResourceType, float]] = None
) -> None:
    """Add worker to load balancer."""
    balancer = get_global_load_balancer()
    balancer.add_worker(worker_id, capacity, resource_capacity)


def get_load_balancer_stats() -> Dict[str, Any]:
    """Get load balancer statistics."""
    balancer = get_global_load_balancer()
    return balancer.get_load_balancer_stats()