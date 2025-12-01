"""
Optimized step executor with pre-analysis and caching.

This module provides an optimized step execution system with pre-analysis caching,
signature caching, execution statistics tracking, and fast execution paths for
common step patterns to reduce execution overhead.
"""

import asyncio
import inspect
import time
import weakref
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Optional, Set, TypeVar
from threading import RLock
from enum import Enum

from .....domain.models import StepResult
from .....signature_tools import analyze_signature
from ..memory.object_pool import get_global_pool
from ..memory.memory_utils import track_object_creation
from flujo.type_definitions.common import JSONObject

T = TypeVar("T")


class StepComplexity(Enum):
    """Step complexity levels for optimization."""

    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


@dataclass
class StepAnalysis:
    """Pre-computed step analysis for optimization opportunities."""

    # Basic step properties
    step_name: str
    step_type: str
    complexity: StepComplexity

    # Execution requirements
    needs_context: bool = False
    needs_resources: bool = False
    has_processors: bool = False
    has_validators: bool = False
    has_plugins: bool = False
    has_fallback: bool = False

    # Performance characteristics
    is_cacheable: bool = True
    estimated_duration_ms: float = 1.0
    estimated_memory_usage: int = 1024  # bytes

    # Optimization flags
    can_use_fast_path: bool = False
    requires_deep_context_copy: bool = True
    supports_streaming: bool = False

    # Agent analysis
    agent_signature: Optional[JSONObject] = None
    agent_accepts_context: bool = False
    agent_accepts_resources: bool = False
    agent_accepts_stream: bool = False

    # Caching metadata
    analysis_timestamp: float = field(default_factory=time.time)
    cache_hits: int = 0


@dataclass
class ExecutionStats:
    """Runtime execution statistics for performance monitoring."""

    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0

    # Timing statistics
    total_duration_ns: int = 0
    min_duration_ns: float = float("inf")
    max_duration_ns: int = 0

    # Memory statistics
    total_memory_allocated: int = 0
    peak_memory_usage: int = 0

    # Cache statistics
    analysis_cache_hits: int = 0
    analysis_cache_misses: int = 0

    # Error tracking
    error_types: dict[str, int] = field(default_factory=dict)
    last_error: Optional[str] = None

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_executions == 0:
            return 0.0
        return self.successful_executions / self.total_executions

    @property
    def average_duration_ns(self) -> float:
        """Calculate average execution duration."""
        if self.total_executions == 0:
            return 0.0
        return self.total_duration_ns / self.total_executions

    @property
    def cache_hit_rate(self) -> float:
        """Calculate analysis cache hit rate."""
        total_requests = self.analysis_cache_hits + self.analysis_cache_misses
        if total_requests == 0:
            return 0.0
        return self.analysis_cache_hits / total_requests


class StepAnalyzer:
    """Analyzes steps for optimization opportunities."""

    def __init__(self) -> None:
        self._signature_cache: dict[Any, JSONObject] = {}
        self._complexity_cache: dict[str, StepComplexity] = {}
        self._lock = RLock()

    def analyze_step(self, step: Any) -> StepAnalysis:
        """Analyze a step for optimization opportunities."""
        with self._lock:
            step_name = getattr(step, "name", "unknown")
            step_type = type(step).__name__

            # Determine complexity
            complexity = self._determine_complexity(step)

            # Analyze agent
            agent_analysis = self._analyze_agent(step)

            # Analyze processors
            has_processors = self._has_processors(step)

            # Analyze validators
            has_validators = self._has_validators(step)

            # Analyze plugins
            has_plugins = self._has_plugins(step)

            # Analyze fallback
            has_fallback = self._has_fallback(step)

            # Determine optimization opportunities
            can_use_fast_path = self._can_use_fast_path(
                complexity, has_processors, has_validators, has_plugins, has_fallback
            )

            # Estimate performance characteristics
            estimated_duration = self._estimate_duration(
                complexity, has_processors, has_validators, has_plugins
            )
            estimated_memory = self._estimate_memory_usage(complexity, step)

            return StepAnalysis(
                step_name=step_name,
                step_type=step_type,
                complexity=complexity,
                needs_context=agent_analysis.get("needs_context", False),
                needs_resources=agent_analysis.get("needs_resources", False),
                has_processors=has_processors,
                has_validators=has_validators,
                has_plugins=has_plugins,
                has_fallback=has_fallback,
                is_cacheable=self._is_cacheable(step),
                estimated_duration_ms=estimated_duration,
                estimated_memory_usage=estimated_memory,
                can_use_fast_path=can_use_fast_path,
                requires_deep_context_copy=self._requires_deep_context_copy(step),
                supports_streaming=agent_analysis.get("supports_streaming", False),
                agent_signature=agent_analysis.get("signature"),
                agent_accepts_context=agent_analysis.get("accepts_context", False),
                agent_accepts_resources=agent_analysis.get("accepts_resources", False),
                agent_accepts_stream=agent_analysis.get("accepts_stream", False),
            )

    def _determine_complexity(self, step: Any) -> StepComplexity:
        """Determine step complexity level."""
        step_type = type(step).__name__

        # Check cache first
        if step_type in self._complexity_cache:
            return self._complexity_cache[step_type]

        complexity_score = 0

        # Base complexity by type
        if step_type in ["Step"]:
            complexity_score += 1
        elif step_type in ["ParallelStep", "ConditionalStep"]:
            complexity_score += 3
        elif step_type in ["LoopStep", "DynamicParallelRouterStep"]:
            complexity_score += 4
        else:
            complexity_score += 2

        # Additional complexity factors
        if hasattr(step, "processors") and (
            getattr(step.processors, "prompt_processors", [])
            or getattr(step.processors, "output_processors", [])
        ):
            complexity_score += 1

        if hasattr(step, "validators") and step.validators:
            complexity_score += 1

        if hasattr(step, "plugins") and step.plugins:
            complexity_score += 1

        if hasattr(step, "fallback_step") and step.fallback_step:
            complexity_score += 2

        # Determine complexity level
        if complexity_score <= 2:
            complexity = StepComplexity.SIMPLE
        elif complexity_score <= 4:
            complexity = StepComplexity.MODERATE
        elif complexity_score <= 6:
            complexity = StepComplexity.COMPLEX
        else:
            complexity = StepComplexity.VERY_COMPLEX

        # Cache result
        self._complexity_cache[step_type] = complexity
        return complexity

    def _analyze_agent(self, step: Any) -> JSONObject:
        """Analyze step agent for optimization opportunities."""
        agent = getattr(step, "agent", None)
        if not agent:
            return {}

        # Check cache first
        agent_id = id(agent)
        if agent_id in self._signature_cache:
            return self._signature_cache[agent_id]

        analysis: JSONObject = {}

        try:
            # Get the executable function
            executable_func = None
            if hasattr(agent, "run"):
                executable_func = agent.run
            elif callable(agent):
                executable_func = agent

            if executable_func:
                # Analyze signature
                try:
                    signature = analyze_signature(executable_func)
                    analysis["signature"] = signature

                    # Check parameter acceptance
                    sig = inspect.signature(executable_func)
                    params = list(sig.parameters.keys())

                    analysis["accepts_context"] = "context" in params
                    analysis["accepts_resources"] = "resources" in params
                    analysis["accepts_stream"] = "stream" in params or hasattr(agent, "stream")
                    analysis["needs_context"] = analysis["accepts_context"]
                    analysis["needs_resources"] = analysis["accepts_resources"]
                    analysis["supports_streaming"] = analysis["accepts_stream"]

                except Exception:
                    # Fallback analysis
                    analysis["accepts_context"] = hasattr(agent, "run") and "context" in str(
                        inspect.signature(agent.run)
                    )
                    analysis["accepts_resources"] = hasattr(agent, "run") and "resources" in str(
                        inspect.signature(agent.run)
                    )
                    analysis["supports_streaming"] = hasattr(agent, "stream")

        except Exception:
            # Safe fallback
            analysis = {
                "accepts_context": True,  # Conservative assumption
                "accepts_resources": True,
                "supports_streaming": False,
                "needs_context": True,
                "needs_resources": True,
            }

        # Cache result
        self._signature_cache[agent_id] = analysis
        return analysis

    def _has_processors(self, step: Any) -> bool:
        """Check if step has processors."""
        if not hasattr(step, "processors"):
            return False

        processors = step.processors
        return bool(
            getattr(processors, "prompt_processors", [])
            or getattr(processors, "output_processors", [])
        )

    def _has_validators(self, step: Any) -> bool:
        """Check if step has validators."""
        return bool(getattr(step, "validators", []))

    def _has_plugins(self, step: Any) -> bool:
        """Check if step has plugins."""
        return bool(getattr(step, "plugins", []))

    def _has_fallback(self, step: Any) -> bool:
        """Check if step has fallback."""
        return bool(getattr(step, "fallback_step", None))

    def _can_use_fast_path(
        self,
        complexity: StepComplexity,
        has_processors: bool,
        has_validators: bool,
        has_plugins: bool,
        has_fallback: bool,
    ) -> bool:
        """Determine if step can use fast execution path."""
        # Fast path only for simple steps without complex features
        return (
            complexity == StepComplexity.SIMPLE
            and not has_processors
            and not has_validators
            and not has_plugins
            and not has_fallback
        )

    def _estimate_duration(
        self,
        complexity: StepComplexity,
        has_processors: bool,
        has_validators: bool,
        has_plugins: bool,
    ) -> float:
        """Estimate step execution duration in milliseconds."""
        base_duration = {
            StepComplexity.SIMPLE: 1.0,
            StepComplexity.MODERATE: 5.0,
            StepComplexity.COMPLEX: 20.0,
            StepComplexity.VERY_COMPLEX: 100.0,
        }[complexity]

        # Add overhead for additional features
        if has_processors:
            base_duration += 2.0
        if has_validators:
            base_duration += 3.0
        if has_plugins:
            base_duration += 5.0

        return base_duration

    def _estimate_memory_usage(self, complexity: StepComplexity, step: Any) -> int:
        """Estimate memory usage in bytes."""
        base_memory = {
            StepComplexity.SIMPLE: 1024,
            StepComplexity.MODERATE: 4096,
            StepComplexity.COMPLEX: 16384,
            StepComplexity.VERY_COMPLEX: 65536,
        }[complexity]

        # Adjust based on step features
        if hasattr(step, "processors"):
            base_memory += 2048
        if hasattr(step, "validators"):
            base_memory += 1024
        if hasattr(step, "plugins"):
            base_memory += 4096

        return base_memory

    def _is_cacheable(self, step: Any) -> bool:
        """Determine if step results can be cached."""
        # Steps with plugins or validators might not be cacheable
        # due to side effects or non-deterministic behavior
        if hasattr(step, "plugins") and step.plugins:
            return False

        # Steps with certain validators might not be cacheable
        if hasattr(step, "validators") and step.validators:
            # Conservative approach: assume validators might have side effects
            return False

        return True

    def _requires_deep_context_copy(self, step: Any) -> bool:
        """Determine if step requires deep context copying."""
        # Steps that might modify context require deep copying
        return (
            hasattr(step, "plugins")
            and step.plugins
            or hasattr(step, "processors")
            and self._has_processors(step)
        )


class OptimizedStepExecutor:
    """
    Optimized step execution with pre-analysis and caching.

    Features:
    - Step analysis caching for reduced overhead
    - Signature caching for parameter optimization
    - Execution statistics tracking
    - Fast execution paths for simple steps
    - Memory-efficient object reuse
    - Performance monitoring and optimization
    """

    def __init__(
        self,
        enable_analysis_cache: bool = True,
        enable_fast_path: bool = True,
        enable_statistics: bool = True,
        cache_size: int = 1000,
        stats_retention_hours: int = 24,
    ):
        self.enable_analysis_cache = enable_analysis_cache
        self.enable_fast_path = enable_fast_path
        self.enable_statistics = enable_statistics
        self.cache_size = cache_size
        self.stats_retention_hours = stats_retention_hours

        # Core components
        self._analyzer = StepAnalyzer()
        self._object_pool = get_global_pool()

        # Caches
        self._analysis_cache: dict[int, StepAnalysis] = {}
        # Weak references for cleanup
        self._weak_refs: Set[weakref.ref[Any]] = set()
        self._last_cleanup = time.time()

        # Statistics
        self._execution_stats: dict[str, ExecutionStats] = defaultdict(ExecutionStats)
        self._global_stats = ExecutionStats()

        # Performance optimization
        self._fast_path_cache: dict[int, bool] = {}
        self._result_pool: deque[StepResult] = deque(
            maxlen=100
        )  # Pool of reusable StepResult objects
        # Performance tracking
        self._execution_times: deque[float] = deque(maxlen=100)
        self._memory_usage: deque[float] = deque(maxlen=100)

        # Thread safety
        self._lock = RLock()

    async def execute_optimized(
        self,
        step: Any,
        data: Any,
        context: Optional[Any] = None,
        resources: Optional[Any] = None,
        **kwargs: Any,
    ) -> StepResult:
        """
        Execute step with optimizations.

        Args:
            step: Step to execute
            data: Input data
            context: Execution context
            resources: Available resources
            **kwargs: Additional execution parameters

        Returns:
            StepResult with execution outcome
        """
        start_time = time.perf_counter_ns()
        step_name = getattr(step, "name", "unknown")

        try:
            # Track object creation
            track_object_creation(step, f"step_execution_{step_name}")

            # Get or create step analysis
            analysis = await self._get_step_analysis(step)

            # Update statistics
            if self.enable_statistics:
                self._update_execution_start_stats(step_name, analysis)

            # Choose execution path
            if self.enable_fast_path and analysis.can_use_fast_path:
                result = await self._execute_fast_path(
                    step, data, context, resources, analysis, **kwargs
                )
            else:
                result = await self._execute_standard_path(
                    step, data, context, resources, analysis, **kwargs
                )

            # Update success statistics
            execution_time_ns = time.perf_counter_ns() - start_time
            if self.enable_statistics:
                self._update_execution_success_stats(step_name, execution_time_ns, result)

            return result

        except Exception as e:
            # Update failure statistics
            execution_time_ns = time.perf_counter_ns() - start_time
            if self.enable_statistics:
                self._update_execution_failure_stats(step_name, execution_time_ns, e)

            # Create failure result
            result = await self._create_failure_result(step, e)
            return result

    async def _get_step_analysis(self, step: Any) -> StepAnalysis:
        """Get or create step analysis with caching."""
        if not self.enable_analysis_cache:
            return self._analyzer.analyze_step(step)

        step_id = id(step)

        with self._lock:
            # Check cache
            if step_id in self._analysis_cache:
                analysis = self._analysis_cache[step_id]
                analysis.cache_hits += 1

                # Update statistics
                if self.enable_statistics:
                    step_name = getattr(step, "name", "unknown")
                    self._execution_stats[step_name].analysis_cache_hits += 1
                    self._global_stats.analysis_cache_hits += 1

                return analysis

            # Cache miss - analyze step
            analysis = self._analyzer.analyze_step(step)

            # Update statistics
            if self.enable_statistics:
                step_name = getattr(step, "name", "unknown")
                self._execution_stats[step_name].analysis_cache_misses += 1
                self._global_stats.analysis_cache_misses += 1

            # Cache result if there's space
            if len(self._analysis_cache) < self.cache_size:
                self._analysis_cache[step_id] = analysis

                # Track with weak reference for cleanup
                try:
                    weak_ref = weakref.ref(step, lambda ref: self._cleanup_analysis_cache(step_id))
                    self._weak_refs.add(weak_ref)
                except TypeError:
                    # Some objects can't be weakly referenced
                    pass

            return analysis

    async def _execute_fast_path(
        self,
        step: Any,
        data: Any,
        context: Optional[Any],
        resources: Optional[Any],
        analysis: StepAnalysis,
        **kwargs: Any,
    ) -> StepResult:
        """Execute step using optimized fast path."""
        # Fast path for simple steps without complex features
        agent = getattr(step, "agent", None)
        if not agent:
            raise ValueError(f"Step {analysis.step_name} has no agent")

        # Get reusable result object
        result = await self._get_result_object()
        result.name = analysis.step_name
        result.attempts = 1

        try:
            # Minimal parameter preparation
            agent_kwargs = {}
            if analysis.agent_accepts_context and context is not None:
                agent_kwargs["context"] = context
            if analysis.agent_accepts_resources and resources is not None:
                agent_kwargs["resources"] = resources

            # Execute agent directly
            if hasattr(agent, "run"):
                if asyncio.iscoroutinefunction(agent.run):
                    output = await agent.run(data, **agent_kwargs)
                else:
                    output = agent.run(data, **agent_kwargs)
            else:
                if asyncio.iscoroutinefunction(agent):
                    output = await agent(data, **agent_kwargs)
                else:
                    output = agent(data, **agent_kwargs)

            # Set result
            result.output = output
            result.success = True
            result.latency_s = 0.0  # Will be set by caller

            return result

        except Exception as e:
            result.success = False
            result.feedback = f"Fast path execution failed: {str(e)}"
            result.output = None
            raise e

    async def _execute_standard_path(
        self,
        step: Any,
        data: Any,
        context: Optional[Any],
        resources: Optional[Any],
        analysis: StepAnalysis,
        **kwargs: Any,
    ) -> StepResult:
        """Execute step using standard path with full feature support."""
        # This would delegate to the existing ExecutorCore implementation
        # For now, we'll create a basic implementation

        result = await self._get_result_object()
        result.name = analysis.step_name
        result.attempts = 1

        try:
            # Basic execution (would be replaced with full ExecutorCore logic)
            agent = getattr(step, "agent", None)
            if not agent:
                raise ValueError(f"Step {analysis.step_name} has no agent")

            # Prepare parameters
            agent_kwargs = {}
            if context is not None:
                agent_kwargs["context"] = context
            if resources is not None:
                agent_kwargs["resources"] = resources

            # Execute agent
            if hasattr(agent, "run"):
                if asyncio.iscoroutinefunction(agent.run):
                    output = await agent.run(data, **agent_kwargs)
                else:
                    output = agent.run(data, **agent_kwargs)
            else:
                if asyncio.iscoroutinefunction(agent):
                    output = await agent(data, **agent_kwargs)
                else:
                    output = agent(data, **agent_kwargs)

            result.output = output
            result.success = True

            return result

        except Exception as e:
            result.success = False
            result.feedback = f"Standard path execution failed: {str(e)}"
            result.output = None
            raise e

    async def _get_result_object(self) -> StepResult:
        """Get a reusable StepResult object."""
        if self._result_pool:
            result = self._result_pool.popleft()
            # Reset the result
            result.output = None
            result.success = False
            result.feedback = None
            result.attempts = 0
            result.latency_s = 0.0
            result.token_counts = 0
            result.cost_usd = 0.0
            result.branch_context = None
            result.metadata_ = {}
            return result
        else:
            # Create new result
            return StepResult(name="", output=None, success=False)

    async def _return_result_object(self, result: StepResult) -> None:
        """Return a StepResult object to the pool."""
        if self._result_pool.maxlen is None or len(self._result_pool) < self._result_pool.maxlen:
            self._result_pool.append(result)

    async def _create_failure_result(self, step: Any, error: Exception) -> StepResult:
        """Create a failure result."""
        result = await self._get_result_object()
        result.name = getattr(step, "name", "unknown")
        result.success = False
        result.feedback = str(error)
        result.attempts = 1
        return result

    def _update_execution_start_stats(self, step_name: str, analysis: StepAnalysis) -> None:
        """Update statistics at execution start."""
        with self._lock:
            stats = self._execution_stats[step_name]
            stats.total_executions += 1
            self._global_stats.total_executions += 1

    def _update_execution_success_stats(
        self, step_name: str, execution_time_ns: int, result: StepResult
    ) -> None:
        """Update statistics for successful execution."""
        with self._lock:
            stats = self._execution_stats[step_name]
            stats.successful_executions += 1
            stats.total_duration_ns += execution_time_ns
            stats.min_duration_ns = min(stats.min_duration_ns, execution_time_ns)
            stats.max_duration_ns = max(stats.max_duration_ns, execution_time_ns)

            # Update global stats
            self._global_stats.successful_executions += 1
            self._global_stats.total_duration_ns += execution_time_ns
            self._global_stats.min_duration_ns = min(
                self._global_stats.min_duration_ns, execution_time_ns
            )
            self._global_stats.max_duration_ns = max(
                self._global_stats.max_duration_ns, execution_time_ns
            )

    def _update_execution_failure_stats(
        self, step_name: str, execution_time_ns: int, error: Exception
    ) -> None:
        """Update statistics for failed execution."""
        with self._lock:
            stats = self._execution_stats[step_name]
            stats.failed_executions += 1
            stats.total_duration_ns += execution_time_ns

            error_type = type(error).__name__
            stats.error_types[error_type] = stats.error_types.get(error_type, 0) + 1
            stats.last_error = str(error)

            # Update global stats
            self._global_stats.failed_executions += 1
            self._global_stats.total_duration_ns += execution_time_ns
            self._global_stats.error_types[error_type] = (
                self._global_stats.error_types.get(error_type, 0) + 1
            )
            self._global_stats.last_error = str(error)

    def _cleanup_analysis_cache(self, step_id: int) -> None:
        """Clean up analysis cache entry."""
        with self._lock:
            self._analysis_cache.pop(step_id, None)

    def get_step_stats(self, step_name: str) -> Optional[ExecutionStats]:
        """Get execution statistics for a specific step."""
        return self._execution_stats.get(step_name)

    def get_global_stats(self) -> ExecutionStats:
        """Get global execution statistics."""
        return self._global_stats

    def get_analysis_cache_stats(self) -> JSONObject:
        """Get analysis cache statistics."""
        with self._lock:
            return {
                "cache_size": len(self._analysis_cache),
                "max_cache_size": self.cache_size,
                "cache_utilization": len(self._analysis_cache) / self.cache_size,
                "weak_refs_count": len(self._weak_refs),
            }

    def clear_caches(self) -> None:
        """Clear all caches."""
        with self._lock:
            self._analysis_cache.clear()
            self._fast_path_cache.clear()
            self._weak_refs.clear()

    def clear_statistics(self) -> None:
        """Clear execution statistics."""
        with self._lock:
            self._execution_stats.clear()
            self._global_stats = ExecutionStats()


# Global optimized step executor instance
_global_step_executor: Optional[OptimizedStepExecutor] = None


def get_global_step_executor() -> OptimizedStepExecutor:
    """Get the global optimized step executor instance."""
    global _global_step_executor
    if _global_step_executor is None:
        _global_step_executor = OptimizedStepExecutor()
    return _global_step_executor


async def execute_step_optimized(
    step: Any,
    data: Any,
    context: Optional[Any] = None,
    resources: Optional[Any] = None,
    **kwargs: Any,
) -> StepResult:
    """Convenience function to execute step with optimizations."""
    executor = get_global_step_executor()
    return await executor.execute_optimized(step, data, context, resources, **kwargs)


def get_step_execution_stats(step_name: str) -> Optional[ExecutionStats]:
    """Convenience function to get step execution statistics."""
    executor = get_global_step_executor()
    return executor.get_step_stats(step_name)


def get_global_execution_stats() -> ExecutionStats:
    """Convenience function to get global execution statistics."""
    executor = get_global_step_executor()
    return executor.get_global_stats()
