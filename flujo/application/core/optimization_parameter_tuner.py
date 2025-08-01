"""
Optimization parameter tuner for ExecutorCore optimization components.

This module provides intelligent parameter tuning based on performance benchmarks,
system characteristics, and optimization targets. It automatically adjusts
object pool sizes, cache configurations, concurrency limits, telemetry sampling
rates, and adaptive resource management thresholds.
"""

import asyncio
import json
import math
import multiprocessing
import os
import psutil
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple, Callable
from threading import RLock
from enum import Enum

from .optimization.memory.object_pool import OptimizedObjectPool
from .optimization.memory.context_manager import OptimizedContextManager
from .optimized_telemetry import OptimizedTelemetry
from .adaptive_resource_manager import AdaptiveResourceManager, ResourceType, AdaptationStrategy
from .performance_monitor import PerformanceMonitor


class TuningStrategy(Enum):
    """Parameter tuning strategies."""
    CONSERVATIVE = "conservative"  # Safe, gradual tuning
    BALANCED = "balanced"         # Balanced performance/stability
    AGGRESSIVE = "aggressive"     # Maximum performance focus
    ADAPTIVE = "adaptive"         # Dynamic based on system state


class ParameterCategory(Enum):
    """Categories of optimization parameters."""
    OBJECT_POOL = "object_pool"
    CONTEXT_MANAGER = "context_manager"
    TELEMETRY = "telemetry"
    CONCURRENCY = "concurrency"
    CACHE = "cache"
    RESOURCE_MANAGEMENT = "resource_management"


@dataclass
class ParameterSpec:
    """Specification for a tunable parameter."""
    name: str
    category: ParameterCategory
    current_value: Any
    min_value: Any
    max_value: Any
    step_size: Any
    description: str
    impact_weight: float = 1.0  # How much this parameter affects performance
    
    def validate_value(self, value: Any) -> bool:
        """Validate if a value is within acceptable range."""
        try:
            if isinstance(self.min_value, (int, float)) and isinstance(self.max_value, (int, float)):
                return self.min_value <= value <= self.max_value
            return True
        except (TypeError, ValueError):
            return True
    
    def get_next_values(self, direction: str = "both") -> List[Any]:
        """Get next values to try for tuning."""
        values = []
        
        if isinstance(self.current_value, (int, float)) and isinstance(self.step_size, (int, float)):
            if direction in ["up", "both"]:
                next_up = self.current_value + self.step_size
                if self.validate_value(next_up):
                    values.append(next_up)
            
            if direction in ["down", "both"]:
                next_down = self.current_value - self.step_size
                if self.validate_value(next_down):
                    values.append(next_down)
        
        return values


@dataclass
class TuningResult:
    """Result of parameter tuning."""
    parameter_name: str
    old_value: Any
    new_value: Any
    performance_improvement: float
    memory_improvement: float
    stability_score: float
    confidence: float
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class SystemProfile:
    """System characteristics profile for tuning."""
    cpu_count: int
    memory_gb: float
    cpu_frequency_mhz: float
    memory_bandwidth_gbps: float
    cache_sizes: Dict[str, int]
    
    @classmethod
    def detect_system(cls) -> 'SystemProfile':
        """Detect current system characteristics."""
        cpu_count = multiprocessing.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Try to get CPU frequency
        try:
            cpu_freq = psutil.cpu_freq()
            cpu_frequency_mhz = cpu_freq.current if cpu_freq else 2400.0
        except (AttributeError, OSError):
            cpu_frequency_mhz = 2400.0  # Default assumption
        
        # Estimate memory bandwidth (rough heuristic)
        memory_bandwidth_gbps = min(memory_gb * 10, 100.0)  # Rough estimate
        
        # Default cache sizes (would need platform-specific detection)
        cache_sizes = {
            "L1": 32 * 1024,      # 32KB
            "L2": 256 * 1024,     # 256KB
            "L3": 8 * 1024 * 1024  # 8MB
        }
        
        return cls(
            cpu_count=cpu_count,
            memory_gb=memory_gb,
            cpu_frequency_mhz=cpu_frequency_mhz,
            memory_bandwidth_gbps=memory_bandwidth_gbps,
            cache_sizes=cache_sizes
        )


class OptimizationParameterTuner:
    """
    Intelligent parameter tuner for optimization components.
    
    Features:
    - System-aware parameter optimization
    - Performance-guided tuning
    - Multi-objective optimization (performance, memory, stability)
    - Adaptive tuning strategies
    - Safe parameter exploration
    - Rollback capabilities
    """
    
    def __init__(
        self,
        strategy: TuningStrategy = TuningStrategy.BALANCED,
        max_tuning_iterations: int = 50,
        performance_threshold: float = 0.05,  # 5% minimum improvement
        stability_threshold: float = 0.95,    # 95% stability required
        enable_telemetry: bool = True
    ):
        self.strategy = strategy
        self.max_tuning_iterations = max_tuning_iterations
        self.performance_threshold = performance_threshold
        self.stability_threshold = stability_threshold
        self.enable_telemetry = enable_telemetry
        
        # System profile
        self.system_profile = SystemProfile.detect_system()
        
        # Parameter specifications
        self.parameters: Dict[str, ParameterSpec] = {}
        self._initialize_parameters()
        
        # Tuning history
        self.tuning_history: List[TuningResult] = []
        self.best_configurations: Dict[str, Any] = {}
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor() if enable_telemetry else None
        
        # Thread safety
        self._lock = RLock()
        
        # Tuning state
        self._current_iteration = 0
        self._baseline_metrics: Optional[Dict[str, float]] = None
    
    def _initialize_parameters(self) -> None:
        """Initialize parameter specifications based on system profile."""
        cpu_count = self.system_profile.cpu_count
        memory_gb = self.system_profile.memory_gb
        
        # Object Pool Parameters
        self.parameters.update({
            "object_pool_max_size": ParameterSpec(
                name="object_pool_max_size",
                category=ParameterCategory.OBJECT_POOL,
                current_value=1000,
                min_value=100,
                max_value=min(10000, int(memory_gb * 1000)),
                step_size=100,
                description="Maximum size of object pools",
                impact_weight=0.8
            ),
            "object_pool_max_pools": ParameterSpec(
                name="object_pool_max_pools",
                category=ParameterCategory.OBJECT_POOL,
                current_value=100,
                min_value=10,
                max_value=min(500, cpu_count * 50),
                step_size=10,
                description="Maximum number of object pools",
                impact_weight=0.6
            ),
            "object_pool_cleanup_threshold": ParameterSpec(
                name="object_pool_cleanup_threshold",
                category=ParameterCategory.OBJECT_POOL,
                current_value=0.8,
                min_value=0.5,
                max_value=0.95,
                step_size=0.05,
                description="Cleanup threshold for object pools",
                impact_weight=0.4
            )
        })
        
        # Context Manager Parameters
        self.parameters.update({
            "context_cache_size": ParameterSpec(
                name="context_cache_size",
                category=ParameterCategory.CONTEXT_MANAGER,
                current_value=1024,
                min_value=128,
                max_value=min(8192, int(memory_gb * 512)),
                step_size=128,
                description="Context cache size",
                impact_weight=0.9
            ),
            "context_merge_cache_size": ParameterSpec(
                name="context_merge_cache_size",
                category=ParameterCategory.CONTEXT_MANAGER,
                current_value=512,
                min_value=64,
                max_value=min(4096, int(memory_gb * 256)),
                step_size=64,
                description="Context merge cache size",
                impact_weight=0.7
            ),
            "context_ttl_seconds": ParameterSpec(
                name="context_ttl_seconds",
                category=ParameterCategory.CONTEXT_MANAGER,
                current_value=3600,
                min_value=300,
                max_value=7200,
                step_size=300,
                description="Context cache TTL in seconds",
                impact_weight=0.5
            )
        })
        
        # Telemetry Parameters
        self.parameters.update({
            "telemetry_sampling_rate": ParameterSpec(
                name="telemetry_sampling_rate",
                category=ParameterCategory.TELEMETRY,
                current_value=1.0,
                min_value=0.01,
                max_value=1.0,
                step_size=0.1,
                description="Telemetry sampling rate",
                impact_weight=0.3
            ),
            "telemetry_batch_size": ParameterSpec(
                name="telemetry_batch_size",
                category=ParameterCategory.TELEMETRY,
                current_value=100,
                min_value=10,
                max_value=1000,
                step_size=50,
                description="Telemetry batch processing size",
                impact_weight=0.4
            ),
            "telemetry_flush_interval": ParameterSpec(
                name="telemetry_flush_interval",
                category=ParameterCategory.TELEMETRY,
                current_value=5.0,
                min_value=1.0,
                max_value=30.0,
                step_size=2.0,
                description="Telemetry flush interval in seconds",
                impact_weight=0.3
            )
        })
        
        # Concurrency Parameters
        self.parameters.update({
            "max_concurrent_executions": ParameterSpec(
                name="max_concurrent_executions",
                category=ParameterCategory.CONCURRENCY,
                current_value=cpu_count * 2,
                min_value=1,
                max_value=cpu_count * 8,
                step_size=max(1, cpu_count // 2),
                description="Maximum concurrent executions",
                impact_weight=1.0
            ),
            "concurrency_adaptation_rate": ParameterSpec(
                name="concurrency_adaptation_rate",
                category=ParameterCategory.CONCURRENCY,
                current_value=0.1,
                min_value=0.01,
                max_value=0.5,
                step_size=0.05,
                description="Concurrency adaptation rate",
                impact_weight=0.6
            )
        })
        
        # Cache Parameters
        self.parameters.update({
            "cache_max_size": ParameterSpec(
                name="cache_max_size",
                category=ParameterCategory.CACHE,
                current_value=1000,
                min_value=100,
                max_value=min(10000, int(memory_gb * 500)),
                step_size=100,
                description="Maximum cache size",
                impact_weight=0.8
            ),
            "cache_ttl_seconds": ParameterSpec(
                name="cache_ttl_seconds",
                category=ParameterCategory.CACHE,
                current_value=3600,
                min_value=300,
                max_value=7200,
                step_size=300,
                description="Cache TTL in seconds",
                impact_weight=0.5
            )
        })
        
        # Resource Management Parameters
        self.parameters.update({
            "resource_monitoring_interval": ParameterSpec(
                name="resource_monitoring_interval",
                category=ParameterCategory.RESOURCE_MANAGEMENT,
                current_value=1.0,
                min_value=0.1,
                max_value=10.0,
                step_size=0.5,
                description="Resource monitoring interval in seconds",
                impact_weight=0.4
            ),
            "resource_adaptation_interval": ParameterSpec(
                name="resource_adaptation_interval",
                category=ParameterCategory.RESOURCE_MANAGEMENT,
                current_value=5.0,
                min_value=1.0,
                max_value=30.0,
                step_size=2.0,
                description="Resource adaptation interval in seconds",
                impact_weight=0.5
            )
        })
    
    def get_optimized_parameters(self) -> Dict[str, Any]:
        """Get current optimized parameter values."""
        with self._lock:
            return {name: spec.current_value for name, spec in self.parameters.items()}
    
    def set_parameter(self, name: str, value: Any) -> bool:
        """Set a parameter value with validation."""
        with self._lock:
            if name not in self.parameters:
                return False
            
            spec = self.parameters[name]
            if not spec.validate_value(value):
                return False
            
            spec.current_value = value
            return True
    
    def get_system_optimized_defaults(self) -> Dict[str, Any]:
        """Get system-optimized default parameters."""
        cpu_count = self.system_profile.cpu_count
        memory_gb = self.system_profile.memory_gb
        
        # Calculate optimized defaults based on system characteristics
        defaults = {}
        
        # Object Pool Optimization
        # Scale pool sizes with available memory
        memory_factor = min(memory_gb / 8.0, 4.0)  # Scale up to 4x for systems with 32GB+
        defaults["object_pool_max_size"] = int(500 * memory_factor)
        defaults["object_pool_max_pools"] = min(200, cpu_count * 25)
        defaults["object_pool_cleanup_threshold"] = 0.75  # More aggressive cleanup
        
        # Context Manager Optimization
        # Scale cache sizes with memory and CPU
        cache_factor = min(memory_gb / 4.0, 8.0)
        defaults["context_cache_size"] = int(512 * cache_factor)
        defaults["context_merge_cache_size"] = int(256 * cache_factor)
        defaults["context_ttl_seconds"] = 1800  # Shorter TTL for better memory usage
        
        # Telemetry Optimization
        # Reduce sampling for better performance
        if self.strategy == TuningStrategy.AGGRESSIVE:
            defaults["telemetry_sampling_rate"] = 0.1  # 10% sampling
        elif self.strategy == TuningStrategy.BALANCED:
            defaults["telemetry_sampling_rate"] = 0.5  # 50% sampling
        else:
            defaults["telemetry_sampling_rate"] = 1.0   # Full sampling
        
        defaults["telemetry_batch_size"] = min(500, cpu_count * 50)
        defaults["telemetry_flush_interval"] = 2.0  # More frequent flushing
        
        # Concurrency Optimization
        # Scale with CPU count but consider memory constraints
        if memory_gb >= 16:
            defaults["max_concurrent_executions"] = cpu_count * 4
        elif memory_gb >= 8:
            defaults["max_concurrent_executions"] = cpu_count * 3
        else:
            defaults["max_concurrent_executions"] = cpu_count * 2
        
        defaults["concurrency_adaptation_rate"] = 0.15  # Moderate adaptation
        
        # Cache Optimization
        defaults["cache_max_size"] = int(1000 * memory_factor)
        defaults["cache_ttl_seconds"] = 1800  # Shorter TTL
        
        # Resource Management Optimization
        defaults["resource_monitoring_interval"] = 0.5  # More frequent monitoring
        defaults["resource_adaptation_interval"] = 3.0  # Faster adaptation
        
        return defaults
    
    def apply_optimized_parameters(self) -> Dict[str, bool]:
        """Apply system-optimized parameters."""
        optimized_params = self.get_system_optimized_defaults()
        results = {}
        
        for name, value in optimized_params.items():
            results[name] = self.set_parameter(name, value)
        
        return results
    
    async def benchmark_configuration(
        self,
        config: Dict[str, Any],
        benchmark_func: Callable,
        iterations: int = 10
    ) -> Dict[str, float]:
        """Benchmark a parameter configuration."""
        # Apply configuration
        old_values = {}
        for name, value in config.items():
            if name in self.parameters:
                old_values[name] = self.parameters[name].current_value
                self.set_parameter(name, value)
        
        try:
            # Run benchmark
            times = []
            memory_usage = []
            
            for _ in range(iterations):
                start_time = time.perf_counter()
                start_memory = psutil.Process().memory_info().rss / 1024 / 1024
                
                try:
                    if asyncio.iscoroutinefunction(benchmark_func):
                        await benchmark_func()
                    else:
                        benchmark_func()
                except Exception:
                    continue
                
                end_time = time.perf_counter()
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024
                
                times.append(end_time - start_time)
                memory_usage.append(end_memory - start_memory)
            
            if not times:
                return {"performance": float('inf'), "memory": float('inf'), "stability": 0.0}
            
            # Calculate metrics
            avg_time = sum(times) / len(times)
            avg_memory = sum(memory_usage) / len(memory_usage)
            stability = len(times) / iterations  # Success rate
            
            return {
                "performance": avg_time,
                "memory": avg_memory,
                "stability": stability
            }
        
        finally:
            # Restore old values
            for name, value in old_values.items():
                self.set_parameter(name, value)
    
    async def tune_parameter_category(
        self,
        category: ParameterCategory,
        benchmark_func: Callable,
        max_iterations: int = 20
    ) -> List[TuningResult]:
        """Tune parameters in a specific category."""
        results = []
        category_params = {
            name: spec for name, spec in self.parameters.items()
            if spec.category == category
        }
        
        if not category_params:
            return results
        
        # Get baseline performance
        baseline_config = {name: spec.current_value for name, spec in category_params.items()}
        baseline_metrics = await self.benchmark_configuration(baseline_config, benchmark_func)
        
        # Try different parameter combinations
        for iteration in range(max_iterations):
            # Select parameter to tune (weighted by impact)
            param_weights = [spec.impact_weight for spec in category_params.values()]
            total_weight = sum(param_weights)
            
            if total_weight == 0:
                break
            
            # Simple greedy selection (could be improved with more sophisticated algorithms)
            best_param = None
            best_improvement = 0
            
            for name, spec in category_params.items():
                # Try different values for this parameter
                for new_value in spec.get_next_values():
                    test_config = baseline_config.copy()
                    test_config[name] = new_value
                    
                    test_metrics = await self.benchmark_configuration(test_config, benchmark_func)
                    
                    # Calculate improvement (lower is better for performance and memory)
                    perf_improvement = (baseline_metrics["performance"] - test_metrics["performance"]) / baseline_metrics["performance"]
                    memory_improvement = (baseline_metrics["memory"] - test_metrics["memory"]) / max(baseline_metrics["memory"], 0.001)
                    
                    # Combined score (weighted)
                    combined_improvement = (perf_improvement * 0.7 + memory_improvement * 0.3)
                    
                    if (combined_improvement > best_improvement and 
                        test_metrics["stability"] >= self.stability_threshold):
                        best_improvement = combined_improvement
                        best_param = (name, new_value, test_metrics)
            
            # Apply best improvement if significant
            if best_param and best_improvement >= self.performance_threshold:
                name, new_value, metrics = best_param
                old_value = self.parameters[name].current_value
                
                self.set_parameter(name, new_value)
                baseline_config[name] = new_value
                baseline_metrics = metrics
                
                # Record result
                result = TuningResult(
                    parameter_name=name,
                    old_value=old_value,
                    new_value=new_value,
                    performance_improvement=best_improvement,
                    memory_improvement=(baseline_metrics["memory"] - metrics["memory"]) / max(baseline_metrics["memory"], 0.001),
                    stability_score=metrics["stability"],
                    confidence=min(best_improvement / self.performance_threshold, 1.0)
                )
                results.append(result)
                
                with self._lock:
                    self.tuning_history.append(result)
            else:
                # No significant improvement found
                break
        
        return results
    
    async def auto_tune_all_parameters(self, benchmark_func: Callable) -> Dict[str, List[TuningResult]]:
        """Automatically tune all parameter categories."""
        all_results = {}
        
        # Define tuning order (most impactful first)
        tuning_order = [
            ParameterCategory.CONCURRENCY,
            ParameterCategory.OBJECT_POOL,
            ParameterCategory.CONTEXT_MANAGER,
            ParameterCategory.CACHE,
            ParameterCategory.TELEMETRY,
            ParameterCategory.RESOURCE_MANAGEMENT
        ]
        
        for category in tuning_order:
            print(f"Tuning {category.value} parameters...")
            results = await self.tune_parameter_category(category, benchmark_func)
            all_results[category.value] = results
            
            if results:
                print(f"  Applied {len(results)} improvements")
                for result in results:
                    print(f"    {result.parameter_name}: {result.old_value} -> {result.new_value} "
                          f"({result.performance_improvement:.1%} improvement)")
            else:
                print(f"  No improvements found for {category.value}")
        
        return all_results
    
    def get_tuning_summary(self) -> Dict[str, Any]:
        """Get summary of tuning results."""
        with self._lock:
            if not self.tuning_history:
                return {"total_improvements": 0, "categories_tuned": 0, "overall_improvement": 0.0}
            
            # Calculate summary statistics
            total_improvements = len(self.tuning_history)
            categories_tuned = len(set(result.parameter_name.split('_')[0] for result in self.tuning_history))
            
            # Calculate overall improvement
            performance_improvements = [r.performance_improvement for r in self.tuning_history if r.performance_improvement > 0]
            overall_improvement = sum(performance_improvements) / len(performance_improvements) if performance_improvements else 0.0
            
            # Get best improvements by category
            category_improvements = {}
            for result in self.tuning_history:
                category = result.parameter_name.split('_')[0]
                if category not in category_improvements or result.performance_improvement > category_improvements[category]:
                    category_improvements[category] = result.performance_improvement
            
            return {
                "total_improvements": total_improvements,
                "categories_tuned": categories_tuned,
                "overall_improvement": overall_improvement,
                "category_improvements": category_improvements,
                "system_profile": asdict(self.system_profile),
                "current_parameters": self.get_optimized_parameters(),
                "tuning_history": [result.to_dict() for result in self.tuning_history[-10:]]  # Last 10 results
            }
    
    def save_tuning_results(self, filename: str = "optimization_tuning_results.json") -> None:
        """Save tuning results to file."""
        summary = self.get_tuning_summary()
        
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"Tuning results saved to {filename}")
    
    def load_tuning_results(self, filename: str = "optimization_tuning_results.json") -> bool:
        """Load tuning results from file."""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            # Apply saved parameters
            if "current_parameters" in data:
                for name, value in data["current_parameters"].items():
                    self.set_parameter(name, value)
            
            return True
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return False


# Global parameter tuner instance
_global_parameter_tuner: Optional[OptimizationParameterTuner] = None


def get_global_parameter_tuner() -> OptimizationParameterTuner:
    """Get the global parameter tuner instance."""
    global _global_parameter_tuner
    if _global_parameter_tuner is None:
        _global_parameter_tuner = OptimizationParameterTuner()
    return _global_parameter_tuner


def apply_system_optimized_parameters() -> Dict[str, bool]:
    """Apply system-optimized parameters globally."""
    tuner = get_global_parameter_tuner()
    return tuner.apply_optimized_parameters()


def get_current_optimization_parameters() -> Dict[str, Any]:
    """Get current optimization parameters."""
    tuner = get_global_parameter_tuner()
    return tuner.get_optimized_parameters()


async def auto_tune_optimization_parameters(benchmark_func: Callable) -> Dict[str, List[TuningResult]]:
    """Auto-tune all optimization parameters."""
    tuner = get_global_parameter_tuner()
    return await tuner.auto_tune_all_parameters(benchmark_func)