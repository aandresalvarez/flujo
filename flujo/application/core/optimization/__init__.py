"""
Optimization components for ExecutorCore.

This package contains all performance optimization components including:
- Memory optimization (object pooling, context management)
- Performance optimization (step execution, algorithm optimizations)
- Scalability optimization (resource management, load balancing)
- Monitoring optimization (telemetry, performance monitoring)
"""

from .memory import OptimizedObjectPool, OptimizedContextManager
from .performance import OptimizedStepExecutor, AlgorithmOptimizations
from .config import (
    OptimizationConfig,
    coerce_optimization_config,
    export_config,
    get_config_manager,
    get_optimization_stats,
    get_performance_recommendations,
)

__all__ = [
    # Memory optimization
    "OptimizedObjectPool",
    "OptimizedContextManager",
    # Performance optimization
    "OptimizedStepExecutor",
    "AlgorithmOptimizations",
    # Configuration
    "OptimizationConfig",
    "coerce_optimization_config",
    "export_config",
    "get_config_manager",
    "get_optimization_stats",
    "get_performance_recommendations",
]
