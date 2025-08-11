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

__all__ = [
    # Memory optimization
    "OptimizedObjectPool",
    "OptimizedContextManager",
    # Performance optimization
    "OptimizedStepExecutor",
    "AlgorithmOptimizations",
]
