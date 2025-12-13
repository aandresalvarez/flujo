"""Deprecated OptimizationConfig stub for backward compatibility.

This module provides a deprecated OptimizationConfig class that can still be
imported but does nothing. It exists to maintain backward compatibility with
scripts and examples that use OptimizationConfig.

The optimization layer has been removed. This stub allows existing code to
continue working while emitting deprecation warnings.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping


class OptimizationConfig:
    """
    Deprecated optimization configuration stub.

    The optimization layer has been removed from Flujo. This class exists
    only for backward compatibility. All optimization features are disabled
    and this configuration has no effect.

    .. deprecated:: 0.4.39
        The optimization layer has been removed. This class will be removed
        in a future version. Remove OptimizationConfig usage from your code.

    Examples:
        >>> from flujo.application.core.executor_core import ExecutorCore, OptimizationConfig
        >>> # This will work but emit a deprecation warning
        >>> config = OptimizationConfig()  # doctest: +SKIP
        >>> executor = ExecutorCore(optimization_config=config)  # doctest: +SKIP
    """

    def __init__(
        self,
        enable_object_pool: bool = False,
        enable_context_optimization: bool = False,
        enable_memory_optimization: bool = False,
        enable_optimized_telemetry: bool = False,
        enable_performance_monitoring: bool = False,
        enable_optimized_error_handling: bool = False,
        enable_circuit_breaker: bool = False,
        enable_automatic_optimization: bool = False,
        max_concurrent_executions: int = 10,
        cache_max_size: int = 1024,
        **kwargs: object,
    ) -> None:
        """Initialize a deprecated OptimizationConfig.

        All parameters are ignored. This method exists only for backward
        compatibility and emits a deprecation warning.
        """
        warnings.warn(
            "OptimizationConfig is deprecated. The optimization layer has been "
            "removed from Flujo. This configuration has no effect and will be "
            "removed in a future version. Remove OptimizationConfig usage from "
            "your code.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Store values for compatibility but they're never used
        self.enable_object_pool = enable_object_pool
        self.enable_context_optimization = enable_context_optimization
        self.enable_memory_optimization = enable_memory_optimization
        self.enable_optimized_telemetry = enable_optimized_telemetry
        self.enable_performance_monitoring = enable_performance_monitoring
        self.enable_optimized_error_handling = enable_optimized_error_handling
        self.enable_circuit_breaker = enable_circuit_breaker
        self.enable_automatic_optimization = enable_automatic_optimization
        self.max_concurrent_executions = max_concurrent_executions
        self.cache_max_size = cache_max_size

    def validate(self) -> list[str]:
        """Validate configuration (always returns empty list - no validation needed)."""
        return []

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary (for compatibility)."""
        return {
            "enable_object_pool": self.enable_object_pool,
            "enable_context_optimization": self.enable_context_optimization,
            "enable_memory_optimization": self.enable_memory_optimization,
            "enable_optimized_telemetry": self.enable_optimized_telemetry,
            "enable_performance_monitoring": self.enable_performance_monitoring,
            "enable_optimized_error_handling": self.enable_optimized_error_handling,
            "enable_circuit_breaker": self.enable_circuit_breaker,
            "enable_automatic_optimization": self.enable_automatic_optimization,
            "max_concurrent_executions": self.max_concurrent_executions,
            "cache_max_size": self.cache_max_size,
        }

    @classmethod
    def from_dict(cls, config_dict: Mapping[str, object]) -> "OptimizationConfig":
        """Create from dictionary (for compatibility)."""
        inst = cls()
        for key, value in config_dict.items():
            try:
                if hasattr(inst, key):
                    setattr(inst, key, value)
            except Exception:
                continue
        return inst


__all__ = ["OptimizationConfig"]
