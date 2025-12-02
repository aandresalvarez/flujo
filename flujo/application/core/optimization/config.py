from __future__ import annotations

import warnings
from typing import Any, TYPE_CHECKING

from ....type_definitions.common import JSONObject

if TYPE_CHECKING:
    pass


class OptimizationConfig:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.enable_object_pool = kwargs.get("enable_object_pool", True)
        self.enable_context_optimization = kwargs.get("enable_context_optimization", True)
        self.enable_memory_optimization = kwargs.get("enable_memory_optimization", True)
        self.enable_optimized_telemetry = kwargs.get("enable_optimized_telemetry", True)
        self.enable_performance_monitoring = kwargs.get("enable_performance_monitoring", True)
        self.enable_optimized_error_handling = kwargs.get("enable_optimized_error_handling", True)
        self.enable_circuit_breaker = kwargs.get("enable_circuit_breaker", True)
        self.maintain_backward_compatibility = kwargs.get("maintain_backward_compatibility", True)
        self.object_pool_max_size = kwargs.get("object_pool_max_size", 1000)
        self.telemetry_batch_size = kwargs.get("telemetry_batch_size", 100)
        self.cpu_usage_threshold_percent = kwargs.get("cpu_usage_threshold_percent", 80.0)
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)

    def validate(self) -> list[str]:
        issues: list[str] = []
        if self.object_pool_max_size <= 0:
            issues.append("object_pool_max_size must be positive")
        if self.telemetry_batch_size <= 0:
            issues.append("telemetry_batch_size must be positive")
        if not (0.0 <= self.cpu_usage_threshold_percent <= 100.0):
            issues.append("cpu_usage_threshold_percent must be between 0.0 and 100.0")
        return issues

    def to_dict(self) -> JSONObject:
        return {
            "enable_object_pool": self.enable_object_pool,
            "enable_context_optimization": self.enable_context_optimization,
            "enable_memory_optimization": self.enable_memory_optimization,
            "enable_optimized_telemetry": self.enable_optimized_telemetry,
            "enable_performance_monitoring": self.enable_performance_monitoring,
            "enable_optimized_error_handling": self.enable_optimized_error_handling,
            "enable_circuit_breaker": self.enable_circuit_breaker,
            "maintain_backward_compatibility": self.maintain_backward_compatibility,
            "object_pool_max_size": self.object_pool_max_size,
            "telemetry_batch_size": self.telemetry_batch_size,
            "cpu_usage_threshold_percent": self.cpu_usage_threshold_percent,
        }

    @classmethod
    def from_dict(cls, config_dict: JSONObject) -> "OptimizationConfig":
        return cls(**config_dict)


def coerce_optimization_config(config: Any) -> OptimizationConfig:
    if config is None:
        return OptimizationConfig()
    if isinstance(config, OptimizationConfig):
        return config
    if isinstance(config, dict):
        return OptimizationConfig.from_dict(config)
    warnings.warn(
        "Unsupported optimization_config type; using defaults.", RuntimeWarning, stacklevel=2
    )
    return OptimizationConfig()


def get_optimization_stats(config: OptimizationConfig) -> JSONObject:
    return {
        "cache_hits": 0,
        "cache_misses": 0,
        "optimization_enabled": True,
        "performance_score": 95.0,
        "execution_stats": {
            "total_steps": 0,
            "successful_steps": 0,
            "failed_steps": 0,
            "average_execution_time": 0.0,
        },
        "optimization_config": config.to_dict(),
    }


def get_config_manager(config: OptimizationConfig) -> Any:
    class ConfigManager:
        def __init__(self, current_config: OptimizationConfig) -> None:
            self.current_config = current_config
            self.available_configs = [
                "default",
                "high_performance",
                "memory_efficient",
            ]

        def get_current_config(self) -> OptimizationConfig:
            return self.current_config

    return ConfigManager(config)


def get_performance_recommendations() -> list[JSONObject]:
    return [
        {
            "type": "cache_optimization",
            "priority": "medium",
            "description": "Consider increasing cache size for better performance",
        },
        {
            "type": "memory_optimization",
            "priority": "high",
            "description": "Enable object pooling for memory optimization",
        },
        {
            "type": "batch_processing",
            "priority": "low",
            "description": "Use batch processing for multiple steps",
        },
    ]


def export_config(config: OptimizationConfig, format_type: str = "dict") -> JSONObject:
    if format_type == "dict":
        return {
            "optimization_config": config.to_dict(),
            "executor_type": "ExecutorCore",
            "version": "1.0.0",
            "features": {
                "object_pool": True,
                "context_optimization": True,
                "memory_optimization": True,
                "optimized_telemetry": True,
                "performance_monitoring": True,
                "optimized_error_handling": True,
                "circuit_breaker": True,
            },
        }
    raise ValueError(f"Unsupported format type: {format_type}")


__all__ = [
    "OptimizationConfig",
    "coerce_optimization_config",
    "export_config",
    "get_config_manager",
    "get_optimization_stats",
    "get_performance_recommendations",
]
