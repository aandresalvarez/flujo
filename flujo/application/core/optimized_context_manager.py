"""
Shim module: re-export optimized context manager from the canonical optimization package.

Following FLUJO_TEAM_GUIDE, we avoid duplication by centralizing implementations
under the optimization modules and re-exporting here as a stable import surface.
"""

from .optimization.memory.context_manager import (  # noqa: F401
    ContextStats,
    CachedContext,
    OptimizedContextManager,
    get_global_context_manager,
    optimized_copy_context,
    optimized_merge_context,
    is_context_immutable,
    ManagedContext,
    ContextPool,
)

__all__ = [
    "ContextStats",
    "CachedContext",
    "OptimizedContextManager",
    "get_global_context_manager",
    "optimized_copy_context",
    "optimized_merge_context",
    "is_context_immutable",
    "ManagedContext",
    "ContextPool",
]
