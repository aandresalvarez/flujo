"""
Flujo DSL package root.

Only foundational symbols are exposed at the top level to avoid circular import issues.

Advanced DSL constructs (Pipeline, LoopStep, ConditionalStep, ParallelStep, MapStep, etc.)
must be imported from their respective modules:
    from flujo.domain.dsl.pipeline import Pipeline
    from flujo.domain.dsl.loop import LoopStep, MapStep
    from flujo.domain.dsl.conditional import ConditionalStep
    from flujo.domain.dsl.parallel import ParallelStep
    from flujo.domain.dsl.cache_step import CacheStep

This avoids import cycles and ensures robust usage.
"""

from typing import TYPE_CHECKING
from .step import StepConfig, Step, step, adapter_step

if TYPE_CHECKING:
    from .pipeline import Pipeline
    from .loop import LoopStep, MapStep
    from .conditional import ConditionalStep
    from .parallel import ParallelStep
    from .tree_search import TreeSearchStep
    from .state_machine import StateMachineStep
    from .step import MergeStrategy, BranchFailureStrategy, BranchKey, HumanInTheLoopStep
    from .dynamic_router import DynamicParallelRouterStep
    from .granular import GranularStep, ResumeError
    from .cache_step import CacheStep

__all__ = [
    "StepConfig",
    "Step",
    "step",
    "adapter_step",
    "Pipeline",
    "LoopStep",
    "MapStep",
    "ConditionalStep",
    "ParallelStep",
    "TreeSearchStep",
    "StateMachineStep",
    "MergeStrategy",
    "BranchFailureStrategy",
    "BranchKey",
    "HumanInTheLoopStep",
    "DynamicParallelRouterStep",
    "GranularStep",
    "ResumeError",
    "CacheStep",
]

# Lazy import registry: maps symbol name to (module, attribute_name)
# This pattern allows convenient top-level imports while deferring module loading
# to avoid circular import issues.
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # Pipeline
    "Pipeline": (".pipeline", "Pipeline"),
    # Loop constructs
    "LoopStep": (".loop", "LoopStep"),
    "MapStep": (".loop", "MapStep"),
    # Control flow
    "ConditionalStep": (".conditional", "ConditionalStep"),
    "ParallelStep": (".parallel", "ParallelStep"),
    "TreeSearchStep": (".tree_search", "TreeSearchStep"),
    "StateMachineStep": (".state_machine", "StateMachineStep"),
    "DynamicParallelRouterStep": (".dynamic_router", "DynamicParallelRouterStep"),
    # Step utilities (re-exported from step.py for convenience)
    "MergeStrategy": (".step", "MergeStrategy"),
    "BranchFailureStrategy": (".step", "BranchFailureStrategy"),
    "BranchKey": (".step", "BranchKey"),
    "HumanInTheLoopStep": (".step", "HumanInTheLoopStep"),
    # Granular execution
    "GranularStep": (".granular", "GranularStep"),
    "ResumeError": (".granular", "ResumeError"),
    "CacheStep": (".cache_step", "CacheStep"),
    # Visualization - moved to flujo.visualization
}


def __getattr__(name: str) -> object:
    """Lazy import handler for DSL symbols.

    This pattern enables `from flujo.domain.dsl import Pipeline, Step, ...`
    while deferring module loading to avoid circular import issues.

    The lookup table above documents all available symbols and their sources.
    """
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        # Dynamic import using importlib for cleaner resolution
        import importlib

        module = importlib.import_module(module_path, package=__name__)
        if not attr_name:
            value = module
        else:
            value = getattr(module, attr_name)
        # Cache in globals for subsequent access
        globals()[name] = value
        return value
    raise AttributeError(f"module 'flujo.domain.dsl' has no attribute '{name}'")
