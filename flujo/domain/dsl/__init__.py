"""
Flujo DSL package root.

Only foundational symbols are exposed at the top level to avoid circular import issues.

Advanced DSL constructs (Pipeline, LoopStep, ConditionalStep, ParallelStep, MapStep, etc.)
must be imported from their respective modules:
    from flujo.domain.dsl.pipeline import Pipeline
    from flujo.domain.dsl.loop import LoopStep, MapStep
    from flujo.domain.dsl.conditional import ConditionalStep
    from flujo.domain.dsl.parallel import ParallelStep

This avoids import cycles and ensures robust usage.
"""

from typing import Any
from .step import StepConfig, Step, step, adapter_step

__all__ = [
    "StepConfig",
    "Step",
    "step",
    "adapter_step",
]

# Lazy import pattern for all other symbols


def __getattr__(name: str) -> Any:
    if name == "Pipeline":
        from .pipeline import Pipeline

        globals()[name] = Pipeline
        return Pipeline
    if name == "LoopStep":
        from .loop import LoopStep

        globals()[name] = LoopStep
        return LoopStep
    if name == "MapStep":
        from .loop import MapStep

        globals()[name] = MapStep
        return MapStep
    if name == "ConditionalStep":
        from .conditional import ConditionalStep

        globals()[name] = ConditionalStep
        return ConditionalStep
    if name == "ParallelStep":
        from .parallel import ParallelStep

        globals()[name] = ParallelStep
        return ParallelStep
    if name == "MergeStrategy":
        from .step import MergeStrategy

        globals()[name] = MergeStrategy
        return MergeStrategy
    if name == "BranchFailureStrategy":
        from .step import BranchFailureStrategy

        globals()[name] = BranchFailureStrategy
        return BranchFailureStrategy
    if name == "BranchKey":
        from .step import BranchKey

        globals()[name] = BranchKey
        return BranchKey
    if name == "HumanInTheLoopStep":
        from .step import HumanInTheLoopStep

        globals()[name] = HumanInTheLoopStep
        return HumanInTheLoopStep
    if name == "DynamicParallelRouterStep":
        from .dynamic_router import DynamicParallelRouterStep

        globals()[name] = DynamicParallelRouterStep
        return DynamicParallelRouterStep
    raise AttributeError(f"module 'flujo.domain.dsl' has no attribute '{name}'")
