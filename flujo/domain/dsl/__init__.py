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

from typing import TYPE_CHECKING
from .step import StepConfig, Step, step, adapter_step

if TYPE_CHECKING:
    from .pipeline import Pipeline
    from .loop import LoopStep, MapStep
    from .conditional import ConditionalStep
    from .parallel import ParallelStep
    from .state_machine import StateMachineStep
    from .step import MergeStrategy, BranchFailureStrategy, BranchKey, HumanInTheLoopStep
    from .dynamic_router import DynamicParallelRouterStep
    from .granular import GranularStep, ResumeError

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
    "StateMachineStep",
    "MergeStrategy",
    "BranchFailureStrategy",
    "BranchKey",
    "HumanInTheLoopStep",
    "DynamicParallelRouterStep",
    "GranularStep",
    "ResumeError",
]

# Lazy import pattern for all other symbols


def __getattr__(name: str) -> object:
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
    if name == "StateMachineStep":
        from .state_machine import StateMachineStep

        globals()[name] = StateMachineStep
        return StateMachineStep
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
    if name == "GranularStep":
        from .granular import GranularStep

        globals()[name] = GranularStep
        return GranularStep
    if name == "ResumeError":
        from .granular import ResumeError

        globals()[name] = ResumeError
        return ResumeError
    raise AttributeError(f"module 'flujo.domain.dsl' has no attribute '{name}'")
