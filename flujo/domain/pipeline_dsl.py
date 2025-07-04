"""
DEPRECATED: Use 'flujo.domain.dsl' instead of 'flujo.domain.pipeline_dsl'.
This module is retained only for backward compatibility after the FSD-1 refactor.
All core DSL symbols have moved to the dedicated 'flujo.domain.dsl' package.
"""

import warnings
from flujo.domain.dsl import (
    StepConfig,
    Step,
    Pipeline,
    LoopStep,
    ConditionalStep,
    ParallelStep,
    MapStep,
    MergeStrategy,
    BranchFailureStrategy,
    BranchKey,
    step,
    adapter_step,
    HumanInTheLoopStep,
)

warnings.warn(
    "'flujo.domain.pipeline_dsl' is deprecated – use 'flujo.domain.dsl' instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Legacy alias
mapper = Step.from_mapper

__all__ = [
    "StepConfig",
    "Step",
    "Pipeline",
    "LoopStep",
    "ConditionalStep",
    "ParallelStep",
    "MapStep",
    "MergeStrategy",
    "BranchFailureStrategy",
    "BranchKey",
    "step",
    "adapter_step",
    "mapper",
    "HumanInTheLoopStep",
]
