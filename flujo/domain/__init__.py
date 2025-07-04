"""
Domain components for flujo.
"""

from .dsl import (
    Step,
    step,
    adapter_step,
    Pipeline,
    StepConfig,
    MapStep,
    ParallelStep,
    MergeStrategy,
    BranchFailureStrategy,
)
from .models import (
    Task,
    Candidate,
    Checklist,
    ChecklistItem,
    PipelineResult,
    StepResult,
    UsageLimits,
)
from .types import HookCallable
from .events import HookPayload
from .backends import ExecutionBackend, StepExecutionRequest
from .processors import AgentProcessors
from .plugins import PluginOutcome, ValidationPlugin
from .validation import Validator, ValidationResult
from .pipeline_validation import ValidationFinding, ValidationReport
from .resources import AppResources

# ``mapper`` alias preserved for backwards compatibility
mapper = Step.from_mapper

__all__ = [
    # Pipeline DSL
    "Step",
    "step",
    "adapter_step",
    "Pipeline",
    "StepConfig",
    "MapStep",
    "ParallelStep",
    "MergeStrategy",
    "BranchFailureStrategy",
    "mapper",
    # Models
    "Task",
    "Candidate",
    "Checklist",
    "ChecklistItem",
    "PipelineResult",
    "StepResult",
    "UsageLimits",
    # Types and Events
    "HookCallable",
    "HookPayload",
    # Backends
    "ExecutionBackend",
    "StepExecutionRequest",
    # Processors and Validation
    "AgentProcessors",
    "PluginOutcome",
    "ValidationPlugin",
    "Validator",
    "ValidationResult",
    "ValidationFinding",
    "ValidationReport",
    # Resources
    "AppResources",
]
