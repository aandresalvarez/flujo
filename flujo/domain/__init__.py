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
    TreeSearchStep,
    MergeStrategy,
    BranchFailureStrategy,
)
from .context_mixins import BaseContext, typed_context
from .models import (
    Task,
    Candidate,
    Checklist,
    ChecklistItem,
    SearchNode,
    SearchState,
    PipelineResult,
    StepResult,
    UsageLimits,
    ExecutedCommandLog,
)
from .agent_result import FlujoAgentResult, FlujoAgentUsage
from .types import HookCallable
from .events import HookPayload
from .backends import ExecutionBackend, StepExecutionRequest
from .processors import AgentProcessors
from .plugins import PluginOutcome, ValidationPlugin
from .validation import Validator, ValidationResult
from .evaluation import (
    EvaluationReport,
    EvaluationScore,
    MultiSignalEvaluator,
    make_multi_signal_evaluator,
)
from .consensus import majority_vote, code_consensus, judge_selection
from .pipeline_validation import ValidationFinding, ValidationReport
from .resources import AppResources
from . import blueprint as blueprint

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
    "TreeSearchStep",
    "MergeStrategy",
    "BranchFailureStrategy",
    "BaseContext",
    "typed_context",
    "mapper",
    # Models
    "Task",
    "Candidate",
    "Checklist",
    "ChecklistItem",
    "SearchNode",
    "SearchState",
    "PipelineResult",
    "StepResult",
    "UsageLimits",
    "ExecutedCommandLog",
    # Agent Results
    "FlujoAgentResult",
    "FlujoAgentUsage",
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
    "EvaluationReport",
    "EvaluationScore",
    "MultiSignalEvaluator",
    "make_multi_signal_evaluator",
    "majority_vote",
    "code_consensus",
    "judge_selection",
    # Resources
    "AppResources",
    "blueprint",
]
