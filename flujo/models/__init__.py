"""Public facade for Flujo domain models.

This module is a stable import path for user-facing types.
"""

from ..domain.models import (
    Task,
    Candidate,
    Checklist,
    ChecklistItem,
    SearchNode,
    SearchState,
    PipelineResult,
    StepResult,
    UsageLimits,
    RefinementCheck,
    SuggestionType,
    ConfigChangeDetail,
    PromptModificationDetail,
    ImprovementSuggestion,
    ImprovementReport,
    HumanInteraction,
    PipelineContext,
)
from ..domain.evaluation import EvaluationReport, EvaluationScore

__all__ = [
    "Task",
    "Candidate",
    "Checklist",
    "ChecklistItem",
    "SearchNode",
    "SearchState",
    "PipelineResult",
    "StepResult",
    "UsageLimits",
    "RefinementCheck",
    "SuggestionType",
    "ConfigChangeDetail",
    "PromptModificationDetail",
    "ImprovementSuggestion",
    "ImprovementReport",
    "HumanInteraction",
    "PipelineContext",
    "EvaluationReport",
    "EvaluationScore",
]
