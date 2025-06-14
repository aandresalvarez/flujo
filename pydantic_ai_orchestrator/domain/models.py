"""Domain models for pydantic-ai-orchestrator.""" 

from typing import Any, List, Optional
from pydantic import BaseModel, Field

class Task(BaseModel):
    """Represents a task to be solved by the orchestrator."""
    prompt: str
    metadata: dict[str, Any] = Field(default_factory=dict)

class ChecklistItem(BaseModel):
    """A single item in a checklist for evaluating a solution."""
    description: str = Field(..., description="The criterion to evaluate.")
    passed: Optional[bool] = Field(None, description="Whether the solution passes this criterion.")
    feedback: Optional[str] = Field(None, description="Feedback if the criterion is not met.")

class Checklist(BaseModel):
    """A checklist for evaluating a solution."""
    items: List[ChecklistItem]

class Candidate(BaseModel):
    """Represents a potential solution and its evaluation metadata."""
    solution: str
    score: float
    checklist: Optional[Checklist] = Field(None, description="Checklist evaluation for this candidate.")

    def __repr__(self) -> str:
        return (
            f"<Candidate score={self.score:.2f} solution={self.solution!r} "
            f"checklist_items={len(self.checklist.items) if self.checklist else 0}>"
        )

    def __str__(self) -> str:
        return (
            f"Candidate(score={self.score:.2f}, solution={self.solution!r}, "
            f"checklist_items={len(self.checklist.items) if self.checklist else 0})"
        )


class StepResult(BaseModel):
    """Result of executing a single pipeline step."""

    name: str
    output: Any | None = None
    success: bool = True
    attempts: int = 0
    latency_s: float = 0.0
    token_counts: int = 0
    cost_usd: float = 0.0
    feedback: str | None = None


class PipelineResult(BaseModel):
    """Aggregated result of running a pipeline."""

    step_history: List[StepResult] = Field(default_factory=list)
    total_cost_usd: float = 0.0

class ImprovementSuggestion(BaseModel):
    """A single suggestion from the SelfImprovementAgent."""

    target_step_name: str
    failure_pattern: str
    suggested_change: str
    example_failing_cases: list[str] = Field(default_factory=list)
    suggested_config_change: str | None = None
    suggested_new_test_case: str | None = None


class ImprovementReport(BaseModel):
    """Aggregated improvement suggestions returned by the agent."""

    suggestions: list[ImprovementSuggestion] = Field(default_factory=list)

