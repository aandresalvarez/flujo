"""Domain models for pydantic-ai-orchestrator.""" 

from pydantic import BaseModel, Field
from typing import List, Optional, Any

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
