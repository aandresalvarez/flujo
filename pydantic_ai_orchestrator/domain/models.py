"""Domain models for pydantic-ai-orchestrator.""" 

from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Any

class Task(BaseModel):
    """User-supplied job to solve."""
    prompt: str
    metadata: dict = Field(default_factory=dict)

class ChecklistItem(BaseModel):
    description: str
    passed: Optional[bool] = None
    feedback: Optional[str] = None     # reason if failed

class ChecklistItemWeight(BaseModel):
    """
    Represents a checklist item and its associated weight for scoring.
    """
    item: str
    weight: float
    __slots__ = ("item", "weight")

class Checklist(BaseModel):
    items: List[ChecklistItem]
    __slots__ = ("items",)

class Candidate(BaseModel):
    solution: str
    score: float
    passed: List[str]
    failed: List[str] 