from __future__ import annotations

from typing import List, Any, Optional
from pydantic import BaseModel, Field
from .agent_protocol import AgentProtocol


class StepConfig(BaseModel):
    """Configuration options for a pipeline step."""

    max_retries: int = 1


class Step(BaseModel):
    """Represents a single step in a pipeline."""

    name: str
    agent: Optional[Any] = None
    config: StepConfig = Field(default_factory=StepConfig)

    model_config = {
        "arbitrary_types_allowed": True,
    }

    def __init__(self, name: str, agent: Optional[Any] = None, **config: Any) -> None:  # type: ignore[override]
        super().__init__(name=name, agent=agent, config=StepConfig(**config))

    def __rshift__(self, other: Step | "Pipeline") -> "Pipeline":
        if isinstance(other, Step):
            return Pipeline(steps=[self, other])
        if isinstance(other, Pipeline):
            return Pipeline(steps=[self, *other.steps])
        raise TypeError("Can only chain Step with Step or Pipeline")

    @classmethod
    def review(cls, agent: AgentProtocol[Any, Any], **config: Any) -> "Step":
        return cls("review", agent, **config)

    @classmethod
    def solution(cls, agent: AgentProtocol[Any, Any], **config: Any) -> "Step":
        return cls("solution", agent, **config)

    @classmethod
    def validate(cls, agent: AgentProtocol[Any, Any], **config: Any) -> "Step":
        """Construct a validation step using the provided agent."""
        return cls("validate", agent, **config)


class Pipeline(BaseModel):
    """A sequential pipeline of steps."""

    steps: List[Step]

    def __rshift__(self, other: Step | "Pipeline") -> "Pipeline":
        if isinstance(other, Step):
            return Pipeline(steps=[*self.steps, other])
        if isinstance(other, Pipeline):
            return Pipeline(steps=[*self.steps, *other.steps])
        raise TypeError("Can only chain Pipeline with Step or Pipeline")

    def __iter__(self):  # pragma: no cover - convenience
        return iter(self.steps)
