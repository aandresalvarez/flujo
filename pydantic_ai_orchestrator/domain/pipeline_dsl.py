from __future__ import annotations

from typing import List, Any, Optional, Callable, Generic, TypeVar
from pydantic import BaseModel, Field
from .agent_protocol import AgentProtocol
from .plugins import ValidationPlugin


InT = TypeVar("InT")
OutT = TypeVar("OutT")


class StepConfig(BaseModel):
    """Configuration options for a pipeline step."""

    max_retries: int = 1
    timeout_s: float | None = None


class Step(BaseModel, Generic[InT, OutT]):
    """Represents a single step in a pipeline."""

    name: str
    agent: Optional[Any] = None
    config: StepConfig = Field(default_factory=StepConfig)
    plugins: List[tuple[ValidationPlugin, int]] = Field(default_factory=list)
    failure_handlers: List[Callable[[], None]] = Field(default_factory=list)

    model_config = {
        "arbitrary_types_allowed": True,
    }

    def __init__(
        self,
        name: str,
        agent: Optional[Any] = None,
        plugins: Optional[List[ValidationPlugin | tuple[ValidationPlugin, int]]] = None,
        on_failure: Optional[List[Callable[[], None]]] = None,
        **config: Any,
    ) -> None:  # type: ignore[override]
        plugin_list: List[tuple[ValidationPlugin, int]] = []
        if plugins:
            for p in plugins:
                if isinstance(p, tuple):
                    plugin_list.append(p)
                else:
                    plugin_list.append((p, 0))

        super().__init__(
            name=name,
            agent=agent,
            config=StepConfig(**config),
            plugins=plugin_list,
            failure_handlers=on_failure or [],
        )

    def __rshift__(self, other: Step[OutT, Any] | "Pipeline") -> "Pipeline":
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

    def add_plugin(self, plugin: ValidationPlugin, priority: int = 0) -> "Step":
        self.plugins.append((plugin, priority))
        return self

    def on_failure(self, handler: Callable[[], None]) -> "Step":
        self.failure_handlers.append(handler)
        return self


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
