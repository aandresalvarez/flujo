from __future__ import annotations

from typing import List, Any, Optional, Callable, TypeVar, Generic
from pydantic import BaseModel, Field
from .agent_protocol import AgentProtocol
from .plugins import ValidationPlugin


class StepConfig(BaseModel):
    """Configuration options for a pipeline step."""

    max_retries: int = 1
    timeout_s: float | None = None


InT = TypeVar("InT")
OutT = TypeVar("OutT")


class Step(BaseModel, Generic[InT, OutT]):
    """Represents a single step in a pipeline."""

    name: str
    agent: Optional[Any] = None
    config: StepConfig = Field(default_factory=StepConfig)
    plugins: List[ValidationPlugin] = Field(default_factory=list)
    failure_handlers: List[Callable[[], None]] = Field(default_factory=list)

    model_config = {
        "arbitrary_types_allowed": True,
    }

    def __init__(
        self,
        name: str,
        agent: Optional[Any] = None,
        plugins: Optional[List[ValidationPlugin]] = None,
        on_failure: Optional[List[Callable[[], None]]] = None,
        **config: Any,
    ) -> None:  # type: ignore[override]
        cfg = config.get("config") if "config" in config else None
        if isinstance(cfg, StepConfig):
            step_cfg = cfg
        else:
            step_cfg = StepConfig(**config)
        super().__init__(
            name=name,
            agent=agent,
            config=step_cfg,
            plugins=plugins or [],
            failure_handlers=on_failure or [],
        )

    def __rshift__(self, other: "Step[Any, Any]" | "Pipeline[Any, Any]") -> "Pipeline[Any, Any]":
        if isinstance(other, Step):
            return Pipeline.model_construct(steps=[self, other])
        if isinstance(other, Pipeline):
            return Pipeline.model_construct(steps=[self, *other.steps])
        raise TypeError("Can only chain Step with Step or Pipeline")

    @classmethod
    def review(cls, agent: AgentProtocol[Any, Any], **config: Any) -> "Step[Any, Any]":
        return cls("review", agent, **config)

    @classmethod
    def solution(cls, agent: AgentProtocol[Any, Any], **config: Any) -> "Step[Any, Any]":
        return cls("solution", agent, **config)

    @classmethod
    def validate(cls, agent: AgentProtocol[Any, Any], **config: Any) -> "Step[Any, Any]":
        """Construct a validation step using the provided agent."""
        return cls("validate", agent, **config)

    def add_plugin(self, plugin: ValidationPlugin) -> "Step":
        self.plugins.append(plugin)
        return self

    def on_failure(self, handler: Callable[[], None]) -> "Step":
        self.failure_handlers.append(handler)
        return self


class Pipeline(BaseModel, Generic[InT, OutT]):
    """A sequential pipeline of steps."""

    steps: List[Step[Any, Any]]

    model_config = {"arbitrary_types_allowed": True}

    def __rshift__(self, other: Step | "Pipeline") -> "Pipeline":
        if isinstance(other, Step):
            return Pipeline.model_construct(steps=[*self.steps, other])
        if isinstance(other, Pipeline):
            return Pipeline.model_construct(steps=[*self.steps, *other.steps])
        raise TypeError("Can only chain Pipeline with Step or Pipeline")

    def __iter__(self):  # pragma: no cover - convenience
        return iter(self.steps)
