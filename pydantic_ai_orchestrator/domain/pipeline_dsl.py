from __future__ import annotations

from typing import (
    Any,
    Callable,
    ClassVar,
    Generic,
    Iterator,
    List,
    Optional,
    Sequence,
    TypeVar,
)
from pydantic import BaseModel, Field, ConfigDict
from .agent_protocol import AsyncAgentProtocol
from .plugins import ValidationPlugin


StepInT = TypeVar("StepInT")
StepOutT = TypeVar("StepOutT")
NewOutT = TypeVar("NewOutT")


class StepConfig(BaseModel):
    """Configuration options for a pipeline step."""

    max_retries: int = 1
    timeout_s: float | None = None


class Step(BaseModel, Generic[StepInT, StepOutT]):
    """Represents a single step in a pipeline."""

    name: str
    agent: Any | None = Field(default=None)
    config: StepConfig = Field(default_factory=StepConfig)
    plugins: List[tuple[ValidationPlugin, int]] = Field(default_factory=list)
    failure_handlers: List[Callable[[], None]] = Field(default_factory=list)

    model_config: ClassVar[ConfigDict] = {
        "arbitrary_types_allowed": True,
    }

    def __init__(
        self,
        name: str,
        agent: Optional[AsyncAgentProtocol[StepInT, StepOutT]] = None,
        plugins: Optional[List[ValidationPlugin | tuple[ValidationPlugin, int]]] = None,
        on_failure: Optional[List[Callable[[], None]]] = None,
        **config: Any,
    ) -> None:
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

    def __rshift__(
        self, other: "Step[StepOutT, NewOutT]" | "Pipeline[StepOutT, NewOutT]"
    ) -> "Pipeline[StepInT, NewOutT]":
        if isinstance(other, Step):
            return Pipeline.from_step(self) >> other
        if isinstance(other, Pipeline):
            return Pipeline.from_step(self) >> other
        raise TypeError("Can only chain Step with Step or Pipeline")

    @classmethod
    def review(cls, agent: AsyncAgentProtocol[Any, Any], **config: Any) -> "Step[Any, Any]":
        """Construct a review step using the provided agent."""
        return cls("review", agent, **config)

    @classmethod
    def solution(cls, agent: AsyncAgentProtocol[Any, Any], **config: Any) -> "Step[Any, Any]":
        """Construct a solution step using the provided agent."""
        return cls("solution", agent, **config)

    @classmethod
    def validate_step(cls, agent: AsyncAgentProtocol[Any, Any], **config: Any) -> "Step[Any, Any]":
        """Construct a validation step using the provided agent."""
        return cls("validate", agent, **config)

    validate = validate_step

    def add_plugin(self, plugin: ValidationPlugin, priority: int = 0) -> "Step[StepInT, StepOutT]":
        """Add a validation plugin to this step."""
        self.plugins.append((plugin, priority))
        return self



    def on_failure(self, handler: Callable[[], None]) -> "Step[StepInT, StepOutT]":
        """Add a failure handler to this step."""
        self.failure_handlers.append(handler)
        return self


class StepFactory:
    @staticmethod
    def create[SF_InT, SF_OutT](
        name: str,
        agent: AsyncAgentProtocol[SF_InT, SF_OutT],
        **config: Any,
    ) -> Step[SF_InT, SF_OutT]:
        return Step[SF_InT, SF_OutT](name, agent, **config)

    @classmethod
    def review[ReviewOutT](
        cls, agent: AsyncAgentProtocol[Any, ReviewOutT], **config: Any
    ) -> Step[Any, ReviewOutT]:
        return cls.create("review", agent, **config)

    @classmethod
    def solution[SolInT, SolOutT](
        cls, agent: AsyncAgentProtocol[SolInT, SolOutT], **config: Any
    ) -> Step[SolInT, SolOutT]:
        return cls.create("solution", agent, **config)

    @classmethod
    def validate[ValInT, ValOutT](
        cls, agent: AsyncAgentProtocol[ValInT, ValOutT], **config: Any
    ) -> Step[ValInT, ValOutT]:
        return cls.create("validate", agent, **config)


PipeInT = TypeVar("PipeInT")
PipeOutT = TypeVar("PipeOutT")
NewPipeOutT = TypeVar("NewPipeOutT")


class Pipeline(BaseModel, Generic[PipeInT, PipeOutT]):
    """A sequential pipeline of steps."""

    steps: Sequence[Step[Any, Any]]

    model_config: ClassVar[ConfigDict] = {
        "arbitrary_types_allowed": True,
    }

    @classmethod
    def from_step(cls, step: Step[PipeInT, PipeOutT]) -> "Pipeline[PipeInT, PipeOutT]":
        return cls[PipeInT, PipeOutT].model_construct(steps=[step])

    def __rshift__(
        self, other: Step[PipeOutT, NewPipeOutT] | "Pipeline[PipeOutT, NewPipeOutT]"
    ) -> "Pipeline[PipeInT, NewPipeOutT]":
        if isinstance(other, Step):
            new_steps = list(self.steps) + [other]
            return Pipeline[PipeInT, NewPipeOutT](steps=new_steps)
        if isinstance(other, Pipeline):
            new_steps = list(self.steps) + list(other.steps)
            return Pipeline[PipeInT, NewPipeOutT](steps=new_steps)
        raise TypeError("Can only chain Pipeline with Step or Pipeline")

    def __iter__(self) -> Iterator[Step[Any, Any]]:
        return iter(self.steps)

# Explicit exports
__all__ = ['Step', 'Pipeline', 'StepConfig', 'StepFactory']
