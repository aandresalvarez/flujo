from __future__ import annotations

# NOTE: This module was extracted from flujo.domain.pipeline_dsl as part of FSD1 refactor
# It contains the core Step DSL primitives (StepConfig, Step, decorators, etc.)
# Original implementation remains largely unchanged aside from relative import updates
# and lazy imports to avoid circular dependencies with other DSL modules.

from typing import (
    Any,
    Callable,
    ClassVar,
    Coroutine,
    Generic,
    List,
    Optional,
    TypeVar,
    Dict,
    ParamSpec,
    Concatenate,
    cast,
    Self,
    TYPE_CHECKING,
    Tuple,
    Set,
    Union,
    overload,
)
from enum import Enum
import uuid
from flujo.domain.models import BaseModel, RefinementCheck  # noqa: F401
from pydantic import Field, ConfigDict, field_validator
from ..processors import AgentProcessors

if TYPE_CHECKING:  # pragma: no cover
    from .loop import LoopStep, MapStep
    from .conditional import ConditionalStep
    from .parallel import ParallelStep
    from flujo.steps.cache_step import CacheStep

if TYPE_CHECKING:  # pragma: no cover
    from ..execution_strategy import ExecutionStrategy
    from ..ir import (
        StepIR,
        StandardStepIR,
        AgentReference,
        HumanInTheLoopStepIR,
        ValidationPluginIR,
        ValidatorIR,
    )
    from flujo.registry import CallableRegistry
    from .pipeline import Pipeline


def _generate_step_uid() -> str:
    """Generate a unique step identifier."""
    return str(uuid.uuid4())


# Type variables
StepInT = TypeVar("StepInT")
StepOutT = TypeVar("StepOutT")
NewOutT = TypeVar("NewOutT")
P = ParamSpec("P")

ContextModelT = TypeVar("ContextModelT", bound=BaseModel)

# BranchKey type alias for ConditionalStep
BranchKey = Any


class MergeStrategy(Enum):
    """Strategies for merging branch contexts back into the main context."""

    NO_MERGE = "no_merge"
    OVERWRITE = "overwrite"
    MERGE_SCRATCHPAD = "merge_scratchpad"


class BranchFailureStrategy(Enum):
    """Policies for handling branch failures in ``ParallelStep``."""

    PROPAGATE = "propagate"
    IGNORE = "ignore"


class StepConfig(BaseModel):
    """Configuration options applied to every step.

    Parameters
    ----------
    max_retries:
        How many times the step should be retried on failure.
    timeout_s:
        Optional timeout in seconds for the agent execution.
    temperature:
        Optional temperature setting for LLM based agents.
    """

    max_retries: int = 1
    timeout_s: float | None = None
    temperature: float | None = None


class Step(BaseModel, Generic[StepInT, StepOutT]):
    """Base class for all pipeline steps.

    This is an abstract base class. Use concrete subclasses like StandardStep,
    LoopStep, ConditionalStep, etc. for actual step implementations.
    """

    name: str = Field(description="Unique name for this step")
    agent: Optional[Any] = Field(default=None, description="Agent to execute this step")
    config: StepConfig = Field(default_factory=StepConfig, description="Step configuration")
    plugins: List[Tuple[Any, int]] = Field(
        default_factory=list, description="Validation plugins with priorities"
    )
    validators: List[Any] = Field(default_factory=list, description="Validators")
    processors: AgentProcessors = Field(default_factory=AgentProcessors, description="Processors")
    persist_feedback_to_context: Optional[str] = Field(
        default=None, description="Context attribute to persist feedback to"
    )
    persist_validation_results_to: Optional[str] = Field(
        default=None, description="Context attribute to persist validation results to"
    )
    updates_context: bool = Field(
        default=False, description="Whether step updates pipeline context"
    )
    meta: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata")
    step_uid: str = Field(
        default_factory=_generate_step_uid, description="Globally unique step identifier"
    )

    @field_validator("step_uid", mode="before")
    @classmethod
    def ensure_str_uid(cls, v: Any) -> str:
        if v is None:
            return _generate_step_uid()
        if not isinstance(v, str):
            return str(v)
        return v

    @classmethod
    def model_validate(cls: type[Self], *args: Any, **kwargs: Any) -> Self:
        """Dispatch validation to the appropriate concrete step class."""
        if cls is not Step:
            return super().model_validate(*args, **kwargs)

        return cast(Self, StandardStep.model_validate(*args, **kwargs))

    model_config: ClassVar[ConfigDict] = {
        "arbitrary_types_allowed": True,
    }

    def __init__(self, **data: Any) -> None:
        # Robustly handle step_uid before passing to Pydantic
        step_uid_raw = data.get("step_uid")
        step_uid: str
        if step_uid_raw is not None:
            if not isinstance(step_uid_raw, str):
                step_uid = str(step_uid_raw)
            else:
                step_uid = step_uid_raw
            data["step_uid"] = step_uid
        super().__init__(**data)
        self._fallback_step: Optional["Step[Any, Any]"] = None
        self._circular_fallback_uid: Optional[str] = None
        # Always ensure plugins is a list of (plugin, priority) tuples
        plugins: List[Tuple[Any, int]] = list(getattr(self, "plugins", []))
        self._plugins: List[Tuple[Any, int]] = plugins
        self.plugins = plugins

    @property
    def fallback_step(self) -> Optional["Step[Any, Any]"]:
        """Get the fallback step for this step."""
        return self._fallback_step

    @fallback_step.setter
    def fallback_step(self, step: Optional["Step[Any, Any]"]) -> None:
        """Set the fallback step for this step."""
        self._fallback_step = step

    def fallback(self, step: "Step[Any, Any]") -> "Step[StepInT, StepOutT]":
        """Set the fallback step and return self for chaining."""
        self.fallback_step = step
        return self

    def add_plugin(self, plugin: Any, priority: int = 0) -> "Step[StepInT, StepOutT]":
        """Add a plugin to this step and return self for chaining."""
        self._plugins.append((plugin, priority))
        self.plugins = self._plugins
        return self

    def failure_handlers(self) -> List[Any]:
        """Return the list of failure handlers (empty for now)."""
        return getattr(self, "_failure_handlers", [])

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_callable(
        cls,
        callable_func: Callable[..., Any],
        name: Optional[str] = None,
        step_uid: Optional[str] = None,
        **kwargs: Any,
    ) -> "StandardStep[Any, Any]":
        """Create a step from a callable function.

        This method wraps the callable in an agent and returns a StandardStep.
        """
        from .agents import _CallableAgent

        # Create the callable agent
        agent: _CallableAgent = _CallableAgent(callable_func)

        # Use the provided name or the callable's name
        step_name = cast(str, name or getattr(callable_func, "__name__", "callable_step"))

        # ``StandardStep`` expects ``step_uid`` to be ``str``. When ``step_uid`` is ``None``
        # we generate one explicitly to satisfy the type checker and avoid ``type: ignore``.
        step_uid_value = step_uid if step_uid is not None else _generate_step_uid()

        return StandardStep(
            name=step_name,
            agent=agent,
            step_uid=step_uid_value,
            **kwargs,
        )

    @classmethod
    def from_mapper(cls, *args: Any, **kwargs: Any) -> "StandardStep[Any, Any]":
        """Compatibility shim: use from_callable for mapper steps."""
        return cls.from_callable(*args, **kwargs)

    # ------------------------------------------------------------------
    # Execution interface
    # ------------------------------------------------------------------

    def get_execution_strategy(self) -> "ExecutionStrategy[Any]":
        """Get the execution strategy for this step."""
        from ..execution_strategy import get_execution_strategy

        return cast("ExecutionStrategy[Any]", get_execution_strategy(self))

    def get_strategy(self) -> "ExecutionStrategy[Any]":
        """Get the execution strategy for this step."""
        return self.get_execution_strategy()

    def get_agent(self) -> Any:
        """Get the agent for this step."""
        return self.agent

    def get_config(self) -> StepConfig:
        """Get the configuration for this step."""
        return self.config

    def get_plugins(self) -> List[Tuple[Any, int]]:
        """Get the plugins for this step."""
        return self.plugins

    def get_validators(self) -> List[Any]:
        """Get the validators for this step."""
        return self.validators

    def get_processors(self) -> AgentProcessors:
        """Get the processors for this step."""
        return self.processors

    def get_persist_feedback_to_context(self) -> Optional[str]:
        """Get the context attribute to persist feedback to."""
        return self.persist_feedback_to_context

    def get_persist_validation_results_to(self) -> Optional[str]:
        """Get the context attribute to persist validation results to."""
        return self.persist_validation_results_to

    def get_updates_context(self) -> bool:
        """Get whether this step updates pipeline context."""
        return self.updates_context

    def get_meta(self) -> Dict[str, Any]:
        """Get the metadata for this step."""
        return self.meta

    def get_step_uid(self) -> str:
        """Get the unique identifier for this step."""
        return self.step_uid

    # ------------------------------------------------------------------
    # Serialization interface
    # ------------------------------------------------------------------

    @classmethod
    def from_model(
        cls,
        ir_model: "StepIR",
        agent_registry: Optional[Dict[str, Any]] = None,
        callable_registry: Optional["CallableRegistry"] = None,
        plugin_registry: Optional[Dict[str, Any]] = None,
    ) -> "Step[Any, Any]":
        """Create a Step from its IR representation."""
        from .conditional import ConditionalStep
        from .parallel import ParallelStep
        from flujo.steps.cache_step import CacheStep
        from .step import HumanInTheLoopStep, StandardStep
        from ..ir import (
            StandardStepIR,
            LoopStepIR,
            ConditionalStepIR,
            ParallelStepIR,
            CacheStepIR,
            HumanInTheLoopStepIR,
        )

        if isinstance(ir_model, StandardStepIR):
            return StandardStep._from_standard_ir(
                ir_model, agent_registry, callable_registry, plugin_registry
            )
        elif isinstance(ir_model, LoopStepIR):
            return LoopStep._from_loop_ir(ir_model, agent_registry, callable_registry)
        elif isinstance(ir_model, ConditionalStepIR):
            return ConditionalStep._from_conditional_ir(ir_model, agent_registry, callable_registry)
        elif isinstance(ir_model, ParallelStepIR):
            return ParallelStep._from_parallel_ir(ir_model, agent_registry, callable_registry)
        elif isinstance(ir_model, CacheStepIR):
            return CacheStep._from_cache_ir(ir_model, agent_registry, callable_registry)
        elif isinstance(ir_model, HumanInTheLoopStepIR):
            return HumanInTheLoopStep._from_hitl_ir(ir_model, agent_registry, callable_registry)
        else:
            raise ValueError(f"Unknown step IR type: {type(ir_model)}")

    @classmethod
    def _from_standard_ir(
        cls,
        ir_model: "StandardStepIR[Any, Any]",
        agent_registry: Optional[Dict[str, Any]] = None,
        callable_registry: Optional["CallableRegistry"] = None,
        plugin_registry: Optional[Dict[str, Any]] = None,
    ) -> "StandardStep[Any, Any]":
        """Create a standard Step from its IR representation."""
        from .step import StandardStep

        # Resolve agent
        agent: Optional[Any] = None
        if ir_model.agent is not None:
            if ir_model.agent.agent_type == "callable":
                if callable_registry is None:
                    raise ValueError("CallableRegistry required for callable-based agents")
                callable_name: str = ir_model.agent.agent_name or ""
                callable_func: Callable[..., Any] = callable_registry.get(callable_name)
                if callable_func is None:
                    raise ValueError(f"Callable '{callable_name}' not found in registry")
                from .agents import _CallableAgent

                agent = _CallableAgent(callable_func)
            else:
                if agent_registry is None:
                    raise ValueError("AgentRegistry required for agent-based steps")
                agent_class: Any = agent_registry.get(ir_model.agent.agent_type)
                if agent_class is None:
                    # Try to find the base class name (before the underscore)
                    base_agent_type: str = ir_model.agent.agent_type.split("_")[0]
                    agent_class = agent_registry.get(base_agent_type)
                    if agent_class is None:
                        raise ValueError(
                            f"Agent type '{ir_model.agent.agent_type}' not found in registry"
                        )
                agent = agent_class()

        # Resolve plugins
        plugins: List[Tuple[Any, int]] = []
        if ir_model.plugins is not None:
            for plugin_ir in ir_model.plugins:
                if plugin_registry is None:
                    raise ValueError("PluginRegistry required for plugin-based steps")
                plugin_class: Any = plugin_registry.get(plugin_ir.plugin_type)
                if plugin_class is None:
                    raise ValueError(f"Plugin type '{plugin_ir.plugin_type}' not found in registry")
                plugin: Any = plugin_class()
                plugins.append((plugin, plugin_ir.priority))

        # Create step config
        config: StepConfig = StepConfig(
            max_retries=ir_model.config.max_retries,
            timeout_s=ir_model.config.timeout_seconds,
            temperature=ir_model.config.temperature,
        )

        # Create step
        step: StandardStep[Any, Any] = StandardStep(
            name=ir_model.name,
            agent=agent,
            config=config,
            plugins=plugins,
            validators=[],
            processors=AgentProcessors(),
            persist_feedback_to_context=ir_model.persist_feedback_to_context,
            persist_validation_results_to=ir_model.persist_validation_results_to,
            updates_context=ir_model.updates_context,
            meta=ir_model.meta,
            step_uid=ir_model.step_uid,
        )

        # Handle circular fallback reference
        if hasattr(ir_model, "meta") and ir_model.meta.get("circular_reference", False):
            step._circular_fallback_uid = ir_model.meta.get("circular_step_uid")

        return step

    # ------------------------------------------------------------------
    # Abstract execution interface
    # ------------------------------------------------------------------

    async def arun(self, data: StepInT, **kwargs: Any) -> StepOutT:
        """Execute this step asynchronously.

        This is the main execution interface for steps. Subclasses should
        override this method to implement their specific execution logic.
        """
        raise NotImplementedError("Subclasses must implement arun")

    # ------------------------------------------------------------------
    # Factory methods for complex step types
    # ------------------------------------------------------------------

    @classmethod
    def loop_until(
        cls,
        *,
        name: str,
        loop_body_pipeline: Any,
        exit_condition_callable: Callable[[Any, Optional[Any]], bool],
        max_loops: int = 5,
        **kwargs: Any,
    ) -> "LoopStep[Any]":
        """Create a loop step that executes until a condition is met."""
        from .loop import LoopStep

        return LoopStep(
            name=name,
            loop_body_pipeline=loop_body_pipeline,
            exit_condition_callable=exit_condition_callable,
            max_loops=max_loops,
            **kwargs,
        )

    @classmethod
    def map_over(cls, *args: Any, **kwargs: Any) -> "MapStep[Any]":
        """Create a map step that applies a pipeline to each item in a collection."""
        from .loop import MapStep

        return MapStep(*args, **kwargs)

    @classmethod
    def parallel(
        cls,
        name: str,
        branches: Dict[str, Any],
        **kwargs: Any,
    ) -> "ParallelStep[Any]":
        """Create a parallel step that executes multiple branches concurrently."""
        from .parallel import ParallelStep

        return ParallelStep(name=name, branches=branches, **kwargs)

    @classmethod
    def human_in_the_loop(
        cls,
        name: str,
        message_for_user: Optional[str] = None,
        input_schema: Optional[Any] = None,
        **kwargs: Any,
    ) -> "HumanInTheLoopStep":
        """Create a human-in-the-loop step."""
        return HumanInTheLoopStep(
            name=name,
            message_for_user=message_for_user,
            input_schema=input_schema,
            **kwargs,
        )

    @classmethod
    def branch_on(
        cls,
        *,
        name: str,
        condition_callable: Callable[[Any, Any], Any],
        branches: Dict[BranchKey, Any],
        default_branch_pipeline: Optional[Any] = None,
        branch_input_mapper: Optional[Callable[[Any, Any], Any]] = None,
        branch_output_mapper: Optional[Callable[[Any, Any, Any], Any]] = None,
        **kwargs: Any,
    ) -> "ConditionalStep[Any]":
        """Create a conditional step that branches based on a condition."""
        from .conditional import ConditionalStep

        return ConditionalStep(
            name=name,
            condition_callable=condition_callable,
            branches=branches,
            default_branch_pipeline=default_branch_pipeline,
            branch_input_mapper=branch_input_mapper,
            branch_output_mapper=branch_output_mapper,
            **kwargs,
        )

    @classmethod
    def solution(cls, agent: Any, **kwargs: Any) -> "StandardStep[Any, Any]":
        """Create a solution step with the given agent."""
        return StandardStep(name="solution", agent=agent, **kwargs)

    @classmethod
    def review(cls, agent: Any, **kwargs: Any) -> "StandardStep[Any, Any]":
        """Create a review step with the given agent."""
        return StandardStep(name="review", agent=agent, **kwargs)

    @classmethod
    def validate_step(cls, *args: Any, **kwargs: Any) -> "StandardStep[Any, Any]":
        """Create a validation step."""
        return StandardStep(name="validate", *args, **kwargs)

    @classmethod
    def refine_until(
        cls,
        *args: Any,
        **kwargs: Any,
    ) -> "LoopStep[Any]":
        """Create a refinement loop step."""
        from .loop import LoopStep

        return LoopStep(*args, **kwargs)

    @classmethod
    def cached(cls, *args: Any, **kwargs: Any) -> "CacheStep[Any, Any]":
        """Create a cached step."""
        from flujo.steps.cache_step import CacheStep

        return CacheStep(*args, **kwargs)

    # ------------------------------------------------------------------
    # Operator overloading
    # ------------------------------------------------------------------

    def __rshift__(
        self, other: Union["Step[Any, Any]", "Pipeline[Any, Any]"]
    ) -> "Pipeline[Any, Any]":
        """Chain this step with another step or pipeline."""
        from .pipeline import Pipeline

        if isinstance(other, Step):
            return Pipeline.model_construct(steps=[self, other])
        if isinstance(other, Pipeline):
            return Pipeline.model_construct(steps=[self, *other.steps])
        raise TypeError("Can only chain Step with Step or Pipeline")

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_model(
        self,
        callable_registry: Optional["CallableRegistry"] = None,
        visited: Optional[Set[str]] = None,
    ) -> "StepIR":
        """Convert this step to its IR representation."""
        # This is the base implementation - subclasses should override
        raise NotImplementedError("Subclasses must implement to_model")


class HumanInTheLoopStep(Step[Any, Any]):
    """A step that pauses the pipeline for human input."""

    message_for_user: str | None = Field(default=None)
    input_schema: Any | None = Field(default=None)

    model_config = {"arbitrary_types_allowed": True}

    async def arun(self, data: Any, **kwargs: Any) -> Any:
        """Execute this step asynchronously."""
        # Implementation would depend on the human-in-the-loop mechanism
        return data

    def to_model(
        self,
        callable_registry: Optional["CallableRegistry"] = None,
        visited: Optional[Set[str]] = None,
    ) -> "HumanInTheLoopStepIR[Any, Any]":
        """Convert this step to its IR representation."""
        from ..ir import HumanInTheLoopStepIR, StepConfigIR, ProcessorIR, StepType

        step_uid_raw_self = self.step_uid
        if not isinstance(step_uid_raw_self, str) or not step_uid_raw_self:
            step_uid_raw_self = str(uuid.uuid4())
        step_uid_str = step_uid_raw_self
        return HumanInTheLoopStepIR[Any, Any](
            name=self.name,
            agent=None,
            config=StepConfigIR(
                max_retries=self.config.max_retries,
                timeout_seconds=self.config.timeout_s,
                temperature=self.config.temperature,
            ),
            plugins=[],
            validators=[],
            processors=ProcessorIR(),
            persist_feedback_to_context=self.persist_feedback_to_context,
            persist_validation_results_to=self.persist_validation_results_to,
            updates_context=self.updates_context,
            meta=self.meta,
            step_uid=step_uid_str,
            step_type=StepType.HUMAN_IN_THE_LOOP,
            message_for_user=self.message_for_user,
            input_schema=self.input_schema,
        )

    @classmethod
    def _from_hitl_ir(
        cls,
        ir_model: "HumanInTheLoopStepIR[Any, Any]",
        agent_registry: Optional[Dict[str, Any]] = None,
        callable_registry: Optional["CallableRegistry"] = None,
    ) -> "HumanInTheLoopStep":
        """Create a HumanInTheLoopStep from its IR representation."""
        # Convert IR config to StepConfig
        config = StepConfig(
            max_retries=ir_model.config.max_retries,
            timeout_s=ir_model.config.timeout_seconds,
            temperature=ir_model.config.temperature,
        )

        # Convert IR plugins to actual plugins
        plugins: List[Tuple[Any, int]] = []
        for plugin_ir in ir_model.plugins:
            # For now, create empty plugins - in real implementation would rehydrate
            plugins.append((None, plugin_ir.priority))

        # Convert IR processors to AgentProcessors
        processors = AgentProcessors()

        step = cls(
            name=ir_model.name,
            agent=None,
            config=config,
            plugins=plugins,
            validators=[],
            processors=processors,
            persist_feedback_to_context=ir_model.persist_feedback_to_context,
            persist_validation_results_to=ir_model.persist_validation_results_to,
            updates_context=ir_model.updates_context,
            meta=ir_model.meta,
            step_uid=ir_model.step_uid,
            message_for_user=ir_model.message_for_user,
            input_schema=ir_model.input_schema,
        )
        return step


# ------------------------------------------------------------------
# Decorators
# ------------------------------------------------------------------


@overload
def step(
    func: Callable[Concatenate[StepInT, P], Coroutine[Any, Any, StepOutT]],
    *,
    name: str | None = None,
    step_uid: Optional[str] = None,
    updates_context: bool = False,
    processors: Optional[AgentProcessors] = None,
    persist_feedback_to_context: Optional[str] = None,
    persist_validation_results_to: Optional[str] = None,
    is_adapter: bool = False,
    **config: Any,
) -> StandardStep[StepInT, StepOutT]: ...


@overload
def step(
    func: None = None,
    *,
    name: str | None = None,
    step_uid: Optional[str] = None,
    updates_context: bool = False,
    processors: Optional[AgentProcessors] = None,
    persist_feedback_to_context: Optional[str] = None,
    persist_validation_results_to: Optional[str] = None,
    is_adapter: bool = False,
    **config: Any,
) -> Callable[
    [Callable[Concatenate[StepInT, P], Coroutine[Any, Any, StepOutT]]],
    StandardStep[StepInT, StepOutT],
]: ...


def step(
    func: Callable[Concatenate[StepInT, P], Coroutine[Any, Any, StepOutT]] | None = None,
    *,
    name: str | None = None,
    step_uid: Optional[str] = None,
    updates_context: bool = False,
    processors: Optional[AgentProcessors] = None,
    persist_feedback_to_context: Optional[str] = None,
    persist_validation_results_to: Optional[str] = None,
    is_adapter: bool = False,
    **config: Any,
) -> (
    Callable[
        [Callable[Concatenate[StepInT, P], Coroutine[Any, Any, StepOutT]]],
        StandardStep[StepInT, StepOutT],
    ]
    | StandardStep[StepInT, StepOutT]
):
    def decorator(
        callable_func: Callable[Concatenate[StepInT, P], Coroutine[Any, Any, StepOutT]],
    ) -> StandardStep[StepInT, StepOutT]:
        from .agents import _CallableAgent

        step_name = cast(str, name or getattr(callable_func, "__name__", "step"))

        step_uid_value = step_uid if step_uid is not None else _generate_step_uid()

        return StandardStep(
            name=step_name,
            agent=_CallableAgent(callable_func),
            step_uid=step_uid_value,
            updates_context=updates_context,
            processors=processors or AgentProcessors(),
            persist_feedback_to_context=persist_feedback_to_context,
            persist_validation_results_to=persist_validation_results_to,
            **config,
        )

    if func is None:
        return decorator
    else:
        return decorator(func)


def adapter_step(
    func: Callable[Concatenate[StepInT, P], Coroutine[Any, Any, StepOutT]] | None = None,
    **kwargs: Any,
) -> Any:
    """Decorator to create an adapter step from a function."""
    return step(func, is_adapter=True, **kwargs)


class StandardStep(Step[StepInT, StepOutT]):
    """A standard step that executes an agent."""

    def __init__(self, **data: Any) -> None:
        # Before passing to Pydantic, handle the optional step_uid
        step_uid_raw = data.pop("step_uid", None)
        if step_uid_raw is not None and not isinstance(step_uid_raw, str):
            data["step_uid"] = str(step_uid_raw)
        elif step_uid_raw is not None:
            data["step_uid"] = step_uid_raw
        # else: let Pydantic default handle it
        super().__init__(**data)
        self._fallback_step: Optional["Step[Any, Any]"] = None
        self._circular_fallback_uid: Optional[str] = None
        # Always ensure plugins is a list of (plugin, priority) tuples
        plugins: List[Tuple[Any, int]] = list(getattr(self, "plugins", []))
        self._plugins: List[Tuple[Any, int]] = plugins
        self.plugins = plugins

    @property
    def fallback_step(self) -> Optional["Step[Any, Any]"]:
        """Get the fallback step for this step."""
        return self._fallback_step

    @fallback_step.setter
    def fallback_step(self, step: Optional["Step[Any, Any]"]) -> None:
        """Set the fallback step for this step."""
        self._fallback_step = step

    def fallback(self, step: "Step[Any, Any]") -> "StandardStep[StepInT, StepOutT]":
        """Set the fallback step and return self for chaining."""
        self.fallback_step = step
        return self

    def add_plugin(self, plugin: Any, priority: int = 0) -> "StandardStep[StepInT, StepOutT]":
        """Add a plugin to this step and return self for chaining."""
        self._plugins.append((plugin, priority))
        self.plugins = self._plugins
        return self

    def failure_handlers(self) -> List[Any]:
        """Return the list of failure handlers (empty for now)."""
        return getattr(self, "_failure_handlers", [])

    def to_model(
        self,
        callable_registry: Optional["CallableRegistry"] = None,
        visited: Optional[Set[str]] = None,
    ) -> "StandardStepIR[Any, Any]":
        """Convert this StandardStep to its IR representation."""
        from ..ir import (
            StandardStepIR,
            AgentReference,
            StepConfigIR,
            ValidationPluginIR,
            ValidatorIR,
            ProcessorIR,
            StepType,
        )

        # Prevent infinite recursion for circular fallback references
        if visited is None:
            visited = set()

        step_id = self.step_uid
        if step_id in visited:
            # For circular references, we need to preserve the fallback relationship
            # but break the cycle in the IR to prevent infinite recursion
            agent_ref_circ1: Optional[AgentReference] = None
            if self.agent is not None:
                target_circ1 = getattr(self.agent, "_agent", self.agent)
                if not hasattr(target_circ1, "_step_callable"):
                    agent_type_circ1 = target_circ1.__class__.__name__
                    if hasattr(target_circ1, "_unique_id"):
                        agent_type_circ1 = f"{agent_type_circ1}_{target_circ1._unique_id}"
                    else:
                        agent_type_circ1 = f"{agent_type_circ1}_{id(target_circ1)}"
                    agent_ref_circ1 = AgentReference(
                        agent_type=agent_type_circ1,
                        agent_config={},
                        agent_name=getattr(target_circ1, "__name__", None),
                    )
            return StandardStepIR[Any, Any](
                name=self.name,
                agent=agent_ref_circ1,
                config=StepConfigIR(
                    max_retries=self.config.max_retries,
                    timeout_seconds=self.config.timeout_s,
                    temperature=self.config.temperature,
                ),
                plugins=[],
                validators=[],
                processors=ProcessorIR(),
                persist_feedback_to_context=self.persist_feedback_to_context,
                persist_validation_results_to=self.persist_validation_results_to,
                updates_context=self.updates_context,
                meta={
                    **self.meta,
                    "circular_reference": True,
                    "circular_step_uid": step_id,
                },
                step_uid=self.step_uid,
                fallback_step=None,  # Break the cycle in IR
                step_type=StepType.STANDARD,
            )

        visited.add(step_id)

        # Resolve agent reference
        agent_ref_circ2: Optional[AgentReference] = None
        if self.agent is not None:
            target_circ2 = getattr(self.agent, "_agent", self.agent)
            if not hasattr(target_circ2, "_step_callable"):
                agent_type_circ2 = target_circ2.__class__.__name__
                if hasattr(target_circ2, "_unique_id"):
                    agent_type_circ2 = f"{agent_type_circ2}_{target_circ2._unique_id}"
                else:
                    agent_type_circ2 = f"{agent_type_circ2}_{id(target_circ2)}"
                agent_ref_circ2 = AgentReference(
                    agent_type=agent_type_circ2,
                    agent_config={},
                    agent_name=getattr(target_circ2, "__name__", None),
                )

        # Resolve plugins
        plugin_irs: List[ValidationPluginIR] = []
        for plugin, priority in self.plugins:
            plugin_type = plugin.__class__.__name__
            if hasattr(plugin, "_unique_id"):
                plugin_type = f"{plugin_type}_{plugin._unique_id}"
            else:
                plugin_type = f"{plugin_type}_{id(plugin)}"
            plugin_irs.append(
                ValidationPluginIR(
                    plugin_type=plugin_type,
                    plugin_config={},
                    priority=priority,
                )
            )

        # Resolve validators
        validator_irs: List[ValidatorIR] = []
        for validator in self.validators:
            validator_type = validator.__class__.__name__
            if hasattr(validator, "_unique_id"):
                validator_type = f"{validator_type}_{validator._unique_id}"
            else:
                validator_type = f"{validator_type}_{id(validator)}"
            validator_irs.append(
                ValidatorIR(
                    validator_type=validator_type,
                    validator_config={},
                )
            )

        # Resolve fallback step
        fallback_ir: Optional[StandardStepIR[Any, Any]] = None
        if self.fallback_step is not None:
            ir = self.fallback_step.to_model(callable_registry, visited)
            if isinstance(ir, StandardStepIR):
                fallback_ir = ir
            else:
                fallback_ir = None

        return StandardStepIR[Any, Any](
            name=self.name,
            agent=agent_ref_circ2,
            config=StepConfigIR(
                max_retries=self.config.max_retries,
                timeout_seconds=self.config.timeout_s,
                temperature=self.config.temperature,
            ),
            plugins=plugin_irs,
            validators=validator_irs,
            processors=ProcessorIR(),
            persist_feedback_to_context=self.persist_feedback_to_context,
            persist_validation_results_to=self.persist_validation_results_to,
            updates_context=self.updates_context,
            meta=self.meta,
            step_uid=self.step_uid,
            fallback_step=fallback_ir,
            step_type=StepType.STANDARD,
        )

    async def arun(self, data: StepInT, **kwargs: Any) -> StepOutT:
        """Execute this step asynchronously."""
        if self.agent is None:
            raise ValueError("No agent configured for this step")

        # This is a placeholder implementation
        # The actual execution would be handled by the execution strategy
        return cast(StepOutT, data)
