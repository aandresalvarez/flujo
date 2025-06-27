from __future__ import annotations

# mypy: ignore-errors

from typing import (
    Any,
    Callable,
    ClassVar,
    Coroutine,
    Generic,
    Iterator,
    List,
    Optional,
    Sequence,
    TypeVar,
    Dict,
    Type,
    ParamSpec,
    Concatenate,
    overload,
    get_type_hints,
    get_origin,
    get_args,
    Iterable,
)
import contextvars
import inspect

from .pipeline_validation import ValidationFinding, ValidationReport
from flujo.domain.models import BaseModel
from flujo.domain.resources import AppResources
from pydantic import Field, ConfigDict
from .agent_protocol import AsyncAgentProtocol
from .plugins import ValidationPlugin
from .validation import Validator
from .types import ContextT
from .processors import AgentProcessors


StepInT = TypeVar("StepInT")
StepOutT = TypeVar("StepOutT")
NewOutT = TypeVar("NewOutT")
P = ParamSpec("P")


# BranchKey type alias for ConditionalStep
BranchKey = Any


class StepConfig(BaseModel):
    """Configuration options for a pipeline step."""

    max_retries: int = 1
    timeout_s: float | None = None
    temperature: float | None = None


class Step(BaseModel, Generic[StepInT, StepOutT]):
    """Represents a single step in a pipeline.

    Use :meth:`arun` to execute the step's agent directly when unit testing.
    """

    name: str
    agent: Any | None = Field(default=None)
    config: StepConfig = Field(default_factory=StepConfig)
    plugins: List[tuple[ValidationPlugin, int]] = Field(default_factory=list)
    validators: List[Validator] = Field(default_factory=list)
    failure_handlers: List[Callable[[], None]] = Field(default_factory=list)
    processors: "AgentProcessors" = Field(default_factory=AgentProcessors)
    persist_feedback_to_context: Optional[str] = Field(
        default=None,
        description=(
            "If step fails, append feedback to this context attribute (must be a list)."
        ),
    )
    persist_validation_results_to: Optional[str] = Field(
        default=None,
        description=(
            "Append ValidationResult objects to this context attribute (must be a list)."
        ),
    )
    updates_context: bool = Field(
        default=False,
        description="Whether the step output should merge into the pipeline context.",
    )
    meta: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata about this step.",
    )

    model_config: ClassVar[ConfigDict] = {
        "arbitrary_types_allowed": True,
    }

    def __init__(
        self,
        name: str,
        agent: Optional[AsyncAgentProtocol[StepInT, StepOutT]] = None,
        plugins: Optional[List[ValidationPlugin | tuple[ValidationPlugin, int]]] = None,
        validators: Optional[List[Validator]] = None,
        on_failure: Optional[List[Callable[[], None]]] = None,
        updates_context: bool = False,
        processors: Optional[AgentProcessors] = None,
        persist_feedback_to_context: Optional[str] = None,
        persist_validation_results_to: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
        **config: Any,
    ) -> None:
        plugin_list: List[tuple[ValidationPlugin, int]] = []
        if plugins:
            for p in plugins:
                if isinstance(p, tuple):
                    plugin_list.append(p)
                else:
                    plugin_list.append((p, 0))

        super().__init__(  # type: ignore[misc]
            name=name,
            agent=agent,
            config=StepConfig(**config),
            plugins=plugin_list,
            validators=validators or [],
            failure_handlers=on_failure or [],
            updates_context=updates_context,
            processors=processors or AgentProcessors(),
            persist_feedback_to_context=persist_feedback_to_context,
            persist_validation_results_to=persist_validation_results_to,
            meta=meta or {},
        )

    def __repr__(self) -> str:  # pragma: no cover - simple utility
        agent_repr: str
        if self.agent is None:
            agent_repr = "None"
        else:
            target = getattr(self.agent, "_agent", self.agent)
            if hasattr(target, "__name__"):
                agent_repr = f"<function {target.__name__}>"
            elif hasattr(self.agent, "_model_name"):
                agent_repr = f"AsyncAgentWrapper(model={getattr(self.agent, '_model_name', 'unknown')})"
            else:
                agent_repr = self.agent.__class__.__name__
        config_repr = ""
        default_config = StepConfig()
        if self.config != default_config:
            config_repr = f", config={self.config!r}"
        return f"Step(name={self.name!r}, agent={agent_repr}{config_repr})"

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - behavior
        """Disallow direct invocation of a Step."""
        from ..exceptions import ImproperStepInvocationError

        raise ImproperStepInvocationError(
            f"Step '{self.name}' cannot be invoked directly. "
            "Steps are configuration objects and must be run within a Pipeline. "
            "For unit testing, use `step.arun()`."
        )

    def __getattr__(self, item: str) -> Any:  # pragma: no cover - behavior
        if item in {"run", "stream"}:
            from ..exceptions import ImproperStepInvocationError

            raise ImproperStepInvocationError(
                f"Step '{self.name}' cannot be invoked directly. "
                "Steps are configuration objects and must be run within a Pipeline. "
                "For unit testing, use `step.arun()`."
            )
        raise AttributeError(item)

    def __rshift__(
        self, other: "Step[StepOutT, NewOutT]" | "Pipeline[StepOutT, NewOutT]"
    ) -> "Pipeline[StepInT, NewOutT]":
        if isinstance(other, Step):
            return Pipeline.from_step(self) >> other
        if isinstance(other, Pipeline):
            return Pipeline.from_step(self) >> other
        raise TypeError("Can only chain Step with Step or Pipeline")

    async def arun(self, data: StepInT, **kwargs: Any) -> StepOutT:
        """Run this step's agent directly for testing purposes.

        Parameters
        ----------
        data: StepInT
            The input data for the step.
        **kwargs: Any
            Additional keyword arguments forwarded to the agent's ``run`` method.

        Returns
        -------
        StepOutT
            The agent's output.

        Raises
        ------
        ValueError
            If the step does not have an agent configured.
        """
        if self.agent is None:
            raise ValueError(f"Step '{self.name}' has no agent to run.")

        return await self.agent.run(data, **kwargs)

    @classmethod
    def review(
        cls,
        agent: AsyncAgentProtocol[Any, Any],
        *,
        validators: Optional[List[Validator]] = None,
        processors: Optional[AgentProcessors] = None,
        persist_feedback_to_context: Optional[str] = None,
        persist_validation_results_to: Optional[str] = None,
        **config: Any,
    ) -> "Step[Any, Any]":
        """Construct a review step using the provided agent."""
        return cls(
            "review",
            agent,
            validators=validators,
            processors=processors,
            persist_feedback_to_context=persist_feedback_to_context,
            persist_validation_results_to=persist_validation_results_to,
            **config,
        )

    @classmethod
    def solution(
        cls,
        agent: AsyncAgentProtocol[Any, Any],
        *,
        validators: Optional[List[Validator]] = None,
        processors: Optional[AgentProcessors] = None,
        persist_feedback_to_context: Optional[str] = None,
        persist_validation_results_to: Optional[str] = None,
        **config: Any,
    ) -> "Step[Any, Any]":
        """Construct a solution step using the provided agent."""
        return cls(
            "solution",
            agent,
            validators=validators,
            processors=processors,
            persist_feedback_to_context=persist_feedback_to_context,
            persist_validation_results_to=persist_validation_results_to,
            **config,
        )

    @classmethod
    def validate_step(
        cls,
        agent: AsyncAgentProtocol[Any, Any],
        *,
        validators: Optional[List[Validator]] = None,
        processors: Optional[AgentProcessors] = None,
        persist_feedback_to_context: Optional[str] = None,
        persist_validation_results_to: Optional[str] = None,
        **config: Any,
    ) -> "Step[Any, Any]":
        """Construct a validation step using the provided agent."""
        return cls(
            "validate",
            agent,
            validators=validators,
            processors=processors,
            persist_feedback_to_context=persist_feedback_to_context,
            persist_validation_results_to=persist_validation_results_to,
            **config,
        )

    @classmethod
    def from_callable(
        cls: type["Step[StepInT, StepOutT]"],
        callable_: Callable[Concatenate[StepInT, P], Coroutine[Any, Any, StepOutT]],
        name: str | None = None,
        updates_context: bool = False,
        processors: Optional[AgentProcessors] = None,
        persist_feedback_to_context: Optional[str] = None,
        persist_validation_results_to: Optional[str] = None,
        **config: Any,
    ) -> "Step[StepInT, StepOutT]":
        """Create a :class:`Step` by wrapping an async callable.

        The input and output types are inferred from the callable's first
        positional parameter and return annotation. Additional keyword-only
        parameters such as ``pipeline_context`` or ``resources`` are supported.
        If type hints are missing, ``Any`` is used.
        """

        func = callable_
        if not inspect.iscoroutinefunction(func):
            # try __call__ for callable objects
            call_method = getattr(func, "__call__", None)
            if call_method is None or not inspect.iscoroutinefunction(call_method):
                raise TypeError("from_callable expects an async callable")

        try:
            hints = get_type_hints(func)
        except Exception:
            hints = {}

        sig = inspect.signature(func)
        params = list(sig.parameters.values())
        first: inspect.Parameter | None = None
        for p in params:
            if p.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                if p.name in {"self", "cls"}:
                    continue
                first = p
                break

        input_type = Any
        if first is None:
            input_type = type(None)
        else:
            input_type = hints.get(first.name, Any)

        output_type = hints.get("return", Any)
        if get_origin(output_type) is Coroutine:
            args = get_args(output_type)
            if len(args) == 3:
                output_type = args[2]
            else:
                output_type = Any

        step_name = name or getattr(func, "__name__", func.__class__.__name__)

        from ..signature_tools import analyze_signature

        spec = analyze_signature(func)

        class _CallableAgent:
            _step_callable = func
            _injection_spec = spec

            async def run(
                self,
                data: Any,
                *,
                context: BaseModel | None = None,
                resources: AppResources | None = None,
                temperature: float | None = None,
                **kwargs: Any,
            ) -> Any:
                from ..application.flujo_engine import _accepts_param

                call_kwargs: Dict[str, Any] = {}
                if spec.needs_context and context is not None and spec.context_kw:
                    call_kwargs[spec.context_kw] = context
                if spec.needs_resources and resources is not None:
                    call_kwargs["resources"] = resources
                if temperature is not None and _accepts_param(func, "temperature"):
                    call_kwargs["temperature"] = temperature
                call_kwargs.update(kwargs)
                if first is None:
                    return await func(**call_kwargs)
                return await func(data, **call_kwargs)

        agent_wrapper = _CallableAgent()

        step = cls(
            step_name,
            agent_wrapper,
            updates_context=updates_context,
            processors=processors,
            persist_feedback_to_context=persist_feedback_to_context,
            persist_validation_results_to=persist_validation_results_to,
            **config,
        )
        object.__setattr__(step, "__step_input_type__", input_type)
        object.__setattr__(step, "__step_output_type__", output_type)
        return step

    @classmethod
    def from_mapper(
        cls: type["Step[StepInT, StepOutT]"],
        mapper: Callable[Concatenate[StepInT, P], Coroutine[Any, Any, StepOutT]],
        name: str | None = None,
        updates_context: bool = False,
        processors: Optional[AgentProcessors] = None,
        persist_feedback_to_context: Optional[str] = None,
        persist_validation_results_to: Optional[str] = None,
        **config: Any,
    ) -> "Step[StepInT, StepOutT]":
        """Create a :class:`Step` from an async mapper function.

        This is a thin wrapper around :meth:`from_callable` for semantic
        clarity when the callable merely maps its input to a new value.
        """

        return cls.from_callable(
            mapper,
            name=name,
            updates_context=updates_context,
            processors=processors,
            persist_feedback_to_context=persist_feedback_to_context,
            persist_validation_results_to=persist_validation_results_to,
            **config,
        )

    @classmethod
    def human_in_the_loop(
        cls,
        name: str,
        message_for_user: str | None = None,
        input_schema: Type[BaseModel] | None = None,
    ) -> "HumanInTheLoopStep":
        """Create a step that pauses execution for human input."""
        return HumanInTheLoopStep(
            name=name,
            message_for_user=message_for_user,
            input_schema=input_schema,
        )

    def add_plugin(
        self, plugin: ValidationPlugin, priority: int = 0
    ) -> "Step[StepInT, StepOutT]":
        """Add a validation plugin to this step."""
        self.plugins.append((plugin, priority))
        return self

    def on_failure(self, handler: Callable[[], None]) -> "Step[StepInT, StepOutT]":
        """Add a failure handler to this step."""
        self.failure_handlers.append(handler)
        return self

    @classmethod
    def loop_until(
        cls,
        name: str,
        loop_body_pipeline: "Pipeline[Any, Any]",
        exit_condition_callable: Callable[[Any, Optional[ContextT]], bool],
        max_loops: int = 5,
        initial_input_to_loop_body_mapper: Optional[
            Callable[[Any, Optional[ContextT]], Any]
        ] = None,
        iteration_input_mapper: Optional[
            Callable[[Any, Optional[ContextT], int], Any]
        ] = None,
        loop_output_mapper: Optional[Callable[[Any, Optional[ContextT]], Any]] = None,
        **config_kwargs: Any,
    ) -> "LoopStep[ContextT]":
        """Factory method to create a :class:`LoopStep`."""
        from .pipeline_dsl import LoopStep

        return LoopStep(
            name=name,
            loop_body_pipeline=loop_body_pipeline,
            exit_condition_callable=exit_condition_callable,
            max_loops=max_loops,
            initial_input_to_loop_body_mapper=initial_input_to_loop_body_mapper,
            iteration_input_mapper=iteration_input_mapper,
            loop_output_mapper=loop_output_mapper,
            **config_kwargs,
        )

    @classmethod
    def branch_on(
        cls,
        name: str,
        condition_callable: Callable[[Any, Optional[ContextT]], BranchKey],
        branches: Dict[BranchKey, "Pipeline[Any, Any]"],
        default_branch_pipeline: Optional["Pipeline[Any, Any]"] = None,
        branch_input_mapper: Optional[Callable[[Any, Optional[ContextT]], Any]] = None,
        branch_output_mapper: Optional[
            Callable[[Any, BranchKey, Optional[ContextT]], Any]
        ] = None,
        **config_kwargs: Any,
    ) -> "ConditionalStep[ContextT]":
        """Factory method to create a :class:`ConditionalStep`."""
        from .pipeline_dsl import ConditionalStep

        return ConditionalStep(
            name=name,
            condition_callable=condition_callable,
            branches=branches,
            default_branch_pipeline=default_branch_pipeline,
            branch_input_mapper=branch_input_mapper,
            branch_output_mapper=branch_output_mapper,
            **config_kwargs,
        )

    @classmethod
    def parallel(
        cls,
        name: str,
        branches: Dict[str, "Step[Any, Any]" | "Pipeline[Any, Any]"],
        **config_kwargs: Any,
    ) -> "ParallelStep[ContextT]":
        """Factory to run branches concurrently and aggregate outputs."""
        from .pipeline_dsl import ParallelStep

        return ParallelStep(name=name, branches=branches, **config_kwargs)

    @classmethod
    def map_over(
        cls,
        name: str,
        pipeline_to_run: "Pipeline[Any, Any]",
        *,
        iterable_input: str,
        **config_kwargs: Any,
    ) -> "MapStep[ContextT]":
        """Factory to process each item of ``iterable_input`` with ``pipeline_to_run``."""

        return MapStep(
            name=name,
            pipeline_to_run=pipeline_to_run,
            iterable_input=iterable_input,
            **config_kwargs,
        )


@overload
def step(
    func: Callable[Concatenate[StepInT, P], Coroutine[Any, Any, StepOutT]],
    *,
    is_adapter: bool = False,
) -> "Step[StepInT, StepOutT]":
    """Transform an async function into a :class:`Step`."""
    ...


@overload
def step(
    *,
    name: str | None = None,
    updates_context: bool = False,
    is_adapter: bool = False,
    **config_kwargs: Any,
) -> Callable[
    [Callable[Concatenate[StepInT, P], Coroutine[Any, Any, StepOutT]]],
    "Step[StepInT, StepOutT]",
]:
    """Decorator form to configure the created :class:`Step`."""
    ...


def step(
    func: (
        Callable[Concatenate[StepInT, P], Coroutine[Any, Any, StepOutT]] | None
    ) = None,
    *,
    name: str | None = None,
    updates_context: bool = False,
    is_adapter: bool = False,
    **config_kwargs: Any,
) -> Any:
    """Decorator that converts an async function into a :class:`Step`.

    It can be used with or without arguments. When used without parentheses,
    ``@step`` directly transforms the decorated async function into a ``Step``.
    When called with keyword arguments, those are forwarded to ``Step.from_callable``.
    """

    decorator_kwargs = {
        "name": name,
        "updates_context": updates_context,
        **config_kwargs,
    }

    def decorator(
        fn: Callable[Concatenate[StepInT, P], Coroutine[Any, Any, StepOutT]],
    ) -> "Step[StepInT, StepOutT]":
        step_obj = Step.from_callable(fn, **decorator_kwargs)
        if is_adapter:
            step_obj.meta["is_adapter"] = True
        return step_obj

    if func is not None:
        return decorator(func)

    return decorator


# Convenience alias to create mapping steps
mapper = Step.from_mapper


def adapter_step(
    func: (
        Callable[Concatenate[StepInT, P], Coroutine[Any, Any, StepOutT]] | None
    ) = None,
    **kwargs: Any,
) -> Any:
    """Alias for :func:`step` that marks the created step as an adapter."""
    return step(func, is_adapter=True, **kwargs)


class HumanInTheLoopStep(Step[Any, Any]):
    """A step that pauses the pipeline for human input."""

    message_for_user: str | None = Field(default=None)
    input_schema: Type[BaseModel] | None = Field(default=None)

    model_config = {"arbitrary_types_allowed": True}

    def __init__(
        self,
        *,
        name: str,
        message_for_user: str | None = None,
        input_schema: Type[BaseModel] | None = None,
        **config: Any,
    ) -> None:
        super().__init__(
            name=name,
            agent=None,
            config=StepConfig(**config),
            plugins=[],
            failure_handlers=[],
        )
        object.__setattr__(self, "message_for_user", message_for_user)
        object.__setattr__(self, "input_schema", input_schema)


class LoopStep(Step[Any, Any], Generic[ContextT]):
    """A specialized step that executes a pipeline in a loop."""

    loop_body_pipeline: "Pipeline[Any, Any]" = Field(
        description="The pipeline to execute in each iteration."
    )
    exit_condition_callable: Callable[[Any, Optional[ContextT]], bool] = Field(
        description=(
            "Callable that takes (last_body_output, pipeline_context) and returns True to exit loop."
        )
    )
    max_loops: int = Field(default=5, gt=0, description="Maximum number of iterations.")

    initial_input_to_loop_body_mapper: Optional[
        Callable[[Any, Optional[ContextT]], Any]
    ] = Field(
        default=None,
        description=(
            "Callable to map LoopStep's input to the first iteration's body input."
        ),
    )
    iteration_input_mapper: Optional[Callable[[Any, Optional[ContextT], int], Any]] = (
        Field(
            default=None,
            description=(
                "Callable to map previous iteration's body output to next iteration's input."
            ),
        )
    )
    loop_output_mapper: Optional[Callable[[Any, Optional[ContextT]], Any]] = Field(
        default=None,
        description=(
            "Callable to map the final successful output to the LoopStep's output."
        ),
    )

    model_config = {"arbitrary_types_allowed": True}

    def __init__(
        self,
        *,
        name: str,
        loop_body_pipeline: "Pipeline[Any, Any]",
        exit_condition_callable: Callable[[Any, Optional[ContextT]], bool],
        max_loops: int = 5,
        initial_input_to_loop_body_mapper: Optional[
            Callable[[Any, Optional[ContextT]], Any]
        ] = None,
        iteration_input_mapper: Optional[
            Callable[[Any, Optional[ContextT], int], Any]
        ] = None,
        loop_output_mapper: Optional[Callable[[Any, Optional[ContextT]], Any]] = None,
        **config_kwargs: Any,
    ) -> None:
        if max_loops <= 0:
            raise ValueError("max_loops must be a positive integer.")

        BaseModel.__init__(  # type: ignore[misc]
            self,
            name=name,
            agent=None,
            config=StepConfig(**config_kwargs),
            plugins=[],
            failure_handlers=[],
            loop_body_pipeline=loop_body_pipeline,
            exit_condition_callable=exit_condition_callable,
            max_loops=max_loops,
            initial_input_to_loop_body_mapper=initial_input_to_loop_body_mapper,
            iteration_input_mapper=iteration_input_mapper,
            loop_output_mapper=loop_output_mapper,
        )


class ConditionalStep(Step[Any, Any], Generic[ContextT]):
    """A step that selects and executes a branch pipeline based on a condition."""

    condition_callable: Callable[[Any, Optional[ContextT]], BranchKey] = Field(
        description=("Callable that returns a key to select a branch.")
    )
    branches: Dict[BranchKey, "Pipeline[Any, Any]"] = Field(
        description="Mapping of branch keys to sub-pipelines."
    )
    default_branch_pipeline: Optional["Pipeline[Any, Any]"] = Field(
        default=None,
        description="Pipeline to execute when no branch key matches.",
    )

    branch_input_mapper: Optional[Callable[[Any, Optional[ContextT]], Any]] = Field(
        default=None,
        description="Maps ConditionalStep input to branch input.",
    )
    branch_output_mapper: Optional[
        Callable[[Any, BranchKey, Optional[ContextT]], Any]
    ] = Field(
        default=None,
        description="Maps branch output to ConditionalStep output.",
    )

    model_config = {"arbitrary_types_allowed": True}

    def __init__(
        self,
        *,
        name: str,
        condition_callable: Callable[[Any, Optional[ContextT]], BranchKey],
        branches: Dict[BranchKey, "Pipeline[Any, Any]"],
        default_branch_pipeline: Optional["Pipeline[Any, Any]"] = None,
        branch_input_mapper: Optional[Callable[[Any, Optional[ContextT]], Any]] = None,
        branch_output_mapper: Optional[
            Callable[[Any, BranchKey, Optional[ContextT]], Any]
        ] = None,
        **config_kwargs: Any,
    ) -> None:
        if not branches:
            raise ValueError("'branches' dictionary cannot be empty.")

        BaseModel.__init__(  # type: ignore[misc]
            self,
            name=name,
            agent=None,
            config=StepConfig(**config_kwargs),
            plugins=[],
            failure_handlers=[],
            condition_callable=condition_callable,
            branches=branches,
            default_branch_pipeline=default_branch_pipeline,
            branch_input_mapper=branch_input_mapper,
            branch_output_mapper=branch_output_mapper,
        )


class ParallelStep(Step[Any, Any], Generic[ContextT]):
    """A step that executes multiple branch pipelines concurrently."""

    branches: Dict[str, "Pipeline[Any, Any]"] = Field(
        description="Mapping of branch names to pipelines to run in parallel."
    )

    model_config = {"arbitrary_types_allowed": True}

    def __init__(
        self,
        *,
        name: str,
        branches: Dict[str, "Step[Any, Any]" | "Pipeline[Any, Any]"],
        **config_kwargs: Any,
    ) -> None:
        if not branches:
            raise ValueError("'branches' dictionary cannot be empty.")

        normalized: Dict[str, Pipeline[Any, Any]] = {}
        for key, branch in branches.items():
            normalized[key] = (
                Pipeline.from_step(branch) if isinstance(branch, Step) else branch
            )

        BaseModel.__init__(  # type: ignore[misc]
            self,
            name=name,
            agent=None,
            config=StepConfig(**config_kwargs),
            plugins=[],
            failure_handlers=[],
            branches=normalized,
        )


class MapStep(LoopStep[ContextT]):
    """A step that maps a pipeline over items in the pipeline context."""

    iterable_input: str = Field()

    def __init__(
        self,
        *,
        name: str,
        pipeline_to_run: "Pipeline[Any, Any]",
        iterable_input: str,
        **config_kwargs: Any,
    ) -> None:
        results_attr = f"__{name}_results"
        items_attr = f"__{name}_items"

        async def _collect(
            item: Any, *, pipeline_context: BaseModel | None = None
        ) -> Any:
            if pipeline_context is None:
                raise ValueError("map_over requires a pipeline context")
            getattr(pipeline_context, results_attr).append(item)
            return item

        collector = Step.from_callable(_collect, name=f"_{name}_collect")
        body = pipeline_to_run >> collector

        BaseModel.__init__(  # type: ignore[misc]
            self,
            name=name,
            agent=None,
            config=StepConfig(**config_kwargs),
            plugins=[],
            failure_handlers=[],
            loop_body_pipeline=body,
            exit_condition_callable=lambda _o, ctx: len(getattr(ctx, results_attr, []))
            >= len(getattr(ctx, items_attr, [])),
            max_loops=1,
            initial_input_to_loop_body_mapper=None,
            iteration_input_mapper=None,
            loop_output_mapper=None,
            iterable_input=iterable_input,
        )
        object.__setattr__(self, "_original_body_pipeline", body)

        async def _noop(item: Any, **_: Any) -> Any:
            return item

        object.__setattr__(
            self,
            "_noop_pipeline",
            Pipeline.from_step(Step.from_callable(_noop, name=f"_{name}_noop")),
        )
        object.__setattr__(self, "_results_attr", results_attr)
        object.__setattr__(self, "_items_attr", items_attr)
        object.__setattr__(
            self,
            "_max_loops_var",
            contextvars.ContextVar(f"{name}_max_loops", default=1),
        )
        object.__setattr__(
            self, "_body_var", contextvars.ContextVar(f"{name}_body", default=body)
        )

        def _initial_mapper(_: Any, ctx: BaseModel | None) -> Any:
            if ctx is None:
                raise ValueError("map_over requires a pipeline context")
            raw_items = getattr(ctx, iterable_input, [])
            if isinstance(raw_items, (str, bytes, bytearray)) or not isinstance(
                raw_items, Iterable
            ):
                raise TypeError(
                    f"pipeline_context.{iterable_input} must be a non-string iterable"
                )
            items = list(raw_items)
            setattr(ctx, items_attr, items)
            setattr(ctx, results_attr, [])
            if items:
                self._max_loops_var.set(len(items))
                self._body_var.set(self._original_body_pipeline)
                return items[0]
            else:
                self._max_loops_var.set(1)
                self._body_var.set(self._noop_pipeline)
                return None

        def _iter_mapper(_: Any, ctx: BaseModel | None, i: int) -> Any:
            if ctx is None:
                raise ValueError("map_over requires a pipeline context")
            items = getattr(ctx, items_attr, [])
            return items[i] if i < len(items) else None

        def _output_mapper(_: Any, ctx: BaseModel | None) -> list[Any]:
            if ctx is None:
                raise ValueError("map_over requires a pipeline context")
            return list(getattr(ctx, results_attr, []))

        object.__setattr__(self, "initial_input_to_loop_body_mapper", _initial_mapper)
        object.__setattr__(self, "iteration_input_mapper", _iter_mapper)
        object.__setattr__(self, "loop_output_mapper", _output_mapper)
        object.__setattr__(self, "iterable_input", iterable_input)

    def __getattribute__(self, name: str) -> Any:  # noqa: D401
        """Return attribute, using context vars for loop state."""
        if name == "max_loops":
            return object.__getattribute__(self, "_max_loops_var").get()
        if name == "loop_body_pipeline":
            return object.__getattribute__(self, "_body_var").get()
        return super().__getattribute__(name)


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
        return cls.model_construct(steps=[step])

    def __rshift__(
        self, other: Step[PipeOutT, NewPipeOutT] | "Pipeline[PipeOutT, NewPipeOutT]"
    ) -> "Pipeline[PipeInT, NewPipeOutT]":
        if isinstance(other, Step):
            new_steps = list(self.steps) + [other]
            return Pipeline.model_construct(steps=new_steps)
        if isinstance(other, Pipeline):
            new_steps = list(self.steps) + list(other.steps)
            return Pipeline.model_construct(steps=new_steps)
        raise TypeError("Can only chain Pipeline with Step or Pipeline")

    def validate(self, *, raise_on_error: bool = False) -> ValidationReport:
        """Validate that all steps have agents and compatible types.

        Args:
            raise_on_error: If ``True`` raise ``ConfigurationError`` when any
                errors are found.

        Returns:
            ValidationReport summarizing any errors and warnings.
        """
        from ..exceptions import ConfigurationError
        from typing import Any, get_origin, get_args, Union

        def _compatible(a: Any, b: Any) -> bool:
            if a is Any or b is Any:
                return True

            origin_a, origin_b = get_origin(a), get_origin(b)

            if origin_b is Union:
                return any(_compatible(a, arg) for arg in get_args(b))
            if origin_a is Union:
                return all(_compatible(arg, b) for arg in get_args(a))

            try:
                return issubclass(a, b)
            except Exception:
                pass

            if origin_a is None and origin_b is None:
                return False
            if origin_a is None:
                try:
                    return issubclass(a, origin_b)
                except Exception:
                    return False
            if origin_b is None:
                try:
                    return issubclass(origin_a, b)
                except Exception:
                    return False

            if origin_a is not origin_b:
                return False

            args_a, args_b = get_args(a), get_args(b)
            if len(args_a) != len(args_b):
                return False
            return all(_compatible(x, y) for x, y in zip(args_a, args_b))

        report = ValidationReport()

        seen_steps: set[int] = set()
        prev_step = None
        prev_out_type: Any = None

        for step in self.steps:
            if id(step) in seen_steps:
                report.warnings.append(
                    ValidationFinding(
                        rule_id="V-A3",
                        severity="warning",
                        message=(
                            "The same Step object instance is used more than once in the pipeline. "
                            "This may cause side effects if the step is stateful."
                        ),
                        step_name=step.name,
                    )
                )
            else:
                seen_steps.add(id(step))

            if step.agent is None:
                report.errors.append(
                    ValidationFinding(
                        rule_id="V-A1",
                        severity="error",
                        message=(
                            "Step '{name}' is missing an agent. Assign one via `Step('name', agent=...)` "
                            "or by using a step factory like `@step` or `Step.from_callable()`."
                        ).format(name=step.name),
                        step_name=step.name,
                    )
                )

            in_type = getattr(step, "__step_input_type__", Any)
            if prev_step is not None and prev_out_type is not None:
                if not _compatible(prev_out_type, in_type):
                    report.errors.append(
                        ValidationFinding(
                            rule_id="V-A2",
                            severity="error",
                            message=(
                                f"Type mismatch: Output of '{prev_step.name}' (returns `{prev_out_type}`) "
                                f"is not compatible with '{step.name}' (expects `{in_type}`). "
                                "For best results, use a static type checker like mypy to catch these issues before runtime."
                            ),
                            step_name=step.name,
                        )
                    )
            prev_step = step
            prev_out_type = getattr(step, "__step_output_type__", Any)

        if raise_on_error and report.errors:
            raise ConfigurationError(
                "Pipeline validation failed: " + report.model_dump_json(indent=2)
            )

        return report

    def iter_steps(self) -> Iterator[Step[Any, Any]]:
        return iter(self.steps)


# Explicit exports
__all__ = [
    "Step",
    "step",
    "mapper",
    "Pipeline",
    "StepConfig",
    "LoopStep",
    "MapStep",
    "ParallelStep",
    "ConditionalStep",
    "HumanInTheLoopStep",
    "BranchKey",
]
