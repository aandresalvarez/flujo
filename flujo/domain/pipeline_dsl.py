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
    Union,
)
import contextvars
import inspect
import logging
from enum import Enum

from .pipeline_validation import ValidationFinding, ValidationReport
from flujo.domain.models import (
    BaseModel,
    RefinementCheck,
)  # noqa: F401
from flujo.domain.resources import AppResources
from pydantic import Field, ConfigDict
from .agent_protocol import AsyncAgentProtocol
from .plugins import ValidationPlugin
from .validation import Validator
from .types import ContextT
from .processors import AgentProcessors
from flujo.caching import CacheBackend, InMemoryCache
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from flujo.steps.cache_step import CacheStep


StepInT = TypeVar("StepInT")
StepOutT = TypeVar("StepOutT")
NewOutT = TypeVar("NewOutT")
P = ParamSpec("P")


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
    fallback_step: Optional[Any] = Field(default=None, exclude=True)
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
        plugins: Optional[list[tuple[ValidationPlugin, int]]] = None,
        validators: Optional[List[Validator]] = None,
        processors: Optional[AgentProcessors] = None,
        persist_feedback_to_context: Optional[str] = None,
        persist_validation_results_to: Optional[str] = None,
        **config: Any,
    ) -> "Step[Any, Any]":
        """Construct a review step using the provided agent."""
        return cls.model_validate(
            {
                "name": "review",
                "agent": agent,
                "plugins": plugins or [],
                "validators": validators or [],
                "processors": processors or AgentProcessors(),
                "persist_feedback_to_context": persist_feedback_to_context,
                "persist_validation_results_to": persist_validation_results_to,
                "config": StepConfig(**config),
            }
        )

    @classmethod
    def solution(
        cls,
        agent: AsyncAgentProtocol[Any, Any],
        *,
        plugins: Optional[list[tuple[ValidationPlugin, int]]] = None,
        validators: Optional[List[Validator]] = None,
        processors: Optional[AgentProcessors] = None,
        persist_feedback_to_context: Optional[str] = None,
        persist_validation_results_to: Optional[str] = None,
        **config: Any,
    ) -> "Step[Any, Any]":
        """Construct a solution step using the provided agent."""
        return cls.model_validate(
            {
                "name": "solution",
                "agent": agent,
                "plugins": plugins or [],
                "validators": validators or [],
                "processors": processors or AgentProcessors(),
                "persist_feedback_to_context": persist_feedback_to_context,
                "persist_validation_results_to": persist_validation_results_to,
                "config": StepConfig(**config),
            }
        )

    @classmethod
    def validate_step(
        cls,
        agent: AsyncAgentProtocol[Any, Any],
        *,
        plugins: Optional[list[tuple[ValidationPlugin, int]]] = None,
        validators: Optional[List[Validator]] = None,
        processors: Optional[AgentProcessors] = None,
        persist_feedback_to_context: Optional[str] = None,
        persist_validation_results_to: Optional[str] = None,
        strict: bool = True,
        **config: Any,
    ) -> "Step[Any, Any]":
        """Construct a validation step using the provided agent."""
        meta = {"is_validation_step": True, "strict_validation": strict}
        return cls.model_validate(
            {
                "name": "validate",
                "agent": agent,
                "plugins": plugins or [],
                "validators": validators or [],
                "processors": processors or AgentProcessors(),
                "persist_feedback_to_context": persist_feedback_to_context,
                "persist_validation_results_to": persist_validation_results_to,
                "meta": meta,
                "config": StepConfig(**config),
            }
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
        parameters such as ``context`` or ``resources`` are supported.
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

        analyze_signature(func)

        class _CallableAgent:
            _step_callable = func
            _injection_spec = analyze_signature(func)

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
                if (
                    analyze_signature(func).needs_context
                    and context is not None
                    and analyze_signature(func).context_kw
                ):
                    call_kwargs[analyze_signature(func).context_kw] = context
                if analyze_signature(func).needs_resources and resources is not None:
                    call_kwargs["resources"] = resources
                if temperature is not None and _accepts_param(func, "temperature"):
                    call_kwargs["temperature"] = temperature
                call_kwargs.update(kwargs)
                if first is None:
                    return await func(**call_kwargs)
                return await func(data, **call_kwargs)

        agent_wrapper = _CallableAgent()

        step = cls.model_validate(
            {
                "name": step_name,
                "agent": agent_wrapper,
                "updates_context": updates_context,
                "processors": processors or AgentProcessors(),
                "persist_feedback_to_context": persist_feedback_to_context,
                "persist_validation_results_to": persist_validation_results_to,
                "config": StepConfig(**config),
            }
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
        return HumanInTheLoopStep.model_validate(
            {
                "name": name,
                "message_for_user": message_for_user,
                "input_schema": input_schema,
            }
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

    def fallback(self, step: "Step") -> "Step[StepInT, StepOutT]":
        """Set a fallback step to execute if this step fails after retries."""
        self.fallback_step = step
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

        # Create the LoopStep directly using Pydantic model instantiation
        return LoopStep.model_validate(
            {
                "name": name,
                "loop_body_pipeline": loop_body_pipeline,
                "exit_condition_callable": exit_condition_callable,
                "max_loops": max_loops,
                "initial_input_to_loop_body_mapper": initial_input_to_loop_body_mapper,
                "iteration_input_mapper": iteration_input_mapper,
                "loop_output_mapper": loop_output_mapper,
                "config": StepConfig(**config_kwargs),
            }
        )

    @classmethod
    def refine_until(
        cls,
        name: str,
        generator_pipeline: "Pipeline[Any, Any]",
        critic_pipeline: "Pipeline[Any, RefinementCheck]",
        max_refinements: int = 5,
        feedback_mapper: Optional[Callable[[Any, RefinementCheck], Any]] = None,
        **config_kwargs: Any,
    ) -> "LoopStep[ContextT]":
        """Factory for a Generator-Critic refinement loop."""
        artifact_key = f"__{name}_artifact"
        original_var: contextvars.ContextVar[Any | None] = contextvars.ContextVar(
            f"{name}_orig", default=None
        )
        artifact_var: contextvars.ContextVar[Any | None] = contextvars.ContextVar(
            f"{name}_artifact", default=None
        )

        async def _store_artifact(
            artifact: Any, *, context: BaseModel | None = None
        ) -> Any:
            artifact_var.set(artifact)
            if context is not None and hasattr(context, "scratchpad"):
                context.scratchpad[artifact_key] = artifact
            return artifact

        saver_step = cls.from_callable(_store_artifact, name=f"_{name}_store")

        loop_body = generator_pipeline >> saver_step >> critic_pipeline

        def _exit_condition(out: Any, _ctx: BaseModel | None) -> bool:
            if isinstance(out, RefinementCheck):
                return out.is_complete
            return True

        def _initial_mapper(inp: Any, ctx: BaseModel | None) -> dict[str, Any]:
            original_var.set(inp)
            if ctx is not None and hasattr(ctx, "scratchpad"):
                ctx.scratchpad.setdefault(artifact_key + "__orig", inp)
            return {"original_input": inp, "feedback": None}

        def _iteration_mapper(
            out: Any, ctx: BaseModel | None, _i: int
        ) -> dict[str, Any]:
            if isinstance(out, RefinementCheck):
                feedback = out.feedback
            else:
                feedback = None
            if ctx is not None and hasattr(ctx, "scratchpad"):
                original = ctx.scratchpad.get(artifact_key + "__orig")
            else:
                original = original_var.get(None)
            if feedback_mapper is None:
                return {"original_input": original, "feedback": feedback}
            return feedback_mapper(
                original,
                (
                    out
                    if isinstance(out, RefinementCheck)
                    else RefinementCheck(is_complete=False, feedback=feedback)
                ),
            )

        def _output_mapper(_out: Any, ctx: BaseModel | None) -> Any:
            if (
                ctx is not None
                and hasattr(ctx, "scratchpad")
                and artifact_key in ctx.scratchpad
            ):
                return ctx.scratchpad.get(artifact_key)
            return artifact_var.get(None)

        return cls.loop_until(
            name=name,
            loop_body_pipeline=loop_body,
            exit_condition_callable=_exit_condition,
            max_loops=max_refinements,
            initial_input_to_loop_body_mapper=_initial_mapper,
            iteration_input_mapper=_iteration_mapper,
            loop_output_mapper=_output_mapper,
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

        # Create the ConditionalStep directly using Pydantic model instantiation
        return ConditionalStep.model_validate(
            {
                "name": name,
                "condition_callable": condition_callable,
                "branches": branches,
                "default_branch_pipeline": default_branch_pipeline,
                "branch_input_mapper": branch_input_mapper,
                "branch_output_mapper": branch_output_mapper,
                "config": StepConfig(**config_kwargs),
            }
        )

    @classmethod
    def parallel(
        cls,
        name: str,
        branches: Dict[str, "Step[Any, Any]" | "Pipeline[Any, Any]"],
        context_include_keys: Optional[List[str]] = None,
        merge_strategy: Union[
            MergeStrategy, Callable[[ContextT, ContextT], None]
        ] = MergeStrategy.NO_MERGE,
        on_branch_failure: BranchFailureStrategy = BranchFailureStrategy.PROPAGATE,
        **config_kwargs: Any,
    ) -> "ParallelStep[ContextT]":
        """Factory to run branches concurrently and aggregate outputs."""
        from .pipeline_dsl import ParallelStep

        # Create the ParallelStep directly using Pydantic model instantiation
        return ParallelStep.model_validate(
            {
                "name": name,
                "branches": branches,
                "context_include_keys": context_include_keys,
                "merge_strategy": merge_strategy,
                "on_branch_failure": on_branch_failure,
                "config": StepConfig(**config_kwargs),
            }
        )

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

        # Separate config from specific fields
        config = StepConfig(**config_kwargs)

        return MapStep(
            name=name,
            config=config,
            pipeline_to_run=pipeline_to_run,
            iterable_input=iterable_input,
        )

    @classmethod
    def cached(
        cls,
        wrapped_step: "Step[Any, Any]",
        cache_backend: Optional[CacheBackend] = None,
    ) -> "CacheStep":
        """Wrap ``wrapped_step`` so its results are cached."""
        from flujo.steps.cache_step import CacheStep

        return CacheStep(
            name=f"Cached({wrapped_step.name})",
            wrapped_step=wrapped_step,
            cache_backend=cache_backend or InMemoryCache(),
        )


@overload
def step(
    func: Callable[Concatenate[StepInT, P], Coroutine[Any, Any, StepOutT]],
    *,
    name: str | None = None,
    updates_context: bool = False,
    processors: Optional[AgentProcessors] = None,
    persist_feedback_to_context: Optional[str] = None,
    persist_validation_results_to: Optional[str] = None,
    is_adapter: bool = False,
    **config_kwargs: Any,
) -> "Step[StepInT, StepOutT]":
    """Transform an async function into a :class:`Step`."""
    ...


@overload
def step(
    *,
    name: str | None = None,
    updates_context: bool = False,
    processors: Optional[AgentProcessors] = None,
    persist_feedback_to_context: Optional[str] = None,
    persist_validation_results_to: Optional[str] = None,
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
    processors: Optional[AgentProcessors] = None,
    persist_feedback_to_context: Optional[str] = None,
    persist_validation_results_to: Optional[str] = None,
    is_adapter: bool = False,
    **config_kwargs: Any,
) -> Any:
    """A decorator that creates a :class:`Step` from an async function.

    This is syntactic sugar for :meth:`Step.from_callable`; see that method's
    documentation for all available parameters. The decorator can be used with
    or without arguments. When called with keyword arguments, they are forwarded
    directly to ``Step.from_callable``.
    """

    decorator_kwargs = {
        "name": name,
        "updates_context": updates_context,
        "processors": processors,
        "persist_feedback_to_context": persist_feedback_to_context,
        "persist_validation_results_to": persist_validation_results_to,
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

# Resolve forward references now that ``Step`` is fully defined
# (handled automatically by Pydantic)


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
    input_schema: Any | None = Field(default=None)

    model_config = {"arbitrary_types_allowed": True}


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

    @classmethod
    def model_validate(cls, *args, **kwargs):
        """Validate branches before creating the instance."""
        # Support both dict input (args[0]) and kwargs
        if args and isinstance(args[0], dict):
            branches = args[0].get("branches", {})
        else:
            branches = kwargs.get("branches", {})
        if not branches:
            raise ValueError("'branches' dictionary cannot be empty.")
        return super().model_validate(*args, **kwargs)


class ParallelStep(Step[Any, Any], Generic[ContextT]):
    """A step that executes multiple branch pipelines concurrently."""

    branches: Dict[str, "Pipeline[Any, Any]"] = Field(
        description="Mapping of branch names to pipelines to run in parallel."
    )
    context_include_keys: Optional[List[str]] = Field(
        default=None,
        description="If provided, only these top-level context fields will be copied to each branch. "
        "If None, the entire context is deep-copied (default behavior).",
    )
    merge_strategy: Union[MergeStrategy, Callable[[ContextT, ContextT], None]] = Field(
        default=MergeStrategy.NO_MERGE,
        description="Strategy for merging successful branch contexts back into the main context.",
    )
    on_branch_failure: BranchFailureStrategy = Field(
        default=BranchFailureStrategy.PROPAGATE,
        description="How the ParallelStep should behave when a branch fails.",
    )

    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def model_validate(cls, *args, **kwargs):
        """Validate and normalize branches before creating the instance."""
        # Support both dict input (args[0]) and kwargs
        if args and isinstance(args[0], dict):
            branches = args[0].get("branches", {})
        else:
            branches = kwargs.get("branches", {})
        if not branches:
            raise ValueError("'branches' dictionary cannot be empty.")
        # Normalize branches: convert Steps to Pipelines
        normalized: Dict[str, "Pipeline[Any, Any]"] = {}
        for key, branch in branches.items():
            if isinstance(branch, Step):
                normalized[key] = Pipeline.from_step(branch)
            else:
                normalized[key] = branch
        # Update the input dict or kwargs with normalized branches
        if args and isinstance(args[0], dict):
            args = (dict(args[0], branches=normalized),) + args[1:]
        else:
            kwargs["branches"] = normalized
        return super().model_validate(*args, **kwargs)


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

        async def _collect(item: Any, *, context: BaseModel | None = None) -> Any:
            if context is None:
                raise ValueError("map_over requires a context")
            getattr(context, results_attr).append(item)
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
                raise ValueError("map_over requires a context")
            raw_items = getattr(ctx, iterable_input, [])
            if isinstance(raw_items, (str, bytes, bytearray)) or not isinstance(
                raw_items, Iterable
            ):
                raise TypeError(
                    f"context.{iterable_input} must be a non-string iterable"
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
                raise ValueError("map_over requires a context")
            items = getattr(ctx, items_attr, [])
            return items[i] if i < len(items) else None

        def _output_mapper(_: Any, ctx: BaseModel | None) -> list[Any]:
            if ctx is None:
                raise ValueError("map_over requires a context")
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
            except Exception as e:
                logging.warning(f"_compatible: issubclass({a}, {b}) raised {e}")
                return False

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
            else:
                from ..signature_tools import analyze_signature

                target = getattr(step.agent, "_agent", step.agent)
                func = getattr(target, "_step_callable", getattr(target, "run", None))

                if func is not None:
                    try:
                        analyze_signature(func)
                        # No longer checking for deprecated pipeline_context since it's been removed
                    except Exception as e:  # pragma: no cover - defensive
                        report.warnings.append(
                            ValidationFinding(
                                rule_id="V-A4-ERR",
                                severity="warning",
                                message=f"Could not analyze signature for agent in step '{step.name}': {e}",
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

    def to_mermaid(self) -> str:
        """Generate a Mermaid graph definition for visualizing this pipeline.

        Returns a valid Mermaid graph TD (Top-Down) definition that represents:
        - Different step types with distinct shapes
        - Configuration annotations (plugins/validators, retries)
        - Control flow structures (loops, branches, parallel execution)
        - Nested pipeline structures using subgraphs

        Returns:
            str: A valid Mermaid graph definition string
        """
        return self.to_mermaid_with_detail_level("auto")

    def to_mermaid_with_detail_level(self, detail_level: str = "auto") -> str:
        """Generate a Mermaid graph definition with configurable detail levels.

        Args:
            detail_level: One of "high", "medium", "low", or "auto".
                         "auto" uses AI to determine the best level based on complexity.

        Returns:
            str: A valid Mermaid graph definition string
        """
        if detail_level == "auto":
            detail_level = self._determine_optimal_detail_level()

        if detail_level == "high":
            return self._generate_high_detail_mermaid()
        elif detail_level == "medium":
            return self._generate_medium_detail_mermaid()
        elif detail_level == "low":
            return self._generate_low_detail_mermaid()
        else:
            raise ValueError(
                f"Invalid detail_level: {detail_level}. Must be 'high', 'medium', 'low', or 'auto'"
            )

    def _determine_optimal_detail_level(self) -> str:
        """Use AI to determine the optimal detail level based on pipeline complexity."""
        complexity_score = self._calculate_complexity_score()

        if complexity_score >= 15:
            return "low"
        elif complexity_score >= 8:
            return "medium"
        else:
            return "high"

    def _calculate_complexity_score(self) -> int:
        """Calculate a complexity score for the pipeline."""
        score = 0

        for step in self.steps:
            # Base score for each step
            score += 1

            # Additional complexity for different step types
            if isinstance(step, LoopStep):
                score += 3  # Loops are complex
                # Add complexity for nested steps in loop
                score += len(step.loop_body_pipeline.steps) * 2
            elif isinstance(step, ConditionalStep):
                score += 2  # Conditionals add complexity
                # Add complexity for each branch
                score += len(step.branches) * 2
            elif isinstance(step, ParallelStep):
                score += 2  # Parallel execution adds complexity
                # Add complexity for each parallel branch
                score += len(step.branches) * 2
            elif isinstance(step, HumanInTheLoopStep):
                score += 1  # Human steps add some complexity

            # Additional complexity for configuration
            if step.config.max_retries > 1:
                score += 1
            if step.plugins or step.validators:
                score += 1

        return score

    def _generate_high_detail_mermaid(self) -> str:
        """Generate high-detail Mermaid diagram with all features."""
        lines = ["graph TD"]
        node_counter = 0
        step_nodes: dict[int, str] = {}

        def get_node_id(step: Step[Any, Any]) -> str:
            step_id = id(step)
            if step_id not in step_nodes:
                nonlocal node_counter
                node_counter += 1
                step_nodes[step_id] = f"s{node_counter}"
            return step_nodes[step_id]

        def add_node(step: Step[Any, Any], node_id: str) -> None:
            """Add a node definition for the given step."""
            # Determine node shape and label based on step type
            if isinstance(step, HumanInTheLoopStep):
                shape = f"[/Human: {step.name}/]"
            elif isinstance(step, LoopStep):
                shape = f'("Loop: {step.name}")'
            elif isinstance(step, ConditionalStep):
                shape = f'{{"Branch: {step.name}"}}'
            elif isinstance(step, ParallelStep):
                shape = f'{{{{"Parallel: {step.name}"}}}}'
            else:
                # Standard Step
                label = step.name
                # Add validation annotation if step has plugins or validators
                if step.plugins or step.validators:
                    label += " 🛡️"
                shape = f'["{label}"]'

            lines.append(f"    {node_id}{shape};")

        def add_edge(
            from_node: str, to_node: str, label: str | None = None, style: str = "-->"
        ) -> None:
            """Add an edge between two nodes."""
            if label:
                lines.append(f'    {from_node} {style} |"{label}"| {to_node};')
            else:
                lines.append(f"    {from_node} {style} {to_node};")

        def process_step(step: Step[Any, Any], prev_node: str | None = None) -> str:
            """Process a single step and return its node ID."""
            node_id = get_node_id(step)
            add_node(step, node_id)

            # Add edge from previous node if it exists
            if prev_node:
                # Use dashed edge for steps with retries
                edge_style = "-.->" if step.config.max_retries > 1 else "-->"
                add_edge(prev_node, node_id, style=edge_style)

            return node_id

        def process_pipeline(
            pipeline: Pipeline[Any, Any],
            prev_node: str | None = None,
            subgraph_name: str | None = None,
        ) -> str:
            """Process a pipeline and return the last node ID."""
            if subgraph_name:
                lines.append(f'    subgraph "{subgraph_name}"')

            last_node = prev_node
            for step in pipeline.steps:
                if isinstance(step, LoopStep):
                    last_node = process_loop_step(step, last_node)
                elif isinstance(step, ConditionalStep):
                    last_node = process_conditional_step(step, last_node)
                elif isinstance(step, ParallelStep):
                    last_node = process_parallel_step(step, last_node)
                else:
                    last_node = process_step(step, last_node)

            if subgraph_name:
                lines.append("    end")

            return last_node

        def process_loop_step(step: LoopStep[Any], prev_node: str | None = None) -> str:
            """Process a LoopStep with its internal pipeline structure."""
            loop_node_id = get_node_id(step)
            add_node(step, loop_node_id)

            if prev_node:
                add_edge(prev_node, loop_node_id)

            # Process the loop body pipeline in a subgraph
            lines.append(f'    subgraph "Loop Body: {step.name}"')
            body_start = process_pipeline(step.loop_body_pipeline)
            lines.append("    end")

            # Connect loop node to body start
            add_edge(loop_node_id, body_start)

            # Connect body end back to loop node
            add_edge(body_start, loop_node_id)

            # Create exit path
            exit_node_id = f"{loop_node_id}_exit"
            lines.append(f'    {exit_node_id}(("Exit"));')
            add_edge(loop_node_id, exit_node_id, "Exit")

            return exit_node_id

        def process_conditional_step(
            step: ConditionalStep[Any], prev_node: str | None = None
        ) -> str:
            """Process a ConditionalStep with its branch pipelines."""
            cond_node_id = get_node_id(step)
            add_node(step, cond_node_id)

            if prev_node:
                add_edge(prev_node, cond_node_id)

            # Process each branch
            branch_end_nodes = []
            for branch_key, branch_pipeline in step.branches.items():
                branch_name = f"Branch: {branch_key}"
                lines.append(f'    subgraph "{branch_name}"')
                branch_end = process_pipeline(branch_pipeline)
                lines.append("    end")

                # Connect conditional to branch start
                add_edge(cond_node_id, branch_end, str(branch_key))
                branch_end_nodes.append(branch_end)

            # Add default branch if it exists
            if step.default_branch_pipeline:
                lines.append('    subgraph "Default Branch"')
                default_end = process_pipeline(step.default_branch_pipeline)
                lines.append("    end")
                add_edge(cond_node_id, default_end, "default")
                branch_end_nodes.append(default_end)

            # Create join node
            join_node_id = f"{cond_node_id}_join"
            lines.append(f"    {join_node_id}(( ));")
            lines.append(f"    style {join_node_id} fill:none,stroke:none")

            # Connect all branches to join node
            for branch_end in branch_end_nodes:
                add_edge(branch_end, join_node_id)

            return join_node_id

        def process_parallel_step(
            step: ParallelStep[Any], prev_node: str | None = None
        ) -> str:
            """Process a ParallelStep with its parallel branches."""
            para_node_id = get_node_id(step)
            add_node(step, para_node_id)

            if prev_node:
                add_edge(prev_node, para_node_id)

            # Process each parallel branch
            branch_end_nodes = []
            for branch_name, branch_pipeline in step.branches.items():
                lines.append(f'    subgraph "Parallel: {branch_name}"')
                branch_end = process_pipeline(branch_pipeline)
                lines.append("    end")

                # Connect parallel node to branch start
                add_edge(para_node_id, branch_end)
                branch_end_nodes.append(branch_end)

            # Create join node
            join_node_id = f"{para_node_id}_join"
            lines.append(f"    {join_node_id}(( ));")
            lines.append(f"    style {join_node_id} fill:none,stroke:none")

            # Connect all branches to join node
            for branch_end in branch_end_nodes:
                add_edge(branch_end, join_node_id)

            return join_node_id

        # Process the main pipeline
        process_pipeline(self)

        return "\n".join(lines)

    def _generate_medium_detail_mermaid(self) -> str:
        """Generate medium-detail Mermaid diagram with simplified structure."""
        lines = ["graph TD"]
        node_counter = 0
        step_nodes: dict[int, str] = {}

        def get_node_id(step: Step[Any, Any]) -> str:
            step_id = id(step)
            if step_id not in step_nodes:
                nonlocal node_counter
                node_counter += 1
                step_nodes[step_id] = f"s{node_counter}"
            return step_nodes[step_id]

        def add_node(step: Step[Any, Any], node_id: str) -> None:
            """Add a node definition for the given step."""
            # Simplified node shapes
            if isinstance(step, HumanInTheLoopStep):
                shape = f'["👤 {step.name}"]'
            elif isinstance(step, LoopStep):
                shape = f'("🔄 {step.name}")'
            elif isinstance(step, ConditionalStep):
                shape = f'{{"🔀 {step.name}"}}'
            elif isinstance(step, ParallelStep):
                shape = f'{{{{"⚡ {step.name}"}}}}'
            else:
                # Standard Step
                label = step.name
                # Add validation annotation if step has plugins or validators
                if step.plugins or step.validators:
                    label += " 🛡️"
                shape = f'["{label}"]'

            lines.append(f"    {node_id}{shape};")

        def add_edge(from_node: str, to_node: str, label: str | None = None) -> None:
            """Add an edge between two nodes."""
            if label:
                lines.append(f'    {from_node} --> |"{label}"| {to_node};')
            else:
                lines.append(f"    {from_node} --> {to_node};")

        def process_step(step: Step[Any, Any], prev_node: str | None = None) -> str:
            """Process a single step and return its node ID."""
            node_id = get_node_id(step)
            add_node(step, node_id)

            if prev_node:
                add_edge(prev_node, node_id)

            return node_id

        def process_pipeline(
            pipeline: Pipeline[Any, Any], prev_node: str | None = None
        ) -> str:
            """Process a pipeline and return the last node ID."""
            last_node = prev_node
            for step in pipeline.steps:
                if isinstance(step, (LoopStep, ConditionalStep, ParallelStep)):
                    # For medium detail, just show the control flow step without subgraphs
                    last_node = process_step(step, last_node)
                else:
                    last_node = process_step(step, last_node)

            return last_node

        # Process the main pipeline
        process_pipeline(self)

        return "\n".join(lines)

    def _generate_low_detail_mermaid(self) -> str:
        """Generate low-detail Mermaid diagram with minimal information."""
        lines = ["graph TD"]
        # node_counter = 0  # Removed unused variable

        # Group steps by type for high-level overview
        simple_steps = []
        control_steps = []

        for step in self.steps:
            if isinstance(
                step, (LoopStep, ConditionalStep, ParallelStep, HumanInTheLoopStep)
            ):
                control_steps.append(step)
            else:
                simple_steps.append(step)

        # Create simplified nodes
        if simple_steps:
            simple_node_id = "s1"
            simple_names = [step.name for step in simple_steps]
            lines.append(
                f'    {simple_node_id}["Processing: {", ".join(simple_names)}"];'
            )
            current_node = simple_node_id
        else:
            current_node = None

        # Add control flow steps
        for i, step in enumerate(control_steps, start=2):
            node_id = f"s{i}"
            if isinstance(step, LoopStep):
                lines.append(f'    {node_id}("🔄 {step.name}")')
            elif isinstance(step, ConditionalStep):
                lines.append(f'    {node_id}{{"🔀 {step.name}"}}')
            elif isinstance(step, ParallelStep):
                lines.append(f"    {node_id}{{{{⚡ {step.name}}}}}")
            elif isinstance(step, HumanInTheLoopStep):
                lines.append(f'    {node_id}["👤 {step.name}"]')

            if current_node:
                lines.append(f"    {current_node} --> {node_id};")
            current_node = node_id

        return "\n".join(lines)


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
