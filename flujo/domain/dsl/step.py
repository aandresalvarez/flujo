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
    Type,
    ParamSpec,
    Concatenate,
    overload,
    Union,
    cast,
    TYPE_CHECKING,
)

# contextvars import removed; using context-scoped storage for refine_until
import inspect
from enum import Enum
import contextvars

from flujo.domain.base_model import BaseModel
from flujo.domain.models import RefinementCheck, UsageLimits  # noqa: F401
from flujo.domain.resources import AppResources
from pydantic import Field, ConfigDict
from ..agent_protocol import AsyncAgentProtocol
from ..plugins import ValidationPlugin
from ..validation import Validator

from ..processors import AgentProcessors
from ...exceptions import StepInvocationError

if TYPE_CHECKING:  # pragma: no cover
    from flujo.infra.caching import CacheBackend
    from .loop import LoopStep, MapStep
    from .conditional import ConditionalStep
    from .parallel import ParallelStep
    from .pipeline import Pipeline
    from .dynamic_router import DynamicParallelRouterStep
    from flujo.steps.cache_step import CacheStep

# Type variables
StepInT = TypeVar("StepInT")
StepOutT = TypeVar("StepOutT")
NewOutT = TypeVar("NewOutT")
P = ParamSpec("P")

ContextModelT = TypeVar("ContextModelT", bound=BaseModel)

# BranchKey type alias for ConditionalStep
BranchKey = Any


class MergeStrategy(Enum):
    """Strategies for merging branch contexts back into the main context.

    The CONTEXT_UPDATE strategy is recommended for most use cases as it provides
    proper validation and handles complex context structures safely. Use NO_MERGE
    for performance-critical scenarios where context updates are not needed.
    """

    NO_MERGE = "no_merge"
    OVERWRITE = "overwrite"
    MERGE_SCRATCHPAD = "merge_scratchpad"
    CONTEXT_UPDATE = "context_update"  # Proper context updates with validation
    ERROR_ON_CONFLICT = "error_on_conflict"  # Explicitly error on any conflicting field
    KEEP_FIRST = "keep_first"  # Keep first occurrence of each key when merging
    # Note: CONTEXT_UPDATE performs conflict detection at policy level and raises on conflicts


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
    top_k:
        Optional top-k sampling parameter for LLM based agents.
    top_p:
        Optional nucleus sampling parameter for LLM based agents.
    preserve_fallback_diagnostics:
        Whether to preserve diagnostic feedback from fallback executions.
        When True, successful fallbacks retain feedback for monitoring/debugging.
        When False, successful fallbacks clear feedback for backward compatibility.
    """

    # Default to one retry (two attempts total) to match legacy/test semantics
    max_retries: int = 1
    timeout_s: float | None = None
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    preserve_fallback_diagnostics: bool = False


class Step(BaseModel, Generic[StepInT, StepOutT]):
    """Declarative node in a pipeline.

    A ``Step`` holds a reference to the agent that will execute, configuration
    such as retries and timeout, and optional plugins.  It does **not** execute
    anything by itself.  Steps are composed into :class:`Pipeline` objects and
    run by the :class:`~flujo.application.runner.Flujo` engine.

    Use :meth:`arun` to invoke the underlying agent directly during unit tests.
    """

    name: str
    agent: Any | None = Field(default=None)
    config: StepConfig = Field(default_factory=StepConfig)
    plugins: List[tuple[ValidationPlugin, int]] = Field(default_factory=list)
    validators: List[Validator] = Field(default_factory=list)
    failure_handlers: List[Callable[[], None]] = Field(default_factory=list)
    processors: "AgentProcessors" = Field(default_factory=AgentProcessors)
    fallback_step: Optional[Any] = Field(default=None, exclude=True)
    usage_limits: Optional[UsageLimits] = Field(
        default=None,
        description="Usage limits for this step (cost and token limits).",
    )
    persist_feedback_to_context: Optional[str] = Field(
        default=None,
        description=("If step fails, append feedback to this context attribute (must be a list)."),
    )
    persist_validation_results_to: Optional[str] = Field(
        default=None,
        description=("Append ValidationResult objects to this context attribute (must be a list)."),
    )
    updates_context: bool = Field(
        default=False,
        description="Whether the step output should merge into the pipeline context.",
    )
    validate_fields: bool = Field(
        default=False,
        description="Whether to validate that step return values match context fields.",
    )
    # Optional sink_to for simple steps: store the step's output directly into
    # a context path (e.g., "counter" or "scratchpad.field"). This is useful
    # when the step returns a scalar value that should be persisted in context
    # without requiring a dict-shaped output.
    sink_to: str | None = Field(
        default=None,
        description=(
            "Context path to automatically store the step output "
            "(e.g., 'counter' or 'scratchpad.value')."
        ),
    )
    meta: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata about this step.",
    )

    # Default to the top-level "object" type to satisfy static typing
    __step_input_type__: type[Any] = object
    __step_output_type__: type[Any] = object

    # Note: Avoid defining an inner `Config` class alongside `model_config`.
    # Pydantic v2 raises if both are present. Use only `model_config` here.

    @property
    def is_complex(self) -> bool:
        # ✅ Base steps are not complex by default.
        return False

    model_config: ClassVar[ConfigDict] = {
        "arbitrary_types_allowed": True,
    }

    # ---------------------------------------------------------------------
    # Utility / dunder helpers
    # ---------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover - simple utility
        agent_repr: str
        if self.agent is None:
            agent_repr = "None"
        else:
            target = getattr(self.agent, "_agent", self.agent)
            if hasattr(target, "__name__"):
                agent_repr = f"<function {target.__name__}>"
            elif hasattr(self.agent, "_model_name"):
                agent_repr = (
                    f"AsyncAgentWrapper(model={getattr(self.agent, '_model_name', 'unknown')})"
                )
            else:
                agent_repr = self.agent.__class__.__name__
        config_repr = ""
        default_config = StepConfig()
        if self.config != default_config:
            config_repr = f", config={self.config!r}"
        return f"Step(name={self.name!r}, agent={agent_repr}{config_repr})"

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - behavior
        """Disallow direct invocation of a Step."""
        from ...exceptions import ImproperStepInvocationError

        raise ImproperStepInvocationError(
            f"Step '{self.name}' cannot be invoked directly. "
            "Steps are configuration objects and must be run within a Pipeline. "
            "For unit testing, use `step.arun()`."
        )

    def __getattr__(self, item: str) -> Any:  # pragma: no cover - behavior
        """Raise a helpful error when trying to access non-existent attributes."""
        # Check if this is a legitimate framework attribute that should exist
        if item in {
            "callable",
            "agent",
            "config",
            "name",
            "processors",
            "validators",
            "fallback_step",
        }:
            # These attributes should exist on the Step object
            # If they don't, it's a legitimate AttributeError
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'")

        # Check if this is an inspection-related attribute that should be accessible
        if item in {
            "__wrapped__",
            "__doc__",
            "__name__",
            "__qualname__",
            "__module__",
            "__annotations__",
            "__defaults__",
            "__kwdefaults__",
        }:
            # These are legitimate Python introspection attributes
            # If they don't exist, raise standard AttributeError
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'")

        # Check if this is a method that indicates direct step invocation
        if item in {"run", "stream"}:
            raise StepInvocationError(self.name)

        # For all other missing attributes, raise standard AttributeError
        # This is the correct behavior for missing attributes
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'")

    # ------------------------------------------------------------------
    # Composition helpers ( >> operator )
    # ------------------------------------------------------------------

    def __rshift__(
        self, other: "Step[StepOutT, NewOutT]" | "Pipeline[StepOutT, NewOutT]"
    ) -> "Pipeline[StepInT, NewOutT]":
        from .pipeline import Pipeline  # local import to avoid circular

        if isinstance(other, Step):
            return Pipeline.from_step(self) >> other
        if isinstance(other, Pipeline):
            return Pipeline.from_step(self) >> other
        raise TypeError("Can only chain Step with Step or Pipeline")

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------

    def arun(self, data: StepInT, **kwargs: Any) -> Coroutine[Any, Any, StepOutT]:
        """Return the agent coroutine to run this step directly in tests."""
        if self.agent is None:
            raise ValueError(f"Step '{self.name}' has no agent to run.")

        return cast(Coroutine[Any, Any, StepOutT], self.agent.run(data, **kwargs))

    def fallback(self, fallback_step: "Step[Any, Any]") -> "Step[StepInT, StepOutT]":
        """Set a fallback step to execute if this step fails.

        Args:
            fallback_step: The step to execute if this step fails

        Returns:
            self for method chaining
        """
        self.fallback_step = fallback_step
        return self

    def add_plugin(self, plugin: "ValidationPlugin") -> "Step[StepInT, StepOutT]":
        """Add a validation plugin to this step.

        Args:
            plugin: The validation plugin to add

        Returns:
            self for method chaining
        """
        self.plugins.append((plugin, 0))  # Priority 0 for default
        return self

    # ------------------------------------------------------------------
    # Convenience class constructors (review / solution / validate_step)
    # ------------------------------------------------------------------

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
        """Construct a review step using the provided agent.

        Also infers input/output types from the agent's signature for pipeline validation.
        """
        step_instance = cast(
            "Step[Any, Any]",
            cls.model_validate(
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
            ),
        )

        # Infer types from agent.run or the agent itself if callable
        try:
            from flujo.signature_tools import analyze_signature

            executable = (
                agent.run
                if hasattr(agent, "run") and callable(getattr(agent, "run"))
                else (agent if callable(agent) else None)
            )
            if executable is not None:
                sig_info = analyze_signature(executable)
                step_instance.__step_input_type__ = getattr(sig_info, "input_type", Any)
                step_instance.__step_output_type__ = getattr(sig_info, "output_type", Any)
        except Exception:
            step_instance.__step_input_type__ = Any
            step_instance.__step_output_type__ = Any

        return step_instance

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
        """Construct a solution step using the provided agent.

        Also infers input/output types from the agent's signature for pipeline validation.
        """
        step_instance = cast(
            "Step[Any, Any]",
            cls.model_validate(
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
            ),
        )

        # Infer types from agent.run or the agent itself if callable
        try:
            from flujo.signature_tools import analyze_signature

            executable = (
                agent.run
                if hasattr(agent, "run") and callable(getattr(agent, "run"))
                else (agent if callable(agent) else None)
            )
            if executable is not None:
                sig_info = analyze_signature(executable)
                step_instance.__step_input_type__ = getattr(sig_info, "input_type", Any)
                step_instance.__step_output_type__ = getattr(sig_info, "output_type", Any)
        except Exception:
            step_instance.__step_input_type__ = Any
            step_instance.__step_output_type__ = Any

        return step_instance

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
        """Construct a validation step using the provided agent.

        Also infers input/output types from the agent's signature for pipeline validation.
        """
        step_instance = cast(
            "Step[Any, Any]",
            cls.model_validate(
                {
                    "name": "validate",
                    "agent": agent,
                    "plugins": plugins or [],
                    "validators": validators or [],
                    "processors": processors or AgentProcessors(),
                    "persist_feedback_to_context": persist_feedback_to_context,
                    "persist_validation_results_to": persist_validation_results_to,
                    "config": StepConfig(**config),
                    "meta": {
                        "is_validation_step": True,
                        "strict_validation": strict,
                    },
                }
            ),
        )

        # Infer types from agent.run or the agent itself if callable
        try:
            from flujo.signature_tools import analyze_signature

            executable = (
                agent.run
                if hasattr(agent, "run") and callable(getattr(agent, "run"))
                else (agent if callable(agent) else None)
            )
            if executable is not None:
                sig_info = analyze_signature(executable)
                step_instance.__step_input_type__ = getattr(sig_info, "input_type", Any)
                step_instance.__step_output_type__ = getattr(sig_info, "output_type", Any)
        except Exception:
            step_instance.__step_input_type__ = Any
            step_instance.__step_output_type__ = Any

        return step_instance

    # ------------------------------------------------------------------
    # Pipeline construction helpers (from_callable, human_in_the_loop, etc.)
    # ------------------------------------------------------------------

    @classmethod
    def from_callable(
        cls: Type["Step[StepInT, StepOutT]"],
        callable_: Callable[Concatenate[StepInT, P], Coroutine[Any, Any, StepOutT]],
        name: str | None = None,
        updates_context: bool = False,
        validate_fields: bool = False,  # New parameter
        sink_to: str | None = None,  # Scalar output destination
        processors: Optional[AgentProcessors] = None,
        persist_feedback_to_context: Optional[str] = None,
        persist_validation_results_to: Optional[str] = None,
        is_adapter: bool = False,
        **config: Any,
    ) -> "Step[StepInT, StepOutT]":
        """Create a Step from an async callable."""

        if name is None:
            name = callable_.__name__

        # Infer injection signature & wrap callable into an agent-like object
        func = callable_
        from flujo.signature_tools import analyze_signature

        class _CallableAgent:  # pylint: disable=too-few-public-methods
            _step_callable = func
            _injection_spec = analyze_signature(func)

            # Store the original function signature for parameter names
            _original_sig = inspect.signature(func)

            async def run(
                self,
                data: Any,
                *,
                context: BaseModel | None = None,
                resources: AppResources | None = None,
                temperature: float | None = None,
                **kwargs: Any,
            ) -> Any:  # noqa: D401
                # Build the arguments to pass to the callable
                call_args: list[Any] = []
                callable_kwargs: dict[str, Any] = {}

                params = list(self._original_sig.parameters.values())
                if params:
                    first_param = params[0]
                    # Pass data positionally for common kinds including *args
                    if first_param.kind in (
                        inspect.Parameter.POSITIONAL_ONLY,
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        inspect.Parameter.VAR_POSITIONAL,
                    ):
                        call_args.append(data)
                    else:
                        # Fallback to keyword by name
                        callable_kwargs[first_param.name] = data

                # Add the injected arguments if the callable needs them
                from flujo.application.core.context_manager import _accepts_param

                if _accepts_param(func, "context") and context is not None:
                    callable_kwargs["context"] = context
                if _accepts_param(func, "resources") and resources is not None:
                    callable_kwargs["resources"] = resources

                # Add any additional kwargs
                callable_kwargs.update(kwargs)

                # Call the original function directly
                return await cast(Callable[..., Any], func)(*call_args, **callable_kwargs)

        # Analyze signature for type info
        from flujo.signature_tools import analyze_signature

        sig_info = analyze_signature(func)
        input_type = sig_info.input_type if hasattr(sig_info, "input_type") else Any
        output_type = sig_info.output_type if hasattr(sig_info, "output_type") else Any

        step_instance = cls.model_validate(
            {
                "name": name,
                "agent": _CallableAgent(),
                "plugins": [],
                "validators": [],
                "processors": processors or AgentProcessors(),
                "persist_feedback_to_context": persist_feedback_to_context,
                "persist_validation_results_to": persist_validation_results_to,
                "updates_context": updates_context,
                "validate_fields": validate_fields,
                "sink_to": sink_to,
                "meta": {"is_adapter": True} if is_adapter else {},
                "config": StepConfig(**config),
            }
        )
        # Set type info for pipeline validation
        step_instance.__step_input_type__ = input_type
        step_instance.__step_output_type__ = output_type
        return step_instance

    @classmethod
    def from_mapper(
        cls: Type["Step[StepInT, StepOutT]"],
        mapper: Callable[Concatenate[StepInT, P], Coroutine[Any, Any, StepOutT]],
        name: str | None = None,
        updates_context: bool = False,
        processors: Optional[AgentProcessors] = None,
        persist_feedback_to_context: Optional[str] = None,
        persist_validation_results_to: Optional[str] = None,
        **config: Any,
    ) -> "Step[StepInT, StepOutT]":
        """Alias for :meth:`from_callable` to improve readability."""
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
        """Construct a HumanInTheLoop step."""
        return HumanInTheLoopStep(
            name=name,
            message_for_user=message_for_user,
            input_schema=input_schema,
        )

    # ------------------------------------------------------------------
    # Higher-order Step factories (loop_until, branch_on, etc.)
    # ------------------------------------------------------------------

    @classmethod
    def loop_until(
        cls,
        name: str,
        loop_body_pipeline: "Pipeline[Any, Any]",
        exit_condition_callable: Callable[[Any, Optional[ContextModelT]], bool],
        max_loops: int = 5,
        initial_input_to_loop_body_mapper: Optional[
            Callable[[Any, Optional[ContextModelT]], Any]
        ] = None,
        iteration_input_mapper: Optional[Callable[[Any, Optional[ContextModelT], int], Any]] = None,
        loop_output_mapper: Optional[Callable[[Any, Optional[ContextModelT]], Any]] = None,
        **config_kwargs: Any,
    ) -> "LoopStep[ContextModelT]":
        from .loop import LoopStep  # local import to avoid circular

        return LoopStep[ContextModelT](
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
    def refine_until(
        cls,
        name: str,
        generator_pipeline: "Pipeline[Any, Any]",
        critic_pipeline: "Pipeline[Any, RefinementCheck]",
        max_refinements: int = 5,
        feedback_mapper: Optional[Callable[[Any, RefinementCheck], Any]] = None,
        **config_kwargs: Any,
    ) -> "Pipeline[Any, Any]":
        """Build a refinement loop pipeline: generator >> capture >> critic >> post-mapper."""
        from .loop import LoopStep  # local import

        # Task-local storage for the last artifact; safe under concurrency
        last_artifact_var: contextvars.ContextVar[Any | None] = contextvars.ContextVar(
            f"{name}_last_artifact", default=None
        )

        # Use context-scoped storage too when available to aid inspection
        async def _capture_artifact(artifact: Any, *, context: BaseModel | None = None) -> Any:
            try:
                if context is not None:
                    object.__setattr__(context, "_last_refine_artifact", artifact)
            except Exception:
                pass
            try:
                last_artifact_var.set(artifact)
            except Exception:
                pass
            return artifact

        capture_step = Step.from_callable(_capture_artifact, name="_capture_artifact")
        generator_then_save = generator_pipeline >> capture_step

        def _exit_condition(out: Any, _ctx: BaseModel | None) -> bool:
            return out.is_complete if isinstance(out, RefinementCheck) else True

        def _initial_mapper(inp: Any, ctx: BaseModel | None) -> dict[str, Any]:
            result = {
                "original_input": inp,
                "feedback": None,
            }
            # Update context with the values using object.__setattr__ to bypass Pydantic validation
            if ctx is not None:
                for key, value in result.items():
                    object.__setattr__(ctx, key, value)
            return result

        def _iteration_mapper(out: Any, ctx: BaseModel | None, _i: int) -> dict[str, Any]:
            if feedback_mapper is None:
                # If no feedback_mapper provided, use the feedback from RefinementCheck directly
                feedback = out.feedback if isinstance(out, RefinementCheck) else None
                result = {
                    "original_input": getattr(
                        ctx, "original_input", None
                    ),  # Safe access to context attribute
                    "feedback": feedback,
                }
            else:
                # Use the feedback_mapper to get both original_input and feedback
                original_input = getattr(ctx, "original_input", None)
                mapped_result = feedback_mapper(original_input, out)
                result = mapped_result

            # Update context with the values using object.__setattr__ to bypass Pydantic validation
            if ctx is not None:
                for key, value in result.items():
                    object.__setattr__(ctx, key, value)
            return result

        # Build the core loop step without output mapping
        core_loop = LoopStep[Any](
            name=name,
            loop_body_pipeline=generator_then_save >> critic_pipeline,
            exit_condition_callable=_exit_condition,
            max_loops=max_refinements,
            initial_input_to_loop_body_mapper=_initial_mapper,
            iteration_input_mapper=_iteration_mapper,
            **config_kwargs,
        )

        # Post-loop mapper that only maps on successful refinement (exit condition)
        async def _post_output_mapper(out: Any, *, context: BaseModel | None = None) -> Any:
            # If the critic indicates completion, return the last captured artifact; otherwise, return the check
            if isinstance(out, RefinementCheck) and out.is_complete:
                try:
                    if context is not None:
                        # 0) Prefer the task-local capture when present
                        try:
                            la = last_artifact_var.get()
                        except Exception:
                            la = None
                        if la is not None:
                            return la
                        # 1) Prefer the exact captured artifact recorded during the last iteration
                        sp = getattr(context, "scratchpad", None)
                        if isinstance(sp, dict):
                            steps = sp.get("steps") or {}
                            if isinstance(steps, dict) and "_capture_artifact" in steps:
                                return steps.get("_capture_artifact")
                        # 2) Fallback to any context-scoped attribute set by the capture step
                        if hasattr(context, "_last_refine_artifact"):
                            return getattr(context, "_last_refine_artifact")
                except Exception:
                    pass
            return out

        mapper_step = cast(
            "Step[Any, Any]",
            cls.from_callable(_post_output_mapper, name=f"{name}_output_mapper", is_adapter=True),
        )
        # Compose pipeline: loop then post mapping step
        return core_loop >> mapper_step

    @classmethod
    def branch_on(
        cls,
        name: str,
        condition_callable: Callable[[Any, Optional[ContextModelT]], BranchKey],
        branches: Dict[BranchKey, "Pipeline[Any, Any]"],
        default_branch_pipeline: Optional["Pipeline[Any, Any]"] = None,
        branch_input_mapper: Optional[Callable[[Any, Optional[ContextModelT]], Any]] = None,
        branch_output_mapper: Optional[
            Callable[[Any, BranchKey, Optional[ContextModelT]], Any]
        ] = None,
        **config_kwargs: Any,
    ) -> "ConditionalStep[ContextModelT]":
        from .conditional import ConditionalStep  # local import

        return ConditionalStep[ContextModelT](
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
        context_include_keys: Optional[List[str]] = None,
        merge_strategy: Union[
            MergeStrategy, Callable[[ContextModelT, ContextModelT], None]
        ] = MergeStrategy.CONTEXT_UPDATE,
        on_branch_failure: BranchFailureStrategy = BranchFailureStrategy.PROPAGATE,
        field_mapping: Optional[Dict[str, List[str]]] = None,
        ignore_branch_names: bool = False,
        **config_kwargs: Any,
    ) -> "ParallelStep[ContextModelT]":
        from .parallel import ParallelStep  # local import

        return ParallelStep[ContextModelT].model_validate(
            {
                "name": name,
                "branches": branches,
                "context_include_keys": context_include_keys,
                "merge_strategy": merge_strategy,
                "on_branch_failure": on_branch_failure,
                "field_mapping": field_mapping,
                "ignore_branch_names": ignore_branch_names,
                **config_kwargs,
            }
        )

    @classmethod
    def dynamic_parallel_branch(
        cls,
        name: str,
        router_agent: Any,
        branches: Dict[str, "Step[Any, Any]" | "Pipeline[Any, Any]"],
        context_include_keys: Optional[List[str]] = None,
        merge_strategy: Union[
            MergeStrategy, Callable[[ContextModelT, ContextModelT], None]
        ] = MergeStrategy.CONTEXT_UPDATE,
        on_branch_failure: BranchFailureStrategy = BranchFailureStrategy.PROPAGATE,
        field_mapping: Optional[Dict[str, List[str]]] = None,
        **config_kwargs: Any,
    ) -> "DynamicParallelRouterStep[ContextModelT]":
        from .dynamic_router import DynamicParallelRouterStep  # local import

        return DynamicParallelRouterStep[ContextModelT].model_validate(
            {
                "name": name,
                "router_agent": router_agent,
                "branches": branches,
                "context_include_keys": context_include_keys,
                "merge_strategy": merge_strategy,
                "on_branch_failure": on_branch_failure,
                "field_mapping": field_mapping,
                **config_kwargs,
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
    ) -> "MapStep[ContextModelT]":
        from .loop import MapStep  # local import

        return MapStep[ContextModelT](
            name=name,
            pipeline_to_run=pipeline_to_run,
            iterable_input=iterable_input,
            **config_kwargs,
        )

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def use_input(self, key: str) -> "Pipeline[Any, StepOutT]":
        """Create a small adapter pipeline that selects a key from a dict input.

        This is a common pattern when working with :meth:`parallel` branches
        where each branch only needs a portion of the upstream output.
        """

        async def _select(data: Any, *, context: BaseModel | None = None) -> Any:
            if isinstance(data, dict):
                return data.get(key)
            raise TypeError("use_input expects a dict-like input")

        adapter = Step.from_callable(_select, name=f"select_{key}", is_adapter=True)
        return Pipeline.from_step(adapter) >> self

    @classmethod
    def gather(
        cls,
        name: str,
        *,
        wait_for: List[str],
        **config_kwargs: Any,
    ) -> "Step[Any, Dict[str, Any]]":
        """Collect outputs from multiple parallel branches.

        The step expects a dictionary input (e.g. from :meth:`parallel`) and
        returns a dictionary containing only the specified keys.
        """

        async def _gather(data: Any, *, context: BaseModel | None = None) -> Dict[str, Any]:
            if not isinstance(data, dict):
                raise TypeError("Gather step expects dict input")
            return {k: data.get(k) for k in wait_for}

        return cast(
            "Step[Any, Dict[str, Any]]",
            cls.from_callable(_gather, name=name, is_adapter=True, **config_kwargs),
        )

    @classmethod
    def cached(
        cls,
        wrapped_step: "Step[Any, Any]",
        cache_backend: Optional[CacheBackend] = None,
    ) -> "CacheStep[Any, Any]":
        from flujo.steps.cache_step import CacheStep

        return CacheStep.cached(wrapped_step, cache_backend)


# ----------------------------------------------------------------------
# Helper decorator factory (step / adapter_step)
# ----------------------------------------------------------------------


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
) -> "Step[StepInT, StepOutT]": ...


@overload
def step(
    *,
    updates_context: bool = False,
    validate_fields: bool = False,  # New parameter for field validation
    sink_to: str | None = None,  # Scalar output destination
    name: Optional[str] = None,
    **config_kwargs: Any,
) -> Callable[
    [Callable[Concatenate[Any, P], Coroutine[Any, Any, Any]]],
    "Step[StepInT, StepOutT]",
]: ...


def step(
    func: (Callable[Concatenate[StepInT, P], Coroutine[Any, Any, StepOutT]] | None) = None,
    *,
    name: str | None = None,
    updates_context: bool = False,
    validate_fields: bool = False,  # New parameter for field validation
    sink_to: str | None = None,  # Scalar output destination
    processors: Optional[AgentProcessors] = None,
    persist_feedback_to_context: Optional[str] = None,
    persist_validation_results_to: Optional[str] = None,
    is_adapter: bool = False,
    **config_kwargs: Any,
) -> Any:
    """Decorator / factory for creating :class:`Step` instances from async callables."""

    def decorator(
        fn: Callable[Concatenate[StepInT, P], Coroutine[Any, Any, StepOutT]],
    ) -> "Step[StepInT, StepOutT]":
        return Step.from_callable(
            fn,
            name=name or fn.__name__,
            updates_context=updates_context,
            validate_fields=validate_fields,  # Pass the new parameter
            sink_to=sink_to,  # Pass sink_to parameter
            processors=processors,
            persist_feedback_to_context=persist_feedback_to_context,
            persist_validation_results_to=persist_validation_results_to,
            is_adapter=is_adapter,
            **config_kwargs,
        )

    # If used without parentheses, func is the callable
    if func is not None:
        return decorator(func)

    return decorator


@overload
def adapter_step(
    func: Callable[Concatenate[StepInT, P], Coroutine[Any, Any, StepOutT]],
    *,
    name: str | None = None,
    updates_context: bool = False,
    processors: Optional[AgentProcessors] = None,
    persist_feedback_to_context: Optional[str] = None,
    persist_validation_results_to: Optional[str] = None,
    **config_kwargs: Any,
) -> "Step[StepInT, StepOutT]": ...


@overload
def adapter_step(
    *,
    name: str | None = None,
    updates_context: bool = False,
    processors: Optional[AgentProcessors] = None,
    persist_feedback_to_context: Optional[str] = None,
    persist_validation_results_to: Optional[str] = None,
    **config_kwargs: Any,
) -> Callable[
    [Callable[Concatenate[StepInT, P], Coroutine[Any, Any, StepOutT]]],
    "Step[StepInT, StepOutT]",
]: ...


def adapter_step(
    func: (Callable[Concatenate[StepInT, P], Coroutine[Any, Any, StepOutT]] | None) = None,
    **kwargs: Any,
) -> Any:
    """Alias for :func:`step` that marks the created step as an adapter."""
    return cast(Any, step)(func, is_adapter=True, **kwargs)


class HumanInTheLoopStep(Step[Any, Any]):
    """A step that pauses the pipeline for human input.

    Attributes:
        message_for_user: Optional message to display to the user
        input_schema: Optional schema for validating user input
        sink_to: Optional context path to automatically store the human response
                 (e.g., "scratchpad.user_answer" or "scratchpad.nested.field")
    """

    message_for_user: str | None = Field(default=None)
    input_schema: Any | None = Field(default=None)
    sink_to: str | None = Field(
        default=None,
        description="Context path to automatically store the human response (e.g., 'scratchpad.user_name')",
    )

    model_config = {"arbitrary_types_allowed": True}

    @property
    def is_complex(self) -> bool:
        # ✅ Override to mark as complex.
        return True


__all__ = [
    # Core classes
    "StepConfig",
    "Step",
    "HumanInTheLoopStep",
    # Decorators / helpers
    "step",
    "adapter_step",
    # Enums / aliases
    "MergeStrategy",
    "BranchFailureStrategy",
    "BranchKey",
]
