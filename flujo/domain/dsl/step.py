from __future__ import annotations
from flujo.type_definitions.common import JSONObject

# NOTE: This module was extracted from flujo.domain.pipeline_dsl as part of FSD1 refactor
# It contains the core Step DSL primitives (StepConfig, Step, decorators, etc.)
# Original implementation remains largely unchanged aside from relative import updates
# and lazy imports to avoid circular dependencies with other DSL modules.

from typing import (
    Any,
    Callable,
    Coroutine,
    Generic,
    List,
    NoReturn,
    Optional,
    TypeVar,
    Dict,
    Type,
    Union,
    TYPE_CHECKING,
    Literal,
    get_origin,
)

try:
    from typing import ParamSpec, Concatenate
except ImportError:
    from typing_extensions import ParamSpec, Concatenate
import contextvars
import inspect
from enum import Enum

from flujo.domain.base_model import BaseModel
from flujo.domain.models import RefinementCheck, UsageLimits  # noqa: F401
from flujo.domain.resources import AppResources
from pydantic import Field, model_validator
from ..agent_protocol import AsyncAgentProtocol
from ..plugins import ValidationPlugin
from ..validation import Validator

from ..processors import AgentProcessors
from ...exceptions import StepInvocationError

ExecutionMode = Literal["sync", "background"]

if TYPE_CHECKING:  # pragma: no cover
    from ..caching import CacheBackend
    from .loop import LoopStep, MapStep
    from .conditional import ConditionalStep
    from .parallel import ParallelStep
    from .pipeline import Pipeline
    from .dynamic_router import DynamicParallelRouterStep
    from .cache_step import CacheStep

# Type variables
StepInT = TypeVar("StepInT")
StepOutT = TypeVar("StepOutT")
NewOutT = TypeVar("NewOutT")
P = ParamSpec("P")

ContextModelT = TypeVar("ContextModelT", bound=BaseModel)

# BranchKey type alias for ConditionalStep.
# Keys must be JSON/YAML-friendly and stable for persistence/serialization.
BranchKey = str | bool | int
InvariantRule = str | Callable[..., bool]

_ModelT = TypeVar("_ModelT")

if TYPE_CHECKING:  # pragma: no cover

    def _typed_model_validator(
        *, mode: Literal["after"]
    ) -> Callable[[Callable[[_ModelT], _ModelT]], Callable[[_ModelT], _ModelT]]: ...

else:

    def _typed_model_validator(
        *, mode: Literal["after"]
    ) -> Callable[[Callable[[_ModelT], _ModelT]], Callable[[_ModelT], _ModelT]]:
        return model_validator(mode=mode)


class MergeStrategy(Enum):
    """Strategies for merging branch contexts back into the main context.

    The CONTEXT_UPDATE strategy is recommended for most use cases as it provides
    proper validation and handles complex context structures safely. Use NO_MERGE
    for performance-critical scenarios where context updates are not needed.
    """

    NO_MERGE = "no_merge"
    OVERWRITE = "overwrite"
    CONTEXT_UPDATE = "context_update"  # Proper context updates with validation
    ERROR_ON_CONFLICT = "error_on_conflict"  # Explicitly error on any conflicting field
    KEEP_FIRST = "keep_first"  # Keep first occurrence of each key when merging
    # Note: CONTEXT_UPDATE performs conflict detection at policy level and raises on conflicts


class BranchFailureStrategy(Enum):
    """Policies for handling branch failures in ``ParallelStep``."""

    PROPAGATE = "propagate"
    IGNORE = "ignore"


_TYPE_FALLBACK: type[object] = object


def _log_type_warning(message: str, step_name: str, *, error: Exception | None = None) -> None:
    """Best-effort telemetry logging without introducing import-time cycles."""
    try:
        from flujo.infra import telemetry as _telemetry

        extra: dict[str, object] = {"step": step_name}
        if error is not None:
            extra["error"] = str(error)
        _telemetry.logfire.warn(message, extra=extra)
    except Exception:
        # Telemetry must never break DSL import or execution
        return


def _normalize_signature_type(candidate: object) -> type[object]:
    """Map signature-derived types to safe, non-Any fallbacks."""
    if candidate is Any or candidate is None or candidate is type(None):  # noqa: E721
        return _TYPE_FALLBACK
    try:
        origin = get_origin(candidate)
    except Exception:
        origin = None
    if origin is dict:
        return dict
    if isinstance(candidate, type):
        return candidate
    return _TYPE_FALLBACK


def _infer_agent_io_types(agent: object, *, step_name: str) -> tuple[type[object], type[object]]:
    """Infer input/output types from an agent or callable, defaulting to object on failure."""
    executable = (
        agent.run
        if hasattr(agent, "run") and callable(getattr(agent, "run"))
        else (agent if callable(agent) else None)
    )
    if executable is None:
        _log_type_warning(
            "Unable to infer step IO types; agent exposes no executable", step_name=step_name
        )
        return (_TYPE_FALLBACK, _TYPE_FALLBACK)

    try:
        from flujo.signature_tools import analyze_signature

        sig_info = analyze_signature(executable)
        return (
            _normalize_signature_type(getattr(sig_info, "input_type", _TYPE_FALLBACK)),
            _normalize_signature_type(getattr(sig_info, "output_type", _TYPE_FALLBACK)),
        )
    except Exception as exc:
        _log_type_warning(
            "Failed to infer step IO types from agent signature; using object fallback",
            step_name=step_name,
            error=exc,
        )
        return (_TYPE_FALLBACK, _TYPE_FALLBACK)


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
    execution_mode: ExecutionMode = "sync"


class Step(BaseModel, Generic[StepInT, StepOutT]):
    """Declarative node in a pipeline.

    A ``Step`` holds a reference to the agent that will execute, configuration
    such as retries and timeout, and optional plugins.  It does **not** execute
    anything by itself.  Steps are composed into :class:`Pipeline` objects and
    run by the :class:`~flujo.application.runner.Flujo` engine.

    Use :meth:`arun` to invoke the underlying agent directly during unit tests.
    """

    name: str
    agent: object | None = Field(default=None)
    config: StepConfig = Field(default_factory=StepConfig)
    plugins: List[tuple[ValidationPlugin, int]] = Field(default_factory=list)
    validators: List[Validator] = Field(default_factory=list)
    static_invariants: list[InvariantRule] = Field(
        default_factory=list,
        description="Hard invariants that must hold for this step's context.",
    )
    discovery_agent: object | None = Field(
        default=None,
        description="Optional discovery agent used by TreeSearchStep to deduce invariants.",
    )
    failure_handlers: List[Callable[[], None]] = Field(default_factory=list)
    processors: "AgentProcessors" = Field(default_factory=AgentProcessors)
    fallback_step: object | None = Field(default=None, exclude=True)
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
    # Optional declarative input (templated). Alias preserves legacy ``input`` param.
    input_: object | None = Field(
        default=None,
        alias="input",
        description="Explicit step input (can be templated).",
    )
    updates_context: bool = Field(
        default=False,
        description="Whether the step output should merge into the pipeline context.",
    )
    validate_fields: bool = Field(
        default=False,
        description="Whether to validate that step return values match context fields.",
    )
    input_keys: list[str] = Field(
        default_factory=list,
        description="Context keys this step requires before execution.",
    )
    output_keys: list[str] = Field(
        default_factory=list,
        description="Context keys this step will populate upon completion.",
    )
    # Optional sink_to for simple steps: store the step's output directly into
    # a context path (e.g., "counter" or "result"). This is useful
    # when the step returns a scalar value that should be persisted in context
    # without requiring a dict-shaped output. Scratchpad is reserved and will fail validation.
    sink_to: str | None = Field(
        default=None,
        description=(
            "Context path to automatically store the step output "
            "(e.g., 'counter' or 'result'). Scratchpad targets are not allowed."
        ),
    )
    meta: JSONObject = Field(
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

    @_typed_model_validator(mode="after")
    def _validate_adapter_metadata(self) -> "Step[StepInT, StepOutT]":
        """Adapters must always declare identity and allowlist token."""
        meta = getattr(self, "meta", None)
        if isinstance(meta, dict) and meta.get("is_adapter"):
            adapter_id = meta.get("adapter_id")
            adapter_allow = meta.get("adapter_allow")
            if not adapter_id or not adapter_allow:
                raise ValueError(
                    "Adapter steps must include adapter_id and adapter_allow (allowlist token)."
                )
        return self

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

    @property
    def input(self) -> object | None:
        """Return the declarative input value (legacy alias)."""
        return self.input_

    def model_post_init(self, __context: Any) -> None:
        # Surface declarative input as templating spec so executors apply it uniformly.
        if self.input_ is not None:
            try:
                meta_dict: dict[str, Any] = dict(self.meta or {})
                meta_dict.setdefault("templated_input", self.input_)
                object.__setattr__(self, "meta", meta_dict)
            except Exception:
                # Do not fail initialization if meta isn't mutable
                pass

    def __call__(self, *args: Any, **kwargs: Any) -> NoReturn:  # pragma: no cover - behavior
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

    def arun(self, data: StepInT, **kwargs: Any) -> Coroutine[object, object, StepOutT]:
        """Return the agent coroutine to run this step directly in tests."""
        if self.agent is None:
            raise ValueError(f"Step '{self.name}' has no agent to run.")

        return self.agent.run(data, **kwargs)

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
        step_instance: "Step[Any, Any]" = cls.model_validate(
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
        input_type, output_type = _infer_agent_io_types(agent, step_name=step_instance.name)
        step_instance.__step_input_type__ = input_type
        step_instance.__step_output_type__ = output_type
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
        step_instance: "Step[Any, Any]" = cls.model_validate(
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
        input_type, output_type = _infer_agent_io_types(agent, step_name=step_instance.name)
        step_instance.__step_input_type__ = input_type
        step_instance.__step_output_type__ = output_type
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
        step_instance: "Step[Any, Any]" = cls.model_validate(
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
        )
        input_type, output_type = _infer_agent_io_types(agent, step_name=step_instance.name)
        step_instance.__step_input_type__ = input_type
        step_instance.__step_output_type__ = output_type
        return step_instance

    # ------------------------------------------------------------------
    # Pipeline construction helpers (from_callable, human_in_the_loop, etc.)
    # ------------------------------------------------------------------

    @classmethod
    def from_callable(
        cls: Type["Step[StepInT, StepOutT]"],
        callable_: Callable[Concatenate[StepInT, P], Coroutine[object, object, StepOutT]],
        name: str | None = None,
        updates_context: bool = False,
        validate_fields: bool = False,  # New parameter
        sink_to: str | None = None,  # Scalar output destination
        processors: Optional[AgentProcessors] = None,
        persist_feedback_to_context: Optional[str] = None,
        persist_validation_results_to: Optional[str] = None,
        is_adapter: bool = False,
        adapter_id: str | None = None,
        adapter_allow: str | None = None,
        config: StepConfig | None = None,
        **config_kwargs: Any,
    ) -> "Step[StepInT, StepOutT]":
        """Create a Step from an async callable."""

        if name is None:
            name = callable_.__name__

        # Infer injection signature & wrap callable into an agent-like object
        func = callable_
        sig_info = None
        try:
            from flujo.signature_tools import analyze_signature

            sig_info = analyze_signature(func)
        except Exception as exc:
            _log_type_warning(
                "Failed to analyze callable signature; using object fallback",
                step_name=name,
                error=exc,
            )

        class _CallableAgent:  # pylint: disable=too-few-public-methods
            _step_callable = func
            _injection_spec = sig_info

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
                from flujo.domain.interfaces import accepts_param as _accepts_param

                if _accepts_param(func, "context") and context is not None:
                    callable_kwargs["context"] = context
                if _accepts_param(func, "resources") and resources is not None:
                    callable_kwargs["resources"] = resources

                # Add any additional kwargs
                callable_kwargs.update(kwargs)

                # Call the original function directly
                return await func(*call_args, **callable_kwargs)

        merged_config: dict[str, Any] = {}
        if config is not None:
            merged_config.update(config.model_dump())
        merged_config.update(config_kwargs)

        # Enforce explicit adapter identity/token to prevent generic, untracked adapters.
        adapter_id_val = merged_config.pop("adapter_id", adapter_id)
        adapter_allow_val = merged_config.pop("adapter_allow", adapter_allow)

        if is_adapter:
            if not adapter_id_val or not adapter_allow_val:
                raise ValueError(
                    "Adapter steps must provide adapter_id and adapter_allow (allowlist token)."
                )
            meta: dict[str, Any] = {
                "is_adapter": True,
                "adapter_id": adapter_id_val,
                "adapter_allow": adapter_allow_val,
            }
        else:
            meta = {}

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
                "meta": meta,
                "config": StepConfig(**merged_config),
            }
        )
        # Set type info for pipeline validation
        step_instance.__step_input_type__ = _normalize_signature_type(
            getattr(sig_info, "input_type", _TYPE_FALLBACK)
        )
        step_instance.__step_output_type__ = _normalize_signature_type(
            getattr(sig_info, "output_type", _TYPE_FALLBACK)
        )
        return step_instance

    @classmethod
    def from_mapper(
        cls: Type["Step[StepInT, StepOutT]"],
        mapper: Callable[Concatenate[StepInT, P], Coroutine[object, object, StepOutT]],
        name: str | None = None,
        updates_context: bool = False,
        sink_to: str | None = None,
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
            sink_to=sink_to,
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
        last_artifact_var: contextvars.ContextVar[object | None] = contextvars.ContextVar(
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
                        outputs = getattr(context, "step_outputs", None)
                        if isinstance(outputs, dict) and "_capture_artifact" in outputs:
                            return outputs.get("_capture_artifact")
                        # 2) Fallback to any context-scoped attribute set by the capture step
                        if hasattr(context, "_last_refine_artifact"):
                            return getattr(context, "_last_refine_artifact")
                except Exception:
                    pass
            return out

        mapper_step: "Step[Any, Any]" = cls.from_callable(
            _post_output_mapper,
            name=f"{name}_output_mapper",
            is_adapter=True,
            adapter_id="generic-adapter",
            adapter_allow="generic",
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
        reduce: Callable[..., object] | None = None,
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
                "reduce": reduce,
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

    @classmethod
    def granular(
        cls,
        name: str,
        agent: Any,
        input_: Any = None,
        *,
        max_turns: int = 20,
        history_max_tokens: int = 128_000,
        blob_threshold_bytes: int = 20_000,
        enforce_idempotency: bool = False,
        **config_kwargs: Any,
    ) -> "Pipeline[Any, Any]":
        """Build a granular execution pipeline for crash-safe, resumable agent runs.

        Wraps a GranularStep in a LoopStep with abort-on-failure semantics.
        Each turn is persisted atomically with CAS guards to prevent double-execution.

        Args:
            name: Step name for identification
            agent: The agent to execute
            input_: Optional input data or template
            max_turns: Maximum number of turns before forced completion (default 20)
            history_max_tokens: Token budget for message history (default 128K)
            blob_threshold_bytes: Payload size triggering blob offload (default 20KB)
            enforce_idempotency: Require idempotency keys on tool calls (default False)
            **config_kwargs: Additional step configuration

        Returns:
            Pipeline containing LoopStep(GranularStep) with on_failure="abort"
        """
        from .loop import LoopStep
        from .granular import GranularStep
        from .pipeline import Pipeline

        # Create the inner granular step
        granular_step = GranularStep(
            name=f"{name}_turn",
            agent=agent,
            input_=input_,
            history_max_tokens=history_max_tokens,
            blob_threshold_bytes=blob_threshold_bytes,
            enforce_idempotency=enforce_idempotency,
            **config_kwargs,
        )

        # Wrap in a pipeline for loop body
        inner_pipeline = Pipeline.from_step(granular_step)

        # Exit when granular_state.is_complete is True
        def _exit_condition(output: Any, ctx: Optional[ContextModelT]) -> bool:
            # Check for completion flag in output or granular_state
            if hasattr(output, "is_complete"):
                return bool(output.is_complete)
            if ctx is not None:
                gs = getattr(ctx, "granular_state", None)
                if isinstance(gs, dict):
                    return bool(gs.get("is_complete", False))
            return False

        # Loop until complete or max_turns
        loop = LoopStep[Any](
            name=name,
            loop_body_pipeline=inner_pipeline,
            exit_condition_callable=_exit_condition,
            max_loops=max_turns,
            on_failure="abort",  # Abort on inner failure per PRD §5.3
        )

        return Pipeline.from_step(loop)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def use_input(self, key: str) -> "Pipeline[dict[str, Any], StepOutT]":
        """Create a small adapter pipeline that selects a key from a dict input.

        This is a common pattern when working with :meth:`parallel` branches
        where each branch only needs a portion of the upstream output.
        """

        async def _select(data: dict[str, Any], *, context: BaseModel | None = None) -> StepInT:
            if key not in data:
                raise KeyError(f"use_input missing key '{key}'")
            return data[key]

        adapter = Step.from_callable(
            _select,
            name=f"select_{key}",
            is_adapter=True,
            adapter_id="generic-adapter",
            adapter_allow="generic",
        )
        return Pipeline.from_step(adapter) >> self

    @classmethod
    def gather(
        cls,
        name: str,
        *,
        wait_for: List[str],
        **config_kwargs: Any,
    ) -> "Step[Any, JSONObject]":
        """Collect outputs from multiple parallel branches.

        The step expects a dictionary input (e.g. from :meth:`parallel`) and
        returns a dictionary containing only the specified keys.
        """

        async def _gather(data: Any, *, context: BaseModel | None = None) -> JSONObject:
            if not isinstance(data, dict):
                raise TypeError("Gather step expects dict input")
            return {k: data.get(k) for k in wait_for}

        return Step.from_callable(
            _gather,
            name=name,
            is_adapter=True,
            adapter_id="generic-adapter",
            adapter_allow="generic",
            **config_kwargs,
        )

    @classmethod
    def cached(
        cls,
        wrapped_step: "Step[Any, Any]",
        cache_backend: Optional[CacheBackend] = None,
    ) -> "CacheStep[Any, Any]":
        from .cache_step import CacheStep

        return CacheStep.cached(wrapped_step, cache_backend)


# ----------------------------------------------------------------------
# Helper decorator factory (step / adapter_step)
# Extracted to step_decorators.py - re-exported here for backward compatibility
# ----------------------------------------------------------------------

from .step_decorators import step, adapter_step  # noqa: E402


class HumanInTheLoopStep(Step[Any, Any]):
    """A step that pauses the pipeline for human input.

    Attributes:
        message_for_user: Optional message to display to the user
        input_schema: Optional schema for validating user input
        sink_to: Optional context path to automatically store the human response
                 (e.g., "hitl_data.user_answer" or "user_input")
    """

    message_for_user: str | None = Field(default=None)
    input_schema: Any | None = Field(default=None)
    sink_to: str | None = Field(
        default=None,
        description="Context path to automatically store the human response (e.g., 'hitl_data.user_name')",
    )

    # model_config inherited from BaseModel

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
