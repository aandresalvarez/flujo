from __future__ import annotations

# NOTE: Extracted LoopStep and MapStep from pipeline_dsl for FSD1 refactor.

from typing import (
    Any,
    Callable,
    Generic,
    List,
    Optional,
    TypeVar,
    Iterable,
    Self,
    ClassVar,
)
import contextvars
from typing import TYPE_CHECKING

from pydantic import Field

from ..models import BaseModel
from .step import Step, StepConfig, StepType, _resolve_ref
from .pipeline import Pipeline  # Import for runtime use in MapStep

if TYPE_CHECKING:
    from ..ir import StepIR

# Generic type var reused
TContext = TypeVar("TContext", bound=BaseModel)

__all__ = ["LoopStep", "MapStep"]


class LoopStep(Step[Any, Any], Generic[TContext]):
    """Execute a sub-pipeline repeatedly until a condition is met.

    ``LoopStep`` runs ``loop_body_pipeline`` one or more times. After each
    iteration ``exit_condition_callable`` is evaluated with the last output and
    the current context. When it returns ``True`` the loop stops and the final
    output is returned.
    """

    step_type: ClassVar[StepType] = StepType.LOOP

    loop_body_pipeline: Any = Field(description="The pipeline to execute in each iteration.")
    exit_condition_callable: Callable[[Any, Optional[TContext]], bool] = Field(
        description=(
            "Callable that takes (last_body_output, pipeline_context) and returns True to exit loop."
        )
    )
    max_loops: int = Field(default=5, gt=0, description="Maximum number of iterations.")

    initial_input_to_loop_body_mapper: Optional[Callable[[Any, Optional[TContext]], Any]] = Field(
        default=None,
        description=("Callable to map LoopStep's input to the first iteration's body input."),
    )
    iteration_input_mapper: Optional[Callable[[Any, Optional[TContext], int], Any]] = Field(
        default=None,
        description=("Callable to map previous iteration's body output to next iteration's input."),
    )
    loop_output_mapper: Optional[Callable[[Any, Optional[TContext]], Any]] = Field(
        default=None,
        description=("Callable to map the final successful output to the LoopStep's output."),
    )

    model_config = {"arbitrary_types_allowed": True}

    # Runtime validation of pipeline type
    @classmethod
    def model_validate(cls: type[Self], *args: Any, **kwargs: Any) -> Self:
        loop_body = kwargs.get("loop_body_pipeline")
        if loop_body is not None and not isinstance(loop_body, Pipeline):
            raise ValueError(
                f"loop_body_pipeline must be a Pipeline instance, got {type(loop_body)}"
            )
        return super().model_validate(*args, **kwargs)

    def __repr__(self) -> str:
        return f"LoopStep(name={self.name!r}, loop_body_pipeline={self.loop_body_pipeline!r})"

    # ------------------------------------------------------------------
    # IR helpers
    # ------------------------------------------------------------------

    def to_model(self) -> "StepIR":
        base = super().to_model()
        base.step_type = self.step_type.value
        base.loop_body = self.loop_body_pipeline.to_model()
        base.exit_condition_callable = self.exit_condition_callable
        base.max_loops = self.max_loops
        base.initial_input_to_loop_body_mapper = self.initial_input_to_loop_body_mapper
        base.iteration_input_mapper = self.iteration_input_mapper
        base.loop_output_mapper = self.loop_output_mapper
        return base

    @classmethod
    def from_model(cls, model: "StepIR") -> "LoopStep[Any]":
        from .pipeline import Pipeline

        body = (
            Pipeline.from_model(model.loop_body)
            if model.loop_body
            else Pipeline.model_construct(steps=[])
        )
        from ..plugins import plugin_registry

        plugins = []
        for p in model.plugins:
            plugin_cls = plugin_registry.get(p.plugin_type)
            if plugin_cls is not None:
                plugins.append((plugin_cls(), p.priority))

        step = cls.model_validate(
            {
                "name": model.name,
                "loop_body_pipeline": body,
                "exit_condition_callable": _resolve_ref(model.exit_condition_callable),
                "max_loops": model.max_loops or 1,
                "initial_input_to_loop_body_mapper": _resolve_ref(
                    model.initial_input_to_loop_body_mapper
                ),
                "iteration_input_mapper": _resolve_ref(model.iteration_input_mapper),
                "loop_output_mapper": _resolve_ref(model.loop_output_mapper),
                "config": model.config,
                "plugins": plugins,
                "validators": [],
                "processors": model.processors,
                "persist_feedback_to_context": model.persist_feedback_to_context,
                "persist_validation_results_to": model.persist_validation_results_to,
                "updates_context": model.updates_context,
                "meta": model.meta,
            }
        )
        step.step_uid = model.step_uid
        return step


class MapStep(LoopStep[TContext]):
    """Map a pipeline over an iterable stored on the context.

    ``MapStep`` wraps ``LoopStep`` to iterate over ``context.<iterable_input>``
    and run ``pipeline_to_run`` for each item. The collected outputs are returned
    as a list.
    """

    step_type: ClassVar[StepType] = StepType.MAP

    iterable_input: str = Field()

    def __init__(
        self,
        *,
        name: str,
        pipeline_to_run: Pipeline[Any, Any],
        iterable_input: str,
        **config_kwargs: Any,
    ) -> None:
        results_var: contextvars.ContextVar[list[Any]] = contextvars.ContextVar(
            f"{name}_results", default=[]
        )
        items_var: contextvars.ContextVar[list[Any]] = contextvars.ContextVar(
            f"{name}_items", default=[]
        )

        body = pipeline_to_run

        # Initialize base Step/DataModel via pydantic BaseModel.__init__
        BaseModel.__init__(
            self,
            **{
                "name": name,
                "agent": None,
                "config": StepConfig(**config_kwargs),
                "plugins": [],
                "failure_handlers": [],
                "loop_body_pipeline": body,
                "exit_condition_callable": lambda _o, ctx: len(results_var.get()) + 1
                >= len(items_var.get()),
                "max_loops": 1,
                "initial_input_to_loop_body_mapper": None,
                "iteration_input_mapper": None,
                "loop_output_mapper": None,
                "iterable_input": iterable_input,
            },
        )
        object.__setattr__(self, "_original_body_pipeline", body)

        async def _noop(item: Any, /, **_: Any) -> Any:  # noqa: D401
            return item

        object.__setattr__(
            self,
            "_noop_pipeline",
            Pipeline.from_step(Step.from_callable(_noop, name=f"_{name}_noop")),
        )
        object.__setattr__(self, "_results_var", results_var)
        object.__setattr__(self, "_items_var", items_var)
        object.__setattr__(
            self,
            "_max_loops_var",
            contextvars.ContextVar(f"{name}_max_loops", default=1),
        )
        object.__setattr__(self, "_body_var", contextvars.ContextVar(f"{name}_body", default=body))

        def _initial_mapper(_: Any, ctx: BaseModel | None) -> Any:  # noqa: D401
            if ctx is None:
                raise ValueError("map_over requires a context")
            raw_items = getattr(ctx, iterable_input, [])
            if isinstance(raw_items, (str, bytes, bytearray)) or not isinstance(
                raw_items, Iterable
            ):
                raise TypeError(f"context.{iterable_input} must be a non-string iterable")
            items = list(raw_items)
            items_var.set(items)
            results_var.set([])
            if items:
                self._max_loops_var.set(len(items))
                self._body_var.set(self._original_body_pipeline)
                return items[0]
            self._max_loops_var.set(1)
            self._body_var.set(self._noop_pipeline)
            return None

        def _iter_mapper(out: Any, ctx: BaseModel | None, i: int) -> Any:
            if ctx is None:
                raise ValueError("map_over requires a context")
            res = results_var.get()
            res.append(out)
            results_var.set(res)
            items = items_var.get()
            return items[i] if i < len(items) else None

        def _output_mapper(out: Any, ctx: BaseModel | None) -> List[Any]:
            if ctx is None:
                raise ValueError("map_over requires a context")
            items = items_var.get()
            res = results_var.get()
            if not items:
                return []
            res.append(out)
            return list(res)

        object.__setattr__(self, "initial_input_to_loop_body_mapper", _initial_mapper)
        object.__setattr__(self, "iteration_input_mapper", _iter_mapper)
        object.__setattr__(self, "loop_output_mapper", _output_mapper)
        object.__setattr__(self, "iterable_input", iterable_input)

    # ------------------------------------------------------------------
    # IR helpers
    # ------------------------------------------------------------------

    def to_model(self) -> "StepIR":
        base = super().to_model()
        base.step_type = self.step_type.value
        base.iterable_input = self.iterable_input
        return base

    @classmethod
    def from_model(cls, model: "StepIR") -> "MapStep[Any]":
        from .pipeline import Pipeline

        body = (
            Pipeline.from_model(model.loop_body)
            if model.loop_body
            else Pipeline.model_construct(steps=[])
        )
        from ..plugins import plugin_registry

        step = cls(
            name=model.name,
            pipeline_to_run=body,
            iterable_input=model.iterable_input or "items",
            **model.config.model_dump(mode="python"),
        )
        plugins = []
        for p in model.plugins:
            plugin_cls = plugin_registry.get(p.plugin_type)
            if plugin_cls is not None:
                plugins.append((plugin_cls(), p.priority))
        step.plugins = plugins
        step.validators = []
        step.processors = model.processors
        step.persist_feedback_to_context = model.persist_feedback_to_context
        step.persist_validation_results_to = model.persist_validation_results_to
        step.updates_context = model.updates_context
        step.meta = model.meta
        step.step_uid = model.step_uid
        return step

    # Provide dynamic attribute resolution for loop state using context vars
    def __getattribute__(self, name: str) -> Any:  # noqa: D401
        if name == "max_loops":
            return object.__getattribute__(self, "_max_loops_var").get()
        if name == "loop_body_pipeline":
            return object.__getattribute__(self, "_body_var").get()
        return super().__getattribute__(name)

    def __repr__(self) -> str:
        return f"MapStep(name={self.name!r}, iterable_input={self.iterable_input!r})"
