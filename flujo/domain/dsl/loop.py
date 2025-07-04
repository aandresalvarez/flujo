from __future__ import annotations

# NOTE: Extracted LoopStep and MapStep from pipeline_dsl for FSD1 refactor.
# mypy: ignore-errors

from typing import (
    Any,
    Callable,
    Generic,
    List,
    Optional,
    TypeVar,
    Iterable,
)
import contextvars

from pydantic import Field

from ..models import BaseModel
from .step import Step, StepConfig  # type: ignore
from .pipeline import Pipeline  # Import for runtime use in MapStep

# Generic type var reused
TContext = TypeVar("TContext", bound=BaseModel)

__all__ = ["LoopStep", "MapStep"]


class LoopStep(Step[Any, Any], Generic[TContext]):
    """A specialized step that executes a pipeline in a loop."""

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
    def model_validate(cls, *args, **kwargs):  # type: ignore[override]
        loop_body = kwargs.get("loop_body_pipeline")
        if loop_body is not None and not isinstance(loop_body, Pipeline):
            raise ValueError(
                f"loop_body_pipeline must be a Pipeline instance, got {type(loop_body)}"
            )
        return super().model_validate(*args, **kwargs)

    def __repr__(self) -> str:
        return f"LoopStep(name={self.name!r}, loop_body_pipeline={self.loop_body_pipeline!r})"


class MapStep(LoopStep[TContext]):
    """A step that maps a pipeline over items in the pipeline context."""

    iterable_input: str = Field()

    def __init__(
        self,
        *,
        name: str,
        pipeline_to_run: Pipeline[Any, Any],
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

        # Initialize base Step/DataModel via pydantic BaseModel.__init__
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

        async def _noop(item: Any, **_: Any) -> Any:  # noqa: D401
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
        object.__setattr__(self, "_body_var", contextvars.ContextVar(f"{name}_body", default=body))

        def _initial_mapper(_: Any, ctx: BaseModel | None) -> Any:  # noqa: D401
            if ctx is None:
                raise ValueError("map_over requires a context")
            raw_items = getattr(ctx, iterable_input, [])
            # Disallow strings & ensure an actual iterable collection
            if isinstance(raw_items, (str, bytes, bytearray)) or not isinstance(
                raw_items, Iterable
            ):
                raise TypeError(f"context.{iterable_input} must be a non-string iterable")
            items = list(raw_items)
            setattr(ctx, items_attr, items)
            setattr(ctx, results_attr, [])
            if items:
                self._max_loops_var.set(len(items))
                self._body_var.set(self._original_body_pipeline)
                return items[0]
            # empty: nothing to iterate
            self._max_loops_var.set(1)
            self._body_var.set(self._noop_pipeline)
            return None

        def _iter_mapper(_: Any, ctx: BaseModel | None, i: int) -> Any:
            if ctx is None:
                raise ValueError("map_over requires a context")
            items = getattr(ctx, items_attr, [])
            return items[i] if i < len(items) else None

        def _output_mapper(_: Any, ctx: BaseModel | None) -> List[Any]:
            if ctx is None:
                raise ValueError("map_over requires a context")
            return list(getattr(ctx, results_attr, []))

        object.__setattr__(self, "initial_input_to_loop_body_mapper", _initial_mapper)
        object.__setattr__(self, "iteration_input_mapper", _iter_mapper)
        object.__setattr__(self, "loop_output_mapper", _output_mapper)
        object.__setattr__(self, "iterable_input", iterable_input)

    # Provide dynamic attribute resolution for loop state using context vars
    def __getattribute__(self, name: str) -> Any:  # noqa: D401
        if name == "max_loops":
            return object.__getattribute__(self, "_max_loops_var").get()
        if name == "loop_body_pipeline":
            return object.__getattribute__(self, "_body_var").get()
        return super().__getattribute__(name)

    def __repr__(self) -> str:
        return f"MapStep(name={self.name!r}, iterable_input={self.iterable_input!r})"
