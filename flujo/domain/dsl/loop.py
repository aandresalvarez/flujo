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
    ClassVar,
)

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from pydantic import Field, model_validator
import contextvars

from ..models import BaseModel
from .step import Step
from .pipeline import Pipeline  # Import for runtime use in MapStep

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

    loop_body_pipeline: object = Field(description="The pipeline to execute in each iteration.")
    exit_condition_callable: Callable[[object, Optional[TContext]], bool] = Field(
        description=(
            "Callable that takes (last_body_output, pipeline_context) and returns True to exit loop."
        )
    )
    max_retries: int = Field(
        default=5, ge=1, description="Number of retries after initial iteration.", alias="max_loops"
    )

    initial_input_to_loop_body_mapper: Optional[Callable[[object, Optional[TContext]], object]] = (
        Field(
            default=None,
            description=("Callable to map LoopStep's input to the first iteration's body input."),
        )
    )
    iteration_input_mapper: Optional[Callable[[object, Optional[TContext], int], object]] = Field(
        default=None,
        description=("Callable to map previous iteration's body output to next iteration's input."),
    )
    loop_output_mapper: Optional[Callable[[object, Optional[TContext]], object]] = Field(
        default=None,
        description=("Callable to map the final successful output to the LoopStep's output."),
    )

    model_config = {"populate_by_name": True}

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy_fields(cls: type[Self], data: object) -> object:
        """Support legacy construction using 'body' and 'exit_expression' keys."""
        if not isinstance(data, dict):
            return data
        working = dict(data)
        # Map legacy `body` list to loop_body_pipeline Pipeline
        try:
            if "loop_body_pipeline" not in working and "body" in working:
                body_steps = working.pop("body")
                if isinstance(body_steps, list):
                    working["loop_body_pipeline"] = Pipeline(steps=body_steps)
        except Exception:
            pass
        # Map legacy exit_expression string to callable
        try:
            if "exit_condition_callable" not in working and "exit_expression" in working:
                expr_raw = working.pop("exit_expression")
                if isinstance(expr_raw, str):
                    expr_str = expr_raw.strip()
                    # Strip Jinja-style delimiters
                    if expr_str.startswith("{{") and expr_str.endswith("}}"):
                        expr_str = expr_str[2:-2].strip()

                    def _resolve_path(path: str, output: object, ctx: Optional[TContext]) -> object:
                        names: dict[str, object] = {
                            "previous_step": output,
                            "output": output,
                            "context": ctx,
                        }
                        target: object = names
                        for part in path.split("."):
                            if isinstance(target, dict):
                                target = target.get(part)
                            else:
                                target = getattr(target, part, None)
                            if target is None:
                                return None
                        return target

                    if " is defined" in expr_str:
                        path = expr_str.split(" is defined", 1)[0].strip()

                        def _legacy_exit(output: object, ctx: Optional[TContext]) -> bool:
                            return _resolve_path(path, output, ctx) is not None

                    else:
                        from flujo.utils.expressions import compile_expression_to_callable

                        expr_fn = compile_expression_to_callable(expr_str)

                        def _legacy_exit(output: object, ctx: Optional[TContext]) -> bool:
                            val = expr_fn(output, ctx)
                            try:
                                return bool(val)
                            except Exception:
                                return False

                    working["exit_condition_callable"] = _legacy_exit
        except Exception:
            pass
        loop_body = working.get("loop_body_pipeline")
        if loop_body is not None and not isinstance(loop_body, Pipeline):
            raise ValueError(
                f"loop_body_pipeline must be a Pipeline instance, got {type(loop_body)}"
            )
        return working

    def get_max_loops(self) -> int:
        """Get the maximum number of loops."""
        return self.max_retries

    def get_loop_body_pipeline(self) -> object:
        """Get the loop body pipeline."""
        return self.loop_body_pipeline

    def get_exit_condition_callable(self) -> Callable[[object, Optional[TContext]], bool]:
        """Get the exit condition callable."""
        return self.exit_condition_callable

    def get_initial_input_to_loop_body_mapper(
        self,
    ) -> Optional[Callable[[object, Optional[TContext]], object]]:
        """Get the initial input mapper."""
        return self.initial_input_to_loop_body_mapper

    def get_iteration_input_mapper(
        self,
    ) -> Optional[Callable[[object, Optional[TContext], int], object]]:
        """Get the iteration input mapper."""
        return self.iteration_input_mapper

    def get_loop_output_mapper(self) -> Optional[Callable[[object, Optional[TContext]], object]]:
        """Get the loop output mapper."""
        return self.loop_output_mapper

    @property
    def max_loops(self) -> int:
        return self.max_retries

    @property
    def is_complex(self) -> bool:
        # ✅ Override to mark as complex.
        return True

    def __repr__(self) -> str:
        return f"LoopStep(name={self.name!r}, loop_body_pipeline={self.loop_body_pipeline!r})"


class MapStep(LoopStep[TContext]):
    """Map a pipeline over an iterable stored on the context.

    ``MapStep`` wraps ``LoopStep`` to iterate over ``context.<iterable_input>``
    and run ``pipeline_to_run`` for each item. The collected outputs are returned
    as a list.
    """

    iterable_input: str = Field(
        description="Name of the context field containing the iterable to map over"
    )
    pipeline_to_run: Pipeline[Any, Any] = Field(description="The pipeline to execute for each item")

    # Internal state for a single execution (excluded from serialization)
    items: Optional[List[object]] = Field(default=None, exclude=True)
    results: Optional[List[object]] = Field(default=None, exclude=True)
    original_body_pipeline: Optional[Pipeline[Any, Any]] = Field(default=None, exclude=True)
    # Context-local state for concurrency safety
    _items_var: ClassVar[contextvars.ContextVar[List[object]]] = contextvars.ContextVar(
        "map_items", default=[]
    )
    _results_var: ClassVar[contextvars.ContextVar[List[object]]] = contextvars.ContextVar(
        "map_results", default=[]
    )
    _max_loops_var: ClassVar[contextvars.ContextVar[int]] = contextvars.ContextVar(
        "map_max_loops", default=1
    )
    _body_var: ClassVar[contextvars.ContextVar[Optional[Pipeline[Any, Any]]]] = (
        contextvars.ContextVar("map_body", default=None)
    )

    # Override the required fields from LoopStep with appropriate defaults
    loop_body_pipeline: Optional[object] = Field(
        default=None, description="The pipeline to execute in each iteration."
    )
    exit_condition_callable: Callable[[object, Optional[TContext]], bool] = Field(
        default=lambda _o, _c: True,
        description="Callable that takes (last_body_output, pipeline_context) and returns True to exit loop.",
    )

    def get_loop_body_pipeline(self) -> Pipeline[Any, Any]:
        """Return the configured pipeline to run per item (the original)."""
        return self.original_body_pipeline or self.pipeline_to_run

    def get_max_loops(self) -> int:
        """Get the maximum number of loops based on iterable size."""
        items = self._items_var.get()
        return len(items) if items else 0

    def get_exit_condition_callable(self) -> Callable[[object, Optional[TContext]], bool]:
        """Get the exit condition callable for mapping."""

        def _exit_condition(output: object, ctx: Optional[TContext]) -> bool:
            # Exit when the current (last) output would complete the results
            items = self._items_var.get()
            results = self._results_var.get()
            if not items:
                return True
            return (len(results) + 1) >= len(items)

        return _exit_condition

    def get_initial_input_to_loop_body_mapper(
        self,
    ) -> Callable[[object, Optional[TContext]], object]:
        """Get the initial input mapper for mapping."""

        def _initial_mapper(input_data: object, ctx: Optional[TContext]) -> object:
            # For MapStep, we need to extract the iterable from context
            if ctx is None:
                raise ValueError("MapStep requires a context")

            # Get the iterable from context
            if not hasattr(ctx, self.iterable_input):
                raise ValueError(f"Context missing required field '{self.iterable_input}'")

            iterable = getattr(ctx, self.iterable_input)
            if isinstance(iterable, str) or not isinstance(iterable, (list, tuple, Iterable)):
                raise TypeError(
                    f"Field '{self.iterable_input}' must be a non-string iterable, got {type(iterable)}"
                )

            # Store items for iteration tracking
            items = list(iterable)
            self.items = items
            self.results = []
            self._items_var.set(items)
            self._results_var.set([])
            self._max_loops_var.set(len(items) if items else 0)

            # Update loop count and return first item if available
            if self.items:
                # Keep max_loops property (aliases to max_retries) in sync for introspection/tests
                self.max_retries = len(self.items)
                return self.items[0]
            return None

        return _initial_mapper

    def get_iteration_input_mapper(
        self,
    ) -> Callable[[object, Optional[TContext], int], object]:
        """Get the iteration input mapper for mapping."""

        def _iteration_mapper(output: object, ctx: Optional[TContext], iteration: int) -> object:
            # Store the result from previous iteration
            results = self._results_var.get()
            results.append(output)
            self._results_var.set(results)
            if self.results is not None:
                self.results.append(output)

            # Return next item if available
            items = self._items_var.get()
            if items and iteration < len(items):
                return items[iteration]

            # No more items to process
            return None

        return _iteration_mapper

    def get_loop_output_mapper(self) -> Callable[[object, Optional[TContext]], List[object]]:
        """Get the loop output mapper for mapping."""

        def _output_mapper(output: object, ctx: Optional[TContext]) -> List[object]:
            # Return collected results only (unit tests expect not to include current output)
            base = self._results_var.get()
            if base:
                return list(base)
            return self.results or []

        return _output_mapper

    def model_post_init(self, __context: object) -> None:
        """Ensure transient runtime state is cleared and defaults initialized."""
        super().model_post_init(__context)
        self.items = None
        self.results = None
        # Preserve original pipeline and expose a no-op placeholder on loop_body_pipeline for introspection
        if self.pipeline_to_run is not None:
            self.original_body_pipeline = self.pipeline_to_run
            self._body_var.set(self.pipeline_to_run)

        # Create a no-op pipeline for loop_body_pipeline to keep CLI/inspections decoupled
        class _NoOpStep(Step[Any, Any]):
            name: str = "noop"

            async def execute(self, input_data: object, context: Optional[object]) -> object:
                return input_data

        self.loop_body_pipeline = Pipeline(steps=[_NoOpStep()])
        # Initialize mapping functions on attributes for direct access in tests
        self.initial_input_to_loop_body_mapper = self.get_initial_input_to_loop_body_mapper()
        self.iteration_input_mapper = self.get_iteration_input_mapper()
        self.loop_output_mapper = self.get_loop_output_mapper()
        self.exit_condition_callable = self.get_exit_condition_callable()
        # Keep initial max_loops to 1 until items are discovered
        self.max_retries = 1

    @property
    def is_complex(self) -> bool:
        # ✅ Override to mark as complex.
        return True

    def __repr__(self) -> str:
        return f"MapStep(name={self.name!r}, iterable_input={self.iterable_input!r}, pipeline_to_run={self.pipeline_to_run!r})"

    @classmethod
    def from_pipeline(
        cls,
        *,
        name: str,
        pipeline: Pipeline[Any, Any],
        iterable_input: str,
        **kwargs: object,
    ) -> "MapStep[TContext]":
        """Create a MapStep from a pipeline.

        Args:
            name: Name of the step
            pipeline: Pipeline to execute for each item
            iterable_input: Name of the context field containing the iterable
            **kwargs: Additional configuration options

        Returns:
            Configured MapStep instance
        """
        return cls(
            name=name,
            pipeline_to_run=pipeline,
            iterable_input=iterable_input,
            **kwargs,
        )
