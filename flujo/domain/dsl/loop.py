from __future__ import annotations

# NOTE: Extracted LoopStep and MapStep from pipeline_dsl for FSD1 refactor.

from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    TypeVar,
    Iterable,
    Self,
    Set,
)
import contextvars

from pydantic import Field

from ..models import BaseModel
from .step import Step, StepConfig
from .pipeline import Pipeline  # Import for runtime use in MapStep
from ..processors import AgentProcessors
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..ir import StepIR, LoopStepIR, MapStepIR
    from flujo.registry import CallableRegistry

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

    def to_model(
        self,
        callable_registry: Optional["CallableRegistry"] = None,
        visited: Optional[Set[str]] = None,
    ) -> "StepIR":
        """Convert this LoopStep to its IR representation."""
        from ..ir import LoopStepIR, StepConfigIR, ProcessorIR, CallableReference

        if callable_registry is None:
            raise ValueError("CallableRegistry required for LoopStep")

        # Create base step IR directly since Step doesn't have to_model()
        base_ir: LoopStepIR[Any, Any] = LoopStepIR[Any, Any](
            step_type=self.step_type,  # type: ignore[attr-defined]
            name=self.name,
            agent=None,  # LoopStep doesn't have an agent
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
            step_uid=self.step_uid,
            loop_body_pipeline=self.loop_body_pipeline.to_model(callable_registry),
            exit_condition_callable=CallableReference(
                ref_id=callable_registry.register(self.exit_condition_callable)
            ),
            max_loops=self.max_loops,
            initial_input_mapper=CallableReference(
                ref_id=callable_registry.register(self.initial_input_to_loop_body_mapper)
            )
            if self.initial_input_to_loop_body_mapper is not None
            else None,
            iteration_input_mapper=CallableReference(
                ref_id=callable_registry.register(self.iteration_input_mapper)
            )
            if self.iteration_input_mapper is not None
            else None,
            loop_output_mapper=CallableReference(
                ref_id=callable_registry.register(self.loop_output_mapper)
            )
            if self.loop_output_mapper is not None
            else None,
        )

        return base_ir

    @classmethod
    def _from_loop_ir(
        cls,
        ir_model: "LoopStepIR[Any, Any]",
        agent_registry: Optional[Dict[str, Any]] = None,
        callable_registry: Optional["CallableRegistry"] = None,
    ) -> "LoopStep[Any]":
        """Create a LoopStep from its IR representation."""
        if callable_registry is None:
            raise ValueError("CallableRegistry required for LoopStep")

        # Resolve callables
        exit_condition_callable = callable_registry.get(ir_model.exit_condition_callable.ref_id)

        initial_input_mapper = None
        if ir_model.initial_input_mapper is not None:
            initial_input_mapper = callable_registry.get(ir_model.initial_input_mapper.ref_id)

        iteration_input_mapper = None
        if ir_model.iteration_input_mapper is not None:
            iteration_input_mapper = callable_registry.get(ir_model.iteration_input_mapper.ref_id)

        loop_output_mapper = None
        if ir_model.loop_output_mapper is not None:
            loop_output_mapper = callable_registry.get(ir_model.loop_output_mapper.ref_id)

        # Rehydrate loop body pipeline
        loop_body_pipeline = Pipeline.from_model(
            ir_model.loop_body_pipeline, agent_registry, callable_registry
        )

        # Create step config
        config = StepConfig(
            max_retries=ir_model.config.max_retries,
            timeout_s=ir_model.config.timeout_seconds,
            temperature=ir_model.config.temperature,
        )

        return cls(
            name=ir_model.name,
            loop_body_pipeline=loop_body_pipeline,
            exit_condition_callable=exit_condition_callable,
            max_loops=ir_model.max_loops,
            initial_input_to_loop_body_mapper=initial_input_mapper,
            iteration_input_mapper=iteration_input_mapper,
            loop_output_mapper=loop_output_mapper,
            config=config,
            plugins=[],
            validators=[],
            processors=AgentProcessors(),
            persist_feedback_to_context=ir_model.persist_feedback_to_context,
            persist_validation_results_to=ir_model.persist_validation_results_to,
            updates_context=ir_model.updates_context,
            meta=ir_model.meta,
            step_uid=ir_model.step_uid,
        )


class MapStep(LoopStep[TContext]):
    """Map a pipeline over an iterable stored on the context.

    ``MapStep`` wraps ``LoopStep`` to iterate over ``context.<iterable_input>``
    and run ``pipeline_to_run`` for each item. The collected outputs are returned
    as a list.
    """

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

    # Provide dynamic attribute resolution for loop state using context vars
    def __getattribute__(self, name: str) -> Any:  # noqa: D401
        if name == "max_loops":
            return object.__getattribute__(self, "_max_loops_var").get()
        if name == "loop_body_pipeline":
            return object.__getattribute__(self, "_body_var").get()
        return super().__getattribute__(name)

    def __repr__(self) -> str:
        return f"MapStep(name={self.name!r}, iterable_input={self.iterable_input!r})"

    def to_model(
        self,
        callable_registry: Optional[CallableRegistry] = None,
        visited: Optional[Set[str]] = None,
    ) -> "StepIR":
        """Convert this MapStep to its IR representation."""
        from ..ir import MapStepIR, CallableReference

        if callable_registry is None:
            raise ValueError("CallableRegistry required for MapStep")

        # Get the base step IR
        base_ir = super().to_model(callable_registry, visited)

        # Register callables
        exit_condition_ref = CallableReference(
            ref_id=callable_registry.register(self.exit_condition_callable)
        )

        initial_mapper_ref = None
        if self.initial_input_to_loop_body_mapper is not None:
            initial_mapper_ref = CallableReference(
                ref_id=callable_registry.register(self.initial_input_to_loop_body_mapper)
            )

        iteration_mapper_ref = None
        if self.iteration_input_mapper is not None:
            iteration_mapper_ref = CallableReference(
                ref_id=callable_registry.register(self.iteration_input_mapper)
            )

        # Convert loop body pipeline
        loop_body_ir = self.loop_body_pipeline.to_model(callable_registry)

        return MapStepIR[Any, Any](
            step_type=base_ir.step_type,
            name=base_ir.name,
            agent=base_ir.agent,
            config=base_ir.config,
            plugins=base_ir.plugins,
            validators=base_ir.validators,
            processors=base_ir.processors,
            persist_feedback_to_context=base_ir.persist_feedback_to_context,
            persist_validation_results_to=base_ir.persist_validation_results_to,
            updates_context=base_ir.updates_context,
            meta=base_ir.meta,
            step_uid=base_ir.step_uid,
            loop_body_pipeline=loop_body_ir,
            iterable_input=self.iterable_input,
            initial_input_mapper=initial_mapper_ref,
            iteration_input_mapper=iteration_mapper_ref,
            exit_condition_callable=exit_condition_ref,
            max_loops=self.max_loops,
        )

    @classmethod
    def _from_map_ir(
        cls,
        ir_model: "MapStepIR[Any, Any]",
        agent_registry: Optional[Dict[str, Any]] = None,
        callable_registry: Optional["CallableRegistry"] = None,
    ) -> "MapStep[Any]":
        """Create a MapStep from its IR representation."""
        if callable_registry is None:
            raise ValueError("CallableRegistry required for MapStep")

        # Resolve callables
        exit_condition_callable = callable_registry.get(ir_model.exit_condition_callable.ref_id)

        initial_input_mapper = None
        if ir_model.initial_input_mapper is not None:
            initial_input_mapper = callable_registry.get(ir_model.initial_input_mapper.ref_id)

        iteration_input_mapper = None
        if ir_model.iteration_input_mapper is not None:
            iteration_input_mapper = callable_registry.get(ir_model.iteration_input_mapper.ref_id)

        loop_output_mapper = None
        if ir_model.loop_output_mapper is not None:
            loop_output_mapper = callable_registry.get(ir_model.loop_output_mapper.ref_id)

        # Rehydrate loop body pipeline
        loop_body_pipeline = Pipeline.from_model(
            ir_model.loop_body_pipeline, agent_registry, callable_registry
        )

        # Create step config
        config = StepConfig(
            max_retries=ir_model.config.max_retries,
            timeout_s=ir_model.config.timeout_seconds,
            temperature=ir_model.config.temperature,
        )

        # Create MapStep with the required parameters
        return cls(
            name=ir_model.name,
            pipeline_to_run=loop_body_pipeline,
            iterable_input=ir_model.iterable_input,
            exit_condition_callable=exit_condition_callable,
            initial_input_to_loop_body_mapper=initial_input_mapper,
            iteration_input_mapper=iteration_input_mapper,
            loop_output_mapper=loop_output_mapper,
            config=config,
            plugins=[],
            validators=[],
            processors=AgentProcessors(),
            persist_feedback_to_context=ir_model.persist_feedback_to_context,
            persist_validation_results_to=ir_model.persist_validation_results_to,
            updates_context=ir_model.updates_context,
            meta=ir_model.meta,
            step_uid=ir_model.step_uid,
        )
