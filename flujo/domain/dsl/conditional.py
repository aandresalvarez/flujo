from __future__ import annotations

from typing import Any, Callable, Dict, Generic, Optional, TypeVar, Self, Set

from pydantic import Field

from ..models import BaseModel
from .step import Step, BranchKey, StepConfig
from typing import TYPE_CHECKING
from ..processors import AgentProcessors
from .pipeline import Pipeline

if TYPE_CHECKING:
    from ..ir import StepIR, ConditionalStepIR
    from flujo.registry import CallableRegistry

TContext = TypeVar("TContext", bound=BaseModel)

__all__ = ["ConditionalStep"]


class ConditionalStep(Step[Any, Any], Generic[TContext]):
    """Route execution to one of several branch pipelines.

    ``condition_callable`` receives the previous step's output and optional
    context and returns a key that selects a branch from ``branches``. Each
    branch is its own :class:`Pipeline`. An optional ``default_branch_pipeline``
    is executed when no key matches.
    """

    condition_callable: Callable[[Any, Optional[TContext]], BranchKey] = Field(
        description=("Callable that returns a key to select a branch.")
    )
    branches: Dict[BranchKey, Any] = Field(description="Mapping of branch keys to sub-pipelines.")
    default_branch_pipeline: Optional[Any] = Field(
        default=None,
        description="Pipeline to execute when no branch key matches.",
    )

    branch_input_mapper: Optional[Callable[[Any, Optional[TContext]], Any]] = Field(
        default=None,
        description="Maps ConditionalStep input to branch input.",
    )
    branch_output_mapper: Optional[Callable[[Any, BranchKey, Optional[TContext]], Any]] = Field(
        default=None,
        description="Maps branch output to ConditionalStep output.",
    )

    model_config = {"arbitrary_types_allowed": True}

    # Ensure non-empty branch mapping and validate pipeline types
    @classmethod
    def model_validate(cls: type[Self], *args: Any, **kwargs: Any) -> Self:
        if args and isinstance(args[0], dict):
            branches = args[0].get("branches", {})
        else:
            branches = kwargs.get("branches", {})
        if not branches:
            raise ValueError("'branches' dictionary cannot be empty.")

        # Runtime validation of pipeline types
        from .pipeline import Pipeline

        for branch_key, branch_pipeline in branches.items():
            if not isinstance(branch_pipeline, Pipeline):
                raise ValueError(
                    f"Branch {branch_key} must be a Pipeline instance, got {type(branch_pipeline)}"
                )

        default_branch = kwargs.get("default_branch_pipeline")
        if default_branch is not None and not isinstance(default_branch, Pipeline):
            raise ValueError(
                f"default_branch_pipeline must be a Pipeline instance, got {type(default_branch)}"
            )

        return super().model_validate(*args, **kwargs)

    def __repr__(self) -> str:
        return f"ConditionalStep(name={self.name!r}, branches={list(self.branches.keys())})"

    def to_model(
        self,
        callable_registry: Optional["CallableRegistry"] = None,
        visited: Optional[Set[str]] = None,
    ) -> "StepIR":
        """Convert this ConditionalStep to its IR representation."""
        from ..ir import ConditionalStepIR, StepConfigIR, ProcessorIR, StepType, CallableReference

        if callable_registry is None:
            raise ValueError("CallableRegistry required for ConditionalStep")

        # Create base step IR directly since Step doesn't have to_model()
        base_ir: ConditionalStepIR[Any, Any] = ConditionalStepIR[Any, Any](
            step_type=StepType.CONDITIONAL,
            name=self.name,
            agent=None,  # ConditionalStep doesn't have an agent
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
            condition_callable=CallableReference(
                ref_id=callable_registry.register(self.condition_callable)
            ),
            branches={k: v.to_model(callable_registry) for k, v in self.branches.items()},
            default_branch=self.default_branch_pipeline.to_model(callable_registry)
            if self.default_branch_pipeline is not None
            else None,
            branch_input_mapper=CallableReference(
                ref_id=callable_registry.register(self.branch_input_mapper)
            )
            if self.branch_input_mapper is not None
            else None,
            branch_output_mapper=CallableReference(
                ref_id=callable_registry.register(self.branch_output_mapper)
            )
            if self.branch_output_mapper is not None
            else None,
        )

        return base_ir

    @classmethod
    def _from_conditional_ir(
        cls,
        ir_model: "ConditionalStepIR[Any, Any]",
        agent_registry: Optional[Dict[str, Any]] = None,
        callable_registry: Optional["CallableRegistry"] = None,
    ) -> "ConditionalStep[Any]":
        """Create a ConditionalStep from its IR representation."""
        if callable_registry is None:
            raise ValueError("CallableRegistry required for ConditionalStep")

        # Resolve callables
        condition_callable = callable_registry.get(ir_model.condition_callable.ref_id)

        branch_input_mapper = None
        if ir_model.branch_input_mapper is not None:
            branch_input_mapper = callable_registry.get(ir_model.branch_input_mapper.ref_id)

        branch_output_mapper = None
        if ir_model.branch_output_mapper is not None:
            branch_output_mapper = callable_registry.get(ir_model.branch_output_mapper.ref_id)

        # Rehydrate branch pipelines
        branches = {}
        for key, pipeline_ir in ir_model.branches.items():
            branches[key] = Pipeline.from_model(pipeline_ir, agent_registry, callable_registry)

        default_branch_pipeline = None
        if ir_model.default_branch is not None:
            default_branch_pipeline = Pipeline.from_model(
                ir_model.default_branch, agent_registry, callable_registry
            )

        # Create step config
        config = StepConfig(
            max_retries=ir_model.config.max_retries,
            timeout_s=ir_model.config.timeout_seconds,
            temperature=ir_model.config.temperature,
        )

        return cls(
            name=ir_model.name,
            condition_callable=condition_callable,
            branches=branches,
            default_branch_pipeline=default_branch_pipeline,
            branch_input_mapper=branch_input_mapper,
            branch_output_mapper=branch_output_mapper,
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
