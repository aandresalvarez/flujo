from __future__ import annotations

from typing import Any, Callable, Dict, Generic, Optional, TypeVar, Self, ClassVar

from pydantic import Field

from ..models import BaseModel
from .step import Step, BranchKey, StepType, _resolve_ref
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..ir import StepIR

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

    step_type: ClassVar[StepType] = StepType.CONDITIONAL

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

    # ------------------------------------------------------------------
    # IR helpers
    # ------------------------------------------------------------------

    def to_model(self) -> "StepIR":
        base = super().to_model()
        base.step_type = self.step_type.value
        base.condition_callable = self.condition_callable
        base.branches = {k: p.to_model() for k, p in self.branches.items()}
        base.default_branch_pipeline = (
            self.default_branch_pipeline.to_model() if self.default_branch_pipeline else None
        )
        base.branch_input_mapper = self.branch_input_mapper
        base.branch_output_mapper = self.branch_output_mapper
        return base

    @classmethod
    def from_model(cls, model: "StepIR") -> "ConditionalStep[Any]":
        from .pipeline import Pipeline

        branches = {k: Pipeline.from_model(p) for k, p in (model.branches or {}).items()}
        default_branch = (
            Pipeline.from_model(model.default_branch_pipeline)
            if model.default_branch_pipeline
            else None
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
                "condition_callable": _resolve_ref(model.condition_callable),
                "branches": branches,
                "default_branch_pipeline": default_branch,
                "branch_input_mapper": _resolve_ref(model.branch_input_mapper),
                "branch_output_mapper": _resolve_ref(model.branch_output_mapper),
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
