from __future__ import annotations

from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    TypeVar,
    Union,
    Self,
    ClassVar,
)

from pydantic import Field

from ..models import BaseModel
from .step import Step, MergeStrategy, BranchFailureStrategy, StepType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..ir import StepIR
from .pipeline import Pipeline  # Import for runtime use in normalization

TContext = TypeVar("TContext", bound=BaseModel)

__all__ = ["ParallelStep"]


class ParallelStep(Step[Any, Any], Generic[TContext]):
    """Execute multiple branch pipelines concurrently.

    Each entry in ``branches`` is run in parallel and the outputs are returned
    as a dictionary keyed by branch name. Context fields can be selectively
    copied to branches via ``context_include_keys`` and merged back using
    ``merge_strategy``.
    """

    step_type: ClassVar[StepType] = StepType.PARALLEL

    branches: Dict[str, Any] = Field(
        description="Mapping of branch names to pipelines to run in parallel."
    )
    context_include_keys: Optional[List[str]] = Field(
        default=None,
        description="If provided, only these top-level context fields will be copied to each branch. "
        "If None, the entire context is deep-copied (default behavior).",
    )
    merge_strategy: Union[MergeStrategy, Callable[[TContext, TContext], None]] = Field(
        default=MergeStrategy.NO_MERGE,
        description="Strategy for merging successful branch contexts back into the main context.",
    )
    on_branch_failure: BranchFailureStrategy = Field(
        default=BranchFailureStrategy.PROPAGATE,
        description="How the ParallelStep should behave when a branch fails.",
    )

    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def model_validate(cls: type[Self], *args: Any, **kwargs: Any) -> Self:
        """Validate and normalize branches before creating the instance."""
        if args and isinstance(args[0], dict):
            branches = args[0].get("branches", {})
        else:
            branches = kwargs.get("branches", {})
        if not branches:
            raise ValueError("'branches' dictionary cannot be empty.")

        normalized: Dict[str, "Pipeline[Any, Any]"] = {}
        for key, branch in branches.items():
            if isinstance(branch, Step):
                normalized[key] = Pipeline.from_step(branch)
            else:
                normalized[key] = branch

        if args and isinstance(args[0], dict):
            args = (dict(args[0], branches=normalized),) + args[1:]
        else:
            kwargs["branches"] = normalized
        return super().model_validate(*args, **kwargs)

    def __repr__(self) -> str:
        return f"ParallelStep(name={self.name!r}, branches={list(self.branches.keys())})"

    # ------------------------------------------------------------------
    # IR helpers
    # ------------------------------------------------------------------

    def to_model(self) -> "StepIR":
        base = super().to_model()
        base.step_type = self.step_type.value
        base.parallel_branches = {k: p.to_model() for k, p in self.branches.items()}
        base.context_include_keys = self.context_include_keys
        base.merge_strategy = self.merge_strategy
        base.on_branch_failure = self.on_branch_failure
        return base

    @classmethod
    def from_model(cls, model: "StepIR") -> "ParallelStep[Any]":
        from .pipeline import Pipeline

        branches = {k: Pipeline.from_model(p) for k, p in (model.parallel_branches or {}).items()}
        from ..plugins import plugin_registry

        plugins = []
        for p in model.plugins:
            plugin_cls = plugin_registry.get(p.plugin_type)
            if plugin_cls is not None:
                plugins.append((plugin_cls(), p.priority))

        step = cls.model_validate(
            {
                "name": model.name,
                "branches": branches,
                "context_include_keys": model.context_include_keys,
                "merge_strategy": model.merge_strategy,
                "on_branch_failure": model.on_branch_failure,
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
