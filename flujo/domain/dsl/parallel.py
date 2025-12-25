from __future__ import annotations

from typing import Awaitable, Callable, Generic, TypeVar

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from pydantic import Field, model_validator

from ..models import BaseModel, StepResult
from .step import Step, MergeStrategy, BranchFailureStrategy
from .pipeline import Pipeline  # Import for runtime use in normalization
from flujo.type_definitions.common import JSONObject

TContext = TypeVar("TContext", bound=BaseModel)
ParallelReducer = Callable[..., StepResult | object | Awaitable[StepResult | object]]

__all__ = ["ParallelStep"]


class ParallelStep(Step[object, object], Generic[TContext]):
    """Execute multiple branch pipelines concurrently.

    Each entry in ``branches`` is run in parallel and the outputs are returned
    as a dictionary keyed by branch name. Context fields can be selectively
    copied to branches via ``context_include_keys`` and merged back using
    ``merge_strategy``.
    """

    branches: JSONObject = Field(
        description="Mapping of branch names to pipelines to run in parallel."
    )
    reduce: ParallelReducer | None = Field(
        default=None,
        description="Optional reducer to merge branch results into a single StepResult/output.",
    )
    context_include_keys: list[str] | None = Field(
        default=None,
        description="If provided, only these top-level context fields will be copied to each branch. "
        "If None, the entire context is deep-copied (default behavior).",
    )
    merge_strategy: MergeStrategy | Callable[[TContext, JSONObject], None] = Field(
        default=MergeStrategy.CONTEXT_UPDATE,
        description="Strategy for merging successful branch contexts back into the main context.",
    )
    on_branch_failure: BranchFailureStrategy = Field(
        default=BranchFailureStrategy.PROPAGATE,
        description="How the ParallelStep should behave when a branch fails.",
    )
    field_mapping: dict[str, list[str]] | None = Field(
        default=None,
        description="Explicit mapping of branch names to context fields that should be merged. "
        "Only used with CONTEXT_UPDATE merge strategy.",
    )
    ignore_branch_names: bool = Field(
        default=False,
        description="When True, branch names are not treated as context fields during merging.",
    )

    # model_config inherited from BaseModel

    @model_validator(mode="before")
    @classmethod
    def _normalize_branches(cls: type[Self], data: object) -> object:
        """Validate and normalize branches before creating the instance."""
        if not isinstance(data, dict):
            return data

        branches = data.get("branches", {})
        if not isinstance(branches, dict):
            return data

        normalized: dict[str, object] = {}
        for key, branch in branches.items():
            if isinstance(branch, Step):
                normalized[key] = Pipeline.from_step(branch)
            else:
                normalized[key] = branch

        return dict(data, branches=normalized)

    @property
    def is_complex(self) -> bool:
        # ✅ Override to mark as complex.
        return True

    def __repr__(self) -> str:
        return f"ParallelStep(name={self.name!r}, branches={list(self.branches.keys())})"
