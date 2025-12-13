from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Generic, TypeVar

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from pydantic import Field

from ..models import BaseModel
from .step import Step, MergeStrategy, BranchFailureStrategy
from .pipeline import Pipeline  # Import for runtime use in normalization
from flujo.type_definitions.common import JSONObject

if TYPE_CHECKING:
    from ...application.core.types import TContext_w_Scratch
else:
    TContext_w_Scratch = BaseModel

TContext = TypeVar("TContext", bound=BaseModel)

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
    context_include_keys: list[str] | None = Field(
        default=None,
        description="If provided, only these top-level context fields will be copied to each branch. "
        "If None, the entire context is deep-copied (default behavior).",
    )
    merge_strategy: MergeStrategy | Callable[[TContext_w_Scratch, JSONObject], None] = Field(
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

    model_config = {"arbitrary_types_allowed": True}

    @property
    def is_complex(self) -> bool:
        # ✅ Override to mark as complex.
        return True

    @classmethod
    def model_validate(
        cls: type[Self],
        obj: object,
        *,
        strict: bool | None = None,
        from_attributes: bool | None = None,
        context: object | None = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ) -> Self:
        """Validate and normalize branches before creating the instance."""
        if not isinstance(obj, dict):
            return super().model_validate(
                obj,
                strict=strict,
                from_attributes=from_attributes,
                context=context,
                by_alias=by_alias,
                by_name=by_name,
            )

        branches = obj.get("branches", {})
        if not branches:
            raise ValueError("'branches' dictionary cannot be empty.")

        normalized: dict[str, object] = {}
        for key, branch in branches.items():
            if isinstance(branch, Step):
                normalized[key] = Pipeline.from_step(branch)
            else:
                normalized[key] = branch

        normalized_obj = dict(obj, branches=normalized)
        return super().model_validate(
            normalized_obj,
            strict=strict,
            from_attributes=from_attributes,
            context=context,
            by_alias=by_alias,
            by_name=by_name,
        )

    def __repr__(self) -> str:
        return f"ParallelStep(name={self.name!r}, branches={list(self.branches.keys())})"
