from __future__ import annotations

from typing import Any, Callable, Generic, List, Optional, TypeVar, Union

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from pydantic import Field

from ..models import BaseModel
from .step import Step, MergeStrategy, BranchFailureStrategy
from .pipeline import Pipeline  # Import for runtime use in normalization
from ...application.core.types import TContext_w_Scratch
from flujo.type_definitions.common import JSONObject

TContext = TypeVar("TContext", bound=BaseModel)

__all__ = ["ParallelStep"]


class ParallelStep(Step[Any, Any], Generic[TContext]):
    """Execute multiple branch pipelines concurrently.

    Each entry in ``branches`` is run in parallel and the outputs are returned
    as a dictionary keyed by branch name. Context fields can be selectively
    copied to branches via ``context_include_keys`` and merged back using
    ``merge_strategy``.
    """

    branches: JSONObject = Field(
        description="Mapping of branch names to pipelines to run in parallel."
    )
    context_include_keys: Optional[List[str]] = Field(
        default=None,
        description="If provided, only these top-level context fields will be copied to each branch. "
        "If None, the entire context is deep-copied (default behavior).",
    )
    merge_strategy: Union[MergeStrategy, Callable[[TContext_w_Scratch, JSONObject], None]] = Field(
        default=MergeStrategy.CONTEXT_UPDATE,
        description="Strategy for merging successful branch contexts back into the main context.",
    )
    on_branch_failure: BranchFailureStrategy = Field(
        default=BranchFailureStrategy.PROPAGATE,
        description="How the ParallelStep should behave when a branch fails.",
    )
    field_mapping: Optional[dict[str, List[str]]] = Field(
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
    def model_validate(cls: type[Self], *args: Any, **kwargs: Any) -> Self:
        """Validate and normalize branches before creating the instance."""
        if args and isinstance(args[0], dict):
            branches = args[0].get("branches", {})
        else:
            branches = kwargs.get("branches", {})
        if not branches:
            raise ValueError("'branches' dictionary cannot be empty.")

        normalized: dict[str, "Pipeline[Any, Any]"] = {}
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
