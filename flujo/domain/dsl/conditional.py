from __future__ import annotations

from typing import Any, Callable, Dict, Generic, Optional, TypeVar, Self

from pydantic import Field

from ..models import BaseModel
from .step import Step, BranchKey
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass  # For type hints only

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
