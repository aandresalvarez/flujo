from __future__ import annotations

from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar, Union, Self

from pydantic import Field

from ..models import BaseModel
from .step import Step, MergeStrategy, BranchFailureStrategy
from .pipeline import Pipeline

TContext = TypeVar("TContext", bound=BaseModel)

__all__ = ["DynamicParallelRouterStep"]


class DynamicParallelRouterStep(Step[Any, Any], Generic[TContext]):
    """Dynamically execute a subset of branches in parallel.

    ``router_agent`` is invoked first and should return a list of branch
    names to execute. Only the selected branches are then run in parallel
    using the same semantics as :class:`ParallelStep`.

    Example
    -------
    >>> router_step = Step.dynamic_parallel_branch(
    ...     name="Router",
    ...     router_agent=my_router_agent,
    ...     branches={"Billing": billing_pipe, "Support": support_pipe},
    ... )
    """

    router_agent: Any = Field(description="Agent that returns branches to run.")
    branches: Dict[str, Pipeline[Any, Any]] = Field(
        description="Mapping of branch names to pipelines."
    )
    context_include_keys: Optional[List[str]] = Field(
        default=None,
        description="Context keys to include when copying context to branches.",
    )
    merge_strategy: Union[MergeStrategy, Callable[[TContext, TContext], None]] = Field(
        default=MergeStrategy.NO_MERGE,
        description="Strategy for merging branch contexts back.",
    )
    on_branch_failure: BranchFailureStrategy = Field(
        default=BranchFailureStrategy.PROPAGATE,
        description="How to handle branch failures.",
    )

    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def model_validate(cls: type[Self], *args: Any, **kwargs: Any) -> Self:  # noqa: D401
        if args and isinstance(args[0], dict):
            branches = args[0].get("branches", {})
        else:
            branches = kwargs.get("branches", {})
        if not branches:
            raise ValueError("'branches' dictionary cannot be empty.")

        normalized: Dict[str, Pipeline[Any, Any]] = {}
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

    def __repr__(self) -> str:  # pragma: no cover - simple repr
        return (
            f"DynamicParallelRouterStep(name={self.name!r}, branches={list(self.branches.keys())})"
        )
