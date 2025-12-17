from __future__ import annotations
from flujo.type_definitions.common import JSONObject

from typing import Callable, Generic, TypeVar

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from pydantic import Field, model_validator

from ..models import BaseModel
from .step import Step, MergeStrategy, BranchFailureStrategy
from .pipeline import Pipeline

TContext = TypeVar("TContext", bound=BaseModel)

__all__ = ["DynamicParallelRouterStep"]


class DynamicParallelRouterStep(Step[object, object], Generic[TContext]):
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

    router_agent: object = Field(description="Agent that returns branches to run.")
    branches: dict[str, Pipeline[object, object]] = Field(
        description="Mapping of branch names to pipelines."
    )
    context_include_keys: list[str] | None = Field(
        default=None,
        description="Context keys to include when copying context to branches.",
    )
    merge_strategy: MergeStrategy | Callable[[TContext, JSONObject], None] = Field(
        default=MergeStrategy.NO_MERGE,
        description="Strategy for merging branch contexts back.",
    )
    on_branch_failure: BranchFailureStrategy = Field(
        default=BranchFailureStrategy.PROPAGATE,
        description="How to handle branch failures.",
    )
    field_mapping: dict[str, list[str]] | None = Field(
        default=None,
        description="Explicit mapping of branch names to context fields that should be merged. "
        "Only used with CONTEXT_UPDATE merge strategy.",
    )

    @property
    def is_complex(self) -> bool:
        return True

    # model_config inherited from BaseModel

    @model_validator(mode="before")
    @classmethod
    def _normalize_branches(cls: type[Self], data: object) -> object:
        if not isinstance(data, dict):
            return data

        branches = data.get("branches", {})
        if not isinstance(branches, dict):
            return data
        if not branches:
            raise ValueError("'branches' dictionary cannot be empty.")

        normalized: dict[str, object] = {}
        for key, branch in branches.items():
            if isinstance(branch, Step):
                normalized[key] = Pipeline.from_step(branch)
            else:
                normalized[key] = branch

        return dict(data, branches=normalized)

    def __repr__(self) -> str:  # pragma: no cover - simple repr
        return (
            f"DynamicParallelRouterStep(name={self.name!r}, branches={list(self.branches.keys())})"
        )
