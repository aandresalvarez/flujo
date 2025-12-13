from __future__ import annotations
from flujo.type_definitions.common import JSONObject

from typing import Callable, Generic, TypeVar

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from pydantic import Field

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

    model_config = {"arbitrary_types_allowed": True}

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
    ) -> Self:  # noqa: D401
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

    def __repr__(self) -> str:  # pragma: no cover - simple repr
        return (
            f"DynamicParallelRouterStep(name={self.name!r}, branches={list(self.branches.keys())})"
        )
