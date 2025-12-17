from __future__ import annotations

from typing import Callable, Generic, TypeVar

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from pydantic import Field, model_validator

from ..models import BaseModel
from .step import Step, BranchKey
from .pipeline import Pipeline

TContext = TypeVar("TContext", bound=BaseModel)

__all__ = ["ConditionalStep"]


def _wrap_step_as_pipeline(step: Step[object, object]) -> Pipeline[object, object]:
    pipeline = Pipeline[object, object].model_construct(steps=[step], hooks=[], on_finish=[])
    try:
        pipeline._initialize_io_types()
    except Exception:
        pass
    return pipeline


def _coerce_pipeline_variant(pipeline: Pipeline[object, object]) -> Pipeline[object, object]:
    coerced = Pipeline[object, object].model_construct(
        steps=list(getattr(pipeline, "steps", []) or []),
        hooks=list(getattr(pipeline, "hooks", []) or []),
        on_finish=list(getattr(pipeline, "on_finish", []) or []),
    )
    try:
        coerced._initialize_io_types()
    except Exception:
        pass
    return coerced


class ConditionalStep(Step[object, object], Generic[TContext]):
    """Route execution to one of several branch pipelines.

    ``condition_callable`` receives the previous step's output and optional
    context and returns a key that selects a branch from ``branches``. Each
    branch is its own :class:`Pipeline`. An optional ``default_branch_pipeline``
    is executed when no key matches.
    """

    condition_callable: Callable[[object, TContext | None], BranchKey] = Field(
        description=("Callable that returns a key to select a branch.")
    )
    branches: dict[BranchKey, Pipeline[object, object]] = Field(
        description="Mapping of branch keys to sub-pipelines."
    )
    default_branch_pipeline: object | None = Field(
        default=None,
        description="Pipeline to execute when no branch key matches.",
    )

    branch_input_mapper: Callable[[object, TContext | None], object] | None = Field(
        default=None,
        description="Maps ConditionalStep input to branch input.",
    )
    branch_output_mapper: Callable[[object, BranchKey, TContext | None], object] | None = Field(
        default=None,
        description="Maps branch output to ConditionalStep output.",
    )

    # model_config inherited from BaseModel

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy_fields(cls: type[Self], data: object) -> object:
        """Support legacy construction using `condition` instead of `condition_callable`."""
        if isinstance(data, dict):
            working = dict(data)
            if "condition" in working and "condition_callable" not in working:
                working["condition_callable"] = working.pop("condition")
            try:
                branches = working.get("branches")
                if isinstance(branches, dict):
                    normalized: dict[BranchKey, Pipeline[object, object]] = {}
                    for branch_key, branch_pipeline in branches.items():
                        if isinstance(branch_pipeline, Step):
                            normalized[branch_key] = _wrap_step_as_pipeline(branch_pipeline)
                        elif isinstance(branch_pipeline, Pipeline):
                            normalized[branch_key] = _coerce_pipeline_variant(branch_pipeline)
                        else:
                            normalized[branch_key] = branch_pipeline
                    working["branches"] = normalized
            except Exception:
                pass
            try:
                default_branch = working.get("default_branch_pipeline")
                if isinstance(default_branch, Step):
                    working["default_branch_pipeline"] = _wrap_step_as_pipeline(default_branch)
            except Exception:
                pass
            return working
        return data

    @property
    def is_complex(self) -> bool:
        # ✅ Override to mark as complex.
        return True

    # Ensure non-empty branch mapping and validate pipeline types
    @classmethod
    def model_validate(
        cls: type[Self],
        obj: object | None = None,
        *,
        strict: bool | None = None,
        from_attributes: bool | None = None,
        context: object | None = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
        **kwargs: object,
    ) -> Self:
        validate_default_branch = "default_branch_pipeline" in kwargs
        if kwargs:
            if isinstance(obj, dict):
                merged = dict(obj)
                merged.update(kwargs)
                obj = merged
            elif obj is None:
                obj = dict(kwargs)

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

        normalized: dict[BranchKey, Pipeline[object, object]] = {}
        for branch_key, branch_pipeline in branches.items():
            if isinstance(branch_pipeline, Step):
                normalized[branch_key] = _wrap_step_as_pipeline(branch_pipeline)
                continue
            if isinstance(branch_pipeline, Pipeline):
                normalized[branch_key] = _coerce_pipeline_variant(branch_pipeline)
                continue
            if not isinstance(branch_pipeline, Pipeline):
                raise ValueError(
                    f"Branch {branch_key} must be a Pipeline instance, got {type(branch_pipeline)}"
                )

        default_branch = obj.get("default_branch_pipeline")
        if isinstance(default_branch, Step):
            default_branch = _wrap_step_as_pipeline(default_branch)
        elif isinstance(default_branch, Pipeline):
            default_branch = _coerce_pipeline_variant(default_branch)
        if (
            validate_default_branch
            and default_branch is not None
            and not isinstance(default_branch, Pipeline)
        ):
            raise ValueError(
                f"default_branch_pipeline must be a Pipeline instance, got {type(default_branch)}"
            )

        normalized_obj = dict(obj, branches=normalized, default_branch_pipeline=default_branch)
        return super().model_validate(
            normalized_obj,
            strict=strict,
            from_attributes=from_attributes,
            context=context,
            by_alias=by_alias,
            by_name=by_name,
        )

    def __repr__(self) -> str:
        return f"ConditionalStep(name={self.name!r}, branches={list(self.branches.keys())})"
