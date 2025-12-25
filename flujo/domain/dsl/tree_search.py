from __future__ import annotations

from typing import Callable, Generic, TypeVar

try:
    from typing import Self
except ImportError:  # pragma: no cover - py<3.11
    from typing_extensions import Self

from pydantic import Field, model_validator

from ..models import BaseModel
from .step import Step

TContext = TypeVar("TContext", bound=BaseModel)

__all__ = ["TreeSearchStep"]


class TreeSearchStep(Step[object, object], Generic[TContext]):
    """Run a quota-aware tree search with proposer/evaluator agents."""

    proposer: object = Field(description="Agent or callable that proposes next steps.")
    evaluator: object = Field(description="Agent or callable that evaluates candidates.")
    cost_function: Callable[..., float] | None = Field(
        default=None,
        description="Optional function to compute g-cost for a candidate.",
    )
    branching_factor: int = Field(
        default=3,
        ge=1,
        description="Number of candidates to request from the proposer each expansion.",
    )
    beam_width: int = Field(
        default=3,
        ge=1,
        description="Beam width for pruning the open set.",
    )
    max_depth: int = Field(
        default=5,
        ge=1,
        description="Maximum depth to expand in the search tree.",
    )
    max_iterations: int | None = Field(
        default=None,
        ge=1,
        description="Optional hard cap on total node expansions.",
    )
    candidate_validator: Callable[[object], bool] | None = Field(
        default=None,
        description="Lightweight validator to pre-filter candidates before evaluation.",
    )
    path_max_tokens: int = Field(
        default=2000,
        ge=200,
        description="Token budget for path summaries passed to agents.",
    )
    goal_score_threshold: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Rubric score threshold considered a goal.",
    )
    require_goal: bool = Field(
        default=False,
        description="If True, TreeSearchStep fails when no goal node is found.",
    )

    @model_validator(mode="before")
    @classmethod
    def _ensure_required_agents(cls: type[Self], data: object) -> object:
        if not isinstance(data, dict):
            return data
        if data.get("proposer") is None:
            raise ValueError("TreeSearchStep requires a proposer")
        if data.get("evaluator") is None:
            raise ValueError("TreeSearchStep requires an evaluator")
        return data

    @property
    def is_complex(self) -> bool:
        return True

    def __repr__(self) -> str:
        return (
            "TreeSearchStep("
            f"name={self.name!r}, branching_factor={self.branching_factor}, "
            f"beam_width={self.beam_width}, max_depth={self.max_depth})"
        )
