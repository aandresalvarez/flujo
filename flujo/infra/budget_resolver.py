from __future__ import annotations

from typing import Dict, Optional, Tuple
from fnmatch import fnmatchcase

from pydantic import BaseModel

from ..domain.models import UsageLimits


class BudgetSource(BaseModel):
    """Describes where a budget was resolved from for observability."""

    source: str
    pattern: Optional[str] = None


def _min_or_none(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)


def _min_int_or_none(a: Optional[int], b: Optional[int]) -> Optional[int]:
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    return a if a <= b else b


def combine_limits(
    code_limits: Optional[UsageLimits], toml_limits: Optional[UsageLimits]
) -> Optional[UsageLimits]:
    """Return the most restrictive combination of two limits.

    Rules:
    - If both None: None (unlimited)
    - If one is None: return the other
    - For each field, take the minimum treating None as unlimited
    """
    if code_limits is None and toml_limits is None:
        return None
    if code_limits is None:
        return toml_limits
    if toml_limits is None:
        return code_limits

    # Compute token limit minimum using integer-aware helper
    resolved_token_limit: Optional[int] = _min_int_or_none(
        code_limits.total_tokens_limit, toml_limits.total_tokens_limit
    )

    return UsageLimits(
        total_cost_usd_limit=_min_or_none(
            code_limits.total_cost_usd_limit, toml_limits.total_cost_usd_limit
        ),
        total_tokens_limit=resolved_token_limit,
    )


def resolve_limits_for_pipeline(
    budgets: Optional["BudgetConfigLike"], pipeline_name: str
) -> Tuple[Optional[UsageLimits], BudgetSource]:
    """Resolve UsageLimits for a pipeline name using precedence:
    1) Exact pipeline name match in budgets.pipeline
    2) First wildcard match in budgets.pipeline (deterministic order)
    3) budgets.default

    Returns (limits, source)
    """
    if budgets is None:
        return None, BudgetSource(source="none")

    # 1) Exact match
    pipeline_map: Dict[str, UsageLimits] = getattr(budgets, "pipeline", {}) or {}
    if pipeline_name in pipeline_map:
        return pipeline_map[pipeline_name], BudgetSource(
            source="budgets.pipeline", pattern=pipeline_name
        )

    # 2) Wildcard matching in stable insertion order
    for pattern, limits in pipeline_map.items():
        # Skip patterns that are actually exact keys (already checked)
        if pattern == pipeline_name:
            continue
        try:
            if any(ch in pattern for ch in ["*", "?", "["]):
                if fnmatchcase(pipeline_name, pattern):
                    return limits, BudgetSource(source="budgets.pipeline", pattern=pattern)
        except Exception:
            # Defensive: ignore malformed patterns
            continue

    # 3) Default fallback
    default_limits: Optional[UsageLimits] = getattr(budgets, "default", None)
    if default_limits is not None:
        return default_limits, BudgetSource(source="budgets.default")

    return None, BudgetSource(source="none")


# Lightweight protocol for the budgets object to avoid import cycles in type checking
class BudgetConfigLike(BaseModel):
    default: Optional[UsageLimits] = None
    pipeline: Dict[str, UsageLimits] = {}
