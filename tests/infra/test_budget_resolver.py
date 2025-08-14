from __future__ import annotations

from flujo.infra.budget_resolver import resolve_limits_for_pipeline, combine_limits
from flujo.domain.models import UsageLimits
from pydantic import BaseModel


class _Budgets(BaseModel):
    default: UsageLimits | None = None
    pipeline: dict[str, UsageLimits] = {}


def test_resolve_exact_match_overrides_default() -> None:
    budgets = _Budgets(
        default=UsageLimits(total_cost_usd_limit=10.0, total_tokens_limit=1000),
        pipeline={
            "analytics": UsageLimits(total_cost_usd_limit=3.0, total_tokens_limit=100),
        },
    )
    limits, src = resolve_limits_for_pipeline(budgets, "analytics")
    assert limits is not None
    assert limits.total_cost_usd_limit == 3.0
    assert limits.total_tokens_limit == 100
    assert src.source == "budgets.pipeline" and src.pattern == "analytics"


def test_resolve_wildcard_when_no_exact() -> None:
    budgets = _Budgets(
        default=UsageLimits(total_cost_usd_limit=10.0, total_tokens_limit=None),
        pipeline={
            "team-*": UsageLimits(total_cost_usd_limit=5.0, total_tokens_limit=500),
        },
    )
    limits, src = resolve_limits_for_pipeline(budgets, "team-alpha")
    assert limits is not None
    assert limits.total_cost_usd_limit == 5.0
    assert limits.total_tokens_limit == 500
    assert src.source == "budgets.pipeline" and src.pattern == "team-*"


def test_resolve_default_when_no_match() -> None:
    budgets = _Budgets(
        default=UsageLimits(total_cost_usd_limit=None, total_tokens_limit=42),
        pipeline={},
    )
    limits, src = resolve_limits_for_pipeline(budgets, "unknown")
    assert limits is not None
    assert limits.total_cost_usd_limit is None
    assert limits.total_tokens_limit == 42
    assert src.source == "budgets.default"


def test_combine_limits_treats_none_as_unlimited() -> None:
    a = UsageLimits(total_cost_usd_limit=5.0, total_tokens_limit=None)
    b = UsageLimits(total_cost_usd_limit=None, total_tokens_limit=100)
    c = combine_limits(a, b)
    assert c.total_cost_usd_limit == 5.0
    assert c.total_tokens_limit == 100
