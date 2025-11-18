from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from .evidence import Badges, copy_badges


@dataclass
class ComparisonRecord:
    winner: int
    reason: str
    details: Dict[str, Any]


def compare(a: Badges | Dict[str, Any], b: Badges | Dict[str, Any]) -> int:
    lhs = copy_badges(a)
    rhs = copy_badges(b)

    hard_rules = _evaluate_hard_rules(lhs, rhs)
    if hard_rules != 0:
        return hard_rules

    corroboration = _compare_support(lhs, rhs)
    if corroboration != 0:
        return corroboration

    quality = _compare_quality(lhs, rhs)
    if quality != 0:
        return quality

    verification = _compare_verify_hits(lhs, rhs)
    if verification != 0:
        return verification

    return _compare_cost(lhs, rhs)


def compare_with_reason(a: Badges | Dict[str, Any], b: Badges | Dict[str, Any]) -> ComparisonRecord:
    lhs = copy_badges(a)
    rhs = copy_badges(b)

    hard = _evaluate_hard_rules(lhs, rhs)
    if hard != 0:
        return ComparisonRecord(
            winner=hard,
            reason="constraints_ok priority",
            details={
                "a_constraints_ok": lhs.constraints_ok,
                "b_constraints_ok": rhs.constraints_ok,
            },
        )

    corroboration = _compare_support(lhs, rhs)
    if corroboration != 0:
        return ComparisonRecord(
            winner=corroboration,
            reason="support minus contradiction",
            details={
                "a_support": lhs.support,
                "a_contradict": lhs.contradict,
                "b_support": rhs.support,
                "b_contradict": rhs.contradict,
            },
        )

    quality = _compare_quality(lhs, rhs)
    if quality != 0:
        return ComparisonRecord(
            winner=quality,
            reason="quality plus recency plus directness",
            details={
                "a_quality": lhs.quality_avg,
                "a_recency": lhs.recency,
                "a_directness": lhs.directness,
                "b_quality": rhs.quality_avg,
                "b_recency": rhs.recency,
                "b_directness": rhs.directness,
            },
        )

    verification = _compare_verify_hits(lhs, rhs)
    if verification != 0:
        return ComparisonRecord(
            winner=verification,
            reason="verification hits",
            details={
                "a_verify_hits": lhs.verify_hits,
                "b_verify_hits": rhs.verify_hits,
            },
        )

    cost = _compare_cost(lhs, rhs)
    return ComparisonRecord(
        winner=cost,
        reason="total cost",
        details={
            "a_cost_tokens": lhs.cost_tokens,
            "a_tool_cost_usd": lhs.tool_cost_usd,
            "b_cost_tokens": rhs.cost_tokens,
            "b_tool_cost_usd": rhs.tool_cost_usd,
        },
    )


def _evaluate_hard_rules(lhs: Badges, rhs: Badges) -> int:
    if lhs.constraints_ok and not rhs.constraints_ok:
        return 1
    if rhs.constraints_ok and not lhs.constraints_ok:
        return -1
    return 0


def _compare_support(lhs: Badges, rhs: Badges) -> int:
    score_a = lhs.support - lhs.contradict
    score_b = rhs.support - rhs.contradict
    if score_a > score_b:
        return 1
    if score_b > score_a:
        return -1
    return 0


def _compare_quality(lhs: Badges, rhs: Badges) -> int:
    score_a = lhs.quality_avg + lhs.recency + lhs.directness
    score_b = rhs.quality_avg + rhs.recency + rhs.directness
    if score_a > score_b:
        return 1
    if score_b > score_a:
        return -1
    return 0


def _compare_verify_hits(lhs: Badges, rhs: Badges) -> int:
    if lhs.verify_hits > rhs.verify_hits:
        return 1
    if rhs.verify_hits > lhs.verify_hits:
        return -1
    return 0


def _compare_cost(lhs: Badges, rhs: Badges) -> int:
    cost_a = lhs.cost_tokens + lhs.tool_cost_usd
    cost_b = rhs.cost_tokens + rhs.tool_cost_usd
    if cost_a < cost_b:
        return 1
    if cost_b < cost_a:
        return -1
    return 0
