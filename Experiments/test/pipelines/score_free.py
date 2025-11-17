from __future__ import annotations

from typing import Any, Dict, List, Tuple

from flujo.domain.dsl import Pipeline, Step

from Experiments.test.app.comparators import ComparisonRecord, compare_with_reason
from Experiments.test.app.evidence import Badges, update_badges


async def plan(goal: str) -> Dict[str, Any]:
    base = Badges()
    exploratory = update_badges(
        base,
        {
            "support": 1,
            "quality_avg": 0.55,
            "recency": 0.6,
            "directness": 0.4,
        },
    )
    precise = update_badges(
        base,
        {
            "constraints_ok": True,
            "support": 2,
            "quality_avg": 0.7,
            "recency": 0.5,
            "directness": 0.6,
        },
    )
    lean = update_badges(
        base,
        {
            "support": 1,
            "quality_avg": 0.6,
            "recency": 0.4,
            "directness": 0.5,
            "cost_tokens": -20,
        },
    )
    options = [
        {
            "id": "exploratory",
            "summary": f"Explore multiple framings before acting on '{goal}'.",
            "badges": exploratory.to_dict(),
        },
        {
            "id": "precise",
            "summary": f"Deliver a focused answer to '{goal}' with citations.",
            "badges": precise.to_dict(),
        },
        {
            "id": "lean",
            "summary": f"Provide a concise baseline answer for '{goal}'.",
            "badges": lean.to_dict(),
        },
    ]
    return {"goal": goal, "options": options, "log": []}


def _log_comparison(
    current_best: Dict[str, Any],
    contender: Dict[str, Any],
    record: ComparisonRecord,
) -> str:
    winner = contender if record.winner > 0 else current_best
    loser = current_best if record.winner > 0 else contender
    return (
        f"compare({contender['id']} vs {current_best['id']}): "
        f"{winner['id']} wins via {record.reason} "
        f"(details={record.details})"
    )


async def duel_options(payload: Dict[str, Any]) -> Dict[str, Any]:
    options: List[Dict[str, Any]] = list(payload.get("options", []))
    if not options:
        return {**payload, "selected": None}
    log: List[str] = list(payload.get("log", []))
    current_best = options[0]
    for contender in options[1:]:
        record = compare_with_reason(contender["badges"], current_best["badges"])
        log.append(_log_comparison(current_best, contender, record))
        if record.winner > 0:
            current_best = contender
    payload["selected"] = current_best
    payload["selected_badges"] = current_best.get("badges", {})
    payload["log"] = log
    return payload


async def verify_selection(payload: Dict[str, Any]) -> Dict[str, Any]:
    badges = Badges(**payload.get("selected_badges", {}))
    updated = update_badges(
        badges,
        {
            "verify_hits": 1,
            "support": 1,
        },
    )
    payload["selected_badges"] = updated.to_dict()
    payload.setdefault("log", []).append("verify: +support (>=2 domains), +verification hit")
    return payload


async def finalize(payload: Dict[str, Any]) -> Dict[str, Any]:
    selected = payload.get("selected") or {}
    result = {
        "goal": payload.get("goal", ""),
        "selection": selected.get("summary"),
        "badge_snapshot": payload.get("selected_badges", {}),
        "duel_log": payload.get("log", []),
    }
    return result


plan_step = Step.from_callable(plan, name="plan")
duel_step = Step.from_callable(duel_options, name="duel_options")
verify_step = Step.from_callable(verify_selection, name="verify_selection")
finalize_step = Step.from_callable(finalize, name="finalize")

pipeline = Pipeline(steps=[plan_step, duel_step, verify_step, finalize_step])

__all__ = ["pipeline"]
