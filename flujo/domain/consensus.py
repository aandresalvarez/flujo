from __future__ import annotations

from typing import Any, Awaitable, Callable, Protocol, Sequence, cast

from .evaluation import make_multi_signal_evaluator
from .models import BaseModel, StepResult
from ..utils.hash import stable_digest

Reducer = Callable[[Sequence[StepResult], BaseModel | None], StepResult | Awaitable[StepResult]]


class _EvaluatorLike(Protocol):
    async def run(self, data: object | None = None, **kwargs: Any) -> object: ...


def _select_pool(results: Sequence[StepResult]) -> list[StepResult]:
    if not results:
        return []
    successful = [res for res in results if res.success]
    return successful or list(results)


def _clone_with_meta(
    winner: StepResult,
    *,
    strategy: str,
    votes: int,
    total: int,
    extra: dict[str, Any] | None = None,
) -> StepResult:
    cloned = winner.model_copy()
    meta = dict(getattr(cloned, "metadata_", {}) or {})
    payload = {"strategy": strategy, "votes": votes, "total": total}
    if extra:
        payload.update(extra)
    meta["consensus"] = payload
    cloned.metadata_ = meta
    return cloned


def majority_vote(results: Sequence[StepResult]) -> StepResult:
    pool = _select_pool(results)
    if not pool:
        return StepResult(
            name="majority_vote",
            success=False,
            feedback="consensus_empty",
        )

    counts: dict[str, list[StepResult]] = {}
    first_index: dict[str, int] = {}
    for idx, res in enumerate(pool):
        key = stable_digest(res.output)
        counts.setdefault(key, []).append(res)
        if key not in first_index:
            first_index[key] = idx

    # Tie-break on earliest appearance (negative index makes max prefer smaller idx).
    best_key = max(counts.keys(), key=lambda k: (len(counts[k]), -first_index[k]))
    winners = counts[best_key]
    return _clone_with_meta(
        winners[0],
        strategy="majority_vote",
        votes=len(winners),
        total=len(pool),
    )


def code_consensus(results: Sequence[StepResult]) -> StepResult:
    pool = _select_pool(results)
    if not pool:
        return StepResult(
            name="code_consensus",
            success=False,
            feedback="consensus_empty",
        )

    string_pool = [res for res in pool if isinstance(res.output, str)]
    if not string_pool:
        return majority_vote(pool)

    counts: dict[str, list[StepResult]] = {}
    first_index: dict[str, int] = {}
    for idx, res in enumerate(string_pool):
        output = res.output
        if not isinstance(output, str):
            continue
        normalized = output.strip()
        key = stable_digest(normalized)
        counts.setdefault(key, []).append(res)
        if key not in first_index:
            first_index[key] = idx

    # Tie-break on earliest appearance (negative index makes max prefer smaller idx).
    best_key = max(counts.keys(), key=lambda k: (len(counts[k]), -first_index[k]))
    winners = counts[best_key]
    if len(winners) <= 1:
        return majority_vote(pool)
    return _clone_with_meta(
        winners[0],
        strategy="code_consensus",
        votes=len(winners),
        total=len(string_pool),
    )


def judge_selection(
    evaluator: object | None = None,
) -> Reducer:
    eval_obj: _EvaluatorLike
    if evaluator is None:
        eval_obj = make_multi_signal_evaluator()
    else:
        eval_obj = cast(_EvaluatorLike, evaluator)

    async def _reduce(
        results: Sequence[StepResult], context: BaseModel | None = None
    ) -> StepResult:
        pool = _select_pool(results)
        if not pool:
            return StepResult(
                name="judge_selection",
                success=False,
                feedback="consensus_empty",
            )

        best: StepResult | None = None
        best_score = float("-inf")
        best_report: object | None = None
        for res in pool:
            try:
                report = await eval_obj.run(
                    {"solution": res.output, "task": getattr(context, "initial_prompt", None)},
                    context=context,
                )
                score = float(getattr(report, "score", 0.0))
            except Exception:
                report = None
                score = 0.0
            if score > best_score:
                best = res
                best_score = score
                best_report = report

        if best is None:
            return StepResult(
                name="judge_selection",
                success=False,
                feedback="consensus_empty",
            )

        extra: dict[str, Any] = {"judge_score": best_score}
        if best_report is not None and hasattr(best_report, "model_dump"):
            try:
                extra["judge_report"] = best_report.model_dump()
            except Exception:
                pass
        return _clone_with_meta(
            best,
            strategy="judge_selection",
            votes=1,
            total=len(pool),
            extra=extra,
        )

    return _reduce
