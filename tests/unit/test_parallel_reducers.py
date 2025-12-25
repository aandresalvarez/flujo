import pytest

from flujo.domain.consensus import code_consensus, judge_selection, majority_vote
from flujo.domain.evaluation import EvaluationScore
from flujo.domain.models import StepResult


def _sr(name: str, output: object, success: bool = True) -> StepResult:
    return StepResult(name=name, output=output, success=success)


def test_majority_vote_picks_most_common() -> None:
    results = [_sr("a", "yes"), _sr("b", "yes"), _sr("c", "no")]
    winner = majority_vote(results)
    assert winner.output == "yes"
    assert winner.metadata_["consensus"]["strategy"] == "majority_vote"


def test_code_consensus_prefers_identical_code() -> None:
    results = [_sr("a", "code"), _sr("b", "code"), _sr("c", "diff")]
    winner = code_consensus(results)
    assert winner.output == "code"
    assert winner.metadata_["consensus"]["strategy"] == "code_consensus"


def test_code_consensus_falls_back_when_unique() -> None:
    results = [_sr("a", "one"), _sr("b", "two")]
    winner = code_consensus(results)
    assert winner.output == "one"
    assert winner.metadata_["consensus"]["strategy"] == "majority_vote"


@pytest.mark.asyncio
async def test_judge_selection_picks_best_score() -> None:
    class _Evaluator:
        async def run(self, data: object | None = None, **_kwargs: object) -> EvaluationScore:
            payload = data if isinstance(data, dict) else {}
            score = 1.0 if payload.get("solution") == "b" else 0.1
            return EvaluationScore(score=score)

    reducer = judge_selection(_Evaluator())
    winner = await reducer([_sr("a", "a"), _sr("b", "b")], None)
    assert winner.output == "b"
    assert winner.metadata_["consensus"]["strategy"] == "judge_selection"
