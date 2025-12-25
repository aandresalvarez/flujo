import pytest

from flujo.domain.evaluation import MultiSignalEvaluator
from flujo.domain.models import Checklist, ChecklistItem
from flujo.domain.validation import validator


@validator
def _non_empty(output: object) -> tuple[bool, str | None]:
    ok = bool(output)
    return ok, None


@validator
def _always_false(_output: object) -> tuple[bool, str | None]:
    return False, "bad"


class _ReviewAgent:
    async def run(self, _data: object | None = None, **_kwargs: object) -> Checklist:
        return Checklist(items=[ChecklistItem(description="has value", passed=None)])


class _ValidatorAgent:
    async def run(self, data: object | None = None, **_kwargs: object) -> Checklist:
        payload = data if isinstance(data, dict) else {}
        checklist = payload.get("checklist")
        if not isinstance(checklist, Checklist):
            return Checklist(items=[])
        return Checklist(
            items=[
                ChecklistItem(description=item.description, passed=True) for item in checklist.items
            ]
        )


@pytest.mark.asyncio
async def test_multi_signal_evaluator_combines_scores() -> None:
    evaluator = MultiSignalEvaluator(objective_validators=[_non_empty])
    report = await evaluator.run(
        {
            "solution": "ok",
            "checklist": Checklist(
                items=[
                    ChecklistItem(description="a", passed=True),
                    ChecklistItem(description="b", passed=False),
                ]
            ),
        }
    )

    assert report.rubric_score == 0.5
    assert report.objective_score == 1.0
    assert report.hard_fail is False
    assert report.score == pytest.approx((0.5 + 2.0) / 3.0)


@pytest.mark.asyncio
async def test_multi_signal_evaluator_hard_fail() -> None:
    evaluator = MultiSignalEvaluator(objective_validators=[_always_false])
    report = await evaluator.run({"solution": "ok"})

    assert report.hard_fail is True
    assert report.score == 0.0


@pytest.mark.asyncio
async def test_multi_signal_evaluator_review_and_validator_agents() -> None:
    evaluator = MultiSignalEvaluator(review_agent=_ReviewAgent(), validator_agent=_ValidatorAgent())
    report = await evaluator.run({"solution": "ok", "task": "do thing"})

    assert report.rubric_score == 1.0
    assert report.objective_score == 0.0
