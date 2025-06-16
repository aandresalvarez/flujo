from flujo.recipes import Default
from flujo.domain.models import Task, Candidate
from flujo.testing.utils import StubAgent
from flujo.domain.models import Checklist, ChecklistItem


def test_orchestrator_run_sync():
    review = StubAgent([Checklist(items=[ChecklistItem(description="x")])])
    solve = StubAgent(["s"])
    validate = StubAgent([Checklist(items=[ChecklistItem(description="x", passed=True)])])
    orch = Default(review, solve, validate, None)

    result = orch.run_sync(Task(prompt="x"))

    assert isinstance(result, Candidate)
    assert result.solution == "s"
