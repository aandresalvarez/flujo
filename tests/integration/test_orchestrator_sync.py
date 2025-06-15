from flujo.recipes import Default
from flujo.domain.models import Task, Candidate
from flujo.testing.utils import StubAgent


def test_orchestrator_run_sync():
    review = StubAgent(["c"])
    solve = StubAgent(["s"])
    validate = StubAgent(["v"])
    orch = Default(review, solve, validate, None)

    result = orch.run_sync(Task(prompt="x"))

    assert isinstance(result, Candidate)
    assert result.solution == "s"
