from flujo.application.orchestrator import Orchestrator
from flujo.domain.models import Task, Candidate
from flujo.testing.utils import StubAgent


def test_orchestrator_run_sync():
    review = StubAgent(["c"])
    solve = StubAgent(["s"])
    validate = StubAgent(["v"])
    orch = Orchestrator(review, solve, validate, None)

    result = orch.run_sync(Task(prompt="x"))

    assert isinstance(result, Candidate)
    assert result.solution == "s"
