from pydantic_ai_orchestrator.application.orchestrator import Orchestrator
from pydantic_ai_orchestrator.domain.models import Task, Candidate
from pydantic_ai_orchestrator.testing.utils import StubAgent


def test_orchestrator_run_sync():
    review = StubAgent(["c"])
    solve = StubAgent(["s"])
    validate = StubAgent(["v"])
    orch = Orchestrator(review, solve, validate, None)

    result = orch.run_sync(Task(prompt="x"))

    assert isinstance(result, Candidate)
    assert result.solution == "s"
