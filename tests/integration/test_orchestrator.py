from pydantic_ai_orchestrator.application.orchestrator import Orchestrator
from pydantic_ai_orchestrator.domain.models import Task, Candidate
from pydantic_ai_orchestrator.testing.utils import StubAgent


async def test_orchestrator_runs_pipeline():
    review = StubAgent(["checklist"])
    solve = StubAgent(["solution"])
    validate = StubAgent(["validated"])
    orch = Orchestrator(review, solve, validate, None)

    result = await orch.run_async(Task(prompt="do"))

    assert isinstance(result, Candidate)
    assert result.solution == "solution"
    assert review.call_count == 1
    assert solve.inputs[0] == "checklist"
    assert validate.inputs[0] == "solution"
