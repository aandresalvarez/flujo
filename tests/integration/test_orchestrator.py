from flujo.recipes import Default
from flujo.domain.models import Task, Candidate
from flujo.testing.utils import StubAgent


async def test_orchestrator_runs_pipeline():
    review = StubAgent(["checklist"])
    solve = StubAgent(["solution"])
    validate = StubAgent(["validated"])
    orch = Default(review, solve, validate, None)

    result = await orch.run_async(Task(prompt="do"))

    assert isinstance(result, Candidate)
    assert result.solution == "solution"
    assert review.call_count == 1
    assert solve.inputs[0] == "checklist"
    assert validate.inputs[0] == "solution"
