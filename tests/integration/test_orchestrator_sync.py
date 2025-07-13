from flujo.recipes.factories import make_default_pipeline, run_default_pipeline
from flujo.domain.models import Task, Candidate
from flujo.testing.utils import StubAgent
from flujo.domain.models import Checklist, ChecklistItem


def test_orchestrator_run_sync():
    review = StubAgent([Checklist(items=[ChecklistItem(description="x")])])
    solve = StubAgent(["s"])
    validate = StubAgent([Checklist(items=[ChecklistItem(description="x", passed=True)])])
    pipeline = make_default_pipeline(
        review_agent=review,
        solution_agent=solve,
        validator_agent=validate,
        reflection_agent=None,
    )

    import asyncio

    result = asyncio.run(run_default_pipeline(pipeline, Task(prompt="x")))

    assert isinstance(result, Candidate)
    assert result.solution == "s"
