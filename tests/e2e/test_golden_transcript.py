import pytest
import vcr
from pydantic_ai_orchestrator.application.orchestrator import Orchestrator
from pydantic_ai_orchestrator.domain.models import Task, Candidate
from pydantic_ai_orchestrator.infra.agents import review_agent, solution_agent, validator_agent, get_reflection_agent

# Note: This test makes real API calls that are recorded to a cassette.
# To re-record, delete the `golden.yaml` file and run the test with a valid
# ORCH_OPENAI_API_KEY environment variable.
@pytest.mark.e2e
@vcr.use_cassette("tests/e2e/cassettes/golden.yaml")
def test_golden_transcript():
    """
    Runs a simple end-to-end test against the real OpenAI API (or a recording)
    to ensure the entire orchestration flow produces a valid, scored candidate.
    """
    orch = Orchestrator(
        review_agent,
        solution_agent,
        validator_agent,
        get_reflection_agent(),
        k_variants=1,
        max_iters=1,
    )
    result = orch.run_sync(Task(prompt="Write a short haiku about a robot learning to paint."))
    
    assert isinstance(result, Candidate)
    assert result.score > 0
    assert len(result.solution) > 10
    assert len(result.checklist.items) > 0 