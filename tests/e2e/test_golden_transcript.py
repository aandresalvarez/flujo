import pytest

try:
    import vcr
except Exception:  # pragma: no cover - skip if dependency missing
    vcr = None
    pytest.skip("vcrpy not installed", allow_module_level=True)
from flujo.application.orchestrator import Orchestrator
from flujo.domain.models import Task, Candidate
from flujo.infra.agents import (
    review_agent,
    solution_agent,
    validator_agent,
    get_reflection_agent,
)

pytest.skip("Skipping golden transcript", allow_module_level=True)


def scrub_auth(request):
    if "authorization" in request.headers:
        request.headers["authorization"] = ["Bearer [REDACTED]"]
    return request


# Note: This test makes real API calls that are recorded to a cassette.
# To re-record, delete the `golden.yaml` file and run the test with a valid
# OPENAI_API_KEY environment variable.
@pytest.mark.e2e
@pytest.mark.skipif(vcr is None, reason="vcrpy not installed")
@vcr.use_cassette("tests/e2e/cassettes/golden.yaml", before_record_request=scrub_auth)
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
    assert isinstance(result.solution, str)
