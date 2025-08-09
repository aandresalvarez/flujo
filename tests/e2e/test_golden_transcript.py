import pytest
import os

# Conditionally skip the test unless explicitly enabled.
if not os.getenv("CI_E2E_RUN"):
    pytest.skip(
        "Skipping E2E golden transcript test; run manually or via E2E workflow.",
        allow_module_level=True,
    )

try:
    import vcr
except ImportError:  # pragma: no cover - skip if dependency missing
    pytest.skip("vcrpy not installed", allow_module_level=True)
from flujo.recipes.factories import make_default_pipeline, run_default_pipeline
from flujo.domain.models import Task, Candidate
from flujo.agents import (
    make_review_agent,
    make_solution_agent,
    make_validator_agent,
    get_reflection_agent,
)


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
    pipeline = make_default_pipeline(
        review_agent=make_review_agent(),
        solution_agent=make_solution_agent(),
        validator_agent=make_validator_agent(),
        reflection_agent=get_reflection_agent(),
        k_variants=1,
        max_iters=1,
    )
    import asyncio

    result = asyncio.run(
        run_default_pipeline(
            pipeline,
            Task(prompt="Write a short haiku about a robot learning to paint."),
        )
    )

    assert isinstance(result, Candidate)
    assert isinstance(result.solution, str)
