import pytest
import os
import tempfile
from flujo import Step, Pipeline, Flujo
from flujo.testing.utils import StubAgent
from flujo.state.backends.sqlite import SQLiteBackend

pytestmark = [pytest.mark.slow]


@pytest.fixture
def temp_db_path():
    """Create a temporary SQLite database path for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    yield db_path

    # Clean up after test
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.mark.asyncio
async def test_pipeline_run_span_finalization_regression(temp_db_path):
    """
    Regression test for Issue #590.
    Ensures that the root 'pipeline_run' span is persisted with status='completed'
    and has a non-null end_time in the database.
    """
    backend = SQLiteBackend(temp_db_path)
    try:
        # Simple pipeline
        step = Step.model_validate({"name": "step1", "agent": StubAgent(["ok"])})
        pipeline = Pipeline(steps=[step], name="regression_pipeline")

        # Use Flujo with the temporary SQLite backend
        # We use a explicit run_id for verification
        run_id = "test-run-590-regression"

        async with Flujo(pipeline, state_backend=backend) as runner:
            result = await runner.run_result_async("input", run_id=run_id)
            assert result.success is True

        # Now verify the persisted trace in the database via the backend
        persisted_trace = await backend.get_trace(run_id)
        assert persisted_trace is not None

        # Check the root span
        assert persisted_trace["name"] == "pipeline_run"
        assert persisted_trace["status"] == "completed", (
            f"Root span status should be 'completed', got {persisted_trace.get('status')}"
        )
        assert persisted_trace["end_time"] is not None, "Root span end_time should not be None"

        # Verify the child span as well
        assert len(persisted_trace["children"]) == 1
        child_span = persisted_trace["children"][0]
        assert child_span["name"] == "step1"
        assert child_span["status"] == "completed"
        assert child_span["end_time"] is not None
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_failed_pipeline_run_span_finalization_regression(temp_db_path):
    """
    Regression test for Issue #590 - Failed run case.
    Ensures that even if a pipeline fails, the root span is marked as 'failed' (or completed/processed)
    and has a non-null end_time in the database.
    """
    backend = SQLiteBackend(temp_db_path)
    try:

        class FailingAgent:
            async def run(self, _input_data):
                raise ValueError("Intentional failure")

        # Simple pipeline
        step = Step.model_validate({"name": "failing_step", "agent": FailingAgent()})
        pipeline = Pipeline(steps=[step], name="failing_regression_pipeline")

        run_id = "test-run-590-failed-regression"

        async with Flujo(pipeline, state_backend=backend) as runner:
            result = await runner.run_result_async("input", run_id=run_id)
            assert result.success is False

        # Now verify the persisted trace in the database
        persisted_trace = await backend.get_trace(run_id)
        assert persisted_trace is not None

        # Check the root span
        assert persisted_trace["name"] == "pipeline_run"
        # FSD-590 fix ensures this is called. TraceManager marks it based on result success.
        assert persisted_trace["status"] == "failed"
        assert persisted_trace["end_time"] is not None
    finally:
        await backend.close()
