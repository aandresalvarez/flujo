"""Test integration of trace saving into pipeline execution flow."""

import pytest
from unittest.mock import MagicMock
from datetime import datetime, timezone
from uuid import uuid4

from flujo.application.core.state_manager import StateManager
from flujo.application.core.execution_manager import ExecutionManager
from flujo.domain.models import PipelineResult
from flujo.domain.dsl import Step, Pipeline
from flujo.testing.utils import StubAgent

# Mark as slow (uses SQLite backend with proper cleanup)
pytestmark = pytest.mark.slow


@pytest.mark.asyncio
async def test_trace_saving_integration(sqlite_backend_factory) -> None:
    """Test that trace saving is integrated into the pipeline execution flow."""
    # Create a backend and state manager
    backend = sqlite_backend_factory("test.db")
    state_manager = StateManager(backend)

    # Create a simple pipeline
    step1 = Step(name="step1", agent=StubAgent(["Hello"]))
    step2 = Step(name="step2", agent=StubAgent(["World"]))
    pipeline = step1 >> step2

    # Create execution manager
    execution_manager = ExecutionManager(pipeline=pipeline, state_manager=state_manager)

    # Create a mock trace tree
    mock_trace_tree = MagicMock()
    mock_trace_tree.span_id = "root_123"
    mock_trace_tree.name = "pipeline_root"
    mock_trace_tree.start_time = 1234567890.0
    mock_trace_tree.end_time = 1234567895.0
    mock_trace_tree.parent_span_id = None
    mock_trace_tree.attributes = {"test": "integration"}
    mock_trace_tree.children = []
    mock_trace_tree.status = "completed"

    # Create pipeline result with trace tree
    result = PipelineResult(step_history=[])
    result.trace_tree = mock_trace_tree

    # Create a run first
    run_id = "test_integration_run"
    await backend.save_run_start(
        {
            "run_id": run_id,
            "pipeline_id": str(uuid4()),
            "pipeline_name": "test_pipeline",
            "pipeline_version": "1.0",
            "status": "running",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    )

    # Execute the pipeline (this should trigger trace saving)
    async for _ in execution_manager.execute_steps(
        start_idx=0,
        data="test_input",
        context=None,
        result=result,
        run_id=run_id,
    ):
        pass

    # Persist final state (this calls record_run_end which saves the trace)
    await execution_manager.persist_final_state(
        run_id=run_id,
        context=None,
        result=result,
        start_idx=0,
        state_created_at=None,
        final_status="completed",
    )

    # Verify that the trace was saved
    saved_trace = await backend.get_trace(run_id)
    assert saved_trace is not None
    assert saved_trace["span_id"] == "root_123"
    assert saved_trace["name"] == "pipeline_root"
    assert saved_trace["attributes"]["test"] == "integration"


@pytest.mark.asyncio
async def test_trace_saving_without_trace_tree(sqlite_backend_factory) -> None:
    """Test that pipeline execution works when no trace tree is present."""
    backend = sqlite_backend_factory("test.db")
    state_manager = StateManager(backend)

    # Create a simple pipeline
    step1 = Step(name="step1", agent=StubAgent(["Hello"]))
    pipeline = Pipeline.from_step(step1)

    execution_manager = ExecutionManager(pipeline=pipeline, state_manager=state_manager)

    # Create pipeline result without trace tree
    result = PipelineResult(step_history=[])
    # result.trace_tree is None by default

    run_id = "test_no_trace_run"
    await backend.save_run_start(
        {
            "run_id": run_id,
            "pipeline_id": str(uuid4()),
            "pipeline_name": "test_pipeline",
            "pipeline_version": "1.0",
            "status": "running",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    )

    # Execute the pipeline
    async for _ in execution_manager.execute_steps(
        start_idx=0,
        data="test_input",
        context=None,
        result=result,
        run_id=run_id,
    ):
        pass

    # Persist final state
    await execution_manager.persist_final_state(
        run_id=run_id,
        context=None,
        result=result,
        start_idx=0,
        state_created_at=None,
        final_status="completed",
    )

    # Verify that no trace was saved (since there was no trace tree)
    saved_trace = await backend.get_trace(run_id)
    assert saved_trace is None


@pytest.mark.asyncio
async def test_trace_saving_error_handling(sqlite_backend_factory) -> None:
    """Test that trace saving errors don't break pipeline execution."""
    backend = sqlite_backend_factory("test.db")
    state_manager = StateManager(backend)

    # Create a pipeline result with a problematic trace tree
    result = PipelineResult(step_history=[])
    result.trace_tree = "invalid_trace_tree"  # This will cause conversion to fail

    run_id = "test_error_handling_run"
    await backend.save_run_start(
        {
            "run_id": run_id,
            "pipeline_id": str(uuid4()),
            "pipeline_name": "test_pipeline",
            "pipeline_version": "1.0",
            "status": "running",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    )

    # This should not raise an exception even though trace saving fails
    await state_manager.record_run_end(run_id, result)

    # Verify that the run was still recorded (trace saving failure didn't break it)
    # The trace should contain sanitized error information in the attributes
    saved_trace = await backend.get_trace(run_id)
    assert saved_trace is not None
    assert saved_trace["name"] == "trace_save_error"
    assert "error_summary" in saved_trace["attributes"]
    assert "error_type" in saved_trace["attributes"]
    assert "Trace serialization failed" in saved_trace["attributes"]["error_summary"]
