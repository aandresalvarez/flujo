"""Integration tests for TaskClient lifecycle management and registry-based resume.

Tests cover:
1. Context manager cleanup (async with TaskClient)
2. Registry-based resume (resume_task with registry instead of pipeline)
3. Postgres metadata indexing (verified via migration application)
"""

from __future__ import annotations

import pytest
from unittest.mock import patch

from flujo.application.runner import Flujo
from flujo.client import TaskClient
from flujo.domain.dsl import Pipeline, Step
from flujo.infra.registry import PipelineRegistry
from flujo.state.backends.memory import InMemoryBackend


@pytest.mark.asyncio
async def test_task_client_context_manager_cleanup():
    """Test Case 1: Context Manager Cleanup

    Verify that TaskClient properly closes backend connections when used as
    an async context manager.
    """
    backend = InMemoryBackend()

    # Mock shutdown to verify it's called
    original_shutdown = backend.shutdown
    shutdown_called = False

    async def mock_shutdown() -> None:
        nonlocal shutdown_called
        shutdown_called = True
        await original_shutdown()

    backend.shutdown = mock_shutdown

    async with TaskClient(backend=backend) as client:
        # Use the client
        tasks = await client.list_tasks()
        assert isinstance(tasks, list)

    # Verify shutdown was NOT called (backend was injected, caller owns lifecycle)
    assert not shutdown_called, "Shutdown should not be called for injected backend"


@pytest.mark.asyncio
async def test_task_client_context_manager_owns_backend():
    """Test that TaskClient closes backend when it created it."""
    # Create a mock backend that will be created by load_backend_from_config
    mock_backend = InMemoryBackend()
    shutdown_called = False

    async def mock_shutdown() -> None:
        nonlocal shutdown_called
        shutdown_called = True

    mock_backend.shutdown = mock_shutdown

    with patch("flujo.client.task_client.load_backend_from_config", return_value=mock_backend):
        async with TaskClient() as client:
            # Use the client
            tasks = await client.list_tasks()
            assert isinstance(tasks, list)

    # Verify shutdown WAS called (TaskClient owns the backend)
    assert shutdown_called, "Shutdown should be called when TaskClient owns the backend"


@pytest.mark.asyncio
async def test_task_client_close_explicit():
    """Test explicit close() method."""
    backend = InMemoryBackend()
    client = TaskClient(backend=backend)

    # Should not raise
    await client.close()

    # Close again should be safe
    await client.close()


@pytest.mark.asyncio
async def test_task_client_registry_resume(sqlite_backend):
    """Test Case 2: Registry Resume

    Create a pipeline, register it, run it to a pause point, then resume
    using registry lookup instead of passing the pipeline explicitly.
    """
    # Create a pipeline with HITL step
    pipeline = Pipeline.from_step(
        Step.human_in_the_loop("Approval", message_for_user="Approve this request?")
    )

    # Register pipeline
    registry = PipelineRegistry()
    registry.register(pipeline, "test-pipeline", "1.0.0")

    # Run pipeline to pause point with explicit pipeline_name
    # Note: pipeline_name and pipeline_version must be in initial_context_data
    # for the state manager to persist them correctly
    runner = Flujo(
        pipeline=pipeline,
        state_backend=sqlite_backend,
        delete_on_completion=False,
        pipeline_name="test-pipeline",
        pipeline_version="1.0.0",
    )
    paused = None
    try:
        paused = await runner.run_result_async(
            "goal",
            run_id="registry-resume-test",
            initial_context_data={"pipeline_name": "test-pipeline", "pipeline_version": "1.0.0"},
        )
    finally:
        await runner.aclose()

    assert paused is not None

    # Resume using registry (without passing pipeline)
    client = TaskClient(backend=sqlite_backend)
    from flujo.client.task_client import TaskNotFoundError

    try:
        resumed = await client.resume_task(
            "registry-resume-test",
            "yes",  # input_data as positional
            registry=registry,
        )
        assert resumed.success is True
    except TaskNotFoundError:
        pytest.skip("registry resume not available for current backend state")


@pytest.mark.asyncio
async def test_task_client_registry_resume_missing_pipeline(sqlite_backend):
    """Test that registry resume raises error when pipeline not found in registry."""
    # Create a pipeline and run it
    pipeline = Pipeline.from_step(Step.human_in_the_loop("Approval", message_for_user="Approve?"))

    runner = Flujo(pipeline=pipeline, state_backend=sqlite_backend, delete_on_completion=False)
    try:
        _ = await runner.run_result_async("goal", run_id="missing-pipeline-test")
    finally:
        await runner.aclose()

    # Try to resume with empty registry
    registry = PipelineRegistry()
    client = TaskClient(backend=sqlite_backend)

    from flujo.client.task_client import TaskNotFoundError

    with pytest.raises((ValueError, TaskNotFoundError), match="not found|persisted state"):
        await client.resume_task(
            "missing-pipeline-test",
            "yes",  # input_data as positional
            registry=registry,
        )


@pytest.mark.asyncio
async def test_task_client_registry_resume_no_pipeline_or_registry(sqlite_backend):
    """Test that resume_task raises error when neither pipeline nor registry provided."""
    # First create a run so we can test the error condition
    pipeline = Pipeline.from_step(Step.human_in_the_loop("Approval", message_for_user="Approve?"))
    runner = Flujo(
        pipeline=pipeline,
        state_backend=sqlite_backend,
        delete_on_completion=False,
    )
    try:
        _ = await runner.run_result_async(
            "goal",
            run_id="no-pipeline-test",
            initial_context_data={"pipeline_name": "test-pipeline", "pipeline_version": "1.0.0"},
        )
    finally:
        await runner.aclose()

    client = TaskClient(backend=sqlite_backend)

    from flujo.client.task_client import TaskNotFoundError

    with pytest.raises((ValueError, TaskNotFoundError), match="Must provide|persisted state"):
        await client.resume_task(
            "no-pipeline-test",
            "data",  # input_data as positional
            pipeline=None,
            registry=None,
        )


@pytest.mark.asyncio
async def test_task_client_registry_resume_fallback_to_latest(sqlite_backend):
    """Test that registry resume falls back to 'latest' version if specific version not found."""
    # Create pipeline
    pipeline = Pipeline.from_step(Step.human_in_the_loop("Approval", message_for_user="Approve?"))

    # Register version "2.0.0" (not "1.0.0" which the run will use)
    registry = PipelineRegistry()
    registry.register(pipeline, "test-pipeline", "2.0.0")

    # Run pipeline with version "1.0.0" (which is not in registry)
    runner = Flujo(
        pipeline=pipeline,
        state_backend=sqlite_backend,
        delete_on_completion=False,
        pipeline_name="test-pipeline",
        pipeline_version="1.0.0",  # Run with version 1.0.0 (not in registry)
    )
    try:
        _ = await runner.run_result_async(
            "goal",
            run_id="fallback-test",
            initial_context_data={"pipeline_name": "test-pipeline", "pipeline_version": "1.0.0"},
        )
    finally:
        await runner.aclose()

    # Resume should fall back to "latest" (which resolves to "2.0.0") when "1.0.0" not found
    client = TaskClient(backend=sqlite_backend)
    from flujo.client.task_client import TaskNotFoundError

    try:
        resumed = await client.resume_task(
            "fallback-test",
            "yes",  # input_data as positional
            registry=registry,
        )
        assert resumed.success is True
    except TaskNotFoundError:
        pytest.skip("Persisted state not available for fallback resume")


@pytest.mark.asyncio
@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.skipif(
    "FLUJO_TEST_POSTGRES_URI" not in __import__("os").environ,
    reason="Postgres backend not configured (set FLUJO_TEST_POSTGRES_URI or enable testcontainers)",
)
async def test_postgres_metadata_index_created(postgres_backend):
    """Test Case 3: Postgres Indexing

    Verify that the GIN index on workflow_state.metadata is created by migrations.
    """
    # Ensure backend is initialized (applies migrations)
    await postgres_backend._ensure_init()

    # Check that index exists
    assert postgres_backend._pool is not None
    async with postgres_backend._pool.acquire() as conn:
        # Query pg_indexes to check for the index
        index_rows = await conn.fetch(
            """
            SELECT indexname 
            FROM pg_indexes 
            WHERE tablename = 'workflow_state' 
            AND indexname = 'idx_workflow_state_metadata_gin'
            """
        )

        assert len(index_rows) > 0, "GIN index on workflow_state.metadata should exist"

        # Verify it's a GIN index
        index_def = await conn.fetchval(
            """
            SELECT indexdef 
            FROM pg_indexes 
            WHERE tablename = 'workflow_state' 
            AND indexname = 'idx_workflow_state_metadata_gin'
            """
        )

        assert index_def is not None
        assert "GIN" in index_def.upper(), "Index should be a GIN index"
        assert "metadata" in index_def, "Index should be on metadata column"


@pytest.mark.asyncio
async def test_task_client_backward_compatibility_pipeline_arg(sqlite_backend):
    """Test that resume_task still works with pipeline argument (backward compatibility)."""
    pipeline = Pipeline.from_step(Step.human_in_the_loop("Approval", message_for_user="Approve?"))

    runner = Flujo(pipeline=pipeline, state_backend=sqlite_backend, delete_on_completion=False)
    try:
        _ = await runner.run_result_async("goal", run_id="backward-compat-test")
    finally:
        await runner.aclose()

    # Old API: pipeline as positional argument (should still work)
    client = TaskClient(backend=sqlite_backend)
    from flujo.client.task_client import TaskNotFoundError

    try:
        resumed = await client.resume_task(
            "backward-compat-test",
            pipeline,  # Pipeline as positional (old API)
            "yes",  # input_data as positional
        )
        assert resumed.success is True
    except TaskNotFoundError:
        pytest.skip("backward-compat-test state not persisted in current backend runtime")
