import os
import pytest
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from flujo.state.backends.sqlite import SQLiteBackend
import time


@pytest.mark.asyncio
async def test_sqlite_backend_large_dataset_performance(tmp_path: Path):
    """Test that SQLiteBackend can handle a large number of workflows efficiently."""
    backend = SQLiteBackend(tmp_path / "state.db")
    now = datetime.utcnow().replace(microsecond=0)
    num_workflows = 5000
    # Insert many workflows
    for i in range(num_workflows):
        state = {
            "run_id": f"run_{i}",
            "pipeline_id": f"pipeline_{i % 10}",
            "pipeline_name": f"Pipeline {i % 10}",
            "pipeline_version": "1.0",
            "current_step_index": i % 5,
            "pipeline_context": {"a": i},
            "last_step_output": f"out{i}",
            "status": "completed" if i % 2 == 0 else "failed",
            "created_at": now - timedelta(minutes=i),
            "updated_at": now - timedelta(minutes=i),
            "total_steps": 5,
            "error_message": None,
            "execution_time_ms": 1000 * (i % 10),
            "memory_usage_mb": 10.0 * (i % 10),
        }
        await backend.save_state(f"run_{i}", state)
    # Query all workflows
    t0 = time.time()
    all_workflows = await backend.list_workflows()
    t1 = time.time()
    assert len(all_workflows) == num_workflows
    # Query by status
    completed = await backend.list_workflows(status="completed")
    failed = await backend.list_workflows(status="failed")
    assert len(completed) + len(failed) == num_workflows
    # Query by pipeline_id
    for j in range(10):
        filtered = await backend.list_workflows(pipeline_id=f"pipeline_{j}")
        assert all(wf["pipeline_id"] == f"pipeline_{j}" for wf in filtered)
    # Performance check: listing all should be reasonably fast
    per_workflow_time_limit = float(
        os.getenv("SQLITE_PER_WORKFLOW_TIME_LIMIT", 0.0004)
    )  # Default to 0.0004s per workflow
    threshold = num_workflows * per_workflow_time_limit
    assert (t1 - t0) < threshold, (
        f"Performance test failed: took {t1 - t0:.2f}s, threshold is {threshold:.2f}s "
        f"({per_workflow_time_limit:.6f}s per workflow)"
    )


@pytest.mark.asyncio
async def test_sqlite_backend_high_concurrency(tmp_path: Path):
    """Test SQLiteBackend under high concurrent load (writers and readers)."""
    backend = SQLiteBackend(tmp_path / "state.db")
    now = datetime.utcnow().replace(microsecond=0)
    num_workflows = 1000
    num_workers = 20

    async def writer(i):
        state = {
            "run_id": f"run_{i}",
            "pipeline_id": f"pipeline_{i % 5}",
            "pipeline_name": f"Pipeline {i % 5}",
            "pipeline_version": "1.0",
            "current_step_index": i % 5,
            "pipeline_context": {"a": i},
            "last_step_output": f"out{i}",
            "status": "completed" if i % 2 == 0 else "failed",
            "created_at": now,
            "updated_at": now,
            "total_steps": 5,
            "error_message": None,
            "execution_time_ms": 1000 * (i % 10),
            "memory_usage_mb": 10.0 * (i % 10),
        }
        await backend.save_state(f"run_{i}", state)

    async def reader():
        # Randomly query by status and pipeline_id
        for _ in range(10):
            await backend.list_workflows(status="completed")
            await backend.list_workflows(status="failed")
            await backend.list_workflows(pipeline_id="pipeline_1")

    # Launch concurrent writers and readers
    await asyncio.gather(
        *(writer(i) for i in range(num_workflows)),
        *(reader() for _ in range(num_workers)),
    )
    # Final check: all workflows should be present
    all_workflows = await backend.list_workflows()
    assert len(all_workflows) == num_workflows


@pytest.mark.asyncio
async def test_sqlite_backend_query_pagination_and_filtering(tmp_path: Path):
    """Test query performance and correctness for pagination and filtering edge cases."""
    backend = SQLiteBackend(tmp_path / "state.db")
    now = datetime.utcnow().replace(microsecond=0)
    num_workflows = 200
    for i in range(num_workflows):
        state = {
            "run_id": f"run_{i}",
            "pipeline_id": f"pipeline_{i % 4}",
            "pipeline_name": f"Pipeline {i % 4}",
            "pipeline_version": "1.0",
            "current_step_index": i % 5,
            "pipeline_context": {"a": i},
            "last_step_output": f"out{i}",
            "status": "completed" if i % 2 == 0 else "failed",
            "created_at": now - timedelta(minutes=i),
            "updated_at": now - timedelta(minutes=i),
            "total_steps": 5,
            "error_message": None,
            "execution_time_ms": 1000 * (i % 10),
            "memory_usage_mb": 10.0 * (i % 10),
        }
        await backend.save_state(f"run_{i}", state)
    # Test pagination
    page_size = 25
    seen_ids = set()
    for offset in range(0, num_workflows, page_size):
        page = await backend.list_workflows(limit=page_size, offset=offset)
        assert len(page) <= page_size
        for wf in page:
            seen_ids.add(wf["run_id"])
    assert len(seen_ids) == num_workflows
    # Test filtering with pagination
    completed_count = 0
    for offset in range(0, num_workflows, page_size):
        page = await backend.list_workflows(status="completed", limit=page_size, offset=offset)
        completed_count += len(page)
        for wf in page:
            assert wf["status"] == "completed"
    assert completed_count == num_workflows // 2
    # Test filtering by pipeline_id
    for j in range(4):
        filtered = await backend.list_workflows(pipeline_id=f"pipeline_{j}")
        assert all(wf["pipeline_id"] == f"pipeline_{j}" for wf in filtered)
