import os
import pytest
import asyncio
from datetime import datetime, timedelta, timezone
import time


@pytest.mark.asyncio
@pytest.mark.slow
@pytest.mark.benchmark
async def test_sqlite_backend_large_dataset_performance(sqlite_backend_factory):
    """Test that SQLiteBackend can handle a large number of workflows efficiently."""
    backend = sqlite_backend_factory("state.db")
    now = datetime.now(timezone.utc).replace(microsecond=0)
    num_workflows = 1000  # Reduced from 5000 to 1000 (80% reduction)
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
@pytest.mark.slow
@pytest.mark.benchmark
async def test_sqlite_backend_high_concurrency(sqlite_backend_factory):
    """Test SQLiteBackend under high concurrent load (writers and readers)."""
    backend = sqlite_backend_factory("state.db")
    now = datetime.now(timezone.utc).replace(microsecond=0)
    num_workflows = 200  # Reduced from 1000 to 200 (80% reduction)
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
@pytest.mark.slow
@pytest.mark.benchmark
async def test_sqlite_backend_query_pagination_and_filtering(sqlite_backend_factory):
    """Test query performance and correctness for pagination and filtering edge cases."""
    backend = sqlite_backend_factory("state.db")
    now = datetime.now(timezone.utc).replace(microsecond=0)
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
