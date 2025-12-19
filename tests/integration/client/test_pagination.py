"""Integration tests for pagination fixes in SQLite and InMemory backends.

These tests verify that pagination (limit/offset) is applied AFTER metadata filtering,
not before, to prevent the "vanishing task" bug where valid results are hidden
because they fall outside the initial SQL LIMIT window.

Test cases:
1. The "Hidden Item" Test - items beyond the first page should be findable with metadata filters
2. Offset Accuracy - pagination should work correctly with metadata filters
3. Dependency Check - PostgresBackend should raise clear error if asyncpg is missing
"""

from __future__ import annotations

import pytest
from datetime import datetime, timedelta, timezone

from flujo.client import TaskClient
from flujo.state.backends.memory import InMemoryBackend
from flujo.type_definitions.common import JSONObject


@pytest.mark.asyncio
async def test_sqlite_pagination_hidden_item_bug(sqlite_backend):
    """Test Case 1: The 'Hidden Item' Test

    Create 20 runs where:
    - Runs 1-15: metadata={"batch": "A"} (older)
    - Runs 16-20: metadata={"batch": "B"} (most recent)

    When querying with metadata_filter={"batch": "A"} and limit=5,
    we should get 5 items (runs 11-15), not 0 items.

    Current Bug Behavior: Returns 0 items (SQL fetches Runs 16-20, Python filters them all out)
    Fixed Behavior: SQL fetches all (or sufficient) rows, Python finds the "A" batch, then paginates
    """
    now = datetime.now(timezone.utc)

    # Create 20 runs: batches A (1-15) and B (16-20)
    # Most recent runs (16-20) have batch="B"
    # Older runs (1-15) have batch="A"
    for i in range(1, 21):
        created_at_dt = now - timedelta(minutes=20 - i)  # Most recent = highest i
        created_at_str = created_at_dt.isoformat()
        run_data: JSONObject = {
            "run_id": f"run_{i:02d}",
            "pipeline_id": f"pipeline_{i}",
            "pipeline_name": "Test Pipeline",
            "pipeline_version": "1.0",
            "status": "completed",
            "created_at": created_at_str,
            "updated_at": created_at_str,
            "execution_time_ms": 1000,
        }
        await sqlite_backend.save_run_start(run_data)

        # Add metadata via workflow_state: batch A for runs 1-15, batch B for runs 16-20
        # Note: save_state expects datetime objects, not strings
        metadata = {"batch": "A" if i <= 15 else "B"}
        workflow_state: JSONObject = {
            "run_id": f"run_{i:02d}",
            "pipeline_id": f"pipeline_{i}",
            "pipeline_name": "Test Pipeline",
            "pipeline_version": "1.0",
            "status": "completed",
            "metadata": metadata,
            "created_at": created_at_dt,  # Pass datetime object, not string
            "updated_at": created_at_dt,  # Pass datetime object, not string
            "pipeline_context": {},
            "current_step_index": 0,
            "total_steps": 1,
        }
        await sqlite_backend.save_state(f"run_{i:02d}", workflow_state)

    client = TaskClient(backend=sqlite_backend)

    # Query for batch A with limit=5 (should get runs 11-15, not 0)
    results = await client.list_tasks(
        metadata_filter={"batch": "A"},
        limit=5,
    )

    # Should return 5 items, not 0
    assert len(results) == 5, f"Expected 5 items, got {len(results)}"

    # All results should have batch="A"
    for result in results:
        assert result.metadata.get("batch") == "A"

    # Verify we got 5 batch A runs (should be runs 11-15, the 5 most recent batch A runs)
    # Note: runs are ordered by created_at DESC, so most recent batch A runs are 15, 14, 13, 12, 11
    run_ids = {result.run_id for result in results}
    expected_run_ids = {f"run_{i:02d}" for i in range(11, 16)}
    assert run_ids == expected_run_ids, f"Expected runs 11-15, got {run_ids}"


@pytest.mark.asyncio
async def test_sqlite_pagination_offset_accuracy(sqlite_backend):
    """Test Case 2: Offset Accuracy

    Create 20 runs with batch="A", then query with:
    - metadata_filter={"batch": "A"}, limit=5, offset=0 -> should get runs 16-20 (most recent)
    - metadata_filter={"batch": "A"}, limit=5, offset=5 -> should get runs 11-15
    - metadata_filter={"batch": "A"}, limit=5, offset=10 -> should get runs 6-10
    """
    now = datetime.now(timezone.utc)

    # Create 20 runs, all with batch="A"
    for i in range(1, 21):
        created_at_dt = now - timedelta(minutes=20 - i)  # Most recent = highest i
        created_at_str = created_at_dt.isoformat()
        run_data: JSONObject = {
            "run_id": f"run_{i:02d}",
            "pipeline_id": f"pipeline_{i}",
            "pipeline_name": "Test Pipeline",
            "pipeline_version": "1.0",
            "status": "completed",
            "created_at": created_at_str,
            "updated_at": created_at_str,
            "execution_time_ms": 1000,
        }
        await sqlite_backend.save_run_start(run_data)

        metadata = {"batch": "A"}
        workflow_state: JSONObject = {
            "run_id": f"run_{i:02d}",
            "pipeline_id": f"pipeline_{i}",
            "pipeline_name": "Test Pipeline",
            "pipeline_version": "1.0",
            "status": "completed",
            "metadata": metadata,
            "created_at": created_at_dt,  # Pass datetime object, not string
            "updated_at": created_at_dt,  # Pass datetime object, not string
            "pipeline_context": {},
            "current_step_index": 0,
            "total_steps": 1,
        }
        await sqlite_backend.save_state(f"run_{i:02d}", workflow_state)

    client = TaskClient(backend=sqlite_backend)

    # Page 1: offset=0, limit=5 -> should get runs 16-20 (most recent 5)
    page1 = await client.list_tasks(
        metadata_filter={"batch": "A"},
        limit=5,
        offset=0,
    )
    assert len(page1) == 5
    run_ids_page1 = {r.run_id for r in page1}
    expected_page1 = {f"run_{i:02d}" for i in range(16, 21)}
    assert run_ids_page1 == expected_page1, f"Page 1: Expected runs 16-20, got {run_ids_page1}"

    # Page 2: offset=5, limit=5 -> should get runs 11-15
    page2 = await client.list_tasks(
        metadata_filter={"batch": "A"},
        limit=5,
        offset=5,
    )
    assert len(page2) == 5
    run_ids_page2 = {r.run_id for r in page2}
    expected_page2 = {f"run_{i:02d}" for i in range(11, 16)}
    assert run_ids_page2 == expected_page2, f"Page 2: Expected runs 11-15, got {run_ids_page2}"

    # Page 3: offset=10, limit=5 -> should get runs 6-10
    page3 = await client.list_tasks(
        metadata_filter={"batch": "A"},
        limit=5,
        offset=10,
    )
    assert len(page3) == 5
    run_ids_page3 = {r.run_id for r in page3}
    expected_page3 = {f"run_{i:02d}" for i in range(6, 11)}
    assert run_ids_page3 == expected_page3, f"Page 3: Expected runs 6-10, got {run_ids_page3}"

    # Verify no overlap between pages
    assert run_ids_page1.isdisjoint(run_ids_page2)
    assert run_ids_page2.isdisjoint(run_ids_page3)
    assert run_ids_page1.isdisjoint(run_ids_page3)


@pytest.mark.asyncio
async def test_sqlite_pagination_no_metadata_filter(sqlite_backend):
    """Verify that pagination still works correctly when NO metadata_filter is provided.

    This ensures we didn't break the fast path (SQL pagination) for non-filtered queries.
    """
    now = datetime.now(timezone.utc)

    # Create 20 runs
    for i in range(1, 21):
        created_at_dt = now - timedelta(minutes=20 - i)
        created_at_str = created_at_dt.isoformat()
        run_data: JSONObject = {
            "run_id": f"run_{i:02d}",
            "pipeline_id": f"pipeline_{i}",
            "pipeline_name": "Test Pipeline",
            "pipeline_version": "1.0",
            "status": "completed",
            "created_at": created_at_str,
            "updated_at": created_at_str,
            "execution_time_ms": 1000,
        }
        await sqlite_backend.save_run_start(run_data)

    client = TaskClient(backend=sqlite_backend)

    # Query without metadata filter - should use fast SQL pagination
    page1 = await client.list_tasks(limit=5, offset=0)
    assert len(page1) == 5

    page2 = await client.list_tasks(limit=5, offset=5)
    assert len(page2) == 5

    # Verify no overlap
    run_ids_page1 = {r.run_id for r in page1}
    run_ids_page2 = {r.run_id for r in page2}
    assert run_ids_page1.isdisjoint(run_ids_page2)


@pytest.mark.asyncio
async def test_inmemory_pagination_hidden_item_bug():
    """Test Case 1 for InMemoryBackend: The 'Hidden Item' Test"""
    backend = InMemoryBackend()
    now = datetime.now(timezone.utc)

    # Create 20 runs: batches A (1-15) and B (16-20)
    for i in range(1, 21):
        created_at_dt = now - timedelta(minutes=20 - i)
        workflow_state: JSONObject = {
            "run_id": f"run_{i:02d}",
            "pipeline_id": f"pipeline_{i}",
            "pipeline_name": "Test Pipeline",
            "pipeline_version": "1.0",
            "status": "completed",
            "metadata": {"batch": "A" if i <= 15 else "B"},
            "created_at": created_at_dt,  # Pass datetime object, not string
            "updated_at": created_at_dt,  # Pass datetime object, not string
            "pipeline_context": {},
            "current_step_index": 0,
            "total_steps": 1,
        }
        await backend.save_state(f"run_{i:02d}", workflow_state)

    client = TaskClient(backend=backend)

    # Query for batch A with limit=5
    results = await client.list_tasks(
        metadata_filter={"batch": "A"},
        limit=5,
    )

    # Should return 5 items
    assert len(results) == 5

    # All results should have batch="A"
    for result in results:
        assert result.metadata.get("batch") == "A"


@pytest.mark.asyncio
async def test_inmemory_pagination_offset_accuracy():
    """Test Case 2 for InMemoryBackend: Offset Accuracy"""
    backend = InMemoryBackend()
    now = datetime.now(timezone.utc)

    # Create 20 runs, all with batch="A"
    for i in range(1, 21):
        created_at_dt = now - timedelta(minutes=20 - i)
        workflow_state: JSONObject = {
            "run_id": f"run_{i:02d}",
            "pipeline_id": f"pipeline_{i}",
            "pipeline_name": "Test Pipeline",
            "pipeline_version": "1.0",
            "status": "completed",
            "metadata": {"batch": "A"},
            "created_at": created_at_dt,  # Pass datetime object, not string
            "updated_at": created_at_dt,  # Pass datetime object, not string
            "pipeline_context": {},
            "current_step_index": 0,
            "total_steps": 1,
        }
        await backend.save_state(f"run_{i:02d}", workflow_state)

    client = TaskClient(backend=backend)

    # Page 1: offset=0, limit=5
    page1 = await client.list_tasks(
        metadata_filter={"batch": "A"},
        limit=5,
        offset=0,
    )
    assert len(page1) == 5

    # Page 2: offset=5, limit=5
    page2 = await client.list_tasks(
        metadata_filter={"batch": "A"},
        limit=5,
        offset=5,
    )
    assert len(page2) == 5

    # Verify no overlap
    run_ids_page1 = {r.run_id for r in page1}
    run_ids_page2 = {r.run_id for r in page2}
    assert run_ids_page1.isdisjoint(run_ids_page2)


def test_postgres_dependency_guard_missing_asyncpg(monkeypatch):
    """Test Case 3: Dependency Check

    Verify that load_backend_from_config raises a clear error when:
    - flujo.toml specifies a Postgres URI
    - asyncpg is not installed

    Should raise ImportError with installation instructions, not a raw ModuleNotFoundError.
    """
    import importlib.util

    # Mock importlib.util.find_spec to simulate asyncpg not being installed
    original_find_spec = importlib.util.find_spec

    def mock_find_spec(name: str):
        if name == "asyncpg":
            return None  # Simulate asyncpg not installed
        return original_find_spec(name)

    monkeypatch.setattr(importlib.util, "find_spec", mock_find_spec)
    monkeypatch.delenv("FLUJO_STATE_URI", raising=False)

    # Mock get_state_uri to return a postgres URI
    monkeypatch.setattr(
        "flujo.cli.config.get_state_uri", lambda **kwargs: "postgres://localhost/test"
    )

    # Mock test environment detection to prevent SQLite fallback
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    # Mock ConfigManager.get_settings since load_backend_from_config calls cfg_manager.get_settings()
    monkeypatch.setattr(
        "flujo.infra.config_manager.ConfigManager.get_settings",
        lambda self, **kwargs: type("Settings", (), {"test_mode": False})(),
    )

    from flujo.cli.config import load_backend_from_config
    import typer

    # Should raise typer.Exit(1) with a clear error message
    with pytest.raises(typer.Exit) as exc_info:
        load_backend_from_config()

    assert exc_info.value.exit_code == 1


def test_postgres_dependency_guard_in_factory(monkeypatch):
    """Verify that BackendFactory.create_state_backend also checks for asyncpg."""
    import importlib.util

    # Mock importlib.util.find_spec to simulate asyncpg not being installed
    original_find_spec = importlib.util.find_spec

    def mock_find_spec(name: str):
        if name == "asyncpg":
            return None
        return original_find_spec(name)

    monkeypatch.setattr(importlib.util, "find_spec", mock_find_spec)
    monkeypatch.delenv("FLUJO_STATE_URI", raising=False)

    # Mock get_state_uri to return a postgres URI
    monkeypatch.setattr(
        "flujo.application.core.factories.get_state_uri",
        lambda **kwargs: "postgres://localhost/test",
    )

    # Mock test_mode to be False so it doesn't return InMemoryBackend
    def mock_get_settings():
        return type("Settings", (), {"test_mode": False})()

    monkeypatch.setattr("flujo.application.core.factories.get_settings", mock_get_settings)

    from flujo.application.core.factories import BackendFactory

    factory = BackendFactory()

    # Should raise ImportError with clear message
    with pytest.raises(ImportError) as exc_info:
        factory.create_state_backend()

    error_msg = str(exc_info.value)
    assert "asyncpg is required" in error_msg
    assert "pip install flujo[postgres]" in error_msg
