"""Unit tests for SQLite trace persistence functionality."""

import asyncio

import pytest
from datetime import datetime, timezone

from flujo.state.backends.sqlite import SQLiteBackend

# Mark as very slow to exclude from fast suites
pytestmark = pytest.mark.veryslow


# Helper to create a run before saving a trace
def create_run(backend: SQLiteBackend, run_id: str) -> dict:
    now = datetime.now(timezone.utc)
    run_data = {
        "run_id": run_id,
        "pipeline_id": f"pid_{run_id}",
        "pipeline_name": f"pipeline_{run_id}",
        "pipeline_version": "1.0",
        "status": "completed",
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
        "end_time": now,
        "total_cost": 0.0,
    }
    return run_data


@pytest.mark.asyncio
async def test_save_and_get_trace_roundtrip(sqlite_backend_factory) -> None:
    backend = sqlite_backend_factory("test.db")
    run_id = "test_run_123"
    await backend.save_run_start(create_run(backend, run_id))
    trace_data = {
        "span_id": "root_123",
        "name": "pipeline_root",
        "start_time": 1234567890.0,
        "end_time": 1234567895.0,
        "parent_span_id": None,
        "attributes": {"run_id": run_id, "initial_input": "test input"},
        "children": [
            {
                "span_id": "root_123_child_0",
                "name": "step1",
                "start_time": 1234567891.0,
                "end_time": 1234567892.0,
                "parent_span_id": "root_123",
                "attributes": {
                    "success": True,
                    "attempts": 1,
                    "latency_s": 1.0,
                    "cost_usd": 0.01,
                    "token_counts": 100,
                },
                "children": [],
            }
        ],
        "status": "completed",
    }
    await backend.save_trace(run_id, trace_data)
    retrieved_trace = await backend.get_trace(run_id)
    assert retrieved_trace is not None
    assert retrieved_trace["span_id"] == "root_123"
    assert retrieved_trace["name"] == "pipeline_root"
    assert retrieved_trace["children"][0]["name"] == "step1"
    assert retrieved_trace["children"][0]["attributes"]["success"] is True


@pytest.mark.asyncio
async def test_get_trace_nonexistent_run(sqlite_backend_factory) -> None:
    backend = sqlite_backend_factory("test.db")
    result = await backend.get_trace("nonexistent_run")
    assert result is None


@pytest.mark.asyncio
async def test_save_trace_overwrites_existing(sqlite_backend_factory) -> None:
    backend = sqlite_backend_factory("test.db")
    run_id = "test_run"
    await backend.save_run_start(create_run(backend, run_id))
    initial_trace = {
        "span_id": "root_1",
        "name": "initial_pipeline",
        "start_time": 1234567890.0,
        "end_time": 1234567895.0,
        "attributes": {"version": "1.0"},
        "children": [],
    }
    await backend.save_trace(run_id, initial_trace)
    updated_trace = {
        "span_id": "root_2",
        "name": "updated_pipeline",
        "start_time": 1234567890.0,
        "end_time": 1234567895.0,
        "attributes": {"version": "2.0"},
        "children": [],
    }
    await backend.save_trace(run_id, updated_trace)
    retrieved_trace = await backend.get_trace(run_id)
    assert retrieved_trace["span_id"] == "root_2"
    assert retrieved_trace["name"] == "updated_pipeline"
    assert retrieved_trace["attributes"]["version"] == "2.0"


@pytest.mark.asyncio
async def test_save_trace_complex_nested_structure(sqlite_backend_factory) -> None:
    backend = sqlite_backend_factory("test.db")
    run_id = "complex_run"
    await backend.save_run_start(create_run(backend, run_id))
    complex_trace = {
        "span_id": "root_complex",
        "name": "complex_pipeline",
        "start_time": 1234567890.0,
        "end_time": 1234567900.0,
        "attributes": {"total_steps": 5},
        "children": [
            {
                "span_id": "root_complex_child_0",
                "name": "loop_step",
                "start_time": 1234567891.0,
                "end_time": 1234567898.0,
                "parent_span_id": "root_complex",
                "attributes": {"iterations": 3},
                "children": [
                    {
                        "span_id": "root_complex_child_0_child_0",
                        "name": "iteration_1",
                        "start_time": 1234567892.0,
                        "end_time": 1234567893.0,
                        "parent_span_id": "root_complex_child_0",
                        "attributes": {"iteration": 1},
                        "children": [],
                    },
                    {
                        "span_id": "root_complex_child_0_child_1",
                        "name": "iteration_2",
                        "start_time": 1234567894.0,
                        "end_time": 1234567895.0,
                        "parent_span_id": "root_complex_child_0",
                        "attributes": {"iteration": 2},
                        "children": [],
                    },
                ],
            },
            {
                "span_id": "root_complex_child_1",
                "name": "final_step",
                "start_time": 1234567899.0,
                "end_time": 1234567900.0,
                "parent_span_id": "root_complex",
                "attributes": {"success": True},
                "children": [],
            },
        ],
    }
    await backend.save_trace(run_id, complex_trace)
    retrieved_trace = await backend.get_trace(run_id)
    assert retrieved_trace["span_id"] == "root_complex"
    assert len(retrieved_trace["children"]) == 2

    # Find children by name instead of relying on array position
    loop_step = next(
        (child for child in retrieved_trace["children"] if child["name"] == "loop_step"), None
    )
    assert loop_step is not None, "Child with name 'loop_step' not found"
    assert len(loop_step["children"]) == 2

    iteration_1 = next(
        (child for child in loop_step["children"] if child["name"] == "iteration_1"), None
    )
    assert iteration_1 is not None, "Child with name 'iteration_1' not found"

    final_step = next(
        (child for child in retrieved_trace["children"] if child["name"] == "final_step"), None
    )
    assert final_step is not None, "Child with name 'final_step' not found"


@pytest.mark.asyncio
async def test_save_trace_with_special_json_types(sqlite_backend_factory) -> None:
    backend = sqlite_backend_factory("test.db")
    run_id = "special_run"
    await backend.save_run_start(create_run(backend, run_id))
    trace_with_special_types = {
        "span_id": "root_special",
        "name": "special_pipeline",
        "start_time": 1234567890.0,
        "end_time": 1234567895.0,
        "attributes": {
            "null_value": None,
            "boolean_true": True,
            "boolean_false": False,
            "integer": 42,
            "float": 3.14159,
            "empty_list": [],
            "empty_dict": {},
            "nested": {"inner_null": None, "inner_bool": True},
        },
        "children": [],
    }
    await backend.save_trace(run_id, trace_with_special_types)
    retrieved_trace = await backend.get_trace(run_id)
    attrs = retrieved_trace["attributes"]
    assert attrs["null_value"] is None
    assert attrs["boolean_true"] is True
    assert attrs["boolean_false"] is False
    assert attrs["integer"] == 42
    assert attrs["float"] == 3.14159
    assert attrs["empty_list"] == []
    assert attrs["empty_dict"] == {}
    assert attrs["nested"]["inner_null"] is None
    assert attrs["nested"]["inner_bool"] is True


@pytest.mark.asyncio
async def test_trace_persistence_with_run_deletion(sqlite_backend_factory) -> None:
    backend = sqlite_backend_factory("test.db")
    run_id = "test_run_cascade"
    await backend.save_run_start(create_run(backend, run_id))
    await backend.save_run_end(run_id, {"status": "completed"})
    trace_data = {
        "span_id": "root_cascade",
        "name": "cascade_test",
        "start_time": 1234567890.0,
        "end_time": 1234567895.0,
        "attributes": {"test": "cascade"},
        "children": [],
    }
    await backend.save_trace(run_id, trace_data)
    assert await backend.get_trace(run_id) is not None
    await backend.delete_run(run_id)
    assert await backend.get_trace(run_id) is None


@pytest.mark.asyncio
async def test_concurrent_trace_operations(sqlite_backend_factory) -> None:
    backend = sqlite_backend_factory("test.db")

    async def save_trace_worker(run_id: str, trace_data: dict) -> None:
        await backend.save_run_start(create_run(backend, run_id))
        await backend.save_trace(run_id, trace_data)

    async def get_trace_worker(run_id: str) -> dict:
        return await backend.get_trace(run_id)

    trace_tasks = []
    for i in range(5):
        run_id = f"concurrent_run_{i}"
        trace_data = {
            "span_id": f"root_{i}",
            "name": f"pipeline_{i}",
            "start_time": 1234567890.0 + i,
            "end_time": 1234567895.0 + i,
            "attributes": {"index": i},
            "children": [],
        }
        task = save_trace_worker(run_id, trace_data)
        trace_tasks.append(task)
    await asyncio.gather(*trace_tasks)
    for i in range(5):
        run_id = f"concurrent_run_{i}"
        trace = await backend.get_trace(run_id)
        assert trace is not None
        assert trace["span_id"] == f"root_{i}"
        assert trace["name"] == f"pipeline_{i}"
        assert trace["attributes"]["index"] == i


@pytest.mark.asyncio
@pytest.mark.slow  # Mark as slow due to large trace structure
async def test_trace_persistence_large_trace(sqlite_backend_factory) -> None:
    backend = sqlite_backend_factory("test.db")
    run_id = "large_run"
    await backend.save_run_start(create_run(backend, run_id))

    def create_nested_spans(depth: int, max_depth: int, parent_id: str) -> list:
        if depth >= max_depth:
            return []
        children = []
        for i in range(3):
            child_id = f"{parent_id}_child_{i}"
            child = {
                "span_id": child_id,
                "name": f"level_{depth}_child_{i}",
                "start_time": 1234567890.0 + depth + i,
                "end_time": 1234567895.0 + depth + i,
                "parent_span_id": parent_id,
                "attributes": {"depth": depth, "child_index": i},
                "children": create_nested_spans(depth + 1, max_depth, child_id),
            }
            children.append(child)
        return children

    large_trace = {
        "span_id": "root_large",
        "name": "large_pipeline",
        "start_time": 1234567890.0,
        "end_time": 1234568000.0,
        "attributes": {"total_depth": 4},
        "children": create_nested_spans(0, 4, "root_large"),
    }
    await backend.save_trace(run_id, large_trace)
    retrieved_trace = await backend.get_trace(run_id)
    assert retrieved_trace["span_id"] == "root_large"
    assert len(retrieved_trace["children"]) == 3
    level1 = retrieved_trace["children"][0]
    assert len(level1["children"]) == 3
    level2 = level1["children"][0]
    assert len(level2["children"]) == 3
    level3 = level2["children"][0]
    assert len(level3["children"]) == 3
    level4 = level3["children"][0]
    assert len(level4["children"]) == 0
