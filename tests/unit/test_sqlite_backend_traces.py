"""Tests for SQLite backend trace storage with normalized spans table."""

import pytest
import asyncio
from uuid import uuid4
from datetime import datetime, timezone

from flujo.type_definitions.common import JSONObject

from flujo.state.backends.sqlite import SQLiteBackend

# Mark as very slow to exclude from fast suites
pytestmark = pytest.mark.veryslow


@pytest.fixture
async def sqlite_backend(tmp_path):
    """Create a SQLite backend for testing."""
    db_path = tmp_path / "test_traces.db"
    backend = SQLiteBackend(db_path)
    await backend._ensure_init()
    yield backend
    await backend.close()


def create_run_data(run_id: str) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "run_id": run_id,
        "pipeline_id": str(uuid4()),
        "pipeline_name": f"pipeline_{run_id}",
        "pipeline_version": "1.0",
        "status": "completed",
        "created_at": now,
        "updated_at": now,
        "end_time": now,
        "total_cost": 0.0,
        "final_context_blob": None,
    }


@pytest.fixture
def sample_trace_tree() -> JSONObject:
    """Create a sample trace tree for testing."""
    return {
        "span_id": "root-123",
        "name": "pipeline_root",
        "start_time": 1000.0,
        "end_time": 1100.0,
        "status": "completed",
        "attributes": {"initial_input": "test input"},
        "children": [
            {
                "span_id": "child-456",
                "name": "step_1",
                "start_time": 1005.0,
                "end_time": 1050.0,
                "status": "completed",
                "attributes": {"success": True, "attempts": 1},
                "children": [],
            },
            {
                "span_id": "child-789",
                "name": "step_2",
                "start_time": 1055.0,
                "end_time": 1095.0,
                "status": "failed",
                "attributes": {"success": False, "attempts": 3},
                "children": [],
            },
        ],
    }


class TestNormalizedTraceStorage:
    """Test the normalized trace storage functionality."""

    async def test_save_and_get_trace(self, sqlite_backend, sample_trace_tree):
        """Test saving and retrieving a complete trace tree."""
        run_id = "test-run-123"

        # Create run first
        await sqlite_backend.save_run_start(create_run_data(run_id))

        # Save trace
        await sqlite_backend.save_trace(run_id, sample_trace_tree)

        # Retrieve trace
        retrieved_trace = await sqlite_backend.get_trace(run_id)

        assert retrieved_trace is not None
        assert retrieved_trace["span_id"] == "root-123"
        assert retrieved_trace["name"] == "pipeline_root"
        assert retrieved_trace["status"] == "completed"
        assert len(retrieved_trace["children"]) == 2

        # Check children
        children = retrieved_trace["children"]
        assert children[0]["span_id"] == "child-456"
        assert children[0]["name"] == "step_1"
        assert children[0]["status"] == "completed"
        assert children[1]["span_id"] == "child-789"
        assert children[1]["name"] == "step_2"
        assert children[1]["status"] == "failed"

    async def test_save_trace_without_prior_run_row(self, sqlite_backend, sample_trace_tree):
        """Saving a trace without an existing run should auto-create the run to satisfy FKs."""
        run_id = "orphan-trace-run"

        # Intentionally do NOT call save_run_start here
        await sqlite_backend.save_trace(run_id, sample_trace_tree)

        # The helper should have created the run row; spans should be retrievable
        trace = await sqlite_backend.get_trace(run_id)
        assert trace is not None
        assert trace["span_id"] == sample_trace_tree["span_id"]

    async def test_save_step_without_prior_run_row(self, sqlite_backend):
        """Saving a step without an existing run should auto-create the run to satisfy FKs."""
        run_id = "orphan-steps-run"

        # Save a single step result without creating the run first
        await sqlite_backend.save_step_result(
            {
                "run_id": run_id,
                "step_name": "dummy",
                "step_index": 0,
                "status": "completed",
                "output": {"ok": True},
                "raw_response": None,
                "cost_usd": 0.0,
                "token_counts": 1,
                "execution_time_ms": 10,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        # Verify step exists and run metadata row was created implicitly
        steps = await sqlite_backend.list_run_steps(run_id)
        assert len(steps) == 1
        assert steps[0]["step_name"] == "dummy"

    async def test_get_spans_with_filtering(self, sqlite_backend, sample_trace_tree):
        """Test retrieving individual spans with filtering."""
        run_id = "test-run-456"

        # Create run first
        await sqlite_backend.save_run_start(create_run_data(run_id))

        await sqlite_backend.save_trace(run_id, sample_trace_tree)

        # Get all spans
        all_spans = await sqlite_backend.get_spans(run_id)
        assert len(all_spans) == 3  # root + 2 children

        # Filter by status
        completed_spans = await sqlite_backend.get_spans(run_id, status="completed")
        assert len(completed_spans) == 2  # root + step_1

        failed_spans = await sqlite_backend.get_spans(run_id, status="failed")
        assert len(failed_spans) == 1  # step_2

        # Filter by name
        step1_spans = await sqlite_backend.get_spans(run_id, name="step_1")
        assert len(step1_spans) == 1
        assert step1_spans[0]["name"] == "step_1"

        # Filter by both status and name
        completed_step1 = await sqlite_backend.get_spans(run_id, status="completed", name="step_1")
        assert len(completed_step1) == 1

    async def test_span_statistics(self, sqlite_backend):
        """Test span statistics aggregation."""
        # Create multiple traces for statistics
        traces = [
            {
                "span_id": "root-1",
                "name": "pipeline_root",
                "start_time": 1000.0,
                "end_time": 1100.0,
                "status": "completed",
                "attributes": {},
                "children": [
                    {
                        "span_id": "child-1",
                        "name": "step_1",
                        "start_time": 1005.0,
                        "end_time": 1050.0,
                        "status": "completed",
                        "attributes": {},
                        "children": [],
                    }
                ],
            },
            {
                "span_id": "root-2",
                "name": "pipeline_root",
                "start_time": 1200.0,
                "end_time": 1300.0,
                "status": "completed",
                "attributes": {},
                "children": [
                    {
                        "span_id": "child-2",
                        "name": "step_1",
                        "start_time": 1205.0,
                        "end_time": 1250.0,
                        "status": "completed",
                        "attributes": {},
                        "children": [],
                    }
                ],
            },
        ]

        # Save traces
        for i, trace in enumerate(traces):
            run_id = f"run-{i}"
            await sqlite_backend.save_run_start(create_run_data(run_id))
            await sqlite_backend.save_trace(run_id, trace)

        # Get statistics
        stats = await sqlite_backend.get_span_statistics()

        assert stats["total_spans"] == 4  # 2 roots + 2 children
        assert "pipeline_root" in stats["by_name"]
        assert "step_1" in stats["by_name"]
        assert stats["by_name"]["pipeline_root"] == 2
        assert stats["by_name"]["step_1"] == 2

        # Check status breakdown
        assert "completed" in stats["by_status"]
        assert stats["by_status"]["completed"] == 4

        # Check duration statistics
        assert "pipeline_root" in stats["avg_duration_by_name"]
        assert "step_1" in stats["avg_duration_by_name"]

        # Verify average durations
        pipeline_root_stats = stats["avg_duration_by_name"]["pipeline_root"]
        assert pipeline_root_stats["count"] == 2
        assert pipeline_root_stats["average"] == 100.0  # (100 + 100) / 2

        step1_stats = stats["avg_duration_by_name"]["step_1"]
        assert step1_stats["count"] == 2
        assert step1_stats["average"] == 45.0  # (45 + 45) / 2

    async def test_cascade_deletion(self, sqlite_backend, sample_trace_tree):
        """Test that spans are deleted when runs are deleted."""
        run_id = "test-run-cascade"

        # Create run first
        await sqlite_backend.save_run_start(create_run_data(run_id))

        await sqlite_backend.save_trace(run_id, sample_trace_tree)

        # Verify spans exist
        spans = await sqlite_backend.get_spans(run_id)
        assert len(spans) == 3

        # Delete the run
        await sqlite_backend.delete_run(run_id)

        # Verify spans are deleted
        spans_after = await sqlite_backend.get_spans(run_id)
        assert len(spans_after) == 0

    async def test_complex_nested_trace(self, sqlite_backend):
        """Test handling of complex nested trace structures."""
        complex_trace = {
            "span_id": "root-complex",
            "name": "pipeline_root",
            "start_time": 1000.0,
            "end_time": 1200.0,
            "status": "completed",
            "attributes": {},
            "children": [
                {
                    "span_id": "loop-1",
                    "name": "loop_step",
                    "start_time": 1005.0,
                    "end_time": 1150.0,
                    "status": "completed",
                    "attributes": {"iterations": 3},
                    "children": [
                        {
                            "span_id": "iter-1",
                            "name": "iteration_1",
                            "start_time": 1010.0,
                            "end_time": 1030.0,
                            "status": "completed",
                            "attributes": {"iteration": 1},
                            "children": [],
                        },
                        {
                            "span_id": "iter-2",
                            "name": "iteration_2",
                            "start_time": 1035.0,
                            "end_time": 1055.0,
                            "status": "completed",
                            "attributes": {"iteration": 2},
                            "children": [],
                        },
                    ],
                },
                {
                    "span_id": "conditional-1",
                    "name": "conditional_step",
                    "start_time": 1155.0,
                    "end_time": 1195.0,
                    "status": "completed",
                    "attributes": {"branch_taken": "main"},
                    "children": [
                        {
                            "span_id": "branch-main",
                            "name": "main_branch",
                            "start_time": 1160.0,
                            "end_time": 1190.0,
                            "status": "completed",
                            "attributes": {},
                            "children": [],
                        }
                    ],
                },
            ],
        }

        run_id = "test-complex-run"

        # Create run first
        await sqlite_backend.save_run_start(create_run_data(run_id))

        await sqlite_backend.save_trace(run_id, complex_trace)

        # Verify reconstruction
        retrieved_trace = await sqlite_backend.get_trace(run_id)
        assert retrieved_trace is not None
        assert retrieved_trace["span_id"] == "root-complex"
        assert len(retrieved_trace["children"]) == 2

        # Check loop step
        loop_step = retrieved_trace["children"][0]
        assert loop_step["name"] == "loop_step"
        assert len(loop_step["children"]) == 2

        # Check conditional step
        conditional_step = retrieved_trace["children"][1]
        assert conditional_step["name"] == "conditional_step"
        assert len(conditional_step["children"]) == 1

    async def test_span_attributes_preservation(self, sqlite_backend):
        """Test that span attributes are properly preserved."""
        trace_with_attrs = {
            "span_id": "root-attrs",
            "name": "pipeline_root",
            "start_time": 1000.0,
            "end_time": 1100.0,
            "status": "completed",
            "attributes": {
                "cost_usd": 0.05,
                "token_counts": {"input": 100, "output": 50},
                "custom_metric": "test_value",
            },
            "children": [],
        }

        run_id = "test-attrs-run"

        # Create run first
        await sqlite_backend.save_run_start(create_run_data(run_id))

        await sqlite_backend.save_trace(run_id, trace_with_attrs)

        # Retrieve and verify attributes
        retrieved_trace = await sqlite_backend.get_trace(run_id)
        assert retrieved_trace["attributes"]["cost_usd"] == 0.05
        assert retrieved_trace["attributes"]["token_counts"]["input"] == 100
        assert retrieved_trace["attributes"]["custom_metric"] == "test_value"

    async def test_empty_trace_handling(self, sqlite_backend):
        """Test handling of empty or invalid trace data."""
        run_id = "test-empty-run"

        # Create run first
        await sqlite_backend.save_run_start(create_run_data(run_id))

        # Test with empty trace
        await sqlite_backend.save_trace(run_id, {})

        # Should return None for empty trace
        retrieved_trace = await sqlite_backend.get_trace(run_id)
        assert retrieved_trace is None

        # Test with None trace
        run_id_none = "test-none-run"
        await sqlite_backend.save_run_start(create_run_data(run_id_none))
        await sqlite_backend.save_trace(run_id_none, None)
        retrieved_trace = await sqlite_backend.get_trace(run_id_none)
        assert retrieved_trace is None

    async def test_concurrent_access(self, sqlite_backend, sample_trace_tree):
        """Test concurrent access to trace storage."""
        run_ids = [f"concurrent-run-{i}" for i in range(5)]

        # Create runs first
        for run_id in run_ids:
            await sqlite_backend.save_run_start(create_run_data(run_id))

        # Save traces concurrently with unique span IDs
        async def save_trace(run_id: str):
            # Create a copy of the trace tree with unique span IDs
            import copy

            unique_trace = copy.deepcopy(sample_trace_tree)
            unique_trace["span_id"] = f"root-{run_id}"
            for i, child in enumerate(unique_trace["children"]):
                child["span_id"] = f"child-{run_id}-{i}"
            await sqlite_backend.save_trace(run_id, unique_trace)

        await asyncio.gather(*[save_trace(run_id) for run_id in run_ids])

        # Retrieve traces concurrently
        async def get_trace(run_id: str):
            return await sqlite_backend.get_trace(run_id)

        retrieved_traces = await asyncio.gather(*[get_trace(run_id) for run_id in run_ids])

        # Verify all traces were saved and retrieved correctly
        for i, trace in enumerate(retrieved_traces):
            assert trace is not None
            assert trace["span_id"] == f"root-concurrent-run-{i}"
            assert len(trace["children"]) == 2
