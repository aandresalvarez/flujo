"""Integration tests for SQLiteBackend concurrency edge cases."""

import asyncio
import os
from pathlib import Path

import pytest
from datetime import datetime, timezone

from flujo.state.backends.sqlite import SQLiteBackend


def skip_if_root():
    """Skip tests if running as root user to avoid permission issues."""
    if getattr(os, "geteuid", lambda: -1)() == 0:
        pytest.skip(
            "permission-based SQLite tests skipped when running as root",
            allow_module_level=True,
        )


skip_if_root()


class TestSQLiteConcurrencyEdgeCases:
    """Test edge cases and error handling in SQLite backend concurrency scenarios."""

    @pytest.mark.asyncio
    async def test_concurrent_save_operations(self, tmp_path: Path) -> None:
        """Test concurrent save operations to ensure thread safety."""
        db_path = tmp_path / "test.db"
        backend = SQLiteBackend(db_path)

        # Create multiple concurrent save operations
        async def save_state(i: int) -> None:
            sample_state = {
                "pipeline_id": f"test_pipeline_{i}",
                "pipeline_name": f"test_pipeline_{i}",
                "pipeline_version": "1.0.0",
                "current_step_index": 0,
                "pipeline_context": {"test": "data"},
                "last_step_output": None,
                "step_history": [],
                "status": "running",
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
            }
            await backend.save_state(f"test_run_{i}", sample_state)

        # Run multiple concurrent save operations
        tasks = [save_state(i) for i in range(10)]
        await asyncio.gather(*tasks)

        # Verify that all states were saved
        for i in range(10):
            loaded_state = await backend.load_state(f"test_run_{i}")
            assert loaded_state is not None
            assert loaded_state["pipeline_id"] == f"test_pipeline_{i}"

    @pytest.mark.asyncio
    async def test_concurrent_load_operations(self, tmp_path: Path) -> None:
        """Test concurrent load operations to ensure thread safety."""
        db_path = tmp_path / "test.db"
        backend = SQLiteBackend(db_path)

        # First, save some states
        for i in range(5):
            sample_state = {
                "pipeline_id": f"test_pipeline_{i}",
                "pipeline_name": f"test_pipeline_{i}",
                "pipeline_version": "1.0.0",
                "current_step_index": 0,
                "pipeline_context": {"test": "data"},
                "last_step_output": None,
                "step_history": [],
                "status": "running",
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
            }
            await backend.save_state(f"test_run_{i}", sample_state)

        # Create multiple concurrent load operations
        async def load_state(i: int) -> None:
            loaded_state = await backend.load_state(f"test_run_{i}")
            assert loaded_state is not None
            assert loaded_state["pipeline_id"] == f"test_pipeline_{i}"

        # Run multiple concurrent load operations
        tasks = [load_state(i) for i in range(5)]
        await asyncio.gather(*tasks)

    @pytest.mark.asyncio
    async def test_concurrent_save_and_load_operations(self, tmp_path: Path) -> None:
        """Test concurrent save and load operations to ensure thread safety."""
        db_path = tmp_path / "test.db"
        backend = SQLiteBackend(db_path)

        # Create multiple concurrent save and load operations
        async def save_and_load_state(i: int) -> None:
            sample_state = {
                "pipeline_id": f"test_pipeline_{i}",
                "pipeline_name": f"test_pipeline_{i}",
                "pipeline_version": "1.0.0",
                "current_step_index": 0,
                "pipeline_context": {"test": "data"},
                "last_step_output": None,
                "step_history": [],
                "status": "running",
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
            }
            await backend.save_state(f"test_run_{i}", sample_state)

            # Immediately load the state
            loaded_state = await backend.load_state(f"test_run_{i}")
            assert loaded_state is not None
            assert loaded_state["pipeline_id"] == f"test_pipeline_{i}"

        # Run multiple concurrent save and load operations
        tasks = [save_and_load_state(i) for i in range(10)]
        await asyncio.gather(*tasks)

        # Verify that all states were saved and can be loaded
        for i in range(10):
            loaded_state = await backend.load_state(f"test_run_{i}")
            assert loaded_state is not None
            assert loaded_state["pipeline_id"] == f"test_pipeline_{i}"
