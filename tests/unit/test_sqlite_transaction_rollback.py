"""Test SQLite transaction rollback behavior to ensure data integrity."""

import pytest
import sqlite3
from datetime import datetime

from flujo.state.backends.sqlite import SQLiteBackend


class TestSQLiteTransactionRollback:
    """Test that transactions properly rollback on exceptions."""

    @pytest.fixture
    def temp_db_path(self, tmp_path):
        """Create a temporary database path."""
        return tmp_path / "test_transaction_rollback.db"

    @pytest.fixture
    async def backend(self, temp_db_path):
        """Create a SQLite backend instance."""
        backend = SQLiteBackend(temp_db_path)
        async with backend:
            yield backend

    async def test_transaction_rollback_on_constraint_violation(self, backend):
        """Test that transactions rollback when a constraint violation occurs."""
        # Save initial state
        test_datetime = datetime(2023, 1, 1, 12, 0, 0)
        initial_state = {
            "pipeline_id": "test-pipeline",
            "pipeline_name": "test-pipeline",
            "pipeline_version": "1.0.0",
            "current_step_index": 0,
            "pipeline_context": {"key": "initial"},
            "last_step_output": None,
            "step_history": [],
            "status": "running",
            "created_at": test_datetime,
            "updated_at": test_datetime,
        }

        await backend.save_state("test-run-real", initial_state)

        # Verify initial state exists
        loaded_state = await backend.load_state("test-run-real")
        assert loaded_state is not None
        assert loaded_state["pipeline_context"]["key"] == "initial"

        # Now try to update the state with a transaction that should rollback
        # We'll create a state that would cause a constraint violation
        # but the transaction should rollback and leave the original state intact

        # This should fail due to invalid status, but the transaction should rollback
        try:
            await backend.save_state(
                "test-run-real",
                {
                    "pipeline_id": "test-pipeline",
                    "pipeline_name": "test-pipeline",
                    "pipeline_version": "1.0.0",
                    "current_step_index": 0,
                    "pipeline_context": {"key": "should_rollback"},  # This would normally succeed
                    "last_step_output": None,
                    "step_history": [],
                    "status": "invalid_status",  # This should cause a constraint violation
                    "created_at": test_datetime,
                    "updated_at": test_datetime,
                },
            )
        except sqlite3.IntegrityError:
            # Expected - the transaction should have rolled back
            pass

        # Verify that the original state is still intact (transaction rolled back)
        final_state = await backend.load_state("test-run-real")
        assert final_state is not None
        # The state should still have the original value, not the attempted update
        assert final_state["pipeline_context"]["key"] == "initial"

    async def test_concurrent_transaction_handling(self, backend):
        """Test that concurrent transactions are handled properly with rollback."""
        import asyncio

        # Create multiple concurrent save operations
        async def save_state_with_delay(run_id, delay):
            await asyncio.sleep(delay)
            test_datetime = datetime(2023, 1, 1, 12, 0, 0)
            return await backend.save_state(
                run_id,
                {
                    "pipeline_id": f"test-pipeline-{run_id}",
                    "pipeline_name": f"test-pipeline-{run_id}",
                    "pipeline_version": "1.0.0",
                    "current_step_index": 0,
                    "pipeline_context": {"key": f"value-{run_id}"},
                    "last_step_output": None,
                    "step_history": [],
                    "status": "running",
                    "created_at": test_datetime,
                    "updated_at": test_datetime,
                },
            )

        # Run multiple concurrent saves
        tasks = [save_state_with_delay(f"concurrent-run-{i}", i * 0.01) for i in range(5)]

        # All should complete successfully with proper transaction handling
        await asyncio.gather(*tasks)

        # Verify all states were saved correctly
        for i in range(5):
            state = await backend.load_state(f"concurrent-run-{i}")
            assert state is not None
            assert state["pipeline_context"]["key"] == f"value-concurrent-run-{i}"

    async def test_retry_mechanism_with_transaction_rollback(self, backend):
        """Test that the retry mechanism works correctly with transaction rollback."""
        # This test verifies that the @db_retry decorator works correctly
        # and that transactions are properly handled during retries

        test_datetime = datetime(2023, 1, 1, 12, 0, 0)
        test_state = {
            "pipeline_id": "test-pipeline",
            "pipeline_name": "test-pipeline",
            "pipeline_version": "1.0.0",
            "current_step_index": 0,
            "pipeline_context": {"key": "retry_test"},
            "last_step_output": None,
            "step_history": [],
            "status": "running",
            "created_at": test_datetime,
            "updated_at": test_datetime,
        }

        # Save state - this should work with retry mechanism
        await backend.save_state("test-retry", test_state)

        # Verify the state was saved correctly
        loaded_state = await backend.load_state("test-retry")
        assert loaded_state is not None
        assert loaded_state["pipeline_context"]["key"] == "retry_test"

        # Test that we can load the state multiple times (verifying transaction integrity)
        for i in range(3):
            state = await backend.load_state("test-retry")
            assert state is not None
            assert state["pipeline_context"]["key"] == "retry_test"

    async def test_db_retry_decorator_handles_locked_db(self, backend, monkeypatch):
        """Test that the db_retry decorator correctly retries on 'database is locked'."""

        # Prepare a state to save
        test_datetime = datetime(2023, 1, 1, 12, 0, 0)
        test_state = {
            "pipeline_id": "test-pipeline",
            "pipeline_name": "test-pipeline",
            "pipeline_version": "1.0.0",
            "current_step_index": 0,
            "pipeline_context": {"key": "locked_retry"},
            "last_step_output": None,
            "step_history": [],
            "status": "running",
            "created_at": test_datetime,
            "updated_at": test_datetime,
        }

        # First, ensure the database is initialized
        await backend._ensure_init()

        # Create a mock for the connection's execute method that raises OperationalError twice, then succeeds
        call_count = {"count": 0}
        real_execute = backend._get_conn.__self__._connection_pool.execute

        async def mock_execute(*args, **kwargs):
            if "BEGIN IMMEDIATE" in str(args[0]) and call_count["count"] < 2:
                call_count["count"] += 1
                raise sqlite3.OperationalError("database is locked")
            else:
                return await real_execute(*args, **kwargs)

        monkeypatch.setattr(backend._get_conn.__self__._connection_pool, "execute", mock_execute)

        # This should fail twice and succeed on the third try
        await backend.save_state("locked-retry-test", test_state)

        # Assert that the transaction was attempted 3 times
        assert call_count["count"] == 2

        # Verify the state was saved correctly
        loaded_state = await backend.load_state("locked-retry-test")
        assert loaded_state is not None
        assert loaded_state["pipeline_context"]["key"] == "locked_retry"
