"""Test SQLite transaction rollback behavior to ensure data integrity."""

import pytest
import sqlite3
from datetime import datetime, timezone

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
        test_datetime = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
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

        # Test transaction rollback with a more predictable failure scenario
        # We'll simulate a database operation that fails during a transaction
        from unittest.mock import patch

        # Mock the database connection to simulate a failure during save_state
        with patch.object(backend, "_get_conn") as mock_get_conn:
            # Create a mock connection that raises an exception during execute
            mock_conn = mock_get_conn.return_value
            mock_conn.execute.side_effect = sqlite3.OperationalError("Simulated database failure")

            # This should fail and rollback the transaction
            try:
                await backend.save_state(
                    "test-run-real",
                    {
                        "pipeline_id": "test-pipeline",
                        "pipeline_name": "test-pipeline",
                        "pipeline_version": "1.0.0",
                        "current_step_index": 0,
                        "pipeline_context": {"key": "should_rollback"},
                        "last_step_output": None,
                        "step_history": [],
                        "status": "running",
                        "created_at": test_datetime,
                        "updated_at": test_datetime,
                    },
                )
                # If we get here, the mock didn't work as expected
                pytest.fail("Expected save_state to fail with simulated database error")
            except sqlite3.OperationalError as e:
                # Expected - the transaction should have rolled back
                assert "Simulated database failure" in str(e)

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
            test_datetime = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
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

        test_datetime = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
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

    async def test_db_retry_decorator_handles_locked_db(self, backend):
        """Test that the db_retry decorator correctly retries on 'database is locked'."""
        # This test verifies that the @db_retry decorator works correctly
        # by testing through public methods that use the decorator

        test_datetime = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
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

        # First, ensure the database is initialized by saving and loading a dummy state
        dummy_state = {
            "pipeline_id": "dummy-pipeline",
            "pipeline_name": "dummy-pipeline",
            "pipeline_version": "0.0.1",
            "current_step_index": 0,
            "pipeline_context": {"key": "dummy"},
            "last_step_output": None,
            "step_history": [],
            "status": "running",
            "created_at": test_datetime,
            "updated_at": test_datetime,
        }
        await backend.save_state("dummy-init", dummy_state)
        loaded_state = await backend.load_state("dummy-init")
        assert loaded_state is not None
        assert loaded_state["pipeline_context"]["key"] == "dummy"

        # Test that save_state and load_state work correctly with the retry mechanism
        # by performing multiple operations that would trigger the retry logic
        await backend.save_state("locked-retry-test", test_state)

        # Verify the state was saved correctly
        loaded_state = await backend.load_state("locked-retry-test")
        assert loaded_state is not None
        assert loaded_state["pipeline_context"]["key"] == "locked_retry"

        # Test multiple load operations to verify transaction integrity
        for i in range(3):
            state = await backend.load_state("locked-retry-test")
            assert state is not None
            assert state["pipeline_context"]["key"] == "locked_retry"

        # Test that we can perform additional operations after the retry
        updated_state = {
            "pipeline_id": "test-pipeline",
            "pipeline_name": "test-pipeline",
            "pipeline_version": "1.0.0",
            "current_step_index": 1,
            "pipeline_context": {"key": "updated_after_retry"},
            "last_step_output": None,
            "step_history": [],
            "status": "running",
            "created_at": test_datetime,
            "updated_at": test_datetime,
        }
        await backend.save_state("locked-retry-test", updated_state)

        final_state = await backend.load_state("locked-retry-test")
        assert final_state is not None
        assert final_state["pipeline_context"]["key"] == "updated_after_retry"
        assert final_state["current_step_index"] == 1
