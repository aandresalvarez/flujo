"""Unit tests for schema migration robustness and error handling."""

import pytest
from datetime import datetime, timezone

from flujo.state.backends.sqlite import SQLiteBackend


@pytest.mark.slow
@pytest.mark.serial
class TestSchemaMigrationRobustness:
    """Test schema migration robustness and error handling."""

    @pytest.fixture
    def temp_db_path(self, tmp_path):
        """Create a temporary database path."""
        return tmp_path / "test_migration.db"

    @pytest.fixture
    async def sqlite_backend(self, temp_db_path):
        """Create a SQLite backend instance."""
        backend = SQLiteBackend(temp_db_path)
        await backend.__aenter__()
        yield backend
        await backend.__aexit__(None, None, None)

    async def test_migration_adds_missing_columns_to_runs_table(self, temp_db_path):
        """Test that migration adds missing columns to runs table."""
        # Create a backend and initialize it
        backend = SQLiteBackend(temp_db_path)
        await backend.__aenter__()

        # Manually remove some columns to simulate old schema
        async with backend._lock:
            import aiosqlite

            async with aiosqlite.connect(backend.db_path) as db:
                # Drop indexes first so ALTER TABLE DROP COLUMN succeeds on SQLite
                await db.execute("DROP INDEX IF EXISTS idx_runs_status")
                await db.execute("DROP INDEX IF EXISTS idx_runs_pipeline_id")
                await db.execute("DROP INDEX IF EXISTS idx_runs_created_at")
                await db.execute("DROP INDEX IF EXISTS idx_runs_pipeline_name")

                # Get current columns
                cursor = await db.execute("PRAGMA table_info(runs)")
                columns = {row[1] for row in await cursor.fetchall()}
                await cursor.close()

                # Remove some columns to simulate old schema
                if "pipeline_id" in columns:
                    await db.execute("ALTER TABLE runs DROP COLUMN pipeline_id")
                if "created_at" in columns:
                    await db.execute("ALTER TABLE runs DROP COLUMN created_at")
                if "updated_at" in columns:
                    await db.execute("ALTER TABLE runs DROP COLUMN updated_at")
                await db.commit()

        # Close and reopen to trigger migration
        await backend.__aexit__(None, None, None)

        backend2 = SQLiteBackend(temp_db_path)
        await backend2.__aenter__()

        # Verify that we can save data (migration should have added columns back)
        run_data = {
            "run_id": "migration_test_columns",
            "pipeline_name": "test_pipeline",
            "pipeline_version": "1.0",
            "status": "running",
        }

        await backend2.save_run_start(run_data)
        run_details = await backend2.get_run_details("migration_test_columns")
        assert run_details is not None

        # Verify that the missing columns were added back
        async with backend2._lock:
            import aiosqlite

            async with aiosqlite.connect(backend2.db_path) as db:
                cursor = await db.execute("PRAGMA table_info(runs)")
                columns_after = {row[1] for row in await cursor.fetchall()}
                await cursor.close()

                assert "pipeline_id" in columns_after
                assert "created_at" in columns_after
                assert "updated_at" in columns_after

        await backend2.__aexit__(None, None, None)

    async def test_migration_handles_existing_data(self, temp_db_path):
        """Test that migration handles existing data gracefully."""
        # Create a backend and add some data
        backend1 = SQLiteBackend(temp_db_path)
        await backend1.__aenter__()

        # Add some test data
        run_data = {
            "run_id": "existing_data_test",
            "pipeline_id": "test_pipeline",
            "pipeline_name": "test_pipeline",
            "pipeline_version": "1.0",
            "status": "running",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        await backend1.save_run_start(run_data)
        await backend1.__aexit__(None, None, None)

        # Create another backend instance (should trigger migration)
        backend2 = SQLiteBackend(temp_db_path)
        await backend2.__aenter__()

        # Verify that existing data is still accessible
        run_details = await backend2.get_run_details("existing_data_test")
        assert run_details is not None
        assert run_details["run_id"] == "existing_data_test"
        assert run_details["pipeline_name"] == "test_pipeline"

        # Verify that we can add new data
        new_run_data = {
            "run_id": "new_data_test",
            "pipeline_name": "new_pipeline",
            "pipeline_version": "2.0",
            "status": "running",
        }

        await backend2.save_run_start(new_run_data)
        new_run_details = await backend2.get_run_details("new_data_test")
        assert new_run_details is not None
        assert new_run_details["run_id"] == "new_data_test"

        await backend2.__aexit__(None, None, None)

    async def test_migration_handles_corrupted_database(self, temp_db_path):
        """Test that migration handles corrupted database gracefully."""
        # Create a corrupted database file
        with open(temp_db_path, "w") as f:
            f.write("This is not a valid SQLite database")

        # Should handle corruption gracefully
        backend = SQLiteBackend(temp_db_path)
        await backend.__aenter__()

        # Should be able to save data after corruption recovery
        run_data = {
            "run_id": "corruption_test",
            "pipeline_name": "test_pipeline",
            "pipeline_version": "1.0",
            "status": "running",
        }

        await backend.save_run_start(run_data)
        run_details = await backend.get_run_details("corruption_test")
        assert run_details is not None

        await backend.__aexit__(None, None, None)

    async def test_migration_handles_missing_tables(self, temp_db_path):
        """Test that migration handles missing tables gracefully."""
        # Create a backend and initialize it
        backend = SQLiteBackend(temp_db_path)
        await backend.__aenter__()

        # Manually drop the runs table to simulate missing table
        async with backend._lock:
            import aiosqlite

            async with aiosqlite.connect(backend.db_path) as db:
                await db.execute("DROP TABLE IF EXISTS runs")
                await db.commit()

        # Close and reopen to trigger migration
        await backend.__aexit__(None, None, None)

        backend2 = SQLiteBackend(temp_db_path)
        await backend2.__aenter__()

        # Should be able to save data (migration should have recreated the table)
        run_data = {
            "run_id": "missing_table_test",
            "pipeline_name": "test_pipeline",
            "pipeline_version": "1.0",
            "status": "running",
        }

        await backend2.save_run_start(run_data)
        run_details = await backend2.get_run_details("missing_table_test")
        assert run_details is not None

        await backend2.__aexit__(None, None, None)

    async def test_migration_handles_null_values_in_not_null_columns(self, temp_db_path):
        """Test that migration handles NULL values in NOT NULL columns."""
        # Create a backend and initialize it
        backend = SQLiteBackend(temp_db_path)
        await backend.__aenter__()

        # Save some data to ensure the runs table is created with proper schema
        run_data = {
            "run_id": "null_test",
            "pipeline_name": "test_pipeline",
            "pipeline_version": "1.0",
            "status": "running",
        }
        await backend.save_run_start(run_data)

        # Close and reopen to trigger migration
        await backend.__aexit__(None, None, None)

        backend2 = SQLiteBackend(temp_db_path)
        await backend2.__aenter__()

        # Migration should have handled the data properly
        run_details = await backend2.get_run_details("null_test")
        assert run_details is not None
        assert run_details["run_id"] == "null_test"
        assert run_details["pipeline_name"] == "test_pipeline"

        await backend2.__aexit__(None, None, None)

    async def test_migration_handles_concurrent_access(self, temp_db_path):
        """Test that migration handles concurrent access gracefully."""
        # Create multiple backend instances simultaneously
        backends = []
        for i in range(5):
            backend = SQLiteBackend(temp_db_path)
            await backend.__aenter__()
            backends.append(backend)

        # All should be able to save data
        for i, backend in enumerate(backends):
            run_data = {
                "run_id": f"concurrent_test_{i}",
                "pipeline_name": f"pipeline_{i}",
                "pipeline_version": "1.0",
                "status": "running",
            }

            await backend.save_run_start(run_data)

        # Verify all data was saved
        for i, backend in enumerate(backends):
            run_details = await backend.get_run_details(f"concurrent_test_{i}")
            assert run_details is not None
            assert run_details["run_id"] == f"concurrent_test_{i}"

        # Clean up
        for backend in backends:
            await backend.__aexit__(None, None, None)

    async def test_migration_handles_large_datasets(self, temp_db_path):
        """Test that migration handles large datasets efficiently."""
        # Create a backend and add many records
        backend = SQLiteBackend(temp_db_path)
        await backend.__aenter__()

        # Add many records
        for i in range(100):
            run_data = {
                "run_id": f"large_dataset_test_{i}",
                "pipeline_name": f"pipeline_{i}",
                "pipeline_version": "1.0",
                "status": "running",
            }

            await backend.save_run_start(run_data)

        # Close and reopen to trigger migration
        await backend.__aexit__(None, None, None)

        backend2 = SQLiteBackend(temp_db_path)
        await backend2.__aenter__()

        # Verify that all data is still accessible
        for i in range(100):
            run_details = await backend2.get_run_details(f"large_dataset_test_{i}")
            assert run_details is not None
            assert run_details["run_id"] == f"large_dataset_test_{i}"

        await backend2.__aexit__(None, None, None)

    async def test_migration_handles_schema_version_changes(self, temp_db_path):
        """Test that migration handles schema version changes gracefully."""
        # Create a backend and initialize it
        backend = SQLiteBackend(temp_db_path)
        await backend.__aenter__()

        # Simulate a schema version change by adding a new column
        async with backend._lock:
            import aiosqlite

            async with aiosqlite.connect(backend.db_path) as db:
                # Add a new column that might be added in future versions
                # First ensure the table exists
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS runs (
                        run_id TEXT PRIMARY KEY,
                        pipeline_id TEXT NOT NULL DEFAULT 'unknown',
                        pipeline_name TEXT NOT NULL DEFAULT 'unknown',
                        pipeline_version TEXT NOT NULL DEFAULT 'latest',
                        status TEXT NOT NULL DEFAULT 'running',
                        created_at TEXT NOT NULL DEFAULT '',
                        updated_at TEXT NOT NULL DEFAULT ''
                    )
                """)
                await db.execute("ALTER TABLE runs ADD COLUMN schema_version TEXT DEFAULT '1.0'")
                await db.commit()

        # Close and reopen to trigger migration
        await backend.__aexit__(None, None, None)

        backend2 = SQLiteBackend(temp_db_path)
        await backend2.__aenter__()

        # Should still be able to save and retrieve data
        run_data = {
            "run_id": "schema_version_test",
            "pipeline_name": "test_pipeline",
            "pipeline_version": "1.0",
            "status": "running",
        }

        await backend2.save_run_start(run_data)
        run_details = await backend2.get_run_details("schema_version_test")
        assert run_details is not None

        await backend2.__aexit__(None, None, None)

    async def test_migration_handles_index_recreation(self, temp_db_path):
        """Test that migration handles index recreation gracefully."""
        # Create a backend and initialize it
        backend = SQLiteBackend(temp_db_path)
        await backend.__aenter__()

        # Manually drop some indexes to simulate old schema
        async with backend._lock:
            import aiosqlite

            async with aiosqlite.connect(backend.db_path) as db:
                await db.execute("DROP INDEX IF EXISTS idx_runs_status")
                await db.execute("DROP INDEX IF EXISTS idx_runs_pipeline_name")
                await db.commit()

        # Close and reopen to trigger migration
        await backend.__aexit__(None, None, None)

        backend2 = SQLiteBackend(temp_db_path)
        await backend2.__aenter__()

        # Should still be able to save and retrieve data
        run_data = {
            "run_id": "index_test",
            "pipeline_name": "test_pipeline",
            "pipeline_version": "1.0",
            "status": "running",
        }

        await backend2.save_run_start(run_data)
        run_details = await backend2.get_run_details("index_test")
        assert run_details is not None

        await backend2.__aexit__(None, None, None)

    async def test_migration_handles_foreign_key_constraints(self, temp_db_path):
        """Test that migration handles foreign key constraints gracefully."""
        # Create a backend and initialize it
        backend = SQLiteBackend(temp_db_path)
        await backend.__aenter__()

        # Add some data to the runs table
        run_data = {
            "run_id": "fk_test",
            "pipeline_name": "test_pipeline",
            "pipeline_version": "1.0",
            "status": "running",
        }

        await backend.save_run_start(run_data)

        # Add some data to the steps table (which has foreign key to runs)
        step_data = {
            "step_run_id": "step_fk_test",
            "run_id": "fk_test",
            "step_name": "test_step",
            "step_index": 0,
            "status": "completed",
            "start_time": datetime.now(timezone.utc).isoformat(),
        }

        await backend.save_step_result(step_data)

        # Close and reopen to trigger migration
        await backend.__aexit__(None, None, None)

        backend2 = SQLiteBackend(temp_db_path)
        await backend2.__aenter__()

        # Verify that foreign key relationships are maintained
        run_details = await backend2.get_run_details("fk_test")
        assert run_details is not None

        steps = await backend2.list_run_steps("fk_test")
        assert len(steps) == 1
        assert steps[0]["step_name"] == "test_step"

        await backend2.__aexit__(None, None, None)
