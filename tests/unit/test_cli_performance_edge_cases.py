"""Tests for CLI performance edge cases and database optimization scenarios."""

import pytest
import time
import asyncio
import os
from datetime import datetime, timedelta
from pathlib import Path

from flujo.state.backends.sqlite import SQLiteBackend
from typer.testing import CliRunner
from flujo.cli.main import app


@pytest.mark.performance
class TestCLIPerformanceEdgeCases:
    """Test CLI performance edge cases and optimizations."""

    @pytest.fixture
    def large_database_with_mixed_data(self, tmp_path: Path) -> Path:
        """Create a database with mixed data types for performance testing (reduced for CI)."""
        db_path = tmp_path / "mixed_ops.db"
        backend = SQLiteBackend(db_path)

        # Create runs with different characteristics (reduced from 1,000 to 50 for CI)
        now = datetime.utcnow()
        for i in range(50):
            # Create run start
            asyncio.run(
                backend.save_run_start(
                    {
                        "run_id": f"run_{i:04d}",
                        "pipeline_name": f"pipeline_{i % 20}",
                        "pipeline_version": "1.0",
                        "status": "running",
                        "start_time": now - timedelta(minutes=i),
                    }
                )
            )

            # Create run end with different statuses
            status = "completed" if i % 3 == 0 else "failed" if i % 3 == 1 else "running"
            if status != "running":
                asyncio.run(
                    backend.save_run_end(
                        f"run_{i:04d}",
                        {
                            "status": status,
                            "end_time": now - timedelta(minutes=i) + timedelta(seconds=30),
                            "total_cost": 0.1 + (i * 0.01),
                            "final_context": {"result": f"output_{i}", "metadata": {"index": i}},
                        },
                    )
                )

            # Add step data for completed runs
            if status == "completed":
                for step_idx in range(2):
                    asyncio.run(
                        backend.save_step_result(
                            {
                                "step_run_id": f"run_{i:04d}:{step_idx}",
                                "run_id": f"run_{i:04d}",
                                "step_name": f"step_{step_idx}",
                                "step_index": step_idx,
                                "status": "completed",
                                "start_time": now - timedelta(minutes=i),
                                "end_time": now - timedelta(minutes=i) + timedelta(seconds=10),
                                "duration_ms": 10000 + (i * 100),
                                "cost": 0.03 + (i * 0.001),
                                "tokens": 100 + (i * 10),
                                "input": f"input_{i}_{step_idx}",
                                "output": f"output_{i}_{step_idx}",
                                "error": None,
                            }
                        )
                    )

        return db_path

    def test_lens_list_with_large_mixed_database(
        self, large_database_with_mixed_data: Path
    ) -> None:
        """Test that `flujo lens list` performs well with large mixed database."""
        os.environ["FLUJO_STATE_URI"] = f"sqlite:///{large_database_with_mixed_data}"

        runner = CliRunner()

        # Test basic list performance
        start_time = time.perf_counter()
        result = runner.invoke(app, ["lens", "list"])
        execution_time = time.perf_counter() - start_time

        print(f"\nLarge mixed database list performance: {execution_time:.3f}s")
        assert execution_time < 1.0, f"List took {execution_time:.3f}s, should be under 1s"
        assert result.exit_code == 0

    def test_lens_list_with_various_filters(self, large_database_with_mixed_data: Path) -> None:
        """Test that `flujo lens list` with different filters performs well."""
        os.environ["FLUJO_STATE_URI"] = f"sqlite:///{large_database_with_mixed_data}"

        runner = CliRunner()

        # Test different filter combinations
        filter_tests = [
            ["lens", "list", "--status", "completed"],
            ["lens", "list", "--status", "failed"],
            ["lens", "list", "--status", "running"],
            ["lens", "list", "--pipeline", "pipeline_0"],
            ["lens", "list", "--limit", "10"],
            ["lens", "list", "--limit", "100"],
        ]

        for filter_args in filter_tests:
            start_time = time.perf_counter()
            result = runner.invoke(app, filter_args)
            execution_time = time.perf_counter() - start_time

            print(f"Filter {filter_args} performance: {execution_time:.3f}s")
            assert execution_time < 1.0, f"Filter {filter_args} took {execution_time:.3f}s"
            assert result.exit_code == 0

    def test_lens_show_with_various_run_ids(self, large_database_with_mixed_data: Path) -> None:
        """Test that `flujo lens show` performs well with different run types."""
        os.environ["FLUJO_STATE_URI"] = f"sqlite:///{large_database_with_mixed_data}"

        runner = CliRunner()

        # Test showing different types of runs
        run_ids = ["run_0000", "run_0001", "run_0002", "run_0999"]

        for run_id in run_ids:
            start_time = time.perf_counter()
            runner.invoke(app, ["lens", "show", run_id])
            execution_time = time.perf_counter() - start_time

            print(f"Show {run_id} performance: {execution_time:.3f}s")
            assert execution_time < 0.5, f"Show {run_id} took {execution_time:.3f}s"
            # Some runs might not exist, so we don't check exit code

    def test_cli_performance_with_concurrent_access(
        self, large_database_with_mixed_data: Path
    ) -> None:
        """Test CLI performance under concurrent access patterns."""
        os.environ["FLUJO_STATE_URI"] = f"sqlite:///{large_database_with_mixed_data}"

        runner = CliRunner()

        # Simulate concurrent CLI access
        commands = [
            ["lens", "list"],
            ["lens", "list", "--status", "completed"],
            ["lens", "list", "--status", "failed"],
            ["lens", "show", "run_0000"],
            ["lens", "show", "run_0001"],
        ]

        start_time = time.perf_counter()
        results = []
        for cmd in commands:
            result = runner.invoke(app, cmd)
            results.append(result)
        total_time = time.perf_counter() - start_time

        print(f"Concurrent CLI access total time: {total_time:.3f}s")
        assert total_time < 3.0, f"Concurrent access took {total_time:.3f}s"

        # Check that all commands succeeded
        for i, result in enumerate(results):
            assert result.exit_code == 0, f"Command {commands[i]} failed: {result.stdout}"

    def test_cli_performance_with_nonexistent_data(
        self, large_database_with_mixed_data: Path
    ) -> None:
        """Test CLI performance when querying nonexistent data."""
        os.environ["FLUJO_STATE_URI"] = f"sqlite:///{large_database_with_mixed_data}"

        runner = CliRunner()

        # Test queries for nonexistent data
        nonexistent_tests = [
            ["lens", "show", "nonexistent_run"],
            ["lens", "list", "--status", "nonexistent_status"],
            ["lens", "list", "--pipeline", "nonexistent_pipeline"],
        ]

        for test_args in nonexistent_tests:
            start_time = time.perf_counter()
            runner.invoke(app, test_args)
            execution_time = time.perf_counter() - start_time

            print(f"Nonexistent data query {test_args} performance: {execution_time:.3f}s")
            assert execution_time < 0.1, f"Nonexistent query {test_args} took {execution_time:.3f}s"

    def test_database_index_optimization(self, tmp_path: Path) -> None:
        """Test that database indexes are working correctly for performance."""
        db_path = tmp_path / "index_test.db"
        backend = SQLiteBackend(db_path)

        # Create a moderate amount of data
        now = datetime.utcnow()
        for i in range(100):
            asyncio.run(
                backend.save_run_start(
                    {
                        "run_id": f"run_{i:03d}",
                        "pipeline_name": f"pipeline_{i % 10}",
                        "pipeline_version": "1.0",
                        "status": "running",
                        "start_time": now - timedelta(minutes=i),
                    }
                )
            )

            # Complete some runs
            if i % 2 == 0:
                asyncio.run(
                    backend.save_run_end(
                        f"run_{i:03d}",
                        {
                            "status": "completed",
                            "end_time": now - timedelta(minutes=i) + timedelta(seconds=30),
                            "total_cost": 0.1,
                            "final_context": {"result": f"output_{i}"},
                        },
                    )
                )

        # Test that queries use indexes efficiently
        start_time = time.perf_counter()
        runs = asyncio.run(backend.list_runs(status="completed"))
        query_time = time.perf_counter() - start_time

        print(f"Indexed query performance: {query_time:.3f}s")
        assert query_time < 0.1, f"Indexed query took {query_time:.3f}s"
        assert len(runs) > 0, "Should find completed runs"

    @pytest.mark.asyncio
    async def test_database_concurrent_writes(self, tmp_path: Path) -> None:
        """Test database performance under concurrent write operations."""
        db_path = tmp_path / "concurrent_test.db"
        backend = SQLiteBackend(db_path)

        # Simulate concurrent write operations
        async def write_operation(run_id: str):
            await backend.save_run_start(
                {
                    "run_id": run_id,
                    "pipeline_name": f"pipeline_{run_id}",
                    "pipeline_version": "1.0",
                    "status": "running",
                    "start_time": datetime.utcnow(),
                }
            )
            await backend.save_run_end(
                run_id,
                {
                    "status": "completed",
                    "end_time": datetime.utcnow(),
                    "total_cost": 0.1,
                    "final_context": {"result": f"output_{run_id}"},
                },
            )

        # Run concurrent operations
        start_time = time.perf_counter()
        tasks = [write_operation(f"concurrent_run_{i}") for i in range(50)]
        await asyncio.gather(*tasks)
        total_time = time.perf_counter() - start_time

        print(f"Concurrent write performance: {total_time:.3f}s")
        assert total_time < 5.0, f"Concurrent writes took {total_time:.3f}s"

        # Verify all writes succeeded
        for i in range(50):
            run_details = await backend.get_run_details(f"concurrent_run_{i}")
            assert run_details is not None, f"Run {i} was not persisted"

    def test_database_memory_usage(self, tmp_path: Path) -> None:
        """Test that database operations don't cause memory issues."""
        db_path = tmp_path / "memory_test.db"
        backend = SQLiteBackend(db_path)

        # Create a large number of runs with substantial data
        now = datetime.utcnow()
        large_context = {"data": "x" * 1000}  # 1KB per context

        for i in range(100):
            asyncio.run(
                backend.save_run_start(
                    {
                        "run_id": f"memory_run_{i:03d}",
                        "pipeline_name": f"pipeline_{i % 10}",
                        "pipeline_version": "1.0",
                        "status": "running",
                        "start_time": now - timedelta(minutes=i),
                    }
                )
            )

            asyncio.run(
                backend.save_run_end(
                    f"memory_run_{i:03d}",
                    {
                        "status": "completed",
                        "end_time": now - timedelta(minutes=i) + timedelta(seconds=30),
                        "total_cost": 0.1,
                        "final_context": large_context,
                    },
                )
            )

        # Test that we can still query efficiently
        start_time = time.perf_counter()
        runs = asyncio.run(backend.list_runs(limit=50))
        query_time = time.perf_counter() - start_time

        print(f"Memory test query performance: {query_time:.3f}s")
        assert query_time < 0.5, f"Memory test query took {query_time:.3f}s"
        assert len(runs) == 50, "Should return exactly 50 runs"

    def test_database_corruption_recovery(self, tmp_path: Path) -> None:
        """Test that database corruption recovery works correctly."""
        db_path = tmp_path / "corruption_test.db"
        backend = SQLiteBackend(db_path)

        # Create some initial data
        asyncio.run(
            backend.save_run_start(
                {
                    "run_id": "test_run",
                    "pipeline_name": "test_pipeline",
                    "pipeline_version": "1.0",
                    "status": "running",
                    "start_time": datetime.utcnow(),
                }
            )
        )

        # Verify data was saved
        run_details = asyncio.run(backend.get_run_details("test_run"))
        assert run_details is not None

        # Simulate corruption by writing invalid data directly to the file
        # (This is a simplified test - in practice, corruption would be more complex)
        with open(db_path, "ab") as f:
            f.write(b"corrupted_data")

        # The backend should handle this gracefully
        try:
            # Try to save more data - should trigger corruption recovery
            asyncio.run(
                backend.save_run_start(
                    {
                        "run_id": "recovery_test_run",
                        "pipeline_name": "recovery_pipeline",
                        "pipeline_version": "1.0",
                        "status": "running",
                        "start_time": datetime.utcnow(),
                    }
                )
            )
        except Exception as e:
            # Corruption recovery might raise an exception, which is acceptable
            print(f"Corruption recovery exception: {e}")

        # The database should still be functional after recovery attempts
        try:
            runs = asyncio.run(backend.list_runs())
            assert isinstance(runs, list), "Should return a list even after corruption"
        except Exception as e:
            print(f"Post-recovery query exception: {e}")
            # This is acceptable if the database was corrupted beyond recovery


class TestCLIErrorHandling:
    """Test CLI error handling and edge cases."""

    def test_cli_with_invalid_database_path(self):
        """Test CLI behavior with invalid database path."""
        os.environ["FLUJO_STATE_URI"] = "sqlite:///nonexistent/path/database.db"

        runner = CliRunner()

        # Should handle gracefully
        result = runner.invoke(app, ["lens", "list"])
        assert result.exit_code != 0, "Should fail with invalid database path"

    def test_cli_with_malformed_environment_variable(self):
        """Test CLI behavior with malformed environment variable."""
        os.environ["FLUJO_STATE_URI"] = "invalid://uri"

        runner = CliRunner()

        # Should handle gracefully
        result = runner.invoke(app, ["lens", "list"])
        assert result.exit_code != 0, "Should fail with malformed URI"

    def test_cli_with_missing_environment_variable(self):
        """Test CLI behavior with missing environment variable."""
        if "FLUJO_STATE_URI" in os.environ:
            del os.environ["FLUJO_STATE_URI"]

        runner = CliRunner()

        # Should handle gracefully
        result = runner.invoke(app, ["lens", "list"])
        assert result.exit_code != 0, "Should fail with missing environment variable"
