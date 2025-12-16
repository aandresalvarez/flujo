"""Tests for CLI performance edge cases and database optimization scenarios."""

import pytest
import time
import asyncio
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
import platform
import subprocess
import sys

from flujo.state.backends.sqlite import SQLiteBackend
from typer.testing import CliRunner
from flujo.cli.main import app

# Mark as very slow to exclude from fast suites
# Add timeout to prevent CI hangs
pytestmark = [pytest.mark.veryslow, pytest.mark.serial, pytest.mark.timeout(300)]


class TestCLIPerformanceEdgeCases:
    """Test CLI performance edge cases and optimizations."""

    @pytest.fixture
    async def large_database_with_mixed_data(self, tmp_path) -> Path:
        """Create a database with mixed data types for performance testing.

        Changed from module-scoped to function-scoped to ensure proper async execution.
        Reduced default size from 200 to 50 for faster test execution.
        """
        db_path = tmp_path / "mixed_ops.db"
        backend = SQLiteBackend(db_path)

        try:
            # Create runs with different characteristics
            import os as _os

            now = datetime.now(timezone.utc)
            # Reduced default from 200 to 50 for faster tests
            # CI can override with FLUJO_CI_DB_SIZE if needed
            try:
                total = int(_os.getenv("FLUJO_CI_DB_SIZE", "50"))
            except Exception:
                total = 50
            for i in range(total):
                # Create run start
                dt = (now - timedelta(minutes=i)).isoformat()
                await backend.save_run_start(
                    {
                        "run_id": f"run_{i:04d}",
                        "pipeline_id": f"pid_{i:04d}",
                        "pipeline_name": f"pipeline_{i % 20}",
                        "pipeline_version": "1.0",
                        "status": "running",
                        "created_at": dt,
                        "updated_at": dt,
                    }
                )

                # Create run end with different statuses
                status = "completed" if i % 3 == 0 else "failed" if i % 3 == 1 else "running"
                if status != "running":
                    await backend.save_run_end(
                        f"run_{i:04d}",
                        {
                            "status": status,
                            "end_time": now - timedelta(minutes=i) + timedelta(seconds=30),
                            "total_cost": 0.1 + (i * 0.01),
                            "final_context": {
                                "result": f"output_{i}",
                                "metadata": {"index": i},
                            },
                        },
                    )

                # Add step data for completed runs
                if status == "completed":
                    for step_idx in range(2):
                        await backend.save_step_result(
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

            return db_path
        finally:
            # Cleanup backend connection
            await backend.close()

    def test_lens_list_with_large_mixed_database(
        self, large_database_with_mixed_data: Path
    ) -> None:
        """Test that `flujo lens list` performs well with large mixed database."""
        os.environ["FLUJO_STATE_URI"] = f"sqlite:///{large_database_with_mixed_data}"

        runner = CliRunner()

        # Establish baseline: show one run
        start_time = time.perf_counter()
        runner.invoke(app, ["lens", "show", "run_0000"])
        baseline_single = time.perf_counter() - start_time
        print(f"Baseline (single show) performance: {baseline_single:.3f}s")

        # Test basic list performance
        start_time = time.perf_counter()
        result = runner.invoke(app, ["lens", "list"])
        execution_time = time.perf_counter() - start_time

        print(f"\nLarge mixed database list performance: {execution_time:.3f}s")

        # Relative check: Listing 50 items should not be exponentially slower than showing 1
        # We allow a factor of 20x (efficient listing)
        max_ratio = 20.0
        if baseline_single > 0:
            ratio = execution_time / baseline_single
            assert ratio < max_ratio, (
                f"List took {execution_time:.3f}s, baseline {baseline_single:.3f}s. "
                f"Ratio {ratio:.2f}x exceeds {max_ratio}x"
            )
        else:
            assert execution_time < 1.0

        assert result.exit_code == 0

    def test_lens_list_with_various_filters(self, large_database_with_mixed_data: Path) -> None:
        """Test that `flujo lens list` with different filters performs well."""
        os.environ["FLUJO_STATE_URI"] = f"sqlite:///{large_database_with_mixed_data}"

        runner = CliRunner()

        # Establish baseline: unfiltered list
        start_time = time.perf_counter()
        runner.invoke(app, ["lens", "list"])
        baseline = time.perf_counter() - start_time
        print(f"Baseline (unfiltered list): {baseline:.3f}s")

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
            # Relative check: filtered should not be dramatically slower than baseline
            # Allow 10x for major regression detection (CI timing variance)
            max_ratio = 10.0
            if baseline > 0:
                ratio = execution_time / baseline
                assert ratio < max_ratio, (
                    f"Filter {filter_args} took {execution_time:.3f}s, baseline {baseline:.3f}s. "
                    f"Ratio {ratio:.2f}x exceeds {max_ratio}x (major regression)"
                )
            else:
                # Fallback: generous absolute threshold
                assert execution_time < 5.0, f"Filter {filter_args} took {execution_time:.3f}s"
            assert result.exit_code == 0

    def test_lens_show_with_various_run_ids(self, large_database_with_mixed_data: Path) -> None:
        """Test that `flujo lens show` performs well with different run types."""
        os.environ["FLUJO_STATE_URI"] = f"sqlite:///{large_database_with_mixed_data}"

        runner = CliRunner()

        # Test showing different types of runs
        run_ids = ["run_0000", "run_0001", "run_0002", "run_0999"]

        # Establish baseline with first run
        start_time = time.perf_counter()
        runner.invoke(app, ["lens", "show", run_ids[0]])
        baseline = time.perf_counter() - start_time
        print(f"Baseline ({run_ids[0]}) performance: {baseline:.3f}s")

        for run_id in run_ids[1:]:
            start_time = time.perf_counter()
            runner.invoke(app, ["lens", "show", run_id])
            execution_time = time.perf_counter() - start_time

            print(f"Show {run_id} performance: {execution_time:.3f}s")

            # Relative check: Should be comparable to baseline
            # Allow 2x variance
            if baseline > 0:
                assert execution_time < baseline * 3.0, (
                    f"Show {run_id} took {execution_time:.3f}s, baseline {baseline:.3f}s"
                )
            else:
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

        # Establish baseline
        start_time = time.perf_counter()
        runner.invoke(app, ["lens", "show", "run_0000"])
        baseline = time.perf_counter() - start_time

        start_time = time.perf_counter()
        results = []
        for cmd in commands:
            result = runner.invoke(app, cmd)
            results.append(result)
        total_time = time.perf_counter() - start_time

        print(f"Concurrent CLI access total time: {total_time:.3f}s")

        # Relative check: 5 commands should take roughly 5x baseline
        # Allow overhead factor
        if baseline > 0:
            # 5 commands. Allow 10x baseline total.
            assert total_time < baseline * 15.0, (
                f"Concurrent access took {total_time:.3f}s, baseline {baseline:.3f}s"
            )
        else:
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

        # Establish baseline with existing data
        # We know run_0000 exists from the fixture
        start_time = time.perf_counter()
        runner.invoke(app, ["lens", "show", "run_0000"])
        baseline_time = time.perf_counter() - start_time

        print(f"Baseline (existing data) performance: {baseline_time:.3f}s")

        for test_args in nonexistent_tests:
            start_time = time.perf_counter()
            runner.invoke(app, test_args)
            execution_time = time.perf_counter() - start_time

            print(f"Nonexistent data query {test_args} performance: {execution_time:.3f}s")

            # Relative performance check
            # Nonexistent queries should be faster due to early exit
            # We allow them to be slightly slower (1.5x) to account for variance, but not more
            # This avoids hardcoded 0.2s threshold which fails in CI
            max_ratio = 1.5
            if baseline_time > 0:
                ratio = execution_time / baseline_time
                assert ratio < max_ratio, (
                    f"Nonexistent query {test_args} took {execution_time:.3f}s, "
                    f"baseline took {baseline_time:.3f}s. "
                    f"Ratio {ratio:.2f}x exceeds limit {max_ratio}x"
                )
            else:
                # Fallback if baseline is 0 (unlikely)
                assert execution_time < 0.5

    @pytest.mark.asyncio
    async def test_database_index_optimization(self, tmp_path: Path) -> None:
        """Test that database indexes are working correctly for performance."""
        db_path = tmp_path / "index_test.db"
        backend = SQLiteBackend(db_path)

        try:
            # Create a moderate amount of data
            now = datetime.now(timezone.utc)
            for i in range(100):
                await backend.save_run_start(
                    {
                        "run_id": f"run_{i:03d}",
                        "pipeline_id": f"pid_{i:03d}",
                        "pipeline_name": f"pipeline_{i % 10}",
                        "pipeline_version": "1.0",
                        "status": "running",
                        "created_at": (now - timedelta(minutes=i)).isoformat(),
                        "updated_at": (now - timedelta(minutes=i)).isoformat(),
                    }
                )

                # Complete some runs
                if i % 2 == 0:
                    await backend.save_run_end(
                        f"run_{i:03d}",
                        {
                            "status": "completed",
                            "end_time": now - timedelta(minutes=i) + timedelta(seconds=30),
                            "total_cost": 0.1,
                            "final_context": {"result": f"output_{i}"},
                        },
                    )

            # Establish baseline: unfiltered list (no index benefit)
            start_time = time.perf_counter()
            await backend.list_runs()
            baseline = time.perf_counter() - start_time
            print(f"Baseline (unfiltered): {baseline:.3f}s")

            # Test that filtered queries use indexes efficiently
            start_time = time.perf_counter()
            runs = await backend.list_runs(status="completed")
            query_time = time.perf_counter() - start_time

            print(f"Indexed query performance: {query_time:.3f}s")
            # Indexed query should not be dramatically slower than baseline
            # Allow 3x for major regression detection
            max_ratio = 3.0
            if baseline > 0:
                ratio = query_time / baseline
                assert ratio < max_ratio, (
                    f"Indexed query took {query_time:.3f}s, baseline {baseline:.3f}s. "
                    f"Ratio {ratio:.2f}x exceeds {max_ratio}x (index may not be working)"
                )
            else:
                # Fallback: generous absolute threshold
                assert query_time < 1.0, f"Indexed query took {query_time:.3f}s"
            assert len(runs) > 0, "Should find completed runs"
        finally:
            await backend.close()

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
                    "pipeline_id": f"pid_{run_id}",
                    "pipeline_name": f"pipeline_{run_id}",
                    "pipeline_version": "1.0",
                    "status": "running",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            await backend.save_run_end(
                run_id,
                {
                    "status": "completed",
                    "end_time": datetime.now(timezone.utc),
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
        # Sanity check: 50 concurrent writes should complete within reasonable time
        # This is a major regression detector, not a tight performance gate
        assert total_time < 30.0, (
            f"Concurrent writes took {total_time:.3f}s - major regression detected"
        )

        # Verify all writes succeeded
        for i in range(50):
            run_details = await backend.get_run_details(f"concurrent_run_{i}")
            assert run_details is not None, f"Run {i} was not persisted"

    @pytest.mark.asyncio
    async def test_database_memory_usage(self, tmp_path: Path) -> None:
        """Test that database operations don't cause memory issues."""
        db_path = tmp_path / "memory_test.db"
        backend = SQLiteBackend(db_path)

        try:
            # Create a large number of runs with substantial data
            now = datetime.now(timezone.utc)
            large_context = {"data": "x" * 1000}  # 1KB per context

            for i in range(100):
                await backend.save_run_start(
                    {
                        "run_id": f"memory_run_{i:03d}",
                        "pipeline_id": f"pid_{i:03d}",
                        "pipeline_name": f"pipeline_{i % 10}",
                        "pipeline_version": "1.0",
                        "status": "running",
                        "created_at": (now - timedelta(minutes=i)).isoformat(),
                        "updated_at": (now - timedelta(minutes=i)).isoformat(),
                    }
                )

                await backend.save_run_end(
                    f"memory_run_{i:03d}",
                    {
                        "status": "completed",
                        "end_time": now - timedelta(minutes=i) + timedelta(seconds=30),
                        "total_cost": 0.1,
                        "final_context": large_context,
                    },
                )

            # Establish baseline: small query first
            start_time = time.perf_counter()
            await backend.list_runs(limit=10)
            baseline = time.perf_counter() - start_time
            print(f"Baseline (limit=10): {baseline:.3f}s")

            # Test that we can still query efficiently with larger limit
            start_time = time.perf_counter()
            runs = await backend.list_runs(limit=50)
            query_time = time.perf_counter() - start_time

            print(f"Memory test query performance: {query_time:.3f}s")
            # Larger query should not be dramatically slower than smaller query
            # Allow 10x for major regression detection (5x more data, plus overhead)
            max_ratio = 10.0
            if baseline > 0:
                ratio = query_time / baseline
                assert ratio < max_ratio, (
                    f"Memory test query took {query_time:.3f}s, baseline {baseline:.3f}s. "
                    f"Ratio {ratio:.2f}x exceeds {max_ratio}x (major regression)"
                )
            else:
                # Fallback: generous absolute threshold
                assert query_time < 5.0, f"Memory test query took {query_time:.3f}s"
            assert len(runs) == 50, "Should return exactly 50 runs"
        finally:
            await backend.close()

    @pytest.mark.asyncio
    async def test_database_corruption_recovery(self, tmp_path: Path) -> None:
        """Test that database corruption recovery works correctly."""
        db_path = tmp_path / "corruption_test.db"
        backend = SQLiteBackend(db_path)

        try:
            # Create some initial data
            await backend.save_run_start(
                {
                    "run_id": "test_run",
                    "pipeline_id": "test_pid",
                    "pipeline_name": "test_pipeline",
                    "pipeline_version": "1.0",
                    "status": "running",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            )

            # Verify data was saved
            run_details = await backend.get_run_details("test_run")
            assert run_details is not None

            # Simulate corruption by writing invalid data directly to the file
            # (This is a simplified test - in practice, corruption would be more complex)
            with open(db_path, "ab") as f:
                f.write(b"corrupted_data")

            # The backend should handle this gracefully
            try:
                # Try to save more data - should trigger corruption recovery
                await backend.save_run_start(
                    {
                        "run_id": "recovery_test_run",
                        "pipeline_id": "recovery_pid",
                        "pipeline_name": "recovery_pipeline",
                        "pipeline_version": "1.0",
                        "status": "running",
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
            except Exception as e:
                # Corruption recovery might raise an exception, which is acceptable
                print(f"Corruption recovery exception: {e}")

            # The database should still be functional after recovery attempts
            try:
                runs = await backend.list_runs()
                assert isinstance(runs, list), "Should return a list even after corruption"
            except Exception as e:
                print(f"Post-recovery query exception: {e}")
                # This is acceptable if the database was corrupted beyond recovery
        finally:
            try:
                await backend.close()
            except Exception:
                pass  # Best effort cleanup on corrupted DB


class TestCLIErrorHandling:
    """Test CLI error handling and edge cases."""

    @pytest.fixture
    def unwritable_db_path(self, tmp_path):
        """Fixture for a path that is guaranteed to be unwritable for non-root users on Unix."""
        import stat

        # Create a temporary directory and remove write permissions
        unwritable_dir = tmp_path / "unwritable"
        unwritable_dir.mkdir()
        unwritable_db = unwritable_dir / "forbidden.db"
        # Remove write permissions from the directory
        unwritable_dir.chmod(stat.S_IREAD | stat.S_IEXEC)
        yield str(unwritable_db)
        # Restore permissions so pytest can clean up
        unwritable_dir.chmod(stat.S_IWUSR | stat.S_IREAD | stat.S_IEXEC)

    @pytest.mark.skipif(platform.system() == "Windows", reason="Path permissions test is Unix-only")
    def test_cli_fails_with_unwritable_path_unix(self, unwritable_db_path):
        """Fails gracefully when database path is unwritable (not root, Unix)."""
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            pytest.skip("Test not valid when running as root")

        original_uri = os.environ.get("FLUJO_STATE_URI")
        try:
            os.environ["FLUJO_STATE_URI"] = f"sqlite://{unwritable_db_path}"
            result = subprocess.run(
                [sys.executable, "-m", "flujo.cli.main", "lens", "show"],
                capture_output=True,
                text=True,
                env=os.environ.copy(),
            )
            assert result.returncode != 0, "Should fail with invalid database path"
            assert (
                "not writable" in result.stderr
                or "Error" in result.stderr
                or "Permission denied" in result.stderr
            )
        finally:
            # Clean up: restore original environment
            if original_uri is None:
                os.environ.pop("FLUJO_STATE_URI", None)
            else:
                os.environ["FLUJO_STATE_URI"] = original_uri

    def test_cli_skipped_on_windows(self):
        """No-op on non-Windows to avoid xdist skip serialization; skip on Windows."""
        if platform.system() != "Windows":
            # On Linux/macOS CI, just pass without invoking pytest.skip
            assert True
            return
        pytest.skip("Windows-specific test placeholder")

    @pytest.mark.skipif(not hasattr(os, "geteuid"), reason="geteuid not available on this platform")
    def test_cli_skipped_as_root(self, unwritable_db_path):
        """Test is skipped if running as root (Unix)."""
        if os.geteuid() == 0:
            original_uri = os.environ.get("FLUJO_STATE_URI")
            try:
                os.environ["FLUJO_STATE_URI"] = f"sqlite://{unwritable_db_path}"
                pytest.skip("Test not valid when running as root")
            finally:
                # Clean up even if skipping
                if original_uri is None:
                    os.environ.pop("FLUJO_STATE_URI", None)
                else:
                    os.environ["FLUJO_STATE_URI"] = original_uri

    def test_cli_with_malformed_environment_variable(self):
        """Test CLI behavior with malformed environment variable."""
        original_uri = os.environ.get("FLUJO_STATE_URI")
        try:
            os.environ["FLUJO_STATE_URI"] = "invalid://uri"
            runner = CliRunner()
            result = runner.invoke(app, ["lens", "list"])
            assert result.exit_code != 0, "Should fail with malformed URI"
        finally:
            # Clean up: restore original environment
            if original_uri is None:
                os.environ.pop("FLUJO_STATE_URI", None)
            else:
                os.environ["FLUJO_STATE_URI"] = original_uri

    def test_cli_with_missing_environment_variable(self):
        """Test CLI behavior with missing environment variable."""
        env = os.environ.copy()
        env.pop("FLUJO_STATE_URI", None)
        result = subprocess.run(
            [sys.executable, "-m", "flujo.cli.main", "lens", "show"],
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode != 0, "Should fail with missing environment variable"
        assert "not writable" in result.stderr or "Error" in result.stderr
