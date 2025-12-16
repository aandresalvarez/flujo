import asyncio
import pytest
import os
from datetime import datetime, timezone
from pathlib import Path

from typer.testing import CliRunner
from flujo.cli.main import app
from flujo.state.backends.sqlite import SQLiteBackend

# Mark as very slow to exclude from fast suites
pytestmark = pytest.mark.veryslow

runner = CliRunner()


@pytest.fixture(autouse=True)
def cleanup_sqlite_backends_sync(monkeypatch):
    """Autouse fixture to ensure all SQLiteBackend instances are properly closed.

    This prevents resource leaks that cause 361-second timeouts.
    Uses SQLiteBackend.close_sync() which properly handles event loop contexts.
    """
    backends = []
    original_init = SQLiteBackend.__init__

    def tracking_init(self, *args, **kwargs):
        backends.append(self)
        return original_init(self, *args, **kwargs)

    monkeypatch.setattr(SQLiteBackend, "__init__", tracking_init)
    yield
    # Clean up all backends created during the test
    for backend in backends:
        try:
            backend.close_sync()
        except Exception:
            pass


def create_run_with_steps(backend, run_id, steps=None, status="completed", final_context=None):
    if steps is None:
        steps = []
    if final_context is None:
        final_context = {}
    asyncio.run(
        backend.save_run_start(
            {
                "run_id": run_id,
                "pipeline_id": f"test-pid-{run_id}",
                "pipeline_name": f"pipeline_{run_id}",
                "pipeline_version": "1.0",
                "status": "running",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        )
    )
    for step in steps:
        asyncio.run(backend.save_step_result(step))
    asyncio.run(
        backend.save_run_end(
            run_id,
            {
                "status": status,
                "end_time": datetime.now(timezone.utc),
                "total_cost": 0.01,
                "final_context": final_context,
            },
        )
    )


def test_lens_commands(tmp_path: Path, monkeypatch) -> None:
    """Test basic lens CLI functionality."""
    db_path = tmp_path / "ops.db"
    backend = SQLiteBackend(db_path)

    # Create test data
    asyncio.run(
        backend.save_run_start(
            {
                "run_id": "r1",
                "pipeline_id": "test-pid-1",
                "pipeline_name": "p",
                "pipeline_version": "v",
                "status": "running",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        )
    )
    asyncio.run(
        backend.save_run_end(
            "r1",
            {
                "status": "completed",
                "end_time": datetime.now(timezone.utc),
                "total_cost": 0.0,
                "final_context": {},
            },
        )
    )

    # Test list command
    os.environ["FLUJO_STATE_URI"] = f"sqlite:////{db_path}"
    result = runner.invoke(app, ["lens", "list"])
    assert result.exit_code == 0
    assert "r1" in result.stdout

    # Test show command
    result = runner.invoke(app, ["lens", "show", "r1"])
    assert result.exit_code == 0
    assert "r1" in result.stdout


def test_lens_commands_with_filters(tmp_path: Path) -> None:
    """Test lens CLI with filtering options."""
    db_path = tmp_path / "ops.db"
    backend = SQLiteBackend(db_path)

    # Create multiple runs with different statuses
    for i in range(5):
        run_id = f"run_{i}"
        status = "completed" if i % 2 == 0 else "failed"

        asyncio.run(
            backend.save_run_start(
                {
                    "run_id": run_id,
                    "pipeline_id": f"test-pid-{run_id}",
                    "pipeline_name": f"pipeline_{i}",
                    "pipeline_version": "1.0",
                    "status": "running",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            )
        )
        asyncio.run(
            backend.save_run_end(
                run_id,
                {
                    "status": status,
                    "end_time": datetime.now(timezone.utc),
                    "total_cost": 0.1,
                    "final_context": {"result": f"output_{i}"},
                },
            )
        )

    os.environ["FLUJO_STATE_URI"] = f"sqlite:////{db_path}"

    # Test list with status filter
    result = runner.invoke(app, ["lens", "list", "--status", "completed"])
    assert result.exit_code == 0
    assert "completed" in result.stdout

    # Test list with pipeline filter
    result = runner.invoke(app, ["lens", "list", "--pipeline", "pipeline_0"])
    assert result.exit_code == 0
    assert "pipeline_0" in result.stdout


def test_lens_show_detailed_run(tmp_path: Path) -> None:
    """Test lens show command with detailed run data."""
    db_path = tmp_path / "ops.db"
    backend = SQLiteBackend(db_path)

    # Create a run with step data
    run_id = "detailed_run"

    # Save run start
    asyncio.run(
        backend.save_run_start(
            {
                "run_id": run_id,
                "pipeline_id": f"test-pid-{run_id}",
                "pipeline_name": "test_pipeline",
                "pipeline_version": "1.0",
                "status": "running",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        )
    )

    # Save step results
    for step_idx in range(3):
        asyncio.run(
            backend.save_step_result(
                {
                    "step_run_id": f"{run_id}:{step_idx}",
                    "run_id": run_id,
                    "step_name": f"step_{step_idx}",
                    "step_index": step_idx,
                    "status": "completed",
                    "start_time": datetime.now(timezone.utc),
                    "end_time": datetime.now(timezone.utc),
                    "duration_ms": 1000,
                    "cost": 0.01,
                    "tokens": 50,
                    "input": f"input_{step_idx}",
                    "output": f"output_{step_idx}",
                    "error": None,
                }
            )
        )

    # Save run end
    asyncio.run(
        backend.save_run_end(
            run_id,
            {
                "status": "completed",
                "end_time": datetime.now(timezone.utc),
                "total_cost": 0.03,
                "final_context": {"final_result": "success"},
            },
        )
    )

    os.environ["FLUJO_STATE_URI"] = f"sqlite:////{db_path}"

    # Test show command
    result = runner.invoke(app, ["lens", "show", run_id])
    assert result.exit_code == 0
    assert run_id in result.stdout
    assert "completed" in result.stdout
    assert "step_0" in result.stdout
    assert "step_1" in result.stdout
    assert "step_2" in result.stdout


def test_lens_show_nonexistent_run(tmp_path: Path) -> None:
    """Test lens show command with nonexistent run."""
    db_path = tmp_path / "ops.db"
    os.environ["FLUJO_STATE_URI"] = f"sqlite:////{db_path}"

    # Test show with nonexistent run
    result = runner.invoke(app, ["lens", "show", "nonexistent_run"])
    assert result.exit_code != 0  # Should fail
    # Check that the error message is in the output (either stdout or stderr)
    assert "not found" in result.output.lower() or "error" in result.output.lower()


def test_lens_commands_with_empty_database(tmp_path: Path) -> None:
    """Test lens commands with empty database."""
    db_path = tmp_path / "empty_ops.db"
    os.environ["FLUJO_STATE_URI"] = f"sqlite:////{db_path}"

    # Test list with empty database
    result = runner.invoke(app, ["lens", "list"])
    assert result.exit_code == 0
    # Should not crash, may show empty table or message

    # Test show with empty database
    result = runner.invoke(app, ["lens", "show", "any_run"])
    assert result.exit_code != 0  # Should fail


def test_lens_commands_with_failed_run(tmp_path: Path) -> None:
    """Test lens commands with failed run data."""
    db_path = tmp_path / "failed_ops.db"
    backend = SQLiteBackend(db_path)

    run_id = "failed_run"

    # Create a failed run
    asyncio.run(
        backend.save_run_start(
            {
                "run_id": run_id,
                "pipeline_id": f"test-pid-{run_id}",
                "pipeline_name": "failing_pipeline",
                "pipeline_version": "1.0",
                "status": "running",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        )
    )

    # Add a failed step
    asyncio.run(
        backend.save_step_result(
            {
                "step_run_id": f"{run_id}:0",
                "run_id": run_id,
                "step_name": "failing_step",
                "step_index": 0,
                "status": "failed",
                "start_time": datetime.now(timezone.utc),
                "end_time": datetime.now(timezone.utc),
                "duration_ms": 500,
                "cost": 0.01,
                "tokens": 25,
                "input": "test_input",
                "output": None,
                "error": "Step failed due to error",
            }
        )
    )

    # Save run end as failed
    asyncio.run(
        backend.save_run_end(
            run_id,
            {
                "status": "failed",
                "end_time": datetime.now(timezone.utc),
                "total_cost": 0.01,
                "final_context": {"error": "Pipeline failed"},
            },
        )
    )

    os.environ["FLUJO_STATE_URI"] = f"sqlite:////{db_path}"

    # Test list shows failed run
    result = runner.invoke(app, ["lens", "list"])
    assert result.exit_code == 0
    assert run_id in result.stdout
    assert "failed" in result.stdout

    # Test show shows failed run details
    result = runner.invoke(app, ["lens", "show", run_id])
    assert result.exit_code == 0
    assert run_id in result.stdout
    assert "failed" in result.stdout
    assert "failing_step" in result.stdout


def test_lens_commands_with_environment_configuration(tmp_path: Path) -> None:
    """Test lens commands with different environment configurations."""
    import os

    db_path = tmp_path / "env_ops.db"
    backend = SQLiteBackend(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(
        backend.save_run_start(
            {
                "run_id": "env_test_run",
                "pipeline_id": "test-pid-env",
                "pipeline_name": "env_test",
                "pipeline_version": "1.0",
                "status": "running",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )
    )
    runner = CliRunner()
    orig_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        # Standards-compliant absolute path: sqlite:////abs/path.db (4 slashes)
        abs_uri = f"sqlite:////{db_path.as_posix()}"
        result = runner.invoke(app, ["lens", "list"], env={"FLUJO_STATE_URI": abs_uri})
        print(f"STDOUT (sqlite:////{{db_path}}): {result.stdout}")
        assert result.exit_code == 0
        assert "env_test_run" in result.stdout
        # Relative path: sqlite:///env_ops.db (should resolve to cwd/env_ops.db)
        rel_uri = "sqlite:///env_ops.db"
        result = runner.invoke(app, ["lens", "list"], env={"FLUJO_STATE_URI": rel_uri})
        print(f"STDOUT (sqlite:///env_ops.db): {result.stdout}")
        assert result.exit_code == 0
        assert "env_test_run" in result.stdout
    finally:
        os.chdir(orig_cwd)


def test_lens_cli_relative_path_resolution(tmp_path: Path, monkeypatch):
    """Regression test: CLI resolves relative sqlite paths in state_uri correctly regardless of CWD."""
    test_dir = tmp_path / "manual_testing"
    test_dir.mkdir()
    db_path = test_dir / "ops.db"
    flujo_toml = test_dir / "flujo.toml"
    flujo_toml.write_text('state_uri = "sqlite:///./ops.db"\n')
    backend = SQLiteBackend(db_path)
    run_id = "relpath_run"
    create_run_with_steps(backend, run_id)
    monkeypatch.chdir(test_dir)
    env = {
        **os.environ,
        "FLUJO_CONFIG_PATH": str(flujo_toml),
        "FLUJO_STATE_URI": f"sqlite:////{db_path}",
    }
    result = runner.invoke(app, ["lens", "list"], env=env)
    assert result.exit_code == 0, result.stdout + result.stderr
    assert run_id in result.stdout


def test_lens_show_with_verbose_options(tmp_path: Path) -> None:
    db_path = tmp_path / "verbose_ops.db"
    backend = SQLiteBackend(db_path)
    run_id = "verbose_run"
    step = {
        "run_id": run_id,
        "step_name": "test_step",
        "step_index": 0,
        "status": "completed",
        "output": {"test_output": "result"},
        "cost_usd": 0.01,
        "token_counts": 50,
        "execution_time_ms": 1000,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    create_run_with_steps(backend, run_id, steps=[step])
    os.environ["FLUJO_STATE_URI"] = f"sqlite:////{db_path}"
    # Test show-output option
    result = runner.invoke(app, ["lens", "show", run_id, "--show-output"])
    assert result.exit_code == 0
    assert "test_output" in result.stdout
    # Test verbose option (should show output)
    result = runner.invoke(app, ["lens", "show", run_id, "--verbose"])
    assert result.exit_code == 0
    assert "test_output" in result.stdout


def test_lens_trace_command(tmp_path: Path) -> None:
    """Test lens trace command with trace data."""
    db_path = tmp_path / "trace_ops.db"
    backend = SQLiteBackend(db_path)

    run_id = "trace_run"

    # Create a run with trace data
    asyncio.run(
        backend.save_run_start(
            {
                "run_id": run_id,
                "pipeline_id": f"test-pid-{run_id}",
                "pipeline_name": "trace_pipeline",
                "pipeline_version": "1.0",
                "status": "running",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        )
    )

    # Save trace data
    trace_data = {
        "run_id": run_id,
        "name": "trace_pipeline",
        "status": "completed",
        "start_time": datetime.now(timezone.utc).timestamp(),
        "end_time": datetime.now(timezone.utc).timestamp(),
        "children": [
            {
                "name": "step1",
                "status": "completed",
                "start_time": datetime.now(timezone.utc).timestamp(),
                "end_time": datetime.now(timezone.utc).timestamp(),
                "attributes": {"iteration": 1},
            }
        ],
    }

    # Note: This assumes the backend has a save_trace method
    # If not available, we'll test the error handling
    try:
        if hasattr(backend, "save_trace"):
            asyncio.run(backend.save_trace(run_id, trace_data))
    except NotImplementedError:
        pass

    asyncio.run(
        backend.save_run_end(
            run_id,
            {
                "status": "completed",
                "end_time": datetime.now(timezone.utc),
                "total_cost": 0.0,
                "final_context": {},
            },
        )
    )

    os.environ["FLUJO_STATE_URI"] = f"sqlite:////{db_path}"

    # Test trace command
    result = runner.invoke(app, ["lens", "trace", run_id])
    # Should either succeed or show appropriate error message
    assert result.exit_code in [0, 1]


def test_lens_trace_command_regression_timestamps(tmp_path: Path) -> None:
    """Regression: test lens trace command with various timestamp formats for duration parsing.

    Fixed: Uses autouse fixture for proper cleanup. Backend is properly tracked and closed.
    Previous issue was complex manual cleanup causing pytest to hang.
    """
    backend = SQLiteBackend(tmp_path / "trace_ops_regression.db")

    def create_run(run_id, pipeline_name="pipeline"):
        asyncio.run(
            backend.save_run_start(
                {
                    "run_id": run_id,
                    "pipeline_id": f"test-pid-{run_id}",
                    "pipeline_name": pipeline_name,
                    "pipeline_version": "1.0",
                    "status": "running",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            )
        )

    def run_and_check(run_id, trace_data, expected_in_output):
        create_run(run_id, pipeline_name=trace_data.get("name", "pipeline"))
        if hasattr(backend, "save_trace"):
            asyncio.run(backend.save_trace(run_id, trace_data))
        os.environ["FLUJO_STATE_URI"] = f"sqlite:////{tmp_path}/trace_ops_regression.db"
        result = runner.invoke(app, ["lens", "trace", run_id])
        assert result.exit_code in [0, 1]
        for expected in expected_in_output:
            assert expected in result.stdout, (
                f"Expected '{expected}' in output. Got: {result.stdout}"
            )

    # 1. Normal float timestamps
    run_id1 = "trace_regression_1"
    trace_data1 = {
        "span_id": "root1",
        "run_id": run_id1,
        "name": "pipeline",
        "status": "completed",
        "start_time": 1000.0,
        "end_time": 1005.5,
        "children": [],
    }
    run_and_check(run_id1, trace_data1, ["(duration: 5.50s)"])

    # 2. Negative timestamps
    run_id2 = "trace_regression_2"
    trace_data2 = dict(trace_data1)
    trace_data2.update(
        {"span_id": "root2", "run_id": run_id2, "start_time": -100.0, "end_time": -90.0}
    )
    run_and_check(run_id2, trace_data2, ["(duration: 10.00s)"])

    # 3. Scientific notation
    run_id3 = "trace_regression_3"
    trace_data3 = dict(trace_data1)
    trace_data3.update(
        {"span_id": "root3", "run_id": run_id3, "start_time": 1e3, "end_time": 1.5e3}
    )
    run_and_check(run_id3, trace_data3, ["(duration: 500.00s)"])

    # 4. Invalid (non-numeric) timestamps
    run_id4 = "trace_regression_4"
    trace_data4 = dict(trace_data1)
    trace_data4.update(
        {
            "span_id": "root4",
            "run_id": run_id4,
            "start_time": "not_a_number",
            "end_time": "also_bad",
        }
    )
    create_run(run_id4, pipeline_name=trace_data4.get("name", "pipeline"))
    if hasattr(backend, "save_trace"):
        asyncio.run(backend.save_trace(run_id4, trace_data4))
    os.environ["FLUJO_STATE_URI"] = f"sqlite:////{tmp_path}/trace_ops_regression.db"
    result = runner.invoke(app, ["lens", "trace", run_id4])
    assert result.exit_code in [0, 1]
    assert "(duration:" not in result.stdout

    # Autouse fixture cleanup_sqlite_backends_sync will handle backend cleanup


def test_lens_commands_error_handling(tmp_path: Path) -> None:
    """Test lens commands with various error conditions."""
    db_path = tmp_path / "error_ops.db"
    os.environ["FLUJO_STATE_URI"] = f"sqlite:////{db_path}"

    # Test with invalid run_id
    result = runner.invoke(app, ["lens", "show", "invalid-run-id"])
    assert result.exit_code != 0

    # Test trace with invalid run_id
    result = runner.invoke(app, ["lens", "trace", "invalid-run-id"])
    assert result.exit_code != 0

    # Test spans with invalid run_id
    result = runner.invoke(app, ["lens", "spans", "invalid-run-id"])
    assert result.exit_code == 0  # Should not fail, just show no spans

    # Test stats command
    result = runner.invoke(app, ["lens", "stats"])
    assert result.exit_code == 0  # Should not fail, just show empty stats
