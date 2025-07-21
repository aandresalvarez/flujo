import os
from pathlib import Path

from typer.testing import CliRunner

from flujo.cli.main import app
from datetime import datetime
from flujo.state.backends.sqlite import SQLiteBackend
import asyncio

runner = CliRunner()


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
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
            }
        )
    )
    asyncio.run(
        backend.save_run_end(
            "r1",
            {
                "status": "completed",
                "end_time": datetime.utcnow(),
                "total_cost": 0.0,
                "final_context": {},
            },
        )
    )

    # Test list command
    os.environ["FLUJO_STATE_URI"] = f"sqlite:///{db_path}"
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
                    "created_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat(),
                }
            )
        )
        asyncio.run(
            backend.save_run_end(
                run_id,
                {
                    "status": status,
                    "end_time": datetime.utcnow(),
                    "total_cost": 0.1,
                    "final_context": {"result": f"output_{i}"},
                },
            )
        )

    os.environ["FLUJO_STATE_URI"] = f"sqlite:///{db_path}"

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
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
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
                    "start_time": datetime.utcnow(),
                    "end_time": datetime.utcnow(),
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
                "end_time": datetime.utcnow(),
                "total_cost": 0.03,
                "final_context": {"final_result": "success"},
            },
        )
    )

    os.environ["FLUJO_STATE_URI"] = f"sqlite:///{db_path}"

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
    os.environ["FLUJO_STATE_URI"] = f"sqlite:///{db_path}"

    # Test show with nonexistent run
    result = runner.invoke(app, ["lens", "show", "nonexistent_run"])
    assert result.exit_code != 0  # Should fail
    assert "not found" in result.stderr.lower() or "error" in result.stderr.lower()


def test_lens_commands_with_empty_database(tmp_path: Path) -> None:
    """Test lens commands with empty database."""
    db_path = tmp_path / "empty_ops.db"
    os.environ["FLUJO_STATE_URI"] = f"sqlite:///{db_path}"

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
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
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
                "start_time": datetime.utcnow(),
                "end_time": datetime.utcnow(),
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
                "end_time": datetime.utcnow(),
                "total_cost": 0.01,
                "final_context": {"error": "Pipeline failed"},
            },
        )
    )

    os.environ["FLUJO_STATE_URI"] = f"sqlite:///{db_path}"

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
    db_path = tmp_path / "env_ops.db"
    backend = SQLiteBackend(db_path)

    # Create test data
    asyncio.run(
        backend.save_run_start(
            {
                "run_id": "env_test_run",
                "pipeline_id": "test-pid-env",
                "pipeline_name": "env_test",
                "pipeline_version": "1.0",
                "status": "running",
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
            }
        )
    )

    # Test with FLUJO_STATE_URI environment variable
    os.environ["FLUJO_STATE_URI"] = f"sqlite:///{db_path}"

    result = runner.invoke(app, ["lens", "list"])
    assert result.exit_code == 0
    assert "env_test_run" in result.stdout

    # Test with different URI format
    os.environ["FLUJO_STATE_URI"] = f"sqlite://{db_path}"

    result = runner.invoke(app, ["lens", "list"])
    assert result.exit_code == 0
    assert "env_test_run" in result.stdout
