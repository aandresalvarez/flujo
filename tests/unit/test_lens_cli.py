import os
from pathlib import Path

from typer.testing import CliRunner

from flujo.cli.main import app
from datetime import datetime
from flujo.state.backends.sqlite import SQLiteBackend
import asyncio

runner = CliRunner()


def test_lens_commands(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "ops.db"
    backend = SQLiteBackend(db_path)
    asyncio.run(
        backend.save_run_start(
            {
                "run_id": "r1",
                "pipeline_name": "p",
                "pipeline_version": "v",
                "status": "running",
                "start_time": datetime.utcnow(),
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
    os.environ["FLUJO_STATE_URI"] = f"sqlite:///{db_path}"
    result = runner.invoke(app, ["lens", "list"])
    assert result.exit_code == 0
    result = runner.invoke(app, ["lens", "show", "r1"])
    assert result.exit_code == 0
