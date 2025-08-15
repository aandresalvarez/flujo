from __future__ import annotations

import json as _json
from pathlib import Path
from typer.testing import CliRunner

from flujo.cli.main import app


def test_solve_smoke_outputs_json(monkeypatch) -> None:
    # Stub environment setup to avoid config/agent complexities
    monkeypatch.setattr(
        "flujo.cli.main.setup_solve_command_environment",
        lambda **kwargs: ({"k": None, "max_iters": None}, {}, (None, None, None, None)),
    )

    class _Settings:
        reflection_limit = 0

    monkeypatch.setattr("flujo.cli.main.load_settings", lambda: _Settings())

    class _Best:
        def model_dump(self):
            return {"score": 0.99, "text": "ok"}

    def _exec(**kwargs):
        return _Best()

    monkeypatch.setattr("flujo.cli.main.execute_solve_pipeline", _exec)

    runner = CliRunner()
    res = runner.invoke(app, ["dev", "experimental", "solve", "hello world"])
    assert res.exit_code == 0
    data = _json.loads(res.stdout)
    assert data.get("score") == 0.99
    assert data.get("text") == "ok"


def test_bench_smoke_prints_table(monkeypatch) -> None:
    # Stub the benchmark to return small arrays
    monkeypatch.setattr(
        "flujo.cli.main.run_benchmark_pipeline",
        lambda prompt, rounds, logfire: ([0.1, 0.2], [0.5, 0.6]),
    )

    # Avoid numpy dependency inside create_benchmark_table by stubbing a simple Table-like object
    class _Table:
        def __init__(self):
            self.title = "Benchmark Results"

        def __rich_console__(self, *args, **kwargs):  # pragma: no cover
            yield "Benchmark Results"

        def __str__(self):  # pragma: no cover
            return self.title

    monkeypatch.setattr("flujo.cli.main.create_benchmark_table", lambda *a, **k: _Table())

    runner = CliRunner()
    res = runner.invoke(app, ["dev", "experimental", "bench", "prompt", "--rounds", "2"])
    assert res.exit_code == 0
    assert "Benchmark Results" in res.stdout


def test_improve_smoke_json_output(monkeypatch, tmp_path: Path) -> None:
    # Stub improve execution to return JSON string
    monkeypatch.setattr(
        "flujo.cli.main.execute_improve",
        lambda pipeline_path, dataset_path, improvement_agent_model, json_output: _json.dumps(
            {"ok": True}
        ),
    )

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "dev",
            "experimental",
            "improve",
            str(tmp_path / "p.py"),
            str(tmp_path / "d.py"),
            "--improvement-model",
            "gpt-dev",
            "--json",
        ],
    )
    assert res.exit_code == 0
    assert '{"ok": true}' in res.stdout
