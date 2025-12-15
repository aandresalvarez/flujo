from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from flujo.cli.main import app
from flujo.domain.models import PipelineResult, StepResult


class _DummyBackend:
    def __init__(self) -> None:
        self.close_sync_calls = 0

    def close_sync(self) -> None:
        self.close_sync_calls += 1


class _DummyRunner:
    def __init__(self) -> None:
        self.close_calls = 0

    def run(self, input_data: str, *, run_id: str | None = None) -> PipelineResult[object]:
        _ = input_data, run_id
        sr = StepResult(name="ok", output="done", success=True, cost_usd=0.0, token_counts=0)
        return PipelineResult(step_history=[sr], total_cost_usd=0.0, total_tokens=0)

    def close(self) -> None:
        self.close_calls += 1


def test_run_closes_runner_and_backend(monkeypatch, tmp_path: Path) -> None:
    yaml_text = """
version: "0.1"
steps:
  - kind: step
    name: ok
    agent: "tests.cli.test_main:Echo"
""".strip()
    p = tmp_path / "pipeline.yaml"
    p.write_text(yaml_text, encoding="utf-8")

    dummy_runner = _DummyRunner()
    dummy_backend = _DummyBackend()

    monkeypatch.setattr("flujo.cli.config.load_backend_from_config", lambda: dummy_backend)
    monkeypatch.setattr("flujo.cli.run_command.create_flujo_runner", lambda *a, **k: dummy_runner)

    runner = CliRunner()
    result = runner.invoke(app, ["run", str(p), "--input", "hi", "--summary"])
    assert result.exit_code == 0, (result.stdout or "") + (result.stderr or "")

    assert dummy_runner.close_calls == 1
    assert dummy_backend.close_sync_calls == 1
