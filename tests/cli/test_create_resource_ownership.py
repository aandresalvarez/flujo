from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from flujo.cli.main import app


class _DummyRunner:
    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class _DummyBackend:
    def __init__(self) -> None:
        self.close_sync_calls = 0

    def close_sync(self) -> None:
        self.close_sync_calls += 1


class _FakeCtx:
    def __init__(self, yaml_text: str) -> None:
        self.generated_yaml = yaml_text


class _FakeResult:
    def __init__(self, yaml_text: str) -> None:
        self.final_pipeline_context = _FakeCtx(yaml_text)
        self.step_history = []
        self.total_cost_usd = 0.0
        self.token_counts = 0


class _DummyReport:
    def __init__(self, is_valid: bool = True) -> None:
        self.is_valid = is_valid
        self.errors = []
        self.warnings = []


def test_create_closes_runner_and_backend(monkeypatch, tmp_path: Path) -> None:
    dummy_runner = _DummyRunner()
    dummy_backend = _DummyBackend()

    monkeypatch.setattr("flujo.cli.config.load_backend_from_config", lambda: dummy_backend)
    monkeypatch.setattr("flujo.architect.builder.build_architect_pipeline", lambda: object())

    monkeypatch.setattr("flujo.cli.main.create_flujo_runner", lambda *a, **k: dummy_runner)
    monkeypatch.setattr(
        "flujo.cli.main.execute_pipeline_with_output_handling",
        lambda *a, **k: _FakeResult('version: "0.1"\\nsteps: []\\n'),
    )
    monkeypatch.setattr("flujo.cli.main.validate_yaml_text", lambda *a, **k: _DummyReport(True))
    monkeypatch.setattr("flujo.cli.main.os.path.isfile", lambda *a, **k: True)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "create",
            "--goal",
            "test",
            "--non-interactive",
            "--allow-side-effects",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, (result.stdout or "") + (result.stderr or "")

    assert (tmp_path / "pipeline.yaml").exists()
    assert dummy_runner.close_calls == 1
    assert dummy_backend.close_sync_calls == 1
