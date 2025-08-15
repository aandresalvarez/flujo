from __future__ import annotations

import logging
from pathlib import Path
from typer.testing import CliRunner

from flujo.cli.main import app


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


def _patch_create_pipeline_with_logging(
    monkeypatch, yaml_text: str, tmp_out: Path, marker: str
) -> None:
    """Patch create execution to emit a flujo logger message during run."""
    # Avoid dependency on bundled architect YAML
    monkeypatch.setattr("flujo.cli.main.load_pipeline_from_yaml_file", lambda *a, **k: object())
    # Mock runner creation
    monkeypatch.setattr("flujo.cli.main.create_flujo_runner", lambda *a, **k: object())

    def _fake_exec(*_args, **_kwargs):
        logging.getLogger("flujo").info(marker)
        return _FakeResult(yaml_text)

    # Mock pipeline execution to yield the YAML and log a marker
    monkeypatch.setattr("flujo.cli.main.execute_pipeline_with_output_handling", _fake_exec)
    # Ensure output directory exists
    tmp_out.mkdir(parents=True, exist_ok=True)


def test_create_suppresses_logs_by_default(tmp_path: Path, monkeypatch, caplog) -> None:
    yaml_text = 'version: "0.1"\nsteps: []\n'
    marker = "DEBUG-MARKER-DEFAULT"
    _patch_create_pipeline_with_logging(monkeypatch, yaml_text, tmp_path, marker)

    # Ensure architect file existence check passes and validation is OK
    monkeypatch.setattr("flujo.cli.main.os.path.isfile", lambda *a, **k: True)
    monkeypatch.setattr("flujo.cli.main.validate_yaml_text", lambda *a, **k: _DummyReport(True))

    caplog.set_level(logging.INFO, logger="flujo")
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "create",
            "--goal",
            "test",
            "--non-interactive",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    # By default, flujo logger is silenced to CRITICAL; marker should not be logged
    assert not any(rec.message == marker for rec in caplog.records)


def test_create_emits_logs_with_debug_flag(tmp_path: Path, monkeypatch, caplog) -> None:
    yaml_text = 'version: "0.1"\nsteps: []\n'
    marker = "DEBUG-MARKER-ENABLED"
    _patch_create_pipeline_with_logging(monkeypatch, yaml_text, tmp_path, marker)

    # Ensure architect file existence check passes and validation is OK
    monkeypatch.setattr("flujo.cli.main.os.path.isfile", lambda *a, **k: True)
    monkeypatch.setattr("flujo.cli.main.validate_yaml_text", lambda *a, **k: _DummyReport(True))

    caplog.set_level(logging.INFO, logger="flujo")
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "create",
            "--goal",
            "test",
            "--non-interactive",
            "--output-dir",
            str(tmp_path),
            "--debug",
        ],
    )

    assert result.exit_code == 0
    # With --debug, flujo logger should emit the marker
    assert any(rec.message == marker for rec in caplog.records)
