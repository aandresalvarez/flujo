from __future__ import annotations

from pathlib import Path
from typer.testing import CliRunner
from flujo.cli.main import app
from flujo.infra.skill_registry import get_skill_registry
import yaml
import pytest


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


def _patch_create_pipeline(monkeypatch, yaml_text: str, tmp_out: Path):
    # Mock loader to avoid dependency on bundled architect YAML
    monkeypatch.setattr("flujo.cli.main.load_pipeline_from_yaml_file", lambda *a, **k: object())
    # Mock runner creation
    monkeypatch.setattr("flujo.cli.main.create_flujo_runner", lambda *a, **k: object())
    # Mock pipeline execution to yield the YAML
    monkeypatch.setattr(
        "flujo.cli.main.execute_pipeline_with_output_handling",
        lambda *a, **k: _FakeResult(yaml_text),
    )
    # Ensure output directory exists
    tmp_out.mkdir(parents=True, exist_ok=True)


def test_create_blocks_side_effects_in_non_interactive(tmp_path: Path, monkeypatch) -> None:
    # Pre-register side-effect skill in registry to avoid import resolution
    reg = get_skill_registry()
    reg.register("danger-skill", object(), side_effects=True)

    yaml_text = """version: "0.1"
steps:
  - kind: step
    name: s1
    agent:
      id: "danger-skill"
"""
    _patch_create_pipeline(monkeypatch, yaml_text, tmp_path)

    # Ensure architect file existence check passes
    monkeypatch.setattr("flujo.cli.main.os.path.isfile", lambda *a, **k: True)
    # Avoid full validation and import resolution
    monkeypatch.setattr("flujo.cli.main.validate_yaml_text", lambda *a, **k: _DummyReport(True))
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
    assert result.exit_code != 0
    assert "--allow-side-effects" in result.stdout


def test_create_allows_side_effects_with_flag_and_writes_file(tmp_path: Path, monkeypatch) -> None:
    reg = get_skill_registry()
    reg.register("danger-skill", object(), side_effects=True)

    yaml_text = """version: "0.1"
steps:
  - kind: step
    name: s1
    agent:
      id: "danger-skill"
"""
    _patch_create_pipeline(monkeypatch, yaml_text, tmp_path)

    out_file = tmp_path / "pipeline.yaml"
    monkeypatch.setattr("flujo.cli.main.os.path.isfile", lambda *a, **k: True)
    monkeypatch.setattr("flujo.cli.main.validate_yaml_text", lambda *a, **k: _DummyReport(True))
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
    assert result.exit_code == 0
    assert out_file.exists()


def test_create_force_overwrite(tmp_path: Path, monkeypatch) -> None:
    yaml_text = """version: "0.1"
steps: []
"""
    _patch_create_pipeline(monkeypatch, yaml_text, tmp_path)

    out_file = tmp_path / "pipeline.yaml"
    out_file.write_text("old")

    # Without --force should fail
    monkeypatch.setattr("flujo.cli.main.os.path.isfile", lambda *a, **k: True)
    monkeypatch.setattr("flujo.cli.main.validate_yaml_text", lambda *a, **k: _DummyReport(True))
    runner = CliRunner()
    res1 = runner.invoke(
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
    assert res1.exit_code != 0

    # With --force should overwrite
    res2 = runner.invoke(
        app,
        [
            "create",
            "--goal",
            "test",
            "--non-interactive",
            "--output-dir",
            str(tmp_path),
            "--force",
        ],
    )
    assert res2.exit_code == 0
    # Compare YAML structures to avoid quoting differences
    try:
        actual_yaml = yaml.safe_load(out_file.read_text())
    except yaml.YAMLError as e:
        pytest.fail(f"Malformed YAML in output file {out_file}:\n{e}")
    try:
        expected_yaml = yaml.safe_load(yaml_text)
    except yaml.YAMLError as e:
        pytest.fail(f"Malformed expected YAML string:\n{e}")
    assert actual_yaml == expected_yaml
