from __future__ import annotations

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


def _patch_create_pipeline(monkeypatch, yaml_text: str, tmp_out: Path) -> None:
    # Avoid dependency on bundled architect YAML
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


def test_create_sanitizes_conditional_and_branches(tmp_path: Path, monkeypatch) -> None:
    # YAML with mistakes:
    # - uses 'conditional' key instead of kind/condition
    # - branches provided as a bare list
    # - missing name in inner steps
    malformed = (
        'version: "0.1"\n'
        "steps:\n"
        "  - conditional: flujo.builtins.select_validity_branch\n"
        "    branches:\n"
        "      - kind: step\n"
        "        agent: { id: 'flujo.builtins.passthrough' }\n"
    )
    _patch_create_pipeline(monkeypatch, malformed, tmp_path)

    # Ensure architect file existence check passes and validation succeeds post-sanitize
    monkeypatch.setattr("flujo.cli.main.os.path.isfile", lambda *a, **k: True)
    monkeypatch.setattr("flujo.cli.main.validate_yaml_text", lambda *a, **k: _DummyReport(True))

    out_file = tmp_path / "pipeline.yaml"
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
    text = out_file.read_text()
    # Sanitizer should convert 'conditional' to kind+condition and wrap branches into a dict
    assert "kind: conditional" in text
    assert "condition:" in text
    assert "branches:" in text and "default:" in text


def test_create_non_interactive_requires_output_dir(monkeypatch, tmp_path: Path) -> None:
    yaml_text = 'version: "0.1"\nsteps: []\n'
    _patch_create_pipeline(monkeypatch, yaml_text, tmp_path)

    runner = CliRunner()
    result = runner.invoke(app, ["create", "--goal", "test", "--non-interactive"])
    assert result.exit_code != 0
    combined = (result.stdout or "") + (getattr(result, "stderr", "") or "")
    assert "--output-dir is required" in combined


def test_create_moves_text_input_into_params_for_extract(monkeypatch, tmp_path: Path) -> None:
    # Step has input string, but extract_from_text requires params.text
    malformed = (
        'version: "0.1"\n'
        "steps:\n"
        "  - kind: step\n"
        "    name: extract_snippet\n"
        "    input: '{{ some_prev.body }}'\n"
        "    agent:\n"
        "      id: 'flujo.builtins.extract_from_text'\n"
        "      params:\n"
        "        schema: { type: object, properties: { x: { type: string } }, required: [x] }\n"
    )
    _patch_create_pipeline(monkeypatch, malformed, tmp_path)

    monkeypatch.setattr("flujo.cli.main.os.path.isfile", lambda *a, **k: True)
    monkeypatch.setattr("flujo.cli.main.validate_yaml_text", lambda *a, **k: _DummyReport(True))

    out_file = tmp_path / "pipeline.yaml"
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
    text = out_file.read_text()
    # Input should have been moved under params.text and removed from step-level
    assert "text: '{{ some_prev.body }}'" in text
    assert "input:" not in text


def test_create_moves_content_input_into_params_for_fs_write(monkeypatch, tmp_path: Path) -> None:
    malformed = (
        'version: "0.1"\n'
        "steps:\n"
        "  - kind: step\n"
        "    name: write\n"
        "    input: '{{ extracted.value }}'\n"
        "    agent:\n"
        "      id: 'flujo.builtins.fs_write_file'\n"
        "      params:\n"
        "        path: 'out.txt'\n"
    )
    _patch_create_pipeline(monkeypatch, malformed, tmp_path)

    monkeypatch.setattr("flujo.cli.main.os.path.isfile", lambda *a, **k: True)
    monkeypatch.setattr("flujo.cli.main.validate_yaml_text", lambda *a, **k: _DummyReport(True))

    out_file = tmp_path / "pipeline.yaml"
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
    text = out_file.read_text()
    # Input should have been moved under params.content and removed from step-level
    assert "content: '{{ extracted.value }}'" in text
    assert "input:" not in text
