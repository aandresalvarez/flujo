from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from flujo.cli.main import app

# Mark all tests in this module as slow (architect tests take >180s)
pytestmark = [pytest.mark.slow]


# Module-level variable to control the simulated user input
async def provide_user_input_yes(_prev: Any) -> str:
    return "Y"


async def provide_user_input_no(_prev: Any) -> str:
    return "n"


# Async helper used by the YAML pipeline to emit a valid YAML
async def emit_yaml(_prev: Any) -> dict[str, str]:
    return {"yaml_text": 'version: "0.1"\nsteps: []\n'}


def map_confirmation(prev_output: Any, _context: Any | None = None) -> str:
    # Synchronous adapter mirroring flujo.builtins.check_user_confirmation
    try:
        text = "" if prev_output is None else str(prev_output)
    except Exception:
        text = ""
    norm = text.strip().lower()
    if norm == "" or norm in {"y", "yes"}:
        return "approved"
    return "denied"


# Minimal collaborative architect pipeline exercising the HITL confirmation branch
COLLAB_YAML_APPROVED = """
version: "0.1"
steps:
  - kind: step
    name: PresentPlanToUser
    agent: { id: "flujo.builtins.passthrough" }
    input: |
      Plan:
        1) Do X using tool A
        2) Do Y using tool B

  - kind: step
    name: ConfirmPlanWithUser
    uses: tests.cli.test_architect_hitl:provide_user_input_yes

  - kind: conditional
    name: CheckConfirmation
    condition: "tests.cli.test_architect_hitl:map_confirmation"
    branches:
      approved:
        - kind: step
          name: GenerateBlueprint
          uses: tests.cli.test_architect_hitl:emit_yaml
      denied:
        - kind: step
          name: AbortPipeline
          agent: { id: "flujo.builtins.passthrough" }
"""


COLLAB_YAML_DENIED = COLLAB_YAML_APPROVED.replace(
    "tests.cli.test_architect_hitl:provide_user_input_yes",
    "tests.cli.test_architect_hitl:provide_user_input_no",
)


def _monkeypatch_architect_pipeline(monkeypatch, *, approved: bool) -> None:
    # Replace architect loader in CLI with our collaborative YAML
    import flujo.cli.main as _cli_main
    from flujo.domain.blueprint.loader import load_pipeline_blueprint_from_yaml

    monkeypatch.setattr(
        _cli_main,
        "load_pipeline_from_yaml_file",
        lambda _p: load_pipeline_blueprint_from_yaml(
            COLLAB_YAML_APPROVED if approved else COLLAB_YAML_DENIED
        ),
        raising=True,
    )


def _patch_validators(monkeypatch) -> None:
    # Make validator always return valid so the CLI proceeds to write the file
    class _DummyReport:
        def __init__(self, is_valid: bool = True) -> None:
            self.is_valid = is_valid
            self.errors = []
            self.warnings = []

    import flujo.cli.main as _cli_main
    import flujo.builtins as _builtins

    monkeypatch.setattr(
        _cli_main, "validate_yaml_text", lambda *a, **k: _DummyReport(True), raising=True
    )

    async def _always_valid_yaml(_text: str, *_a: Any, **_k: Any) -> Any:
        return _DummyReport(True)

    monkeypatch.setattr(_builtins, "validate_yaml", _always_valid_yaml, raising=True)


def test_architect_hitl_happy_path(tmp_path: Path, monkeypatch) -> None:
    _monkeypatch_architect_pipeline(monkeypatch, approved=True)
    _patch_validators(monkeypatch)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "create",
            "--goal",
            "demo",
            "--non-interactive",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    out_yaml = tmp_path / "pipeline.yaml"
    assert out_yaml.exists(), "pipeline.yaml should be written in approved branch"
    text = out_yaml.read_text().strip()
    assert text.startswith('version: "0.1"') or text.startswith("version: '0.1'")


def test_architect_hitl_rejection_path(tmp_path: Path, monkeypatch) -> None:
    _monkeypatch_architect_pipeline(monkeypatch, approved=False)

    # Validators are irrelevant here because no YAML should be produced
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "create",
            "--goal",
            "demo",
            "--non-interactive",
            "--output-dir",
            str(tmp_path),
        ],
    )

    # CLI should fail because no YAML was produced by denied branch
    assert result.exit_code != 0
    out_yaml = tmp_path / "pipeline.yaml"
    assert not out_yaml.exists(), "pipeline.yaml should not be written in denied branch"
