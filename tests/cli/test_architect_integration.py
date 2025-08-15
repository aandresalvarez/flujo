from __future__ import annotations

from pathlib import Path
import logging
from typing import Any, Dict

import pytest
from typer.testing import CliRunner

from flujo.cli.main import app


class _DummyReport:
    def __init__(self, is_valid: bool = True) -> None:
        self.is_valid = is_valid
        self.errors = []
        self.warnings = []


MARKER = "ARCH-DEBUG-MARKER"


@pytest.fixture()
def mock_architect_env(monkeypatch) -> Dict[str, Any]:
    """Patch the architect compile + loader path with dummy agents and helpers.

    Returns a dict with shared constants like the log marker string.
    """

    captured: Dict[str, Any] = {}

    class _ArchitectAgent:
        async def run(self, data: Any, **_: Any) -> Any:
            # Capture the exact input provided to the architect agent
            captured["input"] = data
            logging.getLogger("flujo").info(MARKER)
            return {"yaml_text": 'version: "0.1"\nsteps: []\n'}

    class _RepairAgent:
        async def run(self, data: Any, **_: Any) -> Any:
            # Return a fixed valid YAML regardless of input
            return {"yaml_text": 'version: "0.1"\nsteps: []\n'}

    def _fake_make_agent_async(*, model: str, system_prompt: str, output_type: Any):  # type: ignore[no-untyped-def]
        # Both architect and repair agents return simple dicts
        return _ArchitectAgent()

    # Patch agent factory in compiler (in case it is used)
    monkeypatch.setattr("flujo.domain.blueprint.compiler.make_agent_async", _fake_make_agent_async)

    # Ensure agents are considered precompiled to avoid any schema/model errors
    def _fake_compile_agents(self):  # type: ignore[no-redef]
        self._compiled_agents = {
            "architect_agent": _ArchitectAgent(),
            "repair_agent": _RepairAgent(),
        }

    monkeypatch.setattr(
        "flujo.domain.blueprint.compiler.DeclarativeBlueprintCompiler._compile_agents",
        _fake_compile_agents,
        raising=True,
    )

    # Fallback safety: if loader falls back to plain builder, inject our compiled agents
    import flujo.domain.blueprint.loader as _loader

    _orig_build = _loader.build_pipeline_from_blueprint

    def _build_with_fallback(model, compiled_agents=None, compiled_imports=None):  # type: ignore[no-redef]
        if not compiled_agents:
            compiled_agents = {
                "architect_agent": _ArchitectAgent(),
                "repair_agent": _RepairAgent(),
            }
        return _orig_build(
            model, compiled_agents=compiled_agents, compiled_imports=compiled_imports
        )

    monkeypatch.setattr(
        "flujo.domain.blueprint.loader.build_pipeline_from_blueprint",
        _build_with_fallback,
        raising=True,
    )

    # Make validation a no-op success
    monkeypatch.setattr("flujo.cli.main.validate_yaml_text", lambda *a, **k: _DummyReport(True))

    return {"marker": MARKER, "captured": captured}


def test_create_architect_flow_with_mocked_agents(tmp_path: Path, mock_architect_env) -> None:
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
    assert out_yaml.exists(), "pipeline.yaml should be written"
    assert out_yaml.read_text().strip().startswith('version: "0.1"')
    # Assert the architect agent received input containing available skills
    captured = mock_architect_env["captured"]
    assert "input" in captured
    architect_input = str(captured["input"])
    assert "available_skills" in architect_input


@pytest.mark.parametrize("debug, expect_marker", [(False, False), (True, True)])
def test_create_architect_logging(
    tmp_path: Path, mock_architect_env, caplog, debug: bool, expect_marker: bool
) -> None:
    caplog.set_level(logging.INFO, logger="flujo")
    runner = CliRunner()
    args = [
        "create",
        "--goal",
        "demo",
        "--non-interactive",
        "--output-dir",
        str(tmp_path),
    ]
    if debug:
        args.append("--debug")

    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output

    messages = [r.message for r in caplog.records]
    if expect_marker:
        assert MARKER in messages
    else:
        assert MARKER not in messages
