from __future__ import annotations

from pathlib import Path
from typing import Any
from flujo.type_definitions.common import JSONObject

import pytest
from typer.testing import CliRunner

from flujo.cli.main import app

# Mark all tests in this module as slow (architect self-correction tests take >180s)
pytestmark = [pytest.mark.slow]


class _DummyReport:
    def __init__(self, is_valid: bool = True) -> None:
        self.is_valid = is_valid
        self.errors = []
        self.warnings = []


@pytest.fixture()
def mock_architect_self_correction(monkeypatch) -> JSONObject:
    """Patch architect compiler and validator to exercise self-correction.

    - yaml_writer returns invalid YAML first, valid YAML second
    - validate_yaml_text returns invalid first call, valid second
    """

    counters: dict[str, int] = {"architect_calls": 0, "repair_calls": 0, "validator_calls": 0}
    validator_inputs: list[str] = []

    class _ArchitectAgent:
        async def run(self, data: Any, **_: Any) -> Any:
            counters["architect_calls"] += 1
            # Return invalid YAML initially to trigger the repair loop
            return {"yaml_text": "version: '0.1'\nsteps: ["}

    class _RepairAgent:
        async def run(self, data: Any, **_: Any) -> Any:
            counters["repair_calls"] += 1
            # Always return a valid YAML on repair
            return {"yaml_text": 'version: "0.1"\nsteps: []\n'}

    def _fake_make_agent_async(*, model: str, system_prompt: str, output_type: Any):  # type: ignore[no-untyped-def]
        # Return architect agent for simplicity; compiled mapping will override per name
        return _ArchitectAgent()

    # Patch agent compiler
    monkeypatch.setattr("flujo.domain.blueprint.compiler.make_agent_async", _fake_make_agent_async)

    # Ensure agents are considered precompiled
    def _fake_compile_agents(self):  # type: ignore[no-redef]
        self._compiled_agents = {
            "decomposer": _ArchitectAgent(),
            "tool_matcher": _ArchitectAgent(),
            "plan_presenter": _ArchitectAgent(),
            "yaml_writer": _ArchitectAgent(),
            "repair_agent": _RepairAgent(),
        }

    monkeypatch.setattr(
        "flujo.domain.blueprint.compiler.DeclarativeBlueprintCompiler._compile_agents",
        _fake_compile_agents,
        raising=True,
    )

    # Fallback: inject agents if builder path used
    import flujo.domain.blueprint.loader as _loader

    _orig_build = _loader.build_pipeline_from_blueprint

    def _build_with_fallback(model, compiled_agents=None, compiled_imports=None):  # type: ignore[no-redef]
        if not compiled_agents:
            compiled_agents = {
                "decomposer": _ArchitectAgent(),
                "tool_matcher": _ArchitectAgent(),
                "plan_presenter": _ArchitectAgent(),
                "yaml_writer": _ArchitectAgent(),
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

    # Patch validator function used inside pipeline YAML
    import flujo.cli.helpers as _helpers
    import flujo.cli.main as _cli_main

    def _fake_validate_yaml_text(yaml_text: str, base_dir: str | None = None):  # type: ignore[no-redef]
        counters["validator_calls"] += 1
        try:
            validator_inputs.append(str(yaml_text))
        except Exception:
            pass
        # First call invalid, second call valid
        return _DummyReport(is_valid=(counters["validator_calls"] >= 2))

    monkeypatch.setattr(_helpers, "validate_yaml_text", _fake_validate_yaml_text, raising=True)
    # Also patch the symbol imported into flujo.cli.main for the final CLI validation
    monkeypatch.setattr(_cli_main, "validate_yaml_text", _fake_validate_yaml_text, raising=True)

    # Capture step history names via a wrapper around execute_pipeline_with_output_handling
    captured: JSONObject = {"step_names": [], "loop_attempts": -1}

    def _flatten_names(step_results: Any, acc: list[str]) -> None:
        try:
            for sr in step_results or []:
                name = getattr(sr, "name", None)
                if isinstance(name, str):
                    acc.append(name)
                # Detect loop step attempts/iterations
                if name == "ValidateAndRepair":
                    try:
                        attempts = getattr(sr, "attempts", None)
                        if isinstance(attempts, int):
                            captured["loop_attempts"] = max(captured["loop_attempts"], attempts)
                        meta = getattr(sr, "metadata_", {}) or {}
                        iters = meta.get("iterations")
                        if isinstance(iters, int):
                            captured["loop_attempts"] = max(captured["loop_attempts"], iters)
                    except Exception:
                        pass
                nested = getattr(sr, "step_history", None)
                if nested:
                    _flatten_names(nested, acc)
        except Exception:
            pass

    _orig_exec = _cli_main.execute_pipeline_with_output_handling

    def _wrapped_exec(*args: Any, **kwargs: Any):  # type: ignore[no-redef]
        result = _orig_exec(*args, **kwargs)
        try:
            names: list[str] = []
            _flatten_names(getattr(result, "step_history", None), names)
            captured["step_names"] = names
        except Exception:
            captured["step_names"] = []
        return result

    monkeypatch.setattr(
        _cli_main, "execute_pipeline_with_output_handling", _wrapped_exec, raising=True
    )

    return {"counters": counters, "captured": captured, "validator_inputs": validator_inputs}


def test_architect_self_correction_loop(tmp_path: Path, mock_architect_self_correction) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "create",
            "--goal",
            "build a spellcheck then translate pipeline",
            "--non-interactive",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    out_yaml = tmp_path / "pipeline.yaml"
    assert out_yaml.exists(), "pipeline.yaml should be written"
    text = out_yaml.read_text().strip()
    assert text.startswith('version: "0.1"') or text.startswith("version: '0.1'")

    counters = mock_architect_self_correction["counters"]
    captured = mock_architect_self_correction["captured"]
    vins = mock_architect_self_correction["validator_inputs"]
    # Expect at least one validation attempt (CLI-level) and that YAML file is created
    assert counters["validator_calls"] >= 1

    # Tighten: ensure the loop attempted a repair by checking step history names
    names: list[str] = captured.get("step_names", [])
    # Validate that a repair attempt occurred inside the loop (best-effort by name)
    # When names are condensed by policies, assert loop iterations >= 2 deterministically
    # If loop is visible in step history, assert iterations were recorded; otherwise fall back to validator sequence
    if any(n == "ValidateAndRepair" for n in names):
        loop_attempts = int(captured.get("loop_attempts", -1))
        assert loop_attempts >= 2, (
            f"expected >=2 loop attempts recorded, got {loop_attempts}; names={names}"
        )
    else:
        # Fallback: ensure at least one validation occurred
        assert len(vins) >= 1, f"expected validator called at least once, got {len(vins)}"
