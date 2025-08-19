from __future__ import annotations

from pathlib import Path
from typing import Any
from typer.testing import CliRunner
from flujo.cli.main import app


class _DummyReport:
    def __init__(self, is_valid: bool = True) -> None:
        self.is_valid = is_valid
        self.errors = []
        self.warnings = []


class _FakeCtx:
    def __init__(self, yaml_text: str) -> None:
        self.generated_yaml = yaml_text


class _FakeResult:
    def __init__(self, yaml_text: str) -> None:
        self.final_pipeline_context = _FakeCtx(yaml_text)
        self.step_history = []
        self.total_cost_usd = 0.0
        self.token_counts = 0


def test_create_conversation_writes_pipeline_and_budget(tmp_path: Path, monkeypatch) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=str(tmp_path)):
        # Initialize project
        assert runner.invoke(app, ["init"]).exit_code == 0

        # Mock the agent compilation to avoid blueprint loading issues
        class _FakeAgent:
            async def run(self, data: Any, **_: Any) -> Any:
                return {"yaml_text": "version: '0.1'\nsteps: []"}

        def _fake_compile_agents(self):
            self._compiled_agents = {
                "decomposer": _FakeAgent(),
                "tool_matcher": _FakeAgent(),
                "plan_presenter": _FakeAgent(),
                "yaml_writer": _FakeAgent(),
                "repair_agent": _FakeAgent(),
            }

        monkeypatch.setattr(
            "flujo.domain.blueprint.compiler.DeclarativeBlueprintCompiler._compile_agents",
            _fake_compile_agents,
            raising=True,
        )

        # Stub architect pipeline run - return YAML without name to force CLI to use interactive input
        yaml_text = """version: "0.1"\nsteps:\n  - kind: step\n    name: passthrough\n"""
        monkeypatch.setattr(
            "flujo.cli.helpers.load_pipeline_from_yaml_file", lambda *a, **k: object()
        )
        monkeypatch.setattr("flujo.cli.main.create_flujo_runner", lambda *a, **k: object())
        monkeypatch.setattr(
            "flujo.cli.main.execute_pipeline_with_output_handling",
            lambda *a, **k: _FakeResult(yaml_text),
        )

        # Additional mocks needed for create command
        monkeypatch.setattr("flujo.cli.main.os.path.isfile", lambda *a, **k: True)
        monkeypatch.setattr("flujo.cli.main.validate_yaml_text", lambda *a, **k: _DummyReport(True))

        # Provide interactive answers: goal, name, budget
        user_input = "Ship weekly report bot\nweekly_report\n3.25\n"
        res = runner.invoke(app, ["create"], input=user_input)
        assert res.exit_code == 0

        # Assert pipeline.yaml updated with name (search under temp root)
        ypath = next((p for p in tmp_path.rglob("pipeline.yaml")), None)
        assert ypath is not None
        content = ypath.read_text()
        assert 'name: "weekly_report"' in content

        # Assert flujo.toml updated with budget section
        tpath = next((p for p in tmp_path.rglob("flujo.toml")), None)
        assert tpath is not None
        ttext = tpath.read_text()
        assert '[budgets.pipeline."weekly_report"]' in ttext
        assert "total_cost_usd_limit = 3.25" in ttext
