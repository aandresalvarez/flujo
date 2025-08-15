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


def test_create_conversation_writes_pipeline_and_budget(tmp_path: Path, monkeypatch) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=str(tmp_path)):
        # Initialize project
        assert runner.invoke(app, ["init"]).exit_code == 0

        # Stub architect pipeline run
        yaml_text = """version: "0.1"\nsteps:\n  - kind: step\n    name: passthrough\n"""
        monkeypatch.setattr("flujo.cli.main.load_pipeline_from_yaml_file", lambda *a, **k: object())
        monkeypatch.setattr("flujo.cli.main.create_flujo_runner", lambda *a, **k: object())
        monkeypatch.setattr(
            "flujo.cli.main.execute_pipeline_with_output_handling",
            lambda *a, **k: _FakeResult(yaml_text),
        )

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
