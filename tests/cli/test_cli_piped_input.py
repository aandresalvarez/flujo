from __future__ import annotations

from pathlib import Path
from typer.testing import CliRunner
from flujo.cli.main import app


PIPELINE_YAML = """
version: "0.1"
name: "cli_piped_input_test"
agents:
  typed:
    id: "tests.unit.test_error_messages.need_str"
    model: "local:mock"
    system_prompt: "typed"
    output_schema: { type: string }

steps:
  - kind: step
    name: get_input
    uses: agents.typed
    input_schema: { type: string }
    output_schema: { type: string }
    input: "{{ context.initial_prompt or 'What do you want to do today?' }}"

  - kind: step
    name: process_input
    uses: agents.typed
    input_schema: { type: string }
    input: "Processing: {{ steps.get_input }}"
"""


def _scaffold_project(root: Path) -> None:
    (root / "skills").mkdir(parents=True, exist_ok=True)
    (root / ".flujo").mkdir(parents=True, exist_ok=True)
    (root / "flujo.toml").write_text('state_uri = "memory://"\n')
    (root / "pipeline.yaml").write_text(PIPELINE_YAML)


def test_run_reads_from_stdin_when_dash(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=str(tmp_path)):
        _scaffold_project(Path.cwd())

        # Provide input via stdin and signal it with '-'
        res = runner.invoke(app, ["run", "--input", "-"], input="Hello from stdin")
        assert res.exit_code == 0, res.output
        assert "Final output:" in res.output
        assert "Processing:" in res.output


def test_run_reads_from_env_when_set(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=str(tmp_path)):
        _scaffold_project(Path.cwd())

        # No --input and no stdin; pick up FLUJO_INPUT
        res = runner.invoke(app, ["run"], env={"FLUJO_INPUT": "FromEnv"})
        assert res.exit_code == 0, res.output
        assert "Processing:" in res.output


def test_run_reads_from_piped_stdin_without_flag_yaml(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=str(tmp_path)):
        _scaffold_project(Path.cwd())

        # No --input flag; stdin should be consumed
        res = runner.invoke(app, ["run"], input="PipedInput")
        assert res.exit_code == 0, res.output
        assert "Processing:" in res.output


def test_precedence_env_over_piped_without_flag(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=str(tmp_path)):
        _scaffold_project(Path.cwd())

        res = runner.invoke(app, ["run"], env={"FLUJO_INPUT": "FromEnv"}, input="FromStdin")
        assert res.exit_code == 0, res.output
        assert "Processing:" in res.output
        assert "FromStdin" not in res.output


def test_precedence_flag_over_env_and_piped(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=str(tmp_path)):
        _scaffold_project(Path.cwd())

        res = runner.invoke(
            app,
            ["run", "--input", "FlagWins"],
            env={"FLUJO_INPUT": "FromEnv"},
            input="FromStdin",
        )
        assert res.exit_code == 0, res.output
        assert "Processing:" in res.output
        assert "FromEnv" not in res.output
        assert "FromStdin" not in res.output


def test_python_pipeline_reads_from_stdin_with_dash(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=str(tmp_path)):
        # Use example Python pipeline provided in repo
        from pathlib import Path as _Path

        examples_dir = _Path(__file__).resolve().parents[2] / "examples"
        pipeline_py = examples_dir / "test_pipeline.py"

        # Create minimal project structure (not strictly required for .py, but consistent)
        (Path.cwd() / "skills").mkdir(parents=True, exist_ok=True)
        (Path.cwd() / ".flujo").mkdir(parents=True, exist_ok=True)
        (Path.cwd() / "flujo.toml").write_text('state_uri = "memory://"\n')

        res = runner.invoke(app, ["run", str(pipeline_py), "--input", "-"], input="PyStdin")
        assert res.exit_code == 0, res.output
        # Final output mentions processed echo/transform/finalize with input visible at least once
        assert "PyStdin" in res.output
