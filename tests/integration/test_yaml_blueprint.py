from __future__ import annotations

from pathlib import Path
import textwrap
import subprocess
import sys
from typer.testing import CliRunner
from flujo.cli.main import app


def test_runner_uses_toml_budgets(tmp_path: Path) -> None:
    # flujo.toml with default and wildcard
    (tmp_path / "flujo.toml").write_text(
        """
[budgets.default]
 total_cost_usd_limit = 12.0
 total_tokens_limit = 12345

 [budgets.pipeline]
 "demo" = { total_cost_usd_limit = 3.5, total_tokens_limit = 77 }
        """.strip()
    )

    # Simple pipeline file
    pipe = tmp_path / "pipe.py"
    pipe.write_text(
        textwrap.dedent(
            """
            from flujo.domain.dsl import Step, Pipeline
            async def echo(x: str) -> str: return x
            pipeline = Pipeline.from_step(Step.from_callable(echo, name="echo"))
            """
        ).strip()
    )

    # Execute run; it should succeed. We cannot directly introspect Quota via CLI
    # here, but presence of config and successful run acts as a smoke test.
    res = subprocess.run(
        [sys.executable, "-m", "flujo.cli.main", "run", str(pipe), "--input", "hi"],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0


runner = CliRunner()

YAML_SPEC = """
version: "0.1"
agents:
  typed:
    id: "tests.unit.test_error_messages.need_str"
    model: "local:mock"
    system_prompt: "typed"
    output_schema: { type: string }
steps:
  - kind: step
    name: s1
    uses: agents.typed
    input: "Hello YAML"
  - kind: step
    name: s2
    uses: agents.typed
    input_schema: { type: string }
    input: "{{ previous_step.output }}"
"""


def test_run_yaml_blueprint_tmpfile(tmp_path):
    p = tmp_path / "pipe.yaml"
    p.write_text(YAML_SPEC)
    # Provide input via --input for non-interactive
    result = runner.invoke(
        app,
        [
            "run",
            str(p),
            "--input",
            "Hello YAML",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Pipeline execution completed successfully" in result.output
    assert "Final output:" in result.output
    assert "Hello YAML" in result.output


def test_validate_yaml_blueprint(tmp_path):
    p = tmp_path / "pipe.yaml"
    p.write_text(YAML_SPEC)
    result = runner.invoke(app, ["validate", str(p)])
    assert result.exit_code == 0, result.output
    assert "Pipeline is valid" in result.output or "Warnings:" in result.output
