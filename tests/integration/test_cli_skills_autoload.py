from __future__ import annotations

from pathlib import Path
from typer.testing import CliRunner
from flujo.cli.main import app


def test_cli_autoloads_skills_catalog_from_yaml_dir(tmp_path: Path) -> None:
    # Create skills.yaml in temp dir that points to a repository-local test module
    skills_yaml = tmp_path / "skills.yaml"
    skills_yaml.write_text(
        """
echo_cmd:
  path: "tests.integration.skillmods.echo:make_echo"
  description: "Echo agent factory"
        """.strip()
    )

    # Create a YAML pipeline that references the skill id defined above
    pipeline_yaml = tmp_path / "pipeline.yaml"
    pipeline_yaml.write_text(
        """
version: "0.1"
steps:
  - name: Echo
    agent: { id: "echo_cmd", params: {} }
    input: "hello"
        """.strip()
    )

    runner = CliRunner()
    result = runner.invoke(app, ["run", str(pipeline_yaml), "--json"])

    assert result.exit_code == 0, result.output
    # Ensure the run produced the expected output from the echo agent
    assert '"output": "hello"' in result.output
