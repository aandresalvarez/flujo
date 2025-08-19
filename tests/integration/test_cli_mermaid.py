from pathlib import Path

from flujo.cli.main import app
from typer.testing import CliRunner

runner = CliRunner()


def create_pipeline_file(tmp_path: Path) -> Path:
    file = tmp_path / "pipe.py"
    file.write_text(
        "from flujo.domain import Step\n"
        "from flujo.testing.utils import StubAgent\n"
        "pipeline = Step.model_validate({'name': 'A', 'agent': StubAgent(['a'])}) >> "
        "Step.model_validate({'name': 'B', 'agent': StubAgent(['b'])})\n"
    )
    return file


def test_pipeline_mermaid_command(tmp_path: Path) -> None:
    path = create_pipeline_file(tmp_path)
    result = runner.invoke(app, ["dev", "visualize", "--file", str(path)])
    assert result.exit_code == 0
    assert "```mermaid" in result.stdout
    assert "graph TD" in result.stdout
