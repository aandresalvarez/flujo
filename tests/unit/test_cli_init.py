from __future__ import annotations

from pathlib import Path
from typer.testing import CliRunner

from flujo.cli.main import app


def test_init_creates_scaffold(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):  # chdir into tmp
        result = runner.invoke(app, ["init"])  # fresh dir
        assert result.exit_code == 0

        # Basic scaffold created
        assert Path("flujo.toml").exists()
        assert Path("pipeline.yaml").exists()
        assert Path("skills").is_dir()
        assert Path("skills/__init__.py").exists()
        assert Path("skills/custom_tools.py").exists()
        assert Path(".flujo").is_dir()


def test_init_existing_project_requires_force(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        # Simulate existing project
        Path(".").mkdir(parents=True, exist_ok=True)
        Path("flujo.toml").write_text("OLD")
        Path(".flujo").mkdir(parents=True, exist_ok=True)

        result = runner.invoke(app, ["init"])  # without --force should fail
        assert result.exit_code == 1


def test_init_force_prompts_and_overwrites(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        # Prepare existing project with old content
        Path(".").mkdir(parents=True, exist_ok=True)
        Path("flujo.toml").write_text("OLD_TOML")
        Path("pipeline.yaml").write_text("OLD_PIPELINE")
        Path(".flujo").mkdir(parents=True, exist_ok=True)

        # Use --force; confirm with 'y' input
        result = runner.invoke(app, ["init", "--force"], input="y\n")
        assert result.exit_code == 0

        # Files should be overwritten with template content (not equal to OLD_*)
        assert Path("flujo.toml").read_text() != "OLD_TOML"
        assert Path("pipeline.yaml").read_text() != "OLD_PIPELINE"


def test_init_force_yes_skips_prompt_and_overwrites(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        # Prepare existing project with old content
        Path(".").mkdir(parents=True, exist_ok=True)
        Path("flujo.toml").write_text("OLD_TOML")
        Path("pipeline.yaml").write_text("OLD_PIPELINE")
        Path(".flujo").mkdir(parents=True, exist_ok=True)

        result = runner.invoke(app, ["init", "--force", "--yes"])  # no input needed
        assert result.exit_code == 0
        assert Path("flujo.toml").read_text() != "OLD_TOML"
        assert Path("pipeline.yaml").read_text() != "OLD_PIPELINE"
