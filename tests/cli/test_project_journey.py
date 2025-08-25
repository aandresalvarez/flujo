from __future__ import annotations

from pathlib import Path
from typer.testing import CliRunner
from flujo.cli.main import app
import time


def test_init_scaffolds_project_and_is_idempotent(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=str(tmp_path)):
        # First init
        res1 = runner.invoke(app, ["init"])
        assert res1.exit_code == 0
        # Files and directories created somewhere under temp workspace
        found = list(tmp_path.rglob("flujo.toml"))
        assert found, "flujo.toml not created"
        proj = found[0].parent
        assert (proj / "pipeline.yaml").exists()
        assert (proj / "skills").is_dir()
        assert (proj / ".flujo").is_dir()

        # Second init should error and not overwrite
        res2 = runner.invoke(app, ["init"])
        assert res2.exit_code != 0


def test_init_completes_without_hanging(tmp_path: Path) -> None:
    """Test that flujo init completes quickly without hanging."""
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=str(tmp_path)):
        start_time = time.time()
        res = runner.invoke(app, ["init"])
        end_time = time.time()

        # Should complete in under 5 seconds (much faster than hanging)
        assert end_time - start_time < 5.0, "flujo init took too long, may be hanging"
        assert res.exit_code == 0, f"flujo init failed with exit code {res.exit_code}"

        # Verify files were created
        found = list(tmp_path.rglob("flujo.toml"))
        assert found, "flujo.toml not created"
        proj = found[0].parent
        assert (proj / "pipeline.yaml").exists()
        assert (proj / "skills").is_dir()
        assert (proj / ".flujo").is_dir()


def test_project_aware_validate_and_run(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=str(tmp_path)):
        # Initialize project
        assert runner.invoke(app, ["init"]).exit_code == 0

        # Validate should pick up pipeline.yaml implicitly
        v = runner.invoke(app, ["dev", "validate"])  # no args
        assert v.exit_code == 0

        # Run should also infer pipeline.yaml and succeed with passthrough
        r = runner.invoke(app, ["run", "--input", "hello"])
        assert r.exit_code == 0
        # Expect formatted output mentioning Final output
        assert "Final output:" in (r.stdout + r.stderr)


def test_demo_completes_without_hanging(tmp_path: Path) -> None:
    """Test that flujo demo completes quickly without hanging."""
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=str(tmp_path)):
        start_time = time.time()
        res = runner.invoke(app, ["demo"])
        end_time = time.time()

        # Should complete in under 5 seconds (much faster than hanging)
        assert end_time - start_time < 5.0, "flujo demo took too long, may be hanging"
        assert res.exit_code == 0, f"flujo demo failed with exit code {res.exit_code}"

        # Verify demo files were created
        found = list(tmp_path.rglob("flujo.toml"))
        assert found, "flujo.toml not created"
        proj = found[0].parent
        assert (proj / "pipeline.yaml").exists()
        assert (proj / "skills").is_dir()
        assert (proj / ".flujo").is_dir()

        # Demo should create a more complex pipeline
        pipeline_content = (proj / "pipeline.yaml").read_text()
        assert "research_demo" in pipeline_content
        assert "perform_web_search" in pipeline_content
