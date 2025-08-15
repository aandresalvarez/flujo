from __future__ import annotations

from pathlib import Path
from typer.testing import CliRunner
from flujo.cli.main import app


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
