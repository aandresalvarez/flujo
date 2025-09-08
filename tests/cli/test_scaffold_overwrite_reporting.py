from __future__ import annotations

from pathlib import Path
from typer.testing import CliRunner

from flujo.cli.main import app


def test_init_force_reports_overwrites(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=str(tmp_path)):
        # First init creates a project
        res1 = runner.invoke(app, ["init"])
        assert res1.exit_code == 0, res1.output

        # Ensure marker files exist
        created = list(tmp_path.rglob("flujo.toml"))
        assert created, "flujo.toml not created"
        proj = created[0].parent
        assert (proj / "pipeline.yaml").exists()
        assert (proj / "skills" / "__init__.py").exists()
        assert (proj / "skills" / "custom_tools.py").exists()

        # Force re-init with --force --yes
        res2 = runner.invoke(app, ["init", "--force", "--yes"])
        out = res2.stdout + res2.stderr
        assert res2.exit_code == 0, out
        # Should report re-initialization and overwritten files
        assert "Re-initialized" in out or "initialized" in out
        assert "Overwrote:" in out
        # At least the core templates should be listed as overwritten
        assert "flujo.toml" in out
        assert "pipeline.yaml" in out
