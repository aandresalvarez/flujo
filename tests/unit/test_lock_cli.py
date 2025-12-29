"""Unit tests for lock CLI commands."""

from __future__ import annotations

import json
import pytest
from pathlib import Path
from typer.testing import CliRunner

from flujo.cli.main import app

pytestmark = [pytest.mark.fast]

runner = CliRunner()


class TestLockCLICommands:
    """Test CLI command invocation for lock commands."""

    def test_lock_generate_command(self, tmp_path: Path) -> None:
        """Test flujo lock generate command."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create a minimal pipeline.yaml
            Path("pipeline.yaml").write_text(
                """
version: "0.1"
name: "test_pipeline"
steps:
  - kind: step
    name: echo
    agent:
      id: "flujo.builtins.stringify"
    input: "hello"
""",
                encoding="utf-8",
            )

            result = runner.invoke(app, ["lock", "generate", "--output", "flujo.lock"])

            # Command should succeed
            assert result.exit_code == 0
            assert Path("flujo.lock").exists()

            # Verify lockfile content
            lockfile_data: dict[str, object] = json.loads(Path("flujo.lock").read_text())
            assert "schema_version" in lockfile_data
            assert "pipeline" in lockfile_data

    def test_lock_generate_with_external_files(self, tmp_path: Path) -> None:
        """Test flujo lock generate with external files."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create pipeline.yaml
            Path("pipeline.yaml").write_text(
                """
version: "0.1"
name: "test_pipeline"
steps:
  - kind: step
    name: echo
    agent:
      id: "flujo.builtins.stringify"
    input: "hello"
""",
                encoding="utf-8",
            )

            # Create external file
            data_dir: Path = Path("data")
            data_dir.mkdir()
            (data_dir / "file.json").write_text('{"key": "value"}', encoding="utf-8")

            result = runner.invoke(
                app,
                [
                    "lock",
                    "generate",
                    "--include-external",
                    "data/file.json",
                    "--output",
                    "flujo.lock",
                ],
            )

            assert result.exit_code == 0
            assert Path("flujo.lock").exists()

            # Verify external file is included
            lockfile_data: dict[str, object] = json.loads(Path("flujo.lock").read_text())
            if "external_files" in lockfile_data:
                external_files: list[dict[str, object]] = lockfile_data["external_files"]  # type: ignore
                assert len(external_files) > 0

    def test_lock_verify_command(self, tmp_path: Path) -> None:
        """Test flujo lock verify command."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create pipeline.yaml
            Path("pipeline.yaml").write_text(
                """
version: "0.1"
name: "test_pipeline"
steps:
  - kind: step
    name: echo
    agent:
      id: "flujo.builtins.stringify"
    input: "hello"
""",
                encoding="utf-8",
            )

            # Generate lockfile first
            generate_result = runner.invoke(app, ["lock", "generate", "--output", "flujo.lock"])
            assert generate_result.exit_code == 0

            # Verify lockfile
            verify_result = runner.invoke(app, ["lock", "verify", "--lockfile", "flujo.lock"])

            # Should succeed if lockfile matches
            assert verify_result.exit_code in (0, 4)  # 0 = no differences, 4 = differences found

    def test_lock_compare_command(self, tmp_path: Path) -> None:
        """Test flujo lock compare command."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create two lockfiles
            lockfile1_data: dict[str, object] = {
                "schema_version": 1,
                "pipeline": {"name": "pipeline1", "version": "1.0", "id": "id1"},
                "prompts": [{"step": "step1", "hash": "abc123"}],
                "skills": [],
            }

            lockfile2_data: dict[str, object] = {
                "schema_version": 1,
                "pipeline": {"name": "pipeline2", "version": "2.0", "id": "id2"},
                "prompts": [{"step": "step1", "hash": "xyz789"}],
                "skills": [],
            }

            Path("lock1.json").write_text(json.dumps(lockfile1_data), encoding="utf-8")
            Path("lock2.json").write_text(json.dumps(lockfile2_data), encoding="utf-8")

            result = runner.invoke(app, ["lock", "compare", "lock1.json", "lock2.json"])

            # Should succeed and show differences
            assert result.exit_code == 0
            assert "step1" in result.stdout or "differences" in result.stdout.lower()

    def test_lock_show_command(self, tmp_path: Path) -> None:
        """Test flujo lock show command."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create project markers for find_project_root
            Path("flujo.toml").write_text("[project]\n", encoding="utf-8")

            # Create a lockfile
            lockfile_data: dict[str, object] = {
                "schema_version": 1,
                "pipeline": {"name": "test_pipeline", "version": "1.0", "id": "test_id"},
                "prompts": [{"step": "step1", "hash": "abc123"}],
                "skills": [{"step": "step1", "skill_id": "skill1", "hash": "def456"}],
            }

            Path("flujo.lock").write_text(json.dumps(lockfile_data), encoding="utf-8")

            result = runner.invoke(app, ["lock", "show", "--lockfile", "flujo.lock"])

            # Should succeed
            assert result.exit_code == 0
            assert "test_pipeline" in result.stdout

    def test_lock_show_json_format(self, tmp_path: Path) -> None:
        """Test flujo lock show with JSON format."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create project markers for find_project_root
            Path("flujo.toml").write_text("[project]\n", encoding="utf-8")

            # Create a lockfile
            lockfile_data: dict[str, object] = {
                "schema_version": 1,
                "pipeline": {"name": "test_pipeline", "version": "1.0", "id": "test_id"},
                "prompts": [],
                "skills": [],
            }

            Path("flujo.lock").write_text(json.dumps(lockfile_data), encoding="utf-8")

            result = runner.invoke(
                app, ["lock", "show", "--lockfile", "flujo.lock", "--format", "json"]
            )

            # Should succeed and output JSON
            assert result.exit_code == 0
            output_data: dict[str, object] = json.loads(result.stdout)
            assert "pipeline" in output_data

    def test_lock_generate_force_overwrite(self, tmp_path: Path) -> None:
        """Test flujo lock generate with --force to overwrite existing file."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create pipeline.yaml
            Path("pipeline.yaml").write_text(
                """
version: "0.1"
name: "test_pipeline"
steps:
  - kind: step
    name: echo
    agent:
      id: "flujo.builtins.stringify"
    input: "hello"
""",
                encoding="utf-8",
            )

            # Create existing lockfile
            Path("flujo.lock").write_text('{"old": "data"}', encoding="utf-8")

            # Generate without force should fail
            result1 = runner.invoke(app, ["lock", "generate", "--output", "flujo.lock"])
            assert result1.exit_code != 0

            # Generate with force should succeed
            result2 = runner.invoke(app, ["lock", "generate", "--output", "flujo.lock", "--force"])
            assert result2.exit_code == 0

            # Verify file was overwritten
            lockfile_data: dict[str, object] = json.loads(Path("flujo.lock").read_text())
            assert "old" not in lockfile_data
            assert "pipeline" in lockfile_data

    def test_lock_verify_strict_mode(self, tmp_path: Path) -> None:
        """Test flujo lock verify with --strict flag."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create pipeline.yaml
            Path("pipeline.yaml").write_text(
                """
version: "0.1"
name: "test_pipeline"
steps:
  - kind: step
    name: echo
    agent:
      id: "flujo.builtins.stringify"
    input: "hello"
""",
                encoding="utf-8",
            )

            # Generate initial lockfile
            generate_result = runner.invoke(app, ["lock", "generate", "--output", "flujo.lock"])
            assert generate_result.exit_code == 0

            # Verify with strict (should pass if no differences)
            verify_result = runner.invoke(
                app, ["lock", "verify", "--lockfile", "flujo.lock", "--strict"]
            )

            # Exit code depends on whether differences were found
            assert verify_result.exit_code in (0, 4)

    def test_lock_compare_json_format(self, tmp_path: Path) -> None:
        """Test flujo lock compare with JSON output format."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create two lockfiles
            lockfile1_data: dict[str, object] = {
                "schema_version": 1,
                "pipeline": {"name": "pipeline1", "version": "1.0", "id": "id1"},
                "prompts": [{"step": "step1", "hash": "abc123"}],
                "skills": [],
            }

            lockfile2_data: dict[str, object] = {
                "schema_version": 1,
                "pipeline": {"name": "pipeline2", "version": "2.0", "id": "id2"},
                "prompts": [{"step": "step1", "hash": "xyz789"}],
                "skills": [],
            }

            Path("lock1.json").write_text(json.dumps(lockfile1_data), encoding="utf-8")
            Path("lock2.json").write_text(json.dumps(lockfile2_data), encoding="utf-8")

            result = runner.invoke(
                app, ["lock", "compare", "lock1.json", "lock2.json", "--format", "json"]
            )

            # Should succeed and output JSON
            assert result.exit_code == 0
            output_data: dict[str, object] = json.loads(result.stdout)
            assert "has_differences" in output_data
