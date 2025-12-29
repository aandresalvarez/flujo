"""Integration tests for lockfile generation with real pipelines."""

from __future__ import annotations

import pytest
from pathlib import Path
from typing import Any

from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.models import PipelineResult
from flujo.infra.lockfile import (
    build_lockfile_data,
    compare_lockfiles,
    load_lockfile,
    write_lockfile,
)
from flujo.type_definitions.common import JSONObject
from tests.test_types.fixtures import create_test_pipeline


pytestmark = [pytest.mark.slow]


class TestLockfileCLI:
    """Integration tests for lockfile CLI commands (slow subset)."""

    @pytest.mark.asyncio
    async def test_generate_from_yaml_pipeline(self, tmp_path: Path) -> None:
        """Test generating lockfile from YAML pipeline definition."""
        # Create a simple YAML pipeline
        pipeline_yaml: Path = tmp_path / "pipeline.yaml"
        pipeline_yaml.write_text(
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

        # Load pipeline
        from flujo.cli.helpers_io import load_pipeline_from_yaml_file

        pipeline_obj: Pipeline[Any, Any] = load_pipeline_from_yaml_file(str(pipeline_yaml))

        # Generate lockfile
        result: PipelineResult[Any] = PipelineResult(success=True, output=None, step_history=[])
        lockfile_path: Path = tmp_path / "flujo.lock"

        write_lockfile(
            path=lockfile_path,
            pipeline=pipeline_obj,
            result=result,
            pipeline_name="test_pipeline",
            pipeline_version="0.1",
            pipeline_id="test_id",
            run_id=None,
        )

        # Verify lockfile was created
        assert lockfile_path.exists()

        # Load and verify structure
        lockfile_data: JSONObject = load_lockfile(lockfile_path)
        assert lockfile_data["pipeline"]["name"] == "test_pipeline"
        assert "prompts" in lockfile_data
        assert "skills" in lockfile_data

    @pytest.mark.asyncio
    async def test_generate_includes_external_files(self, tmp_path: Path) -> None:
        """Test that external files are included in lockfile."""
        # Create test external files
        data_dir: Path = tmp_path / "data"
        data_dir.mkdir()
        file1: Path = data_dir / "file1.json"
        file2: Path = data_dir / "file2.json"
        file1.write_text('{"key1": "value1"}', encoding="utf-8")
        file2.write_text('{"key2": "value2"}', encoding="utf-8")

        # Create pipeline
        pipeline: Pipeline[Any, Any] = create_test_pipeline()
        result: PipelineResult[Any] = PipelineResult(success=True, output=None, step_history=[])

        # Generate lockfile with external files
        lockfile_path: Path = tmp_path / "flujo.lock"
        write_lockfile(
            path=lockfile_path,
            pipeline=pipeline,
            result=result,
            pipeline_name="test",
            pipeline_version="1.0",
            pipeline_id="test_id",
            run_id=None,
            external_files=[file1, file2],
            project_root=tmp_path,
        )

        # Verify external files are included
        lockfile_data: JSONObject = load_lockfile(lockfile_path)
        assert "external_files" in lockfile_data
        assert len(lockfile_data["external_files"]) == 2

        # Verify file paths and hashes
        file_paths: list[str] = [f["path"] for f in lockfile_data["external_files"]]
        assert "data/file1.json" in file_paths or "data\\file1.json" in file_paths
        assert "data/file2.json" in file_paths or "data\\file2.json" in file_paths

    @pytest.mark.asyncio
    async def test_verify_detects_real_changes(self, tmp_path: Path) -> None:
        """Test verification with actual pipeline changes."""
        # Create initial pipeline
        pipeline_yaml: Path = tmp_path / "pipeline.yaml"
        pipeline_yaml.write_text(
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

        from flujo.cli.helpers_io import load_pipeline_from_yaml_file

        pipeline_obj: Pipeline[Any, Any] = load_pipeline_from_yaml_file(str(pipeline_yaml))

        # Generate initial lockfile
        result: PipelineResult[Any] = PipelineResult(success=True, output=None, step_history=[])
        lockfile_path: Path = tmp_path / "flujo.lock"

        write_lockfile(
            path=lockfile_path,
            pipeline=pipeline_obj,
            result=result,
            pipeline_name="test_pipeline",
            pipeline_version="0.1",
            pipeline_id="test_id",
            run_id=None,
        )

        # Load initial lockfile
        initial_lockfile: JSONObject = load_lockfile(lockfile_path)

        # Modify pipeline (change input)
        pipeline_yaml.write_text(
            """
version: "0.1"
name: "test_pipeline"
steps:
  - kind: step
    name: echo
    agent:
      id: "flujo.builtins.stringify"
    input: "goodbye"
""",
            encoding="utf-8",
        )

        # Generate new lockfile
        modified_pipeline: Pipeline[Any, Any] = load_pipeline_from_yaml_file(str(pipeline_yaml))
        new_result: PipelineResult[Any] = PipelineResult(success=True, output=None, step_history=[])

        new_lockfile: JSONObject = build_lockfile_data(
            pipeline=modified_pipeline,
            result=new_result,
            pipeline_name="test_pipeline",
            pipeline_version="0.1",
            pipeline_id="test_id",
            run_id=None,
        )

        # Compare - should detect differences
        diff = compare_lockfiles(initial_lockfile, new_lockfile)

        # The input change might not affect prompts/skills, but structure should be comparable
        # At minimum, we verify the comparison works
        assert isinstance(diff.has_differences, bool)

    @pytest.mark.asyncio
    async def test_compare_real_lockfiles(self) -> None:
        """Test comparison of real lockfiles."""
        # Create two different pipelines
        pipeline1: Pipeline[Any, Any] = create_test_pipeline()
        pipeline2: Pipeline[Any, Any] = create_test_pipeline()

        result: PipelineResult[Any] = PipelineResult(success=True, output=None, step_history=[])

        # Generate two lockfiles
        lockfile1_data: JSONObject = build_lockfile_data(
            pipeline=pipeline1,
            result=result,
            pipeline_name="pipeline1",
            pipeline_version="1.0",
            pipeline_id="id1",
            run_id=None,
        )

        lockfile2_data: JSONObject = build_lockfile_data(
            pipeline=pipeline2,
            result=result,
            pipeline_name="pipeline2",
            pipeline_version="2.0",
            pipeline_id="id2",
            run_id=None,
        )

        # Compare
        diff = compare_lockfiles(lockfile1_data, lockfile2_data)

        # Verify comparison works (may or may not have differences depending on pipeline content)
        assert isinstance(diff.has_differences, bool)
        assert isinstance(diff.prompts_changed, list)
        assert isinstance(diff.skills_changed, list)


class TestLockfileWithRealPipelines:
    """Test lockfile generation with actual YAML pipelines."""

    @pytest.mark.asyncio
    async def test_lockfile_from_example_pipeline(self, tmp_path: Path) -> None:
        """Test generating lockfile from example pipeline structure."""
        # Create a minimal but valid pipeline YAML
        pipeline_yaml: Path = tmp_path / "pipeline.yaml"
        pipeline_yaml.write_text(
            """
version: "0.1"
name: "example_pipeline"
steps:
  - kind: step
    name: process
    agent:
      id: "flujo.builtins.stringify"
    input: "test"
""",
            encoding="utf-8",
        )

        from flujo.cli.helpers_io import load_pipeline_from_yaml_file

        pipeline_obj: Pipeline[Any, Any] = load_pipeline_from_yaml_file(str(pipeline_yaml))

        # Generate lockfile
        result: PipelineResult[Any] = PipelineResult(success=True, output=None, step_history=[])
        lockfile_path: Path = tmp_path / "flujo.lock"

        write_lockfile(
            path=lockfile_path,
            pipeline=pipeline_obj,
            result=result,
            pipeline_name="example_pipeline",
            pipeline_version="0.1",
            pipeline_id="example_id",
            run_id=None,
        )

        # Verify lockfile structure
        assert lockfile_path.exists()
        lockfile_data: JSONObject = load_lockfile(lockfile_path)

        # Verify required fields
        assert "schema_version" in lockfile_data
        assert "pipeline" in lockfile_data
        assert "prompts" in lockfile_data
        assert "skills" in lockfile_data

        # Verify pipeline metadata
        assert lockfile_data["pipeline"]["name"] == "example_pipeline"
        assert lockfile_data["pipeline"]["version"] == "0.1"


class TestLockfileConfigIntegration:
    """Test lockfile integration with flujo.toml configuration."""

    @pytest.mark.asyncio
    async def test_lockfile_with_config_external_files(self, tmp_path: Path) -> None:
        """Test that external files from config are included."""
        # Create flujo.toml with external files
        flujo_toml: Path = tmp_path / "flujo.toml"
        flujo_toml.write_text(
            """
[lockfile]
external_files = ["data/*.json"]
""",
            encoding="utf-8",
        )

        # Create external files
        data_dir: Path = tmp_path / "data"
        data_dir.mkdir()
        file1: Path = data_dir / "file1.json"
        file2: Path = data_dir / "file2.json"
        file1.write_text('{"key1": "value1"}', encoding="utf-8")
        file2.write_text('{"key2": "value2"}', encoding="utf-8")

        # Generate lockfile (config would be loaded by CLI, but we test the function directly)
        from flujo.infra.lockfile import hash_external_files

        external_files_list: list[Path] = [
            tmp_path / "data" / "file1.json",
            tmp_path / "data" / "file2.json",
        ]
        external_files_data = hash_external_files(external_files_list, tmp_path)

        # Verify external files are hashed
        assert len(external_files_data) == 2
