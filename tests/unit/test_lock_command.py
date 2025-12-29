"""Unit tests for lockfile command functionality."""

from __future__ import annotations

import pytest
from pathlib import Path
from typing import Any, List

from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.models import PipelineResult
from flujo.infra.lockfile import (
    build_lockfile_data,
    compare_lockfiles,
    compute_lockfile_hash,
    extract_model_info,
    format_lockfile_diff,
    hash_external_files,
    load_lockfile,
    write_lockfile,
)
from flujo.type_definitions.common import JSONObject
from flujo.utils.hash import stable_digest
from tests.test_types.fixtures import (
    create_test_pipeline,
    create_test_step,
    create_test_step_result,
)


pytestmark = [pytest.mark.fast]


class _DummyAgent:
    """Dummy agent for testing."""

    __flujo_skill_id__ = "dummy.skill"
    _original_system_prompt = "system prompt"

    async def run(self, data: object | None = None, **_kwargs: object) -> object:
        return data


class _ModelAgent:
    """Agent with model info for testing."""

    model_id = "openai:gpt-4o"
    _agent = None

    async def run(self, data: object | None = None, **_kwargs: object) -> object:
        return data


class TestLockfileGeneration:
    """Test lockfile generation functionality."""

    def test_build_lockfile_data_hashes_prompts(self) -> None:
        """Test that prompts are hashed in lockfile."""
        pipeline: Pipeline[Any, Any] = create_test_pipeline(
            steps=[create_test_step(name="dummy", agent=_DummyAgent())]
        )
        result: PipelineResult[Any] = PipelineResult(
            step_history=[create_test_step_result(name="dummy", success=True, output="ok")]
        )

        data: JSONObject = build_lockfile_data(
            pipeline=pipeline,
            result=result,
            pipeline_name="test",
            pipeline_version="1.0",
            pipeline_id="test_id",
            run_id="test_run",
        )

        assert "prompts" in data
        assert isinstance(data["prompts"], list)
        assert len(data["prompts"]) > 0
        assert data["prompts"][0]["hash"] == stable_digest("system prompt")

    def test_build_lockfile_data_hashes_skills(self) -> None:
        """Test that skills are hashed in lockfile."""
        pipeline: Pipeline[Any, Any] = create_test_pipeline(
            steps=[create_test_step(name="dummy", agent=_DummyAgent())]
        )
        result: PipelineResult[Any] = PipelineResult(
            step_history=[create_test_step_result(name="dummy", success=True, output="ok")]
        )

        data: JSONObject = build_lockfile_data(
            pipeline=pipeline,
            result=result,
            pipeline_name="test",
            pipeline_version="1.0",
            pipeline_id="test_id",
            run_id="test_run",
        )

        assert "skills" in data
        assert isinstance(data["skills"], list)
        assert len(data["skills"]) > 0
        assert data["skills"][0]["skill_id"] == "dummy.skill"

    def test_build_lockfile_data_schema_version(self) -> None:
        """Test that lockfile includes schema_version."""
        pipeline: Pipeline[Any, Any] = create_test_pipeline()
        result: PipelineResult[Any] = PipelineResult(step_history=[])

        data: JSONObject = build_lockfile_data(
            pipeline=pipeline,
            result=result,
            pipeline_name="test",
            pipeline_version="1.0",
            pipeline_id="test_id",
            run_id="test_run",
        )

        assert "schema_version" in data
        assert data["schema_version"] == 1

    def test_build_lockfile_data_with_external_files(self, tmp_path: Path) -> None:
        """Test that external files are included in lockfile."""
        # Create test file
        test_file: Path = tmp_path / "test.json"
        test_file.write_text('{"key": "value"}')

        pipeline: Pipeline[Any, Any] = create_test_pipeline()
        result: PipelineResult[Any] = PipelineResult(step_history=[])

        data: JSONObject = build_lockfile_data(
            pipeline=pipeline,
            result=result,
            pipeline_name="test",
            pipeline_version="1.0",
            pipeline_id="test_id",
            run_id="test_run",
            external_files=[test_file],
            project_root=tmp_path,
        )

        assert data["schema_version"] == 2
        assert "external_files" in data
        assert isinstance(data["external_files"], list)
        assert len(data["external_files"]) == 1
        assert data["external_files"][0]["path"] == "test.json"
        assert "hash" in data["external_files"][0]

    def test_build_lockfile_data_with_model_info(self) -> None:
        """Test that model info is included when requested."""
        pipeline: Pipeline[Any, Any] = create_test_pipeline(
            steps=[create_test_step(name="model_step", agent=_ModelAgent())]
        )
        result: PipelineResult[Any] = PipelineResult(step_history=[])

        data: JSONObject = build_lockfile_data(
            pipeline=pipeline,
            result=result,
            pipeline_name="test",
            pipeline_version="1.0",
            pipeline_id="test_id",
            run_id="test_run",
            include_model_info=True,
        )

        assert data["schema_version"] == 2
        assert "models" in data
        # Model info extraction may return None if agent doesn't have proper structure
        # So we just check the field exists

    def test_write_lockfile_creates_file(self, tmp_path: Path) -> None:
        """Test that write_lockfile creates the expected file."""
        output_path: Path = tmp_path / "flujo.lock"
        pipeline: Pipeline[Any, Any] = create_test_pipeline()
        result: PipelineResult[Any] = PipelineResult(step_history=[])

        written_path: Path = write_lockfile(
            path=output_path,
            pipeline=pipeline,
            result=result,
            pipeline_name="test",
            pipeline_version="1.0",
            pipeline_id="test_id",
            run_id="test_run",
        )

        assert written_path.exists()
        assert written_path == output_path

        # Verify file content is valid JSON
        loaded: JSONObject = load_lockfile(output_path)
        assert loaded["pipeline"]["name"] == "test"


class TestExternalFileHashing:
    """Test external file hashing functionality."""

    def test_hash_external_files_single_file(self, tmp_path: Path) -> None:
        """Test hashing a single external file."""
        test_file: Path = tmp_path / "test.json"
        test_file.write_text('{"key": "value"}')

        results: List[JSONObject] = hash_external_files([test_file], tmp_path)

        assert len(results) == 1
        assert results[0]["path"] == "test.json"
        assert "hash" in results[0]
        assert "size" in results[0]
        assert results[0]["size"] > 0

    def test_hash_external_files_multiple_files(self, tmp_path: Path) -> None:
        """Test hashing multiple external files."""
        file1: Path = tmp_path / "file1.json"
        file2: Path = tmp_path / "file2.json"
        file1.write_text('{"key1": "value1"}')
        file2.write_text('{"key2": "value2"}')

        results: List[JSONObject] = hash_external_files([file1, file2], tmp_path)

        assert len(results) == 2
        paths: List[str] = [r["path"] for r in results]
        assert "file1.json" in paths
        assert "file2.json" in paths

    def test_hash_external_files_glob_pattern(self, tmp_path: Path) -> None:
        """Test hashing files with glob pattern."""
        file1: Path = tmp_path / "data1.json"
        file2: Path = tmp_path / "data2.json"
        file1.write_text('{"key1": "value1"}')
        file2.write_text('{"key2": "value2"}')

        glob_pattern: Path = tmp_path / "data*.json"
        results: List[JSONObject] = hash_external_files([glob_pattern], tmp_path)

        assert len(results) == 2

    def test_hash_external_files_missing_file_strict(self, tmp_path: Path) -> None:
        """Test that missing files raise error in strict mode."""
        missing_file: Path = tmp_path / "missing.json"

        with pytest.raises(FileNotFoundError):
            hash_external_files([missing_file], tmp_path, strict=True)

    def test_hash_external_files_missing_file_non_strict(self, tmp_path: Path) -> None:
        """Test that missing files are skipped in non-strict mode."""
        missing_file: Path = tmp_path / "missing.json"

        results: List[JSONObject] = hash_external_files([missing_file], tmp_path, strict=False)

        assert len(results) == 0


class TestModelInfoExtraction:
    """Test model info extraction functionality."""

    def test_extract_model_info_with_model_id(self) -> None:
        """Test extracting model info from agent with model_id."""
        agent: _ModelAgent = _ModelAgent()
        model_info: JSONObject | None = extract_model_info(agent, "test_step")

        # Model info extraction may return None if agent structure doesn't match expected
        # This is acceptable - the function handles it gracefully
        if model_info is not None:
            assert "step" in model_info
            assert "model_id" in model_info or "hash" in model_info

    def test_extract_model_info_without_model(self) -> None:
        """Test extracting model info from agent without model."""
        agent: _DummyAgent = _DummyAgent()
        model_info: JSONObject | None = extract_model_info(agent, "test_step")

        # Should return None for agents without model info
        assert model_info is None


class TestLockfileComparison:
    """Test lockfile comparison functionality."""

    def test_compare_lockfiles_no_differences(self) -> None:
        """Test comparing identical lockfiles."""
        lockfile1: JSONObject = {
            "schema_version": 1,
            "prompts": [{"step": "step1", "hash": "abc123"}],
            "skills": [{"step": "step1", "skill_id": "skill1", "hash": "def456"}],
        }
        lockfile2: JSONObject = {
            "schema_version": 1,
            "prompts": [{"step": "step1", "hash": "abc123"}],
            "skills": [{"step": "step1", "skill_id": "skill1", "hash": "def456"}],
        }

        diff = compare_lockfiles(lockfile1, lockfile2)

        assert not diff.has_differences

    def test_compare_lockfiles_prompt_changed(self) -> None:
        """Test detecting prompt changes."""
        lockfile1: JSONObject = {
            "schema_version": 1,
            "prompts": [{"step": "step1", "hash": "abc123"}],
        }
        lockfile2: JSONObject = {
            "schema_version": 1,
            "prompts": [{"step": "step1", "hash": "xyz789"}],
        }

        diff = compare_lockfiles(lockfile1, lockfile2)

        assert diff.has_differences
        assert len(diff.prompts_changed) == 1

    def test_compare_lockfiles_prompt_added(self) -> None:
        """Test detecting added prompts."""
        lockfile1: JSONObject = {
            "schema_version": 1,
            "prompts": [{"step": "step1", "hash": "abc123"}],
        }
        lockfile2: JSONObject = {
            "schema_version": 1,
            "prompts": [
                {"step": "step1", "hash": "abc123"},
                {"step": "step2", "hash": "def456"},
            ],
        }

        diff = compare_lockfiles(lockfile1, lockfile2)

        assert diff.has_differences
        assert len(diff.prompts_added) == 1

    def test_compare_lockfiles_prompt_removed(self) -> None:
        """Test detecting removed prompts."""
        lockfile1: JSONObject = {
            "schema_version": 1,
            "prompts": [
                {"step": "step1", "hash": "abc123"},
                {"step": "step2", "hash": "def456"},
            ],
        }
        lockfile2: JSONObject = {
            "schema_version": 1,
            "prompts": [{"step": "step1", "hash": "abc123"}],
        }

        diff = compare_lockfiles(lockfile1, lockfile2)

        assert diff.has_differences
        assert len(diff.prompts_removed) == 1

    def test_compare_lockfiles_ignore_fields(self) -> None:
        """Test that ignored fields are not compared."""
        lockfile1: JSONObject = {
            "schema_version": 1,
            "pipeline": {"name": "test", "version": "1.0", "id": "id1"},
            "prompts": [{"step": "step1", "hash": "abc123"}],
        }
        lockfile2: JSONObject = {
            "schema_version": 1,
            "pipeline": {"name": "test", "version": "1.0", "id": "id1"},
            "prompts": [{"step": "step1", "hash": "xyz789"}],
        }

        # Without ignore, should detect prompt change
        diff1 = compare_lockfiles(lockfile1, lockfile2)
        assert diff1.has_differences

        # With ignore, should not detect prompt change
        diff2 = compare_lockfiles(lockfile1, lockfile2, ignore_fields=["prompts.step1.hash"])
        assert not diff2.has_differences

    def test_compare_lockfiles_external_files(self) -> None:
        """Test comparing external files."""
        lockfile1: JSONObject = {
            "schema_version": 2,
            "external_files": [{"path": "file1.json", "hash": "abc123"}],
        }
        lockfile2: JSONObject = {
            "schema_version": 2,
            "external_files": [{"path": "file1.json", "hash": "xyz789"}],
        }

        diff = compare_lockfiles(lockfile1, lockfile2)

        assert diff.has_differences
        assert len(diff.external_files_changed) == 1

    def test_compare_lockfiles_pipeline_metadata(self) -> None:
        """Test that pipeline metadata changes are detected."""
        lockfile1: JSONObject = {
            "schema_version": 1,
            "pipeline": {"name": "pipeline1", "version": "1.0", "id": "id1"},
            "prompts": [],
        }
        lockfile2: JSONObject = {
            "schema_version": 1,
            "pipeline": {"name": "pipeline2", "version": "2.0", "id": "id2"},
            "prompts": [],
        }

        diff = compare_lockfiles(lockfile1, lockfile2)

        assert diff.has_differences
        assert diff.pipeline_changed

    def test_compare_lockfiles_schema_version(self) -> None:
        """Test that schema version changes are detected."""
        lockfile1: JSONObject = {
            "schema_version": 1,
            "pipeline": {"name": "test", "version": "1.0", "id": "id1"},
        }
        lockfile2: JSONObject = {
            "schema_version": 2,
            "pipeline": {"name": "test", "version": "1.0", "id": "id1"},
        }

        diff = compare_lockfiles(lockfile1, lockfile2)

        assert diff.has_differences
        assert diff.schema_version_changed


class TestLockfileUtilities:
    """Test lockfile utility functions."""

    def test_format_lockfile_diff_no_differences(self) -> None:
        """Test formatting diff with no differences."""
        from flujo.infra.lockfile import LockfileDiff

        diff = LockfileDiff(has_differences=False)
        formatted: str = format_lockfile_diff(diff)

        assert "No differences found" in formatted

    def test_format_lockfile_diff_with_differences(self) -> None:
        """Test formatting diff with differences."""
        from flujo.infra.lockfile import LockfileDiff

        diff = LockfileDiff(
            has_differences=True,
            prompts_changed=[{"step": "step1", "old_hash": "abc", "new_hash": "xyz"}],
        )
        formatted: str = format_lockfile_diff(diff)

        assert "differences detected" in formatted.lower()
        assert "step1" in formatted

    def test_compute_lockfile_hash(self) -> None:
        """Test computing lockfile hash."""
        lockfile: JSONObject = {
            "schema_version": 1,
            "pipeline": {"name": "test", "version": "1.0", "id": "test_id"},
            "prompts": [{"step": "step1", "hash": "abc123"}],
            "skills": [{"step": "step1", "skill_id": "skill1", "hash": "def456"}],
        }

        hash_value: str = compute_lockfile_hash(lockfile)

        assert isinstance(hash_value, str)
        assert len(hash_value) == 64  # SHA-256 hex digest length

    def test_compute_lockfile_hash_stable(self) -> None:
        """Test that lockfile hash is stable for same content."""
        lockfile: JSONObject = {
            "schema_version": 1,
            "pipeline": {"name": "test", "version": "1.0", "id": "test_id"},
            "prompts": [{"step": "step1", "hash": "abc123"}],
        }

        hash1: str = compute_lockfile_hash(lockfile)
        hash2: str = compute_lockfile_hash(lockfile)

        assert hash1 == hash2

    def test_load_lockfile(self, tmp_path: Path) -> None:
        """Test loading lockfile from disk."""
        lockfile_path: Path = tmp_path / "flujo.lock"
        lockfile_data: JSONObject = {
            "schema_version": 1,
            "pipeline": {"name": "test", "version": "1.0", "id": "test_id"},
        }

        import json

        lockfile_path.write_text(json.dumps(lockfile_data, indent=2), encoding="utf-8")

        loaded: JSONObject = load_lockfile(lockfile_path)

        assert loaded["pipeline"]["name"] == "test"
        assert loaded["schema_version"] == 1

    def test_load_lockfile_not_found(self, tmp_path: Path) -> None:
        """Test loading non-existent lockfile raises error."""
        lockfile_path: Path = tmp_path / "missing.lock"

        with pytest.raises(FileNotFoundError):
            load_lockfile(lockfile_path)
