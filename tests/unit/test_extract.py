"""
Unit tests for extract.py script functionality.
Tests all the fixes and edge cases to ensure robust behavior.
"""

import os
import pytest
from unittest.mock import patch
import sys

# Add the project root to the path to import extract
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from extract import (
    should_exclude_path,
    should_exclude_file,
    build_tree_structure,
    DEFAULT_EXCLUDE_FOLDERS,
    DEFAULT_EXCLUDE_FOLDER_PATTERNS,
    EXCLUDED_FILE_PATTERNS,
    GIT_SPECIFIC_FILES,
)


class TestExtractPathExclusion:
    """Test path exclusion logic with various scenarios."""

    def test_exact_folder_exclusion(self):
        """Test that exact folder names are excluded."""
        path_parts = ["project", "src", "__pycache__", "module"]
        result = should_exclude_path(
            path_parts, DEFAULT_EXCLUDE_FOLDERS, DEFAULT_EXCLUDE_FOLDER_PATTERNS
        )
        assert result is True

    def test_folder_pattern_exclusion(self):
        """Test that glob patterns for folders work correctly."""
        path_parts = ["project", "src", "mypackage.egg-info", "module"]
        result = should_exclude_path(
            path_parts, DEFAULT_EXCLUDE_FOLDERS, DEFAULT_EXCLUDE_FOLDER_PATTERNS
        )
        assert result is True

    def test_no_exclusion(self):
        """Test that valid paths are not excluded."""
        path_parts = ["project", "src", "valid_module"]
        result = should_exclude_path(
            path_parts, DEFAULT_EXCLUDE_FOLDERS, DEFAULT_EXCLUDE_FOLDER_PATTERNS
        )
        assert result is False

    def test_specific_path_exclusion_exact_match(self):
        """Test that specific path exclusions work with exact matches."""
        path_parts = ["project", "tests"]
        exclude_specific_paths = ["tests"]
        result = should_exclude_path(
            path_parts,
            DEFAULT_EXCLUDE_FOLDERS,
            DEFAULT_EXCLUDE_FOLDER_PATTERNS,
            exclude_specific_paths,
            "project",
        )
        assert result is True

    def test_specific_path_exclusion_subdirectory(self):
        """Test that specific path exclusions work with direct subdirectories."""
        path_parts = ["project", "tests", "unit"]
        exclude_specific_paths = ["tests"]
        result = should_exclude_path(
            path_parts,
            DEFAULT_EXCLUDE_FOLDERS,
            DEFAULT_EXCLUDE_FOLDER_PATTERNS,
            exclude_specific_paths,
            "project",
        )
        assert result is True

    def test_specific_path_exclusion_no_match(self):
        """Test that similar prefixes don't cause false exclusions."""
        path_parts = ["project", "tests_backup"]
        exclude_specific_paths = ["tests"]
        result = should_exclude_path(
            path_parts,
            DEFAULT_EXCLUDE_FOLDERS,
            DEFAULT_EXCLUDE_FOLDER_PATTERNS,
            exclude_specific_paths,
            "project",
        )
        assert result is False

    def test_specific_path_exclusion_nested_no_match(self):
        """Test that nested paths with similar prefixes don't cause false exclusions."""
        path_parts = ["project", "tests_backup", "unit"]
        exclude_specific_paths = ["tests"]
        result = should_exclude_path(
            path_parts,
            DEFAULT_EXCLUDE_FOLDERS,
            DEFAULT_EXCLUDE_FOLDER_PATTERNS,
            exclude_specific_paths,
            "project",
        )
        assert result is False


class TestExtractFileExclusion:
    """Test file exclusion logic."""

    def test_file_pattern_exclusion(self):
        """Test that file patterns are correctly excluded."""
        # Test various file patterns
        test_cases = [
            ("module.pyc", True),
            ("test.pyo", True),
            ("data.sqlite", True),
            ("app.log", True),
            ("settings.json", True),
            ("*.ipynb", True),
            ("docker-compose.override.yml", True),
            ("uv.lock", True),
            ("output.md", True),
            # Files that should NOT be excluded
            ("main.py", False),
            ("config.toml", False),
            ("README.md", False),
            ("docker-compose.yml", False),  # This should be included
            ("Dockerfile", False),  # This should be included
            ("entrypoint.sh", False),  # This should be included
            (".env.docker", False),  # This should be included (fix for docker env files)
        ]

        for filename, should_exclude in test_cases:
            result = should_exclude_file(filename, EXCLUDED_FILE_PATTERNS)
            assert result == should_exclude, f"File {filename} exclusion test failed"


class TestExtractDockerFileHandling:
    """Test that Docker files are handled correctly."""

    def test_docker_files_not_excluded(self):
        """Test that Docker files are not incorrectly excluded."""
        docker_files = [
            "Dockerfile",
            "docker-compose.yml",
            "docker-compose.yaml",
            "entrypoint.sh",
            ".env.docker",
        ]

        for filename in docker_files:
            result = should_exclude_file(filename, EXCLUDED_FILE_PATTERNS)
            assert result is False, f"Docker file {filename} should not be excluded"


class TestExtractTreeStructure:
    """Test tree structure building functionality."""

    @patch("extract.os.walk")
    def test_build_tree_structure_basic(self, mock_walk):
        """Test basic tree structure building."""
        # Mock file system structure
        mock_walk.return_value = [
            ("/project", ["src", "tests"], ["main.py", "config.toml"]),
            ("/project/src", [], ["module.py"]),
            ("/project/tests", [], ["test_module.py"]),
        ]

        tree = build_tree_structure(
            "/project",
            DEFAULT_EXCLUDE_FOLDERS,
            DEFAULT_EXCLUDE_FOLDER_PATTERNS,
            [".py", ".toml"],
            ["Dockerfile"],
            EXCLUDED_FILE_PATTERNS,
        )

        assert "." in tree
        assert "src" in tree
        assert "tests" in tree
        assert "main.py" in tree["."]
        assert "config.toml" in tree["."]
        assert "module.py" in tree["src"]
        assert "test_module.py" in tree["tests"]

    @patch("extract.os.walk")
    def test_build_tree_structure_with_exclusions(self, mock_walk):
        """Test tree structure building with exclusions."""
        # Mock file system structure with excluded folders
        mock_walk.return_value = [
            ("/project", ["src", "__pycache__", "tests"], ["main.py"]),
            ("/project/src", [], ["module.py"]),
            ("/project/__pycache__", [], ["module.pyc"]),  # This should be excluded
            ("/project/tests", [], ["test_module.py"]),
        ]

        tree = build_tree_structure(
            "/project",
            DEFAULT_EXCLUDE_FOLDERS,
            DEFAULT_EXCLUDE_FOLDER_PATTERNS,
            [".py", ".toml"],
            ["Dockerfile"],
            EXCLUDED_FILE_PATTERNS,
        )

        # __pycache__ should be excluded
        assert "__pycache__" not in tree
        # Other folders should be included
        assert "." in tree
        assert "src" in tree
        assert "tests" in tree

    @patch("extract.os.walk")
    def test_build_tree_structure_with_file_pattern_exclusions(self, mock_walk):
        """Test tree structure building with file pattern exclusions."""
        # Mock file system structure with excluded files
        mock_walk.return_value = [
            ("/project", ["src"], ["main.py", "main.pyc", "config.toml"]),
            ("/project/src", [], ["module.py", "module.pyo"]),
        ]

        tree = build_tree_structure(
            "/project",
            DEFAULT_EXCLUDE_FOLDERS,
            DEFAULT_EXCLUDE_FOLDER_PATTERNS,
            [".py", ".toml"],
            ["Dockerfile"],
            EXCLUDED_FILE_PATTERNS,
        )

        # .pyc and .pyo files should be excluded
        assert "main.pyc" not in tree["."]
        assert "module.pyo" not in tree["src"]
        # Valid files should be included
        assert "main.py" in tree["."]
        assert "config.toml" in tree["."]
        assert "module.py" in tree["src"]


class TestExtractIntegration:
    """Integration tests for the extract script."""

    def test_default_exclude_folders_structure(self):
        """Test that DEFAULT_EXCLUDE_FOLDERS contains only exact matches."""
        for folder in DEFAULT_EXCLUDE_FOLDERS:
            # Ensure no glob patterns in folder exclusions
            assert "*" not in folder, f"Folder exclusion '{folder}' contains glob pattern"

    def test_default_exclude_folder_patterns_structure(self):
        """Test that DEFAULT_EXCLUDE_FOLDER_PATTERNS contains only glob patterns."""
        for pattern in DEFAULT_EXCLUDE_FOLDER_PATTERNS:
            # Ensure glob patterns are in pattern exclusions
            assert "*" in pattern, f"Folder pattern '{pattern}' should contain glob pattern"

    def test_excluded_file_patterns_structure(self):
        """Test that EXCLUDED_FILE_PATTERNS contains appropriate patterns."""
        # Test that docker env files are not incorrectly excluded
        assert "*.env.docker" not in EXCLUDED_FILE_PATTERNS, (
            "Docker env files should not be excluded"
        )

        # Test that important patterns are present
        important_patterns = ["*.pyc", "*.log", "*.sqlite", "*.ipynb"]
        for pattern in important_patterns:
            assert pattern in EXCLUDED_FILE_PATTERNS, f"Important pattern '{pattern}' missing"

    def test_git_specific_files_structure(self):
        """Test that GIT_SPECIFIC_FILES contains appropriate files."""
        expected_git_files = {".gitignore", ".gitattributes", ".gitmodules"}
        assert GIT_SPECIFIC_FILES == expected_git_files


class TestExtractEdgeCases:
    """Test edge cases and error conditions."""

    def test_should_exclude_path_with_empty_parts(self):
        """Test path exclusion with empty path parts."""
        result = should_exclude_path([], DEFAULT_EXCLUDE_FOLDERS, DEFAULT_EXCLUDE_FOLDER_PATTERNS)
        assert result is False

    def test_should_exclude_file_with_empty_patterns(self):
        """Test file exclusion with empty patterns."""
        result = should_exclude_file("test.py", set())
        assert result is False

    def test_should_exclude_file_with_none_patterns(self):
        """Test file exclusion with None patterns."""
        result = should_exclude_file("test.py", None)
        assert result is False

    def test_should_exclude_path_with_none_parameters(self):
        """Test path exclusion with None parameters."""
        result = should_exclude_path(["test"], None, None)
        assert result is False


if __name__ == "__main__":
    pytest.main([__file__])
